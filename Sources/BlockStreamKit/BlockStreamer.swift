// BlockStreamKit, part 2: the runtime (NEUROSTREAM-ACTIONS HV2).
// Extracted from the two receipted consumers — wan-core-mlx-swift b45b879
// (Module-injection bind, receipts probes/hv2_wan_blockstreamer.out) and
// ltx-2-mlx-swift bf0a2ba (functional weight-dict bind, receipts
// probes/hv2_ltx_blockstreamer.out) — after the second consumer pinned which
// parts are model-agnostic. What stays CONSUMER-side: installing the returned
// arrays into the model (dict construction vs `Module.update(parameters:)`),
// routing the model's block loop through `acquireGroup`/`releaseGroup`
// (public here — any hand-written loop that bypasses them reads unrefilled
// slots and produces GARBAGE, not a crash: the bernini `forwardMultiseg`
// lesson), loading non-block "global" tensors, and activation semantics
// (expert switches, two-stage thresholds).
//
// Streams a model's transformer blocks from per-block granule files through
// TWO resident block-group slots, refilled by a background pread thread while
// the GPU computes the other slot. Mechanics carried verbatim from the HV2
// prototype (do not re-derive):
//   - Slot-backed arrays are created ONCE at bind; refills mutate the backings
//     through the P-C alias (`asData(.noCopyIfContiguous)` — same pointer as
//     `asMTLBuffer(noCopy:true)`, works below page alignment).
//   - ⚠️ Swift `Data` INLINES payloads ≤14 bytes, so the alias silently misses
//     tiny tensors (refills write into a transient copy) — sub-256 B tensors
//     are bind-time residents via the COPYING constructor, never slot-backed.
//   - Plain params whose granule dtype ≠ computeDtype also load resident WITH
//     the cast (streaming them raw diverges from resident-load semantics);
//     quant components (packed weight / scales / biases) stream raw.
//   - STEP-MAJOR loop order: the cyclic group sequence advances continuously
//     across forwards; numGroups must be EVEN or the slot parity desyncs from
//     the (i/G)%2 binding across sweeps.
//   - Compute `eval`s at every group boundary BEFORE releasing the slot — the
//     handoff that makes refills invisible to the lazy graph (and the reason
//     streamed output is bit-identical to resident).
//   - IO runs on a plain nonisolated background thread and never touches MLX.
//   - Granules are read F_NOCACHE (bounded memory is the point; S stays honest).
//
// Self-calibrating runtime gate: stream hides iff S ≥ C(N) ⇔ N ≥ B·F/(2·S),
// with S measured on the first refill sweeps and F/C measured IN-REGIME during
// the first gating forward. Policy `.auto` falls back to fully-resident
// automatically (output-invisibly). `gateEvaluationThresholdTokens` defers the
// verdict until a forward at ≥ that many tokens (multi-stage pipelines gate on
// their LARGEST declared stage); `gateSuspended` exempts warmup/calibration
// sweeps whose N is pathological by design.
//
// Acceptance doctrine for consumers (from the LTX receipts): parity vs
// resident is memcmp WHERE THE RESIDENT PATH SELF-REPEATS BIT-EXACTLY at the
// same computeDtype; where it does not (some quantized kernel paths), the bar
// is the resident path's own measured repeat band — always paired with the
// poisoned-slot negative control so the compare keeps teeth.

import CryptoKit
import Foundation
import MLX
import Metal
import os

public enum BlockStreamError: LocalizedError {
    case noMetalDevice
    case aliasUnavailable(String)
    case pointerInstability(String)
    case contract(String)
    case state(String)

    public var errorDescription: String? {
        switch self {
        case .noMetalDevice: return "BlockStreamer: no Metal device"
        case .aliasUnavailable(let m):
            return "BlockStreamer: asMTLBuffer(noCopy:true) unavailable for \(m) — seam broken"
        case .pointerInstability(let m):
            return "BlockStreamer: slot backing pointer moved (\(m)) — refusing to stream"
        case .contract(let m): return "BlockStreamer contract violation: \(m)"
        case .state(let m): return "BlockStreamer state error: \(m)"
        }
    }
}

/// Safetensors dtype string → MLX DType (the subset block checkpoints use).
func mlxDType(safetensors dtype: String) throws -> DType {
    switch dtype {
    case "BF16": return .bfloat16
    case "F16": return .float16
    case "F32": return .float32
    case "U32": return .uint32
    case "U8": return .uint8
    case "I32": return .int32
    case "I64": return .int64
    default:
        throw BlockStreamError.contract("unsupported safetensors dtype \(dtype)")
    }
}

/// The raw backing pointer of an evaluated, contiguous MLXArray — the P-C alias
/// seam. A silent copy (non-contiguous backing) is detected by double
/// extraction: a wrapper returns the same address twice, a copy cannot.
/// (⚠️ NOT sufficient for ≤14-byte tensors — Data's inline storage can reuse
/// the same stack slot across identical calls; that's why sub-threshold
/// tensors never go through this path at all.)
func backingPointer(_ a: MLXArray, context: @autoclosure () -> String) throws
    -> UnsafeMutableRawPointer
{
    func extract() -> UnsafeMutableRawPointer? {
        let d = a.asData(access: .noCopyIfContiguous)
        return d.data.withUnsafeBytes { raw in
            raw.baseAddress.map { UnsafeMutableRawPointer(mutating: $0) }
        }
    }
    guard let p1 = extract(), let p2 = extract(), p1 == p2 else {
        throw BlockStreamError.aliasUnavailable(context())
    }
    return p1
}

/// Opt-in configuration for block streaming (consumer-constructed, per config).
public struct BlockStreamingOptions: Sendable {
    /// Blocks per granule group (slot capacity). Must divide the block count,
    /// and blockCount/groupSize must be EVEN (slot parity).
    public var groupSize: Int
    /// `.auto` — evaluate the runtime gate on the first gating forward and fall
    /// back to fully-resident when it doesn't clear. `.forceStream` — never
    /// fall back (receipts, parity tests).
    public var gatePolicy: GatePolicy
    /// Extra safety margin on the gate: require S ≥ C·margin.
    public var gateMargin: Double
    /// Suppress the one-line lifecycle prints.
    public var quiet: Bool

    public enum GatePolicy: String, Sendable {
        case auto
        case forceStream
    }

    public init(
        groupSize: Int = 2,
        gatePolicy: GatePolicy = .auto,
        gateMargin: Double = 1.0,
        quiet: Bool = false
    ) {
        self.groupSize = groupSize
        self.gatePolicy = gatePolicy
        self.gateMargin = gateMargin
        self.quiet = quiet
    }
}

/// What the gate measured and decided — surfaced for receipts and logs.
public struct StreamGateReport: Sendable {
    public var n: Int  // tokens in the gating forward
    public var sGiBs: Double  // measured refill bandwidth
    public var cGiBs: Double  // measured GPU weight-consumption rate C(N)
    public var nMin: Int  // N_min = N · C / S at the measured rates
    public var step1ComputeSeconds: Double
    public var step1StallSeconds: Double
    public var streaming: Bool  // the decision
}

/// How bind-time residents relate to the model's load semantics.
public enum ResidentCastPolicy: Sendable {
    /// Load raw granule bytes (consumers that never cast at load — wan-core).
    /// Partition then keeps every ≥-threshold tensor streamed regardless of dtype.
    case none
    /// Cast plain (non-quant) params to computeDtype, exactly as a resident
    /// `init` would (ltx-2's DiT.init semantics); such tensors residentize.
    case castPlainParams
}

/// One block's parameter arrays as bound by the kit: slot-backed streamed
/// tensors + bind-time residents, keys RELATIVE to the block (manifest keys).
/// The consumer maps them onto its model however it stores weights.
public struct BoundBlockArrays {
    public let block: Int
    public let arrays: [(key: String, array: MLXArray)]
}

public final class BlockStreamer: @unchecked Sendable {

    public enum Verdict: String, Sendable {
        case undecided
        case streaming
        case fellBack
    }

    /// Tensors smaller than this stay RESIDENT (loaded once per block at bind)
    /// instead of slot-backed — the Data-inlining trap plus syscall economy.
    public static let smallTensorResidentBytes = 256

    // Immutable layout after init.
    public let options: BlockStreamingOptions
    /// One granule directory per SET (dense models pass one; multi-expert
    /// models pass one per expert — sets share the slots, and exactly one
    /// set's IO is active at a time; `ensureActive(set:)` switches).
    public let granuleDirs: [URL]
    public let manifests: [GranuleManifest]
    public var granuleDir: URL { granuleDirs[0] }
    public var manifest: GranuleManifest { manifests[0] }
    public let blockCount: Int
    public let groupSize: Int
    public let numGroups: Int
    let allTemplate: [GranuleTensor]

    // Partitioned at bind (needs computeDtype) — see the header doctrine.
    private(set) var template: [GranuleTensor] = []
    private(set) var residentTemplate: [GranuleTensor] = []
    private var streamedKeys: Set<String> = []
    private(set) var tensorsPerBlock = 0
    private var boundComputeDtype: DType = .bfloat16
    private var boundCastPolicy: ResidentCastPolicy = .castPlainParams
    /// Raw streamed bytes per full sweep (bind-time residents excluded).
    public private(set) var sweepBytes = 0
    /// Resident slot footprint: 2 slots × groupSize blocks of parameters.
    public private(set) var slotResidentBytes = 0

    private let device: any MTLDevice
    private let lock = OSAllocatedUnfairLock()

    // Slot storage — allocated at bind, stable for the streamer's lifetime.
    private var slotArrays: [[MLXArray]] = []
    private var slotPtrs: [[UnsafeMutableRawPointer]] = []

    // Consumer seams, stored at bind: how to install resident arrays on
    // fallback (set, block, pairs), and how to detach the model's loop routing.
    private var installResident: ((Int, Int, [(String, MLXArray)]) throws -> Void)?
    private var onDetach: (() -> Void)?
    /// The granule set whose IO is (or was last) active. -1 before first activation.
    public private(set) var activeSet = -1

    private final class IOState: @unchecked Sendable {
        struct Refill {
            let fd: Int32
            let dst: UnsafeMutableRawPointer
            let offset: Int
            let nbytes: Int
        }

        let fds: [Int32]
        let plan: [[[Refill]]]  // [slot][group][refill]
        let free: [DispatchSemaphore]
        let ready: [DispatchSemaphore]
        let done = DispatchSemaphore(value: 0)
        let lock = OSAllocatedUnfairLock()
        var cancelled = false
        var ioBusySeconds: Double = 0
        var ioBytes: Int = 0
        var failure: String? = nil

        init(fds: [Int32], plan: [[[Refill]]]) {
            self.fds = fds
            self.plan = plan
            // Created at 0 and primed by signal() so dealloc is legal at any
            // rest value (a semaphore below its creation value at dealloc
            // traps in libdispatch).
            self.free = [DispatchSemaphore(value: 0), DispatchSemaphore(value: 0)]
            self.ready = [DispatchSemaphore(value: 0), DispatchSemaphore(value: 0)]
            free[0].signal()
            free[1].signal()
        }

        func note(busy: Double, bytes: Int) {
            lock.lock()
            ioBusySeconds += busy
            ioBytes += bytes
            lock.unlock()
        }

        var isCancelled: Bool {
            lock.lock()
            defer { lock.unlock() }
            return cancelled
        }

        func cancel() {
            lock.lock()
            cancelled = true
            lock.unlock()
        }
    }

    private var io: IOState?
    private var computeSeq = 0

    // Gate state.
    private var _verdict: Verdict = .undecided
    private var _gateReport: StreamGateReport? = nil
    private var forwardCompute: Double = 0
    private var forwardStall: Double = 0
    private var forwardTokens: Int = 0
    private var forwardGating = true
    private var groupOpenedAt: Double = 0

    /// Warmup / calibration sweeps set this so their pathological N never
    /// feeds the gate.
    public var gateSuspended = false

    /// Defer the gate verdict until the first forward with at least this many
    /// tokens (multi-stage: the pipeline sets the LARGEST stage's tokens).
    public var gateEvaluationThresholdTokens = 0

    public var verdict: Verdict {
        lock.lock()
        defer { lock.unlock() }
        return _verdict
    }

    public var gateReport: StreamGateReport? {
        lock.lock()
        defer { lock.unlock() }
        return _gateReport
    }

    /// Cumulative refill bandwidth over the current activation (the measured S).
    public var measuredSGiBs: Double {
        guard let io else { return 0 }
        io.lock.lock()
        defer { io.lock.unlock() }
        return io.ioBusySeconds > 0
            ? Double(io.ioBytes) / io.ioBusySeconds / 1_073_741_824 : 0
    }

    // MARK: - Init / bind

    public convenience init(granuleDir: URL, options: BlockStreamingOptions = .init()) throws {
        try self.init(granuleDirs: [granuleDir], options: options)
    }

    public init(granuleDirs: [URL], options: BlockStreamingOptions = .init()) throws {
        guard !granuleDirs.isEmpty else {
            throw BlockStreamError.contract("no granule directories")
        }
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw BlockStreamError.noMetalDevice
        }
        self.device = device
        self.options = options
        self.granuleDirs = granuleDirs
        self.manifests = try granuleDirs.map { try GranuleManifest.load(from: $0) }
        // Sets share slots, so every manifest must describe the identical layout.
        let first = manifests[0]
        for m in manifests.dropFirst() {
            let a = first.blocks[0].tensors.map { "\($0.key)|\($0.dtype)|\($0.shape)" }
            let b = m.blocks[0].tensors.map { "\($0.key)|\($0.dtype)|\($0.shape)" }
            guard m.blockCount == first.blockCount, a == b else {
                throw BlockStreamError.contract(
                    "granule sets differ in layout — cannot share slots")
            }
        }
        let manifest = first

        self.blockCount = manifest.blockCount
        self.groupSize = options.groupSize
        guard groupSize > 0, blockCount % groupSize == 0 else {
            throw BlockStreamError.contract(
                "groupSize \(groupSize) must divide blockCount \(blockCount)")
        }
        self.numGroups = blockCount / groupSize
        guard numGroups % 2 == 0 else {
            throw BlockStreamError.contract(
                "numGroups \(numGroups) must be even (slot parity) — adjust groupSize")
        }
        self.allTemplate = manifest.blocks[0].tensors
    }

    public func blockRange(_ group: Int) -> Range<Int> {
        (group * groupSize)..<((group + 1) * groupSize)
    }

    /// Allocate the slots, partition streamed vs bind-resident tensors, and
    /// return every block's parameter arrays for the consumer to install.
    ///
    /// - `computeDtype`: the model's compute dtype — decides which plain
    ///   params must residentize-with-cast (quant components stream raw).
    /// - `provenanceCheckpoint`: when given, the granule tree is refused if it
    ///   wasn't laid out from exactly this file (stale-tree trap).
    /// - `installResident`: stored; called per block by the `.auto` fallback
    ///   to swap freshly-loaded resident arrays into the model.
    /// - `onDetach`: stored; called when streaming ends (fallback or
    ///   `detach()`) so the consumer removes its loop routing.
    ///
    /// After installing, the consumer MUST call `verifyInstalled(lookup:)` —
    /// an install that copied instead of aliasing would read stale slots
    /// forever and no forward would ever crash.
    /// Single-set convenience (dense models): see `bind(computeDtype:castPolicy:...)`.
    public func bind(
        computeDtype: DType,
        provenanceCheckpoint: URL? = nil,
        installResident: @escaping (Int, [(String, MLXArray)]) throws -> Void,
        onDetach: @escaping () -> Void
    ) throws -> [BoundBlockArrays] {
        let perSet = try bind(
            computeDtype: computeDtype, castPolicy: .castPlainParams,
            provenanceCheckpoints: provenanceCheckpoint.map { [$0] },
            installResident: { _, block, pairs in try installResident(block, pairs) },
            onDetach: onDetach)
        return perSet[0]
    }

    public func bind(
        computeDtype: DType,
        castPolicy: ResidentCastPolicy,
        provenanceCheckpoints: [URL]? = nil,
        installResident: @escaping (Int, Int, [(String, MLXArray)]) throws -> Void,
        onDetach: @escaping () -> Void
    ) throws -> [[BoundBlockArrays]] {
        guard slotArrays.isEmpty else {
            throw BlockStreamError.state("bind() called twice")
        }
        if let checkpoints = provenanceCheckpoints {
            guard checkpoints.count == manifests.count else {
                throw BlockStreamError.contract(
                    "\(checkpoints.count) provenance checkpoints vs \(manifests.count) sets")
            }
            for (m, c) in zip(manifests, checkpoints) {
                try m.validateProvenance(against: c)
            }
        }
        boundComputeDtype = computeDtype
        boundCastPolicy = castPolicy
        self.installResident = installResident
        self.onDetach = onDetach

        // Partition (see header): stream iff ≥ threshold AND usable as-is.
        let keySet = Set(allTemplate.map(\.key))
        func isQuantComponent(_ key: String) -> Bool {
            key.hasSuffix(".scales") || key.hasSuffix(".biases")
                || (key.hasSuffix(".weight")
                    && keySet.contains(String(key.dropLast("weight".count)) + "scales"))
        }
        var streamed: [GranuleTensor] = []
        var resident: [GranuleTensor] = []
        for t in allTemplate {
            let raw = try mlxDType(safetensors: t.dtype)
            let usableAsIs =
                castPolicy == .none || raw == computeDtype || isQuantComponent(t.key)
            if t.nbytes >= Self.smallTensorResidentBytes, usableAsIs {
                streamed.append(t)
            } else {
                resident.append(t)
            }
        }
        template = streamed
        residentTemplate = resident
        streamedKeys = Set(streamed.map(\.key))
        tensorsPerBlock = streamed.count
        sweepBytes = streamed.reduce(0) { $0 + $1.nbytes } * blockCount
        slotResidentBytes = 2 * groupSize * streamed.reduce(0) { $0 + $1.nbytes }
        let totalBlockBytes = allTemplate.reduce(0) { $0 + $1.nbytes }
        guard sweepBytes >= totalBlockBytes * blockCount / 2 else {
            throw BlockStreamError.contract(
                "only \(sweepBytes / max(blockCount, 1)) of \(totalBlockBytes) bytes/block "
                    + "are streamable at computeDtype \(computeDtype) — checkpoint/compute "
                    + "dtype mismatch defeats streaming")
        }

        // Allocate slot arrays (zeros, eval'd, aliased — the prototype recipe).
        for _ in 0..<2 {
            var arrays: [MLXArray] = []
            var ptrs: [UnsafeMutableRawPointer] = []
            for _ in 0..<groupSize {
                for t in template {
                    let a = MLXArray.zeros(t.shape, dtype: try mlxDType(safetensors: t.dtype))
                    eval(a)
                    let ptr = try backingPointer(a, context: "slot \(t.key)")
                    if UInt(bitPattern: ptr) % 16384 == 0, t.nbytes % 16384 == 0,
                        let buf = a.asMTLBuffer(device: device, noCopy: true),
                        buf.contents() != ptr
                    {
                        throw BlockStreamError.pointerInstability(
                            "asData vs asMTLBuffer disagree for \(t.key)")
                    }
                    arrays.append(a)
                    ptrs.append(ptr)
                }
            }
            slotArrays.append(arrays)
            slotPtrs.append(ptrs)
        }

        // Per-set, per-block array sets: slot-backed streamed tensors are
        // SHARED across sets (block i of every set aliases the same slot
        // array); bind-time residents are per-set values.
        var out: [[BoundBlockArrays]] = []
        var blockResidentBytes = 0
        for (setIndex, setManifest) in manifests.enumerated() {
            var setOut: [BoundBlockArrays] = []
            for block in setManifest.blocks {
                let i = block.index
                let slot = (i / groupSize) % 2
                let base = (i % groupSize) * tensorsPerBlock
                var arrays: [(String, MLXArray)] = []
                for (t, entry) in template.enumerated() {
                    arrays.append((entry.key, slotArrays[slot][base + t]))
                }
                if !residentTemplate.isEmpty {
                    let fd = try GranuleIO.openRead(
                        granuleDirs[setIndex].appendingPathComponent(block.file).path,
                        noCache: false)
                    defer { close(fd) }
                    for t in block.tensors where !streamedKeys.contains(t.key) {
                        var bytes = Data(count: t.nbytes)
                        try bytes.withUnsafeMutableBytes { raw in
                            try GranuleIO.preadFull(
                                fd: fd, into: raw.baseAddress!, count: t.nbytes, offset: t.offset)
                        }
                        let raw = try mlxDType(safetensors: t.dtype)
                        var a = MLXArray(bytes, t.shape, dtype: raw)
                        if castPolicy == .castPlainParams, !isQuantComponent(t.key),
                            raw != computeDtype
                        {
                            a = a.asType(computeDtype)
                        }
                        eval(a)
                        arrays.append((t.key, a))
                        blockResidentBytes += t.nbytes
                    }
                }
                setOut.append(BoundBlockArrays(block: i, arrays: arrays))
            }
            out.append(setOut)
        }

        lock.lock()
        _verdict = options.gatePolicy == .forceStream ? .streaming : .undecided
        lock.unlock()
        say(String(
            format: "bound %d blocks · group=%d · slots 2×%.0f MiB (%.2f GiB resident) "
                + "· sweep %.2f GiB · block-resident %d KiB (%d tensors/block)",
            blockCount, groupSize,
            Double(slotResidentBytes) / 2 / 1_048_576,
            Double(slotResidentBytes) / 1_073_741_824,
            Double(sweepBytes) / 1_073_741_824,
            blockResidentBytes / 1024, residentTemplate.count))
        return out
    }

    /// Pointer-stability re-verification through the CONSUMER's own storage:
    /// every streamed key of every block, as the model will actually read it,
    /// must alias the exact recorded slot pointer.
    public func verifyInstalled(lookup: (Int, String) throws -> MLXArray?) throws {
        try verifyInstalledSets { _, i, key in try lookup(i, key) }
    }

    /// Multi-set: (set, block, key) — the SAME slot pointers must be reachable
    /// through every set's installed storage.
    public func verifyInstalledSets(lookup: (Int, Int, String) throws -> MLXArray?) throws {
        for setIndex in 0..<manifests.count {
            try verifySet(setIndex, lookup: lookup)
        }
    }

    private func verifySet(_ setIndex: Int, lookup: (Int, Int, String) throws -> MLXArray?) throws {
        for i in 0..<blockCount {
            let slot = (i / groupSize) % 2
            let base = (i % groupSize) * tensorsPerBlock
            for (t, entry) in template.enumerated() {
                guard let a = try lookup(setIndex, i, entry.key) else {
                    throw BlockStreamError.contract(
                        "consumer lookup returned nil for set \(setIndex) block \(i) \(entry.key)")
                }
                let ptr = try backingPointer(a, context: "\(entry.key) post-install")
                guard ptr == slotPtrs[slot][base + t] else {
                    throw BlockStreamError.pointerInstability("block \(i) \(entry.key)")
                }
            }
        }
    }

    // MARK: - Activation and the IO thread

    /// (Re)start the prefetch thread at group 0 for set 0 (dense models).
    /// Idempotent while that set's IO is live.
    public func ensureActive() {
        ensureActive(set: 0)
    }

    /// Make `set`'s granule files the streamed source, (re)starting the
    /// prefetch thread at group 0. Switching sets (an expert boundary) stops
    /// the old thread and starts the new one; same-set calls are idempotent.
    public func ensureActive(set: Int) {
        if set == activeSet, io != nil { return }
        stopIO()
        startIO(set: set)
    }

    private func startIO(set: Int) {
        precondition(io == nil)
        let setManifest = manifests[set]
        let setDir = granuleDirs[set]
        let fds: [Int32] = setManifest.blocks.map { block in
            let path = setDir.appendingPathComponent(block.file).path
            guard let fd = try? GranuleIO.openRead(path) else {
                fatalError("BlockStreamer: cannot open granule \(path)")
            }
            return fd
        }
        var plan: [[[IOState.Refill]]] = []
        for slot in 0..<2 {
            var groups: [[IOState.Refill]] = []
            for g in 0..<numGroups {
                var refills: [IOState.Refill] = []
                for lb in 0..<groupSize {
                    let bi = g * groupSize + lb
                    let streamed = setManifest.blocks[bi].tensors.filter {
                        streamedKeys.contains($0.key)
                    }
                    for (t, entry) in streamed.enumerated() {
                        refills.append(
                            IOState.Refill(
                                fd: fds[bi],
                                dst: slotPtrs[slot][lb * tensorsPerBlock + t],
                                offset: entry.offset,
                                nbytes: entry.nbytes))
                    }
                }
                groups.append(refills)
            }
            plan.append(groups)
        }
        let state = IOState(fds: fds, plan: plan)
        io = state
        activeSet = set
        computeSeq = 0

        let numGroups = self.numGroups
        let thread = Thread {
            var seq = 0
            while !state.isCancelled {
                let slot = seq % 2
                state.free[slot].wait()
                if state.isCancelled { break }
                let g = seq % numGroups
                let t0 = CFAbsoluteTimeGetCurrent()
                var bytes = 0
                for refill in state.plan[slot][g] {
                    do {
                        try GranuleIO.preadFull(
                            fd: refill.fd, into: refill.dst,
                            count: refill.nbytes, offset: refill.offset)
                    } catch {
                        state.lock.lock()
                        state.failure = "\(error)"
                        state.cancelled = true
                        state.lock.unlock()
                        state.ready[slot].signal()
                        state.done.signal()
                        return
                    }
                    bytes += refill.nbytes
                }
                state.note(busy: CFAbsoluteTimeGetCurrent() - t0, bytes: bytes)
                state.ready[slot].signal()
                seq += 1
            }
            state.done.signal()
        }
        thread.name = "BlockStreamKit.io"
        thread.qualityOfService = .userInitiated
        thread.start()
        say("streaming set \(set) from \(setDir.lastPathComponent)")
    }

    private func stopIO() {
        guard let state = io else { return }
        state.cancel()
        state.free[0].signal()
        state.free[1].signal()
        state.done.wait()
        for fd in state.fds { close(fd) }
        io = nil
    }

    /// Stop the prefetch thread and close granule files. Slots and bindings
    /// stay — a later forward reactivates.
    public func finish() {
        stopIO()
    }

    deinit {
        stopIO()
    }

    // MARK: - Group window (compute side, PUBLIC — the routed-loop contract)

    /// Wait until the current group's slot is refilled. Returns the slot index.
    @discardableResult
    public func acquireGroup() -> Int {
        guard let state = io else {
            fatalError("BlockStreamer: acquireGroup with no active IO")
        }
        let slot = computeSeq % 2
        let t0 = CFAbsoluteTimeGetCurrent()
        state.ready[slot].wait()
        if let failure = state.failure {
            fatalError("BlockStreamer IO failed: \(failure)")
        }
        if let poison = pendingPoison, poison.target == acquireCount {
            let entry = template[poison.tensor]
            memset(
                slotPtrs[slot][poison.localBlock * tensorsPerBlock + poison.tensor],
                0x55, min(poison.bytes, entry.nbytes))
            pendingPoison = nil
            say("poisoned slot \(slot) at acquire #\(acquireCount) (negative control)")
        }
        acquireCount += 1
        let now = CFAbsoluteTimeGetCurrent()
        forwardStall += now - t0
        groupOpenedAt = now
        return slot
    }

    /// Hand the group's slot back to the IO thread. The caller MUST have
    /// `eval`'d everything that reads the slot's arrays before calling this.
    public func releaseGroup() {
        guard let state = io else { return }
        let slot = computeSeq % 2
        forwardCompute += CFAbsoluteTimeGetCurrent() - groupOpenedAt
        state.free[slot].signal()
        computeSeq += 1
    }

    // MARK: - Gate

    public func beginForward(tokens: Int) {
        forwardCompute = 0
        forwardStall = 0
        forwardTokens = tokens
        forwardGating = !gateSuspended
    }

    /// Per-forward wrap-up: on the first GATING forward at or above the token
    /// threshold, evaluate the runtime gate from in-regime measurements and
    /// (policy `.auto`) fall back to fully resident when it doesn't clear.
    public func endForward() {
        lastForwardComputeSeconds = forwardCompute
        lastForwardStallSeconds = forwardStall
        guard verdict == .undecided, forwardGating,
            forwardTokens >= gateEvaluationThresholdTokens
        else { return }

        let s = measuredSGiBs
        let c = forwardCompute > 0
            ? Double(sweepBytes) / forwardCompute / 1_073_741_824 : .infinity
        let nMin = s > 0 ? Int((Double(forwardTokens) * c / s).rounded(.up)) : Int.max
        let clears = s >= c * options.gateMargin
        let report = StreamGateReport(
            n: forwardTokens, sGiBs: s, cGiBs: c, nMin: nMin,
            step1ComputeSeconds: forwardCompute,
            step1StallSeconds: forwardStall,
            streaming: clears)
        say(String(
            format: "gate: N=%d · S=%.2f GiB/s vs C(N)=%.2f GiB/s → N_min≈%d → %@",
            report.n, s, c, nMin,
            clears ? "STREAM (hidden)" : "FALL BACK (resident)"))

        lock.lock()
        _gateReport = report
        _verdict = clears ? .streaming : .fellBack
        lock.unlock()

        if !clears {
            do {
                try fallBackResident()
            } catch {
                say("fallback FAILED (\(error)) — continuing streamed")
                lock.lock()
                _verdict = .streaming
                lock.unlock()
            }
        }
    }

    public private(set) var lastForwardComputeSeconds: Double = 0
    public private(set) var lastForwardStallSeconds: Double = 0

    // MARK: - Fully-resident fallback

    /// Load every block resident from the granule files (COPYING constructor,
    /// plain params cast to the bound computeDtype), hand each block to the
    /// consumer's `installResident`, then detach. Streamed artifacts computed
    /// so far are bit-exact and remain valid. After this the streamer holds
    /// dead slots — release it (the slot-stranding lesson).
    public func fallBackResident() throws {
        guard let install = installResident else {
            throw BlockStreamError.state("fallBackResident before bind")
        }
        stopIO()
        let t0 = CFAbsoluteTimeGetCurrent()
        var loadedBytes = 0
        let keySet = Set(allTemplate.map(\.key))
        for setIndex in 0..<manifests.count {
        for (i, granule) in manifests[setIndex].blocks.enumerated() {
            let fd = try GranuleIO.openRead(
                granuleDirs[setIndex].appendingPathComponent(granule.file).path)
            defer { close(fd) }
            var pairs: [(String, MLXArray)] = []
            var arrays: [MLXArray] = []
            for t in granule.tensors {
                var bytes = Data(count: t.nbytes)
                try bytes.withUnsafeMutableBytes { raw in
                    try GranuleIO.preadFull(
                        fd: fd, into: raw.baseAddress!, count: t.nbytes, offset: t.offset)
                }
                let raw = try mlxDType(safetensors: t.dtype)
                var a = MLXArray(bytes, t.shape, dtype: raw)
                let quantComponent =
                    t.key.hasSuffix(".scales") || t.key.hasSuffix(".biases")
                    || (t.key.hasSuffix(".weight")
                        && keySet.contains(String(t.key.dropLast("weight".count)) + "scales"))
                if boundCastPolicy == .castPlainParams, !quantComponent,
                    raw != boundComputeDtype
                {
                    a = a.asType(boundComputeDtype)
                }
                pairs.append((t.key, a))
                arrays.append(a)
                loadedBytes += t.nbytes
            }
            eval(arrays)
            try install(setIndex, i, pairs)
        }
        }
        let dt = CFAbsoluteTimeGetCurrent() - t0
        say(String(
            format: "fell back resident: %.2f GiB in %.1fs (%.2f GiB/s)",
            Double(loadedBytes) / 1_073_741_824, dt,
            Double(loadedBytes) / dt / 1_073_741_824))
        onDetach?()
        lock.lock()
        _verdict = .fellBack
        lock.unlock()
    }

    // MARK: - Manifest v2: integrity + sidecar globals

    /// Recompute every granule file's SHA-256 (globals sidecar included) and
    /// compare against the manifest — the post-download check for HOSTED trees,
    /// where v1's source-checkpoint provenance anchor doesn't exist. Reads the
    /// full tree; call it once after materialization, not per launch. Throws on
    /// any mismatch, and on v1 trees (no hashes recorded).
    public func verifyIntegrity() throws {
        let bufBytes = 32 << 20
        var bufRaw: UnsafeMutableRawPointer? = nil
        guard posix_memalign(&bufRaw, 16384, bufBytes) == 0, let buf = bufRaw else {
            throw BlockStreamError.state("posix_memalign failed")
        }
        defer { free(buf) }

        for (setIndex, m) in manifests.enumerated() {
            var files = m.blocks
            if let g = m.globals { files.append(g) }
            for granule in files {
                guard let expected = granule.sha256 else {
                    throw BlockStreamError.contract(
                        "set \(setIndex) \(granule.file) has no sha256 — v1 tree; "
                            + "re-lay with a v2 kit for integrity verification")
                }
                let path = granuleDirs[setIndex].appendingPathComponent(granule.file).path
                let fd = try GranuleIO.openRead(path)
                defer { close(fd) }
                var hasher = SHA256()
                var offset = 0
                while offset < granule.bytes {
                    let n = min(bufBytes, granule.bytes - offset)
                    try GranuleIO.preadFull(fd: fd, into: buf, count: n, offset: offset)
                    hasher.update(bufferPointer: UnsafeRawBufferPointer(start: buf, count: n))
                    offset += n
                }
                let digest = hasher.finalize().map { String(format: "%02x", $0) }.joined()
                guard digest == expected else {
                    throw BlockStreamError.contract(
                        "set \(setIndex) \(granule.file) hash mismatch — corrupt or "
                            + "tampered tree (expected \(expected.prefix(12))…, "
                            + "got \(digest.prefix(12))…)")
                }
            }
        }
    }

    /// Load the globals SIDECAR of a v2 tree as raw arrays (COPYING
    /// constructor, granule dtypes verbatim — the consumer applies its own
    /// dialect strip and cast). Keys are the FULL on-disk names.
    public func loadGlobalTensors(set: Int = 0) throws -> [(key: String, array: MLXArray)] {
        guard let g = manifests[set].globals else {
            throw BlockStreamError.contract(
                "no globals sidecar in set \(set) — v1 tree; load globals from the "
                    + "source checkpoint or re-lay with a v2 kit")
        }
        let fd = try GranuleIO.openRead(
            granuleDirs[set].appendingPathComponent(g.file).path, noCache: false)
        defer { close(fd) }
        var out: [(String, MLXArray)] = []
        for t in g.tensors {
            var bytes = Data(count: t.nbytes)
            try bytes.withUnsafeMutableBytes { raw in
                try GranuleIO.preadFull(
                    fd: fd, into: raw.baseAddress!, count: t.nbytes, offset: t.offset)
            }
            let a = MLXArray(bytes, t.shape, dtype: try mlxDType(safetensors: t.dtype))
            out.append((t.key, a))
        }
        return out
    }

    // MARK: - Test / receipt hooks

    private struct PendingPoison {
        let target: Int
        let localBlock: Int
        let tensor: Int
        let bytes: Int
    }
    private var pendingPoison: PendingPoison? = nil
    private var acquireCount = 0

    /// Arm a one-shot poisoned-slot negative control (a parity compare that
    /// cannot see a bad refill is not evidence).
    public func armPoison(
        afterAcquires: Int, localBlock: Int = 0, tensor: Int = 0, bytes: Int = 1 << 20
    ) {
        pendingPoison = PendingPoison(
            target: acquireCount + afterAcquires, localBlock: localBlock,
            tensor: tensor, bytes: bytes)
    }

    /// Stop IO and detach WITHOUT loading anything resident — the receipts'
    /// between-arms reset. The consumer's arrays keep aliasing this streamer's
    /// slots, so no forward may run between `detach()` and model release.
    public func detach() {
        stopIO()
        onDetach?()
    }

    private func say(_ message: String) {
        if !options.quiet {
            print("[BlockStreamer] \(message)")
            fflush(stdout)
        }
    }
}
