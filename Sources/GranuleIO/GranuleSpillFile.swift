// GranuleIO: the activation spill file — one tensor set as ONE contiguous
// self-describing file (BLOCKSTREAM-EXPANSION-EVAL §1.3): a fixed 16-byte
// preamble + one JSON table + 16 KiB-aligned payloads, written straight from
// each array's unified-memory backing and read back by pread-ing straight
// into a freshly materialized array's backing (the streamer's P-C alias seam).
//
// ⚠️ History, honestly (probes/v12b3g_bank_granule_io.out, 2026-08-03): this
// was built as the "contiguous single-buffer bank format" V12-B3 named after
// decomposing SeedVR2's +48% bank-eviction cost as SERIALIZATION-bound. The
// composite A/B through that port's shipping path FALSIFIED the decomposition:
// on a 128 GB host the safetensors round trip already ran at disk rate, the
// format changed wall clock by less than arm noise, and the +48% itself was
// session drift (in-regime cost ≈+7%). What this format still buys, by
// construction rather than by measurement: honest F_NOCACHE IO in both
// directions — a buffered-IO store pushes the entire spill volume through the
// page cache, which a memory-constrained host (the only kind that evicts)
// cannot absorb. That regime is the open receipt.
//
// What it deliberately is NOT:
//   - Not a granule tree: no manifest.json, no uniform-block requirement —
//     spill sets differ per site (edge tiles change shapes) and come and go
//     per index, so each file is self-describing.
//   - Not integrity-hashed: manifest v2's per-file SHA-256 exists for HOSTED
//     trees read many times; a spill file is written and re-read moments later
//     by the same process, and hashing multi-GiB traffic at SHA-256 rates
//     would hand back much of the serialization win. Refusal of foreign or
//     damaged files is STRUCTURAL: magic, version, header decode, per-tensor
//     dtype/shape/byte-count agreement, and range checks against the file size.
//   - Not durable interchange: version 1 is an intra-process scratch format.
//     Consumers layer their own semantics (e.g. the bank tail-count sentinel)
//     via `userInfo` and must keep enforcing them on read.
//
// Layout (little-endian):
//   [0..8)    magic "GRNSPILL"
//   [8..12)   UInt32 version = 1
//   [12..16)  UInt32 header JSON byte count H
//   [16..16+H) header JSON (SpillHeader: alignment, userInfo, tensors)
//   zero pad to alignUp(16+H, alignment) = data start
//   payloads at their `GranuleTensor.offset` — RELATIVE TO DATA START (unlike
//   granule trees, whose offsets are file-absolute; relative offsets keep the
//   header's own length out of its content), each alignment-aligned, in table
//   order, zero-padded between.

import Foundation
import MLX

public enum GranuleSpillFile {

    public static let magic: [UInt8] = Array("GRNSPILL".utf8)
    public static let version: UInt32 = 1
    /// Cap on the header table (a structural refusal bound, not a real limit —
    /// a million-tensor table fits comfortably).
    static let maxHeaderBytes = 64 << 20

    struct SpillHeader: Codable {
        let alignment: Int
        let userInfo: [String: String]
        let tensors: [GranuleTensor]  // offsets relative to data start
    }

    public struct Contents {
        public let userInfo: [String: String]
        /// In file/table order, exactly as written.
        public let tensors: [(key: String, array: MLXArray)]
    }

    // MARK: - Write

    /// Serialize `tensors` (evaluating them first — an unevaluated tail would
    /// otherwise pin its whole producing graph) into one contiguous file.
    /// Payload bytes leave each array's unified-memory backing directly; the
    /// only per-tensor host cost is the write syscall itself.
    ///
    /// - Returns: total file bytes (header + padding + payloads).
    @discardableResult
    public static func write(
        tensors: [(key: String, array: MLXArray)],
        userInfo: [String: String] = [:],
        to url: URL
    ) throws -> Int {
        eval(tensors.map(\.array))

        // Table first: relative offsets depend only on payload sizes.
        let alignment = GranuleFileIO.alignment
        var entries: [GranuleTensor] = []
        var cursor = 0
        for (key, array) in tensors {
            cursor = alignUp(cursor, alignment)
            entries.append(
                GranuleTensor(
                    key: key, dtype: try safetensorsDType(array.dtype),
                    shape: array.shape, offset: cursor, nbytes: array.nbytes))
            cursor += array.nbytes
        }
        let header = SpillHeader(alignment: alignment, userInfo: userInfo, tensors: entries)
        let encoder = JSONEncoder()
        encoder.outputFormatting = .sortedKeys
        let json = try encoder.encode(header)

        var preamble = Data(capacity: 16 + json.count)
        preamble.append(contentsOf: magic)
        preamble.append(contentsOf: withUnsafeBytes(of: version.littleEndian) { Array($0) })
        preamble.append(
            contentsOf: withUnsafeBytes(of: UInt32(json.count).littleEndian) { Array($0) })
        preamble.append(json)
        let dataStart = alignUp(preamble.count, alignment)

        let fd = try GranuleFileIO.openWrite(url.path)
        defer { close(fd) }
        let pad = [UInt8](repeating: 0, count: alignment)

        try preamble.withUnsafeBytes { raw in
            try GranuleFileIO.writeFull(fd: fd, from: raw.baseAddress!, count: raw.count)
        }
        var written = preamble.count
        func padTo(_ target: Int) throws {
            while written < target {
                let n = min(alignment, target - written)
                try pad.withUnsafeBytes { raw in
                    try GranuleFileIO.writeFull(fd: fd, from: raw.baseAddress!, count: n)
                }
                written += n
            }
        }
        try padTo(dataStart)

        for ((_, array), entry) in zip(tensors, entries) {
            try padTo(dataStart + entry.offset)
            guard entry.nbytes > 0 else { continue }
            // No-copy where the backing is contiguous (the usual case for a
            // settled tail); a transparent copy otherwise — for WRITING, unlike
            // refill, a copy is merely a memcpy, never a correctness hazard.
            let holder = array.asData(access: .noCopyIfContiguous)
            try holder.data.withUnsafeBytes { raw in
                guard raw.count == entry.nbytes else {
                    throw GranuleIOError.malformedSpill(
                        "asData returned \(raw.count) bytes for \(entry.key); expected \(entry.nbytes)")
                }
                try GranuleFileIO.writeFull(fd: fd, from: raw.baseAddress!, count: raw.count)
            }
            written += entry.nbytes
        }
        return written
    }

    // MARK: - Read

    /// Read a file written by ``write(tensors:userInfo:to:)``. Payloads ≥ the
    /// small-tensor threshold are pread straight into each array's freshly
    /// materialized backing (one host copy total); smaller ones take the
    /// copying constructor (the Data-inlining trap).
    ///
    /// - Throws: ``GranuleIOError/malformedSpill(_:)`` for anything that is not
    ///   a structurally valid version-1 spill file. Refusal here is load-
    ///   bearing: consumers install these tensors positionally.
    public static func read(from url: URL) throws -> Contents {
        let fd = try GranuleFileIO.openRead(url.path)
        defer { close(fd) }
        var st = stat()
        guard fstat(fd, &st) == 0 else {
            throw GranuleLayoutError.io("fstat(\(url.path)) errno \(errno)")
        }
        let fileSize = Int(st.st_size)
        guard fileSize >= 16 else {
            throw GranuleIOError.malformedSpill("\(fileSize) bytes — no room for a preamble")
        }

        var preamble = [UInt8](repeating: 0, count: 16)
        try preamble.withUnsafeMutableBytes { raw in
            try GranuleFileIO.preadFull(fd: fd, into: raw.baseAddress!, count: 16, offset: 0)
        }
        guard Array(preamble[0..<8]) == magic else {
            throw GranuleIOError.malformedSpill("bad magic — not a spill file")
        }
        let fileVersion = preamble[8..<12].withUnsafeBytes {
            UInt32(littleEndian: $0.loadUnaligned(as: UInt32.self))
        }
        guard fileVersion == version else {
            throw GranuleIOError.malformedSpill("version \(fileVersion); this reader is v\(version)")
        }
        let headerLen = Int(
            preamble[12..<16].withUnsafeBytes {
                UInt32(littleEndian: $0.loadUnaligned(as: UInt32.self))
            })
        guard headerLen > 0, headerLen <= maxHeaderBytes, 16 + headerLen <= fileSize else {
            throw GranuleIOError.malformedSpill("implausible header length \(headerLen)")
        }

        var json = Data(count: headerLen)
        try json.withUnsafeMutableBytes { raw in
            try GranuleFileIO.preadFull(
                fd: fd, into: raw.baseAddress!, count: headerLen, offset: 16)
        }
        let header: SpillHeader
        do {
            header = try JSONDecoder().decode(SpillHeader.self, from: json)
        } catch {
            throw GranuleIOError.malformedSpill("header decode failed: \(error)")
        }
        guard header.alignment > 0 else {
            throw GranuleIOError.malformedSpill("alignment \(header.alignment)")
        }
        let dataStart = alignUp(16 + headerLen, header.alignment)

        // Validate the whole table BEFORE materializing anything, so a bad
        // file is refused without side effects.
        var checked: [(GranuleTensor, DType)] = []
        for entry in header.tensors {
            let dtype: DType
            do {
                dtype = try mlxDType(safetensors: entry.dtype)
            } catch {
                throw GranuleIOError.malformedSpill("tensor \(entry.key): dtype \(entry.dtype)")
            }
            guard entry.shape.allSatisfy({ $0 >= 0 }) else {
                throw GranuleIOError.malformedSpill("tensor \(entry.key): shape \(entry.shape)")
            }
            let count = entry.shape.reduce(1, *)
            guard entry.nbytes == count * dtype.size else {
                throw GranuleIOError.malformedSpill(
                    "tensor \(entry.key): \(entry.nbytes) bytes for shape \(entry.shape) "
                        + "\(entry.dtype) — the file disagrees with itself")
            }
            guard entry.offset >= 0, dataStart + entry.offset + entry.nbytes <= fileSize else {
                throw GranuleIOError.malformedSpill(
                    "tensor \(entry.key): range \(entry.offset)+\(entry.nbytes) exceeds file")
            }
            checked.append((entry, dtype))
        }

        // Materialize all alias-path arrays in ONE eval, then refill in file
        // order (sequential offsets — the IO-friendly order).
        var arrays: [MLXArray?] = Array(repeating: nil, count: checked.count)
        var aliased: [(index: Int, entry: GranuleTensor)] = []
        for (i, (entry, dtype)) in checked.enumerated() {
            if entry.nbytes < granuleSmallTensorBytes {
                var bytes = Data(count: entry.nbytes)
                if entry.nbytes > 0 {
                    try bytes.withUnsafeMutableBytes { raw in
                        try GranuleFileIO.preadFull(
                            fd: fd, into: raw.baseAddress!, count: entry.nbytes,
                            offset: dataStart + entry.offset)
                    }
                }
                arrays[i] = MLXArray(bytes, entry.shape, dtype: dtype)
            } else {
                arrays[i] = MLXArray.zeros(entry.shape, dtype: dtype)
                aliased.append((i, entry))
            }
        }
        eval(aliased.compactMap { arrays[$0.index] })
        for (i, entry) in aliased {
            let ptr = try backingPointer(arrays[i]!, context: "spill \(entry.key)")
            try GranuleFileIO.preadFull(
                fd: fd, into: ptr, count: entry.nbytes, offset: dataStart + entry.offset)
        }

        return Contents(
            userInfo: header.userInfo,
            tensors: zip(header.tensors, arrays).map { ($0.key, $1!) })
    }

    private static func alignUp(_ value: Int, _ alignment: Int) -> Int {
        (value + alignment - 1) / alignment * alignment
    }
}
