// GranuleIO: the MLX-facing edge of the IO substrate — safetensors↔MLX dtype
// bridging and the P-C alias seam. Split out of BlockStreamer.swift when the
// `GranuleIO` product was carved off (activation spill/refill consumers need
// these without the streamer).

import Foundation
import MLX

public enum GranuleIOError: LocalizedError {
    case aliasUnavailable(String)
    case unsupportedDtype(String)
    case malformedSpill(String)

    public var errorDescription: String? {
        switch self {
        case .aliasUnavailable(let m):
            return "GranuleIO: asData(.noCopyIfContiguous) alias unavailable for \(m) — seam broken"
        case .unsupportedDtype(let m): return "GranuleIO: unsupported dtype \(m)"
        case .malformedSpill(let m): return "GranuleIO: malformed spill file: \(m)"
        }
    }
}

/// Safetensors dtype string → MLX DType (the subset block checkpoints use).
public func mlxDType(safetensors dtype: String) throws -> DType {
    switch dtype {
    case "BF16": return .bfloat16
    case "F16": return .float16
    case "F32": return .float32
    case "U32": return .uint32
    case "U8": return .uint8
    case "I32": return .int32
    case "I64": return .int64
    default:
        throw GranuleIOError.unsupportedDtype(dtype)
    }
}

/// MLX DType → safetensors dtype string — the writer-side inverse of
/// `mlxDType(safetensors:)`, kept to exactly the same subset so a spill file
/// never records a dtype its own reader would refuse.
public func safetensorsDType(_ dtype: DType) throws -> String {
    switch dtype {
    case .bfloat16: return "BF16"
    case .float16: return "F16"
    case .float32: return "F32"
    case .uint32: return "U32"
    case .uint8: return "U8"
    case .int32: return "I32"
    case .int64: return "I64"
    default:
        throw GranuleIOError.unsupportedDtype("\(dtype)")
    }
}

/// Tensors smaller than this never go through the alias seam — the Data
/// ≤14-byte inlining trap plus syscall economy. Streamer bind-residents and
/// spill-file reads share the one threshold.
public let granuleSmallTensorBytes = 256

/// The raw backing pointer of an evaluated, contiguous MLXArray — the P-C alias
/// seam. A silent copy (non-contiguous backing) is detected by double
/// extraction: a wrapper returns the same address twice, a copy cannot.
/// (⚠️ NOT sufficient for ≤14-byte tensors — Data's inline storage can reuse
/// the same stack slot across identical calls; that's why sub-threshold
/// tensors never go through this path at all.)
public func backingPointer(_ a: MLXArray, context: @autoclosure () -> String) throws
    -> UnsafeMutableRawPointer
{
    func extract() -> UnsafeMutableRawPointer? {
        let d = a.asData(access: .noCopyIfContiguous)
        return d.data.withUnsafeBytes { raw in
            raw.baseAddress.map { UnsafeMutableRawPointer(mutating: $0) }
        }
    }
    guard let p1 = extract(), let p2 = extract(), p1 == p2 else {
        throw GranuleIOError.aliasUnavailable(context())
    }
    return p1
}
