// swift-tools-version: 6.2
// mlx-block-stream-swift — SSD weight streaming for block-structured MLX models
// (NEUROSTREAM-ACTIONS HV2, extracted after TWO consumers pinned the seam:
// wan-core-mlx-swift b45b879 (Module-injection bind) and ltx-2-mlx-swift
// bf0a2ba (functional weight-dict bind)). The model-agnostic ~80%: granule
// layout + manifest, the two-slot pread double-buffer, the self-calibrating
// runtime gate with output-invisible resident fallback, and the receipt hooks
// (poisoned-slot negative control). The irreducibly model-specific ~20% —
// installing arrays into the model, routing the block loop through the group
// windows, activation semantics (expert switch / two-stage threshold) — stays
// in each consumer behind `BlockStreamDelegate` + the public group-window API.
//
// Receipts inherited from the consumers: probes/hv2_stream_proto.out,
// hv2_wan_blockstreamer.out, hv2_ltx_blockstreamer.out (mlxengine-todo).

import PackageDescription

let package = Package(
    name: "BlockStreamKit",
    platforms: [
        .macOS(.v26)
    ],
    products: [
        .library(name: "BlockStreamKit", targets: ["BlockStreamKit"])
    ],
    dependencies: [
        // Floor matches the fleet (mlx-engine-swift ≥0.27.0 floors mlx-swift at
        // ≥0.31.5; wan-core validates 0.31.4 via its own resolve — upToNextMinor
        // from 0.31.3 lets every consumer keep its pin).
        .package(url: "https://github.com/ml-explore/mlx-swift.git", .upToNextMinor(from: "0.31.3"))
    ],
    targets: [
        .target(
            name: "BlockStreamKit",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift")
            ],
            path: "Sources/BlockStreamKit"
        )
    ]
)
