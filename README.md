# mlx-block-stream-swift (BlockStreamKit)

SSD weight streaming for block-structured MLX models on Apple silicon
(NEUROSTREAM-ACTIONS HV2). Per-block granule files → two resident block-group
slots, background pread refill, self-calibrating runtime gate
(`N ≥ B·F/(2·S)`) with automatic output-invisible resident fallback.

Extracted 2026-08-02 from the two receipted consumers after the second pinned
the seam: wan-core-mlx-swift `b45b879` (Module-injection bind) and
ltx-2-mlx-swift `bf0a2ba` (functional weight-dict bind). Receipts:
mlxengine-todo `probes/hv2_{stream_proto,wan_blockstreamer,ltx_blockstreamer}.out`.

The kit owns the model-agnostic core: granule layout + manifest + provenance,
slots + the P-C alias seam (with the ≤14-byte `Data`-inlining trap handled via
bind-time residents), the step-major IO thread, the gate, fallback mechanics,
and the poisoned-slot receipt hook. Consumers own: installing the returned
arrays into their model, routing their block loop through
`acquireGroup`/`releaseGroup` (a hand-written loop that bypasses them reads
unrefilled slots — garbage, not a crash), globals loading, and activation
semantics (expert switches, multi-stage gate thresholds).
