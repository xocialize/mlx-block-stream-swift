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

## Products

- **`BlockStreamKit`** — the streamer (everything above). Re-exports `GranuleIO`,
  so existing consumers keep their single import.
- **`GranuleIO`** — the IO substrate WITHOUT the streamer (v0.4.0,
  BLOCKSTREAM-EXPANSION-EVAL §1.3): granule layout/manifest + safetensors header
  parse, raw F_NOCACHE file IO (`GranuleFileIO`), the dtype bridge and P-C alias
  seam, and `GranuleSpillFile` — one tensor set as ONE contiguous self-describing
  16 KiB-aligned file, written straight from unified memory and pread straight
  back into freshly materialized arrays. For ACTIVATION state that must leave
  memory and come back bit-identical (no gate, no slots, no fallback — those are
  weight-streaming concepts). Spill files are structurally validated
  (magic/version/table/range), NOT hashed — manifest v2's per-file SHA-256 is
  for hosted trees read many times, not per-chunk multi-GiB spill traffic.

  ⚠️ Measured caveat (2026-08-03, `probes/v12b3g_bank_granule_io.out`): the
  first candidate customer — SeedVR2's VAE bank eviction — A/B'd this format
  against its safetensors store through the shipping path and found the
  COMPOSITE indifferent on a 128 GB host: the round trip was already at disk
  rate, and V12-B3's "serialization-bound / +48%" decomposition did not
  reproduce (in-regime eviction cost ≈+7%, the +48% was session drift). The
  spill file is correct (bit-identical through two eviction boundaries at 40
  tile positions, all refusal gates) and lives on that port's
  `granule-bank-store` branch; the open case for it is memory-constrained
  hosts, where a buffered-IO store floods the page cache with the spill
  traffic and F_NOCACHE invariance should matter — unmeasured until a 16 GB
  arm exists. Don't cite this product as a proven speedup; cite it as the
  honest-IO spill substrate.
