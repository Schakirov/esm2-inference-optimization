# Milestone 5: torch.compile & static shapes

## TL;DR

`scripts/04_bench_compile.py` compiles the ESM2 encoder with `torch.compile` and measures the
two numbers people usually conflate separately: **cold start** (first call at a shape — wall
clock, includes compilation) and **steady state** (CUDA-event-timed latency after warmup). On
the L4 in bf16, compilation delivers a real **1.3×–2.2× steady-state speedup** across shapes —
largest on the small overhead-bound shape (seq 128, batch 1: **2.15×** default, **2.20×**
reduce-overhead) and a solid **1.63×** on the largest compute-bound shape (seq 512, batch 8).
The catch is honest and large: each shape costs **~28 s to compile**, and with static shapes
every new `(batch, seq_len)` recompiles — so the win only pays off after hundreds-to-thousands
of calls per shape. Results: `results/raw/compile_20260613T150026Z.csv`.

## Short explanation

For each (mode, shape) cell the harness records cold start, steady-state latency, real-token
throughput, and `speedup_vs_eager` (eager steady ÷ compiled steady at the same shape). Modes
benchmarked: `eager` (reference), `default` (Inductor), and `reduce-overhead` (CUDA graphs).
Shapes are compiled statically (`dynamic=False`) so recompilation cost is visible. `eager` is
always run first to supply the speedup baseline; Dynamo state is reset between modes.

## Longer explanation

### Files added / changed

- `scripts/04_bench_compile.py` — CLI benchmark (`--quick`, `--dtype`, `--modes`, `--seq-lens`,
  `--batch-sizes`, `--dynamic`, `--warmup`, `--iters`, `--output`).
- `src/esm2_perf/results.py` — new `COMPILE_COLUMNS` schema.
- `tests/test_results.py` — schema tests for `COMPILE_COLUMNS` (and `BATCHING_COLUMNS`),
  plus a no-duplicate-columns check. 45 tests pass.
- `docs/METHODOLOGY.md` — new "torch.compile: cold start vs steady state" section.

### Results

Source: `results/raw/compile_20260613T150026Z.csv` (bf16, `dynamic=False`). Steady latency is
the median; cold start is the first call at that shape.

| mode | seq | bs | cold start (ms) | steady (ms) | tok/s | speedup |
|------|----:|---:|----------------:|------------:|------:|--------:|
| eager | 128 | 1 | 22 | 20.62 | 6,305 | — |
| default | 128 | 1 | 28,087 | 9.61 | 13,533 | **2.15×** |
| reduce-overhead | 128 | 1 | 28,472 | 9.37 | 13,881 | **2.20×** |
| eager | 128 | 8 | 39 | 46.55 | 22,344 | — |
| default | 128 | 8 | 28,671 | 34.07 | 30,523 | 1.37× |
| reduce-overhead | 128 | 8 | 28,325 | 33.78 | 30,783 | 1.38× |
| eager | 512 | 1 | 27 | 25.77 | 19,943 | — |
| default | 512 | 1 | 28,423 | 19.83 | 25,919 | 1.30× |
| reduce-overhead | 512 | 1 | 27,937 | 19.88 | 25,853 | 1.30× |
| eager | 512 | 8 | 220 | 225.72 | 18,217 | — |
| default | 512 | 8 | 29,412 | 138.69 | 29,650 | 1.63× |
| reduce-overhead | 512 | 8 | 27,799 | 137.71 | 29,860 | **1.64×** |

Observations:

- **Compile wins everywhere, most on the small shape.** Every compiled cell beats eager. The
  largest speedup is the overhead-bound seq 128 / batch 1 (2.15–2.20×): eager there is pinned
  on the ~20 ms launch/overhead floor (see [Milestone 2](02-baseline-esm2-inference.md)), and
  fusion + graph capture cut exactly that overhead. The compute-bound seq 512 / batch 8 still
  gains 1.63×, so this is real kernel improvement, not only launch savings.
- **reduce-overhead ≈ default, marginally better.** CUDA graphs edge out plain Inductor by a
  hair (2.20× vs 2.15×, 1.64× vs 1.63×) — a small extra cut to launch overhead, not a
  separate tier of win for this model on this GPU.
- **Cold start is ~28 s per shape and recurs.** With `dynamic=False`, each distinct
  `(batch, seq_len)` triggers its own ~28 s compile. Amortization is steep: the 512/8 cell
  saves ~87 ms/call, so it breaks even after ~340 calls; the 128/1 cell saves ~11 ms/call and
  needs ~2,500 calls. Compile pays off for high-volume, shape-stable serving — not for one-off
  or highly shape-diverse runs.
- **Memory is essentially unchanged** (~1.22–1.39 GB), in line with eager.

### Why this matters / honest framing

The steady-state speedups are genuine and worth having, but the headline cannot be "2.2×
faster" without the cold-start asterisk. Under static shapes the compile cost is paid *per
shape*; a variable-length protein workload hits many shapes, so in production you would either
compile with `dynamic=True` (fewer recompiles, often a smaller per-shape win), bucket sequence
lengths to a few fixed shapes, or pre-warm the compile cache for the shapes you serve.

## Validation

- `pytest -q` → 45 passed (42 prior + 3 new schema tests).
- `python scripts/04_bench_compile.py --quick` → eager + default at seq 128/bs 4 succeeded
  (1.32× steady speedup, ~35 s cold start).
- Full run (3 modes × seq {128,512} × batch {1,8}) → 12 rows, no OOM, no per-cell errors.

## Limitations

- Single dtype (bf16) and `dynamic=False` only in the committed run; `--dynamic` is supported
  but its (likely smaller) per-shape speedups are not yet tabled here.
- Cold start is wall-clock and includes Python/Inductor host work; it varies with machine load
  and is not claimed to the millisecond — it is reported to convey *order of magnitude* (~28 s).
- `max-autotune` was not run (much longer compile for a marginal expected gain); it is a
  supported `--modes` option.
- Speedups are steady-state, encoder-forward-only, on this L4 + PyTorch 2.12.0+cu130 stack;
  other GPUs / versions will differ.

## Next steps

Milestone 6 — a custom **Triton masked-mean-pooling kernel**: implement, unit-test against the
Milestone 3 reference, and benchmark against the PyTorch pooling — reporting honestly whether
it beats the eager/compiled baseline. **Awaiting human approval before starting.**
