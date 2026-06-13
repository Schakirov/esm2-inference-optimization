# Human review: torch.compile & static-shape benchmark

## TL;DR

The fifth major milestone — *torch.compile benchmark complete* — is done.
`scripts/04_bench_compile.py` compiles the ESM2 encoder and measures cold start (first call at
a shape, includes compilation) and steady state (CUDA-event-timed after warmup) separately. On
the L4 in bf16, `torch.compile` gives a real **1.3×–2.2× steady-state speedup** — biggest on
the overhead-bound seq 128 / batch 1 cell — at a **~28 s per-shape compile cost** that recurs
for every new shape under static shapes. 45 unit tests pass. Results in
`results/raw/compile_20260613T150026Z.csv`. Nothing pushed.

## What changed

- `scripts/04_bench_compile.py` — eager vs `default` (Inductor) vs `reduce-overhead` (CUDA
  graphs); cold-start vs steady-state; `--dynamic` flag for shape-generic compile.
- `src/esm2_perf/results.py` — `COMPILE_COLUMNS` schema.
- `tests/test_results.py` — schema field tests + no-duplicate-columns check.
- `docs/milestones/05-torch-compile.md`, `docs/METHODOLOGY.md` (compile section),
  `docs/commits/0006-torch-compile.md`.
- `results/raw/compile_20260613T150026Z.csv` — 12-row result.

## What I should understand before continuing

- **Cold start vs steady state are different numbers, measured with different clocks.** Cold
  start is the first call at a shape on the wall clock — for compiled modes it is dominated by
  host-side compilation (~28 s), which device-side CUDA events would not capture. Steady state
  is the post-warmup CUDA-event latency. Quoting steady-state speedup without the cold-start
  cost would be the classic `torch.compile` overclaim.
- **Where the speedup comes from.** The largest win is the overhead-bound shape (seq 128 /
  batch 1, 2.15–2.20×): eager sits on the ~20 ms launch floor, and fusion + CUDA graphs cut
  exactly that. The compute-bound seq 512 / batch 8 still gains 1.63×, so it is real kernel
  improvement, not only launch savings.
- **Static shapes recompile.** With `dynamic=False`, every distinct `(batch, seq_len)` pays its
  own ~28 s compile. A variable-length protein workload touches many shapes — so in production
  you would use `dynamic=True`, bucket lengths to a few fixed shapes, or pre-warm the cache.
- **reduce-overhead ≈ default.** CUDA graphs edge out plain Inductor only marginally here, not a
  separate tier.

## Commands I should run manually

```bash
# Fast smoke (eager + default compile, one small shape) — note the ~30s cold start
.venv/bin/python scripts/04_bench_compile.py --quick

# A real slice (3 modes, two shapes) — several minutes due to compiles
.venv/bin/python scripts/04_bench_compile.py --modes eager default reduce-overhead --seq-lens 128 512 --batch-sizes 1 8 --output results/raw/_review.csv

# Inspect the committed results
column -s, -t results/raw/compile_20260613T150026Z.csv | less -S

# Unit tests
.venv/bin/python -m pytest -q
```

## Questions I should be able to answer

- Why measure cold start on the wall clock but steady state with CUDA events? (Compile cost is
  host-side Python/Inductor work; device timing would miss it. Steady state is device work.)
- Why is the speedup largest at seq 128 / batch 1? (It is overhead-bound on the ~20 ms launch
  floor; compile/CUDA-graphs cut launch overhead, which dominates there.)
- When does compiling actually pay off? (High-volume, shape-stable serving: the 512/8 cell
  breaks even after ~340 calls; the 128/1 cell after ~2,500. One-off or shape-diverse runs do
  not amortize the ~28 s/shape compile.)
- Why is reduce-overhead barely better than default here? (CUDA graphs remove a little extra
  launch overhead, but this model on this GPU does not have much left to remove after Inductor.)
- How would I make compile practical for variable-length inputs? (`dynamic=True`, length
  bucketing to a few shapes, or pre-warming the compile cache for served shapes.)

## Possible bugs or misleading benchmark artifacts

- **Cold start is approximate.** Wall-clock and includes host load; reported to convey order of
  magnitude (~28 s), not to the millisecond.
- **Steady state excludes the compile.** It is the right number for sustained serving but would
  badly mislead anyone running a model once — hence the cold-start column sitting right next to
  it.
- **Static shapes only in the committed run.** `dynamic=True` is supported but not yet tabled;
  its per-shape speedup is typically smaller.
- Encoder-forward-only, single dtype (bf16), this exact PyTorch 2.12.0+cu130 / L4 stack — other
  GPUs or versions will differ, possibly a lot.

## Human notes

_Add review outcome / approval here before Milestone 6 begins._
