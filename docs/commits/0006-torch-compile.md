# Commit 6: torch.compile & static-shape benchmark

## TL;DR

Add `scripts/04_bench_compile.py` and a `COMPILE_COLUMNS` schema, benchmarking eager vs
`torch.compile` (default Inductor and reduce-overhead/CUDA-graphs) with cold-start and
steady-state measured separately. 3 new schema tests (45 pass) and a 12-row result CSV.
Headline: 1.3×–2.2× steady-state speedup, largest on the overhead-bound shape, at ~28 s
per-shape compile cost.

## Short explanation

For each (mode, shape) cell the harness times the first call at that shape on the wall clock
(cold start, includes compilation) and the post-warmup steady state with CUDA events, then
reports `speedup_vs_eager` at the same shape. Shapes are compiled statically so recompilation
cost is visible; eager runs first as the baseline.

## Longer explanation

Files changed:

- `scripts/04_bench_compile.py` — CLI benchmark (`--quick`, `--dtype`, `--modes`, `--seq-lens`,
  `--batch-sizes`, `--dynamic`, `--warmup`, `--iters`, `--output`).
- `src/esm2_perf/results.py` — `COMPILE_COLUMNS` schema.
- `tests/test_results.py` — `COMPILE_COLUMNS` + `BATCHING_COLUMNS` field tests and a
  no-duplicate-columns check.
- `docs/milestones/05-torch-compile.md`, `docs/METHODOLOGY.md` (compile section),
  `docs/review_notes/05-torch-compile-human-review.md`.
- `results/raw/compile_20260613T150026Z.csv` — 12-row result.

Design decisions:

- **Cold start on the wall clock, steady state on CUDA events.** Compile cost is host-side
  Python/Inductor work that device timing would miss; steady state is device work, so each is
  measured with the right clock.
- **Eager first, then compiled.** Eager steady latencies are stored per shape and used as the
  `speedup_vs_eager` denominator; Dynamo state is reset between modes so each mode's cold start
  reflects only its own compilation.
- **Static shapes by default.** `dynamic=False` makes per-shape recompilation explicit — the
  point of the "static shapes" study; `--dynamic` is available for the shape-generic path.
- **Honest accounting.** A one-time eager warmup absorbs CUDA init so eager cold start is fair;
  `speedup_vs_eager < 1` would be recorded rather than hidden.

## Commands run

- `.venv/bin/python -m pytest -q` — 45 passed.
- `.venv/bin/python scripts/04_bench_compile.py --quick` — succeeded (smoke; 1.32× steady).
- `.venv/bin/python scripts/04_bench_compile.py --modes eager default reduce-overhead
  --seq-lens 128 512 --batch-sizes 1 8` — succeeded; wrote 12-row CSV, no OOM.

## Validation

All tests pass. Steady-state speedups: 2.15× (default) / 2.20× (reduce-overhead) at seq 128
bs 1; 1.63× / 1.64× at seq 512 bs 8; 1.3× on the seq 512 bs 1 cell. Cold start ~28 s per shape
under static shapes; amortization needs ~340 calls (512/8) to ~2,500 calls (128/1). Memory
unchanged (~1.22–1.39 GB).

## Next steps

Milestone 6 — custom Triton masked-mean-pooling kernel (awaiting human approval).
