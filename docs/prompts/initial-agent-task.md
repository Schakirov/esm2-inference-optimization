You are working in auto mode inside an AWS `g6.2xlarge` instance with an NVIDIA L4 GPU. The repository goal is to build a public portfolio project demonstrating credible ML performance engineering / kernel optimization for biotech AI, focused on ESM2 protein-language-model inference and embedding extraction.

Project thesis:

> Reproducible performance engineering for ESM2 protein embedding extraction on NVIDIA L4: baselines, profiling, batching/padding optimization, `torch.compile` experiments, correctness checks, and one small custom Triton kernel for bio-relevant embedding postprocessing.

Important constraints:

* Do **not** push to GitHub.
* You may create commits locally.
* Commit reasonably often after coherent, working increments.
* Never commit secrets, AWS credentials, tokens, cache files, model weights, virtualenvs, profiler megafiles, or large downloaded artifacts.
* Keep the repo publishable and readable.
* Prefer correctness, reproducibility, and clear documentation over heroic-looking but fragile custom kernels.
* Make the repo useful even if some later optimizations do not beat PyTorch/Inductor.
* Use Hugging Face `transformers` to load ESM2 rather than cloning Meta ESM as the working base.
* Target model for first real benchmark: `facebook/esm2_t33_650M_UR50D`.
* Hardware target: single NVIDIA L4, 24 GB VRAM.
* Main workload: protein sequence embedding extraction, including per-token output and masked mean-pooled sequence embeddings.

Before making changes:

1. Inspect the current repo.
2. Run `git status`.
3. Detect Python, CUDA, PyTorch, GPU, disk space, and available memory.
4. Create or update `.gitignore` to exclude large/generated files.
5. Create a short implementation plan in `docs/PLAN.md`.
6. Then proceed milestone by milestone.

Documentation requirements:

For every commit, create or update a corresponding document under:

```text
docs/commits/
```

Use names like:

```text
0001-repo-scaffold.md
0002-gpu-environment-check.md
0003-baseline-benchmark.md
```

Each commit document must contain:

```markdown
# Commit N: <title>

## TL;DR

Short 2–4 sentence summary.

## Short explanation

What changed and why.

## Longer explanation

Detailed technical explanation of design decisions, files changed, assumptions, limitations, and tradeoffs.

## Commands run

List important commands and whether they succeeded.

## Validation

What was tested, what passed, what failed, and what remains unverified.

## Next steps

Immediate next useful tasks.
```

For every milestone, create or update a document under:

```text
docs/milestones/
```

Use names like:

```text
01-environment-and-scaffold.md
02-baseline-esm2-inference.md
03-batching-and-padding.md
04-compile-and-static-shapes.md
05-triton-pooling-kernel.md
06-final-report.md
```

Each milestone document must start with:

```markdown
# Milestone X: <title>

## TL;DR

## Short explanation

## Longer explanation

## Results

## Validation

## Limitations

## Next steps
```

Also maintain:

```text
README.md
docs/PLAN.md
docs/RESULTS.md
docs/METHODOLOGY.md
docs/LIMITATIONS.md
docs/REPRODUCIBILITY.md
```

The README should eventually be portfolio-quality and include:

* project motivation;
* why ESM2/protein embeddings are relevant;
* hardware and software environment;
* benchmark task;
* baseline results;
* optimization results;
* correctness checks;
* limitations;
* reproduction commands;
* what this project does **not** claim.

Milestone 0 — Safety, repo inspection, and scaffold

Tasks:

* Inspect repo structure.
* Run `git status`.
* Create `.gitignore`.
* Create folders:

```text
src/esm2_perf/
scripts/
tests/
docs/
docs/commits/
docs/milestones/
results/
results/raw/
results/processed/
```

* Add `pyproject.toml` or `requirements.txt`.
* Add basic package structure.
* Add `README.md` draft.
* Add `docs/PLAN.md`.

Commit after this milestone.

Suggested commit message:

```text
Initialize ESM2 L4 performance project scaffold
```

Milestone 1 — Environment and GPU check

Create:

```text
scripts/00_check_environment.py
```

It should print:

* Python version;
* platform;
* PyTorch version;
* CUDA version according to PyTorch;
* whether CUDA is available;
* GPU name;
* GPU capability;
* total GPU memory;
* current allocated/reserved memory;
* selected package versions;
* optional `nvidia-smi` summary if available.

Also create:

```text
docs/milestones/01-environment-and-scaffold.md
```

Run the script and save a small text output under:

```text
results/raw/environment.txt
```

Commit after successful run.

Suggested commit message:

```text
Add reproducible L4 environment check
```

Milestone 2 — Baseline ESM2 inference benchmark

Create a clean baseline benchmark script:

```text
scripts/01_bench_baseline.py
```

Requirements:

* Load `facebook/esm2_t33_650M_UR50D` via Hugging Face.
* Generate synthetic protein sequences using the 20 standard amino acids.
* Use `torch.inference_mode()`.
* Benchmark at least:

```text
dtype: fp16, bf16 if supported, optionally fp32 for small cases
sequence lengths: 128, 256, 512, 1022
batch sizes: 1, 2, 4, 8, 16 where possible
```

* Use CUDA events for timing.
* Include warmup iterations.
* Synchronize correctly.
* Capture OOMs gracefully.
* Save CSV results to:

```text
results/raw/baseline_<timestamp>.csv
```

Columns should include at least:

```text
model_name
gpu_name
dtype
batch_size
seq_len
actual_tokens
latency_ms
tokens_per_sec
sequences_per_sec
max_memory_allocated_gb
oom
notes
```

Add a small helper module if useful:

```text
src/esm2_perf/timing.py
src/esm2_perf/sequences.py
src/esm2_perf/results.py
```

Add tests for synthetic sequence generation and result writing.

Run a **small smoke benchmark first**, not the full matrix, to verify functionality. Then run a reasonable baseline matrix that will not take forever.

Commit after the baseline works and results are saved.

Suggested commit message:

```text
Add baseline ESM2 inference benchmark
```

Milestone 3 — Correctness harness and embedding pooling

Create:

```text
src/esm2_perf/pooling.py
tests/test_pooling.py
scripts/02_check_correctness.py
```

Implement:

* extraction of last hidden states;
* removal or masking of special/padding tokens;
* masked mean pooling over real amino-acid tokens;
* correctness comparison between equivalent implementations.

The correctness script should compare:

* eager fp32 small case;
* fp16/bf16 outputs against fp32 with reasonable tolerances;
* pooled embeddings from different implementations.

Document numerical tolerances and why exact equality is not expected.

Commit after tests pass.

Suggested commit message:

```text
Add ESM2 embedding correctness checks
```

Milestone 4 — Variable-length protein batching and padding efficiency

Create:

```text
src/esm2_perf/batching.py
scripts/03_bench_batching.py
```

Implement and benchmark:

* random batching;
* length-sorted batching;
* bucketed batching, for example buckets around 128, 256, 512, 1022;
* dynamic padding inside batch/bucket.

Use synthetic variable-length sequences, and optionally add support for FASTA input via Biopython if straightforward.

Metrics:

```text
total real amino-acid tokens
total padded tokens
padding waste ratio
wall-clock time
real amino-acid tokens/sec
padded tokens/sec
sequences/sec
max GPU memory
```

This is important: distinguish **real biological tokens/sec** from padded compute tokens/sec.

Save results to:

```text
results/raw/batching_<timestamp>.csv
```

Update docs and README with the concept of padding waste.

Commit after working benchmark and docs.

Suggested commit message:

```text
Benchmark ESM2 variable-length batching strategies
```

Milestone 5 — `torch.compile` and static-shape bucket experiments

Create:

```text
scripts/04_bench_compile.py
docs/milestones/04-compile-and-static-shapes.md
```

Compare:

* eager baseline;
* `torch.compile` baseline;
* compile with dynamic-ish shapes;
* compile per fixed sequence bucket.

Be careful:

* warmup/compilation time should be separated from steady-state inference time;
* report compile overhead separately if possible;
* do not hide failed or slower results;
* document recompilation issues if observed.

The point is to show understanding of static vs dynamic shapes, not to force a win.

Save CSV:

```text
results/raw/compile_<timestamp>.csv
```

Commit after results and docs.

Suggested commit message:

```text
Add torch.compile static-shape benchmark
```

Milestone 6 — Custom Triton masked mean-pooling kernel

Create:

```text
src/esm2_perf/triton_pooling.py
tests/test_triton_pooling.py
scripts/05_bench_triton_pooling.py
```

Implement a small Triton kernel for masked mean pooling over ESM2 hidden states:

Input:

```text
hidden: [batch, seq_len, hidden_dim]
attention_mask or residue_mask: [batch, seq_len]
```

Output:

```text
pooled: [batch, hidden_dim]
```

Requirements:

* compare against PyTorch reference implementation;
* test multiple batch sizes, sequence lengths, hidden sizes if possible;
* use realistic ESM2 hidden size 1280 for the 650M model;
* check numerical tolerances;
* benchmark isolated pooling speed;
* optionally benchmark end-to-end model + pooling, clearly showing that pooling is only part of total runtime.

Do not overclaim. If PyTorch is faster or similar, document that honestly and explain why.

Commit after tests and benchmark work.

Suggested commit message:

```text
Add Triton masked mean pooling benchmark
```

Milestone 7 — Profiling notes

Add lightweight profiling support:

```text
scripts/06_profile_pytorch.py
docs/PROFILING.md
```

Use `torch.profiler` first. Optionally include instructions for Nsight Systems / Nsight Compute, but do not require them.

Capture:

* top CUDA kernels if available;
* CPU overhead;
* model forward vs tokenization vs pooling;
* where time is spent.

Do not commit huge profiler traces. Commit only small summarized text/markdown outputs.

Commit after profiling docs are useful.

Suggested commit message:

```text
Add profiling workflow and notes
```

Milestone 8 — Result processing and portfolio README

Create:

```text
scripts/07_summarize_results.py
```

It should read CSVs and produce markdown tables for:

```text
docs/RESULTS.md
```

Update README to be portfolio-quality.

README should include:

```markdown
# ESM2 L4 Inference Optimization

## TL;DR

## Why this project exists

## Why ESM2 and protein embeddings

## Hardware

## Methods

## Results

## Correctness

## Profiling

## What improved performance

## What did not improve performance

## Limitations

## Reproduce

## Future work
```

Commit after final docs.

Suggested commit message:

```text
Write portfolio results and reproduction docs
```

Milestone 9 — Final cleanup

Before final commit:

* Run tests.
* Run formatting if configured.
* Run `git status`.
* Check that no large files are staged.
* Check that no secrets are present.
* Check that all docs start with TL;DR where required.
* Ensure all generated results are reasonably small.
* Ensure the repo can be understood by someone reading only README + docs/METHODOLOGY.md + docs/RESULTS.md.

Commit final cleanup.

Suggested commit message:

```text
Finalize ESM2 L4 performance engineering portfolio repo
```

Development style:

* Prefer small, readable Python modules.
* Add docstrings where useful.
* Use type hints where they improve clarity.
* Keep benchmark scripts CLI-friendly.
* Print commands needed for reproduction.
* Make failures graceful: if GPU is unavailable, scripts should say so clearly.
* Use timestamps in result filenames.
* Add `--quick` mode to benchmark scripts for smoke tests.
* Add `--model-name`, `--dtype`, `--batch-sizes`, `--seq-lens`, and `--output` CLI args where reasonable.
* Keep tests small and fast.
* Use deterministic seeds for synthetic sequence generation.

Commit discipline:

After each coherent milestone or submilestone:

1. Run relevant tests or smoke commands.
2. Update the relevant `docs/commits/NNNN-*.md`.
3. Update milestone docs if applicable.
4. Run `git status`.
5. Stage only appropriate files.
6. Commit locally.
7. Do **not** push.

At the end, print:

* list of commits made;
* current `git status`;
* where results are saved;
* which milestones are complete;
* which milestones are partial;
* exact reproduction commands.



Additional instruction: human review gates

After each major milestone, stop and wait for my explicit instruction before starting the next major milestone.

A “major milestone” means:

* environment/scaffold complete;
* baseline ESM2 benchmark complete;
* correctness/pooling harness complete;
* batching/padding benchmark complete;
* torch.compile benchmark complete;
* Triton pooling kernel complete;
* profiling workflow complete;
* final README/results report complete.

Do not stop after every small internal change. You may make multiple local commits inside one milestone if that is useful, but stop at the milestone boundary.

At each milestone boundary:

1. Run relevant tests or smoke benchmarks.
2. Update the relevant `docs/milestones/XX-*.md`.
3. Update or create one file under `docs/review_notes/`, for example:

```text
docs/review_notes/02-baseline-human-review.md
```

4. The review note must include:

```markdown
# Human review: <milestone title>

## TL;DR

What was completed.

## What changed

Files and functionality added.

## What I should understand before continuing

Explain the key technical ideas in interview-defensible language.

## Commands I should run manually

List 2–5 commands that let me verify the milestone.

## Questions I should be able to answer

List concrete interview-style questions.

## Possible bugs or misleading benchmark artifacts

List things that could make the result invalid or overstated.

## Human notes

Leave this section empty for me to fill in.
```

5. Make a local git commit if the milestone is coherent and working.
6. Print:

   * commit hash;
   * files changed;
   * tests/commands run;
   * result files produced;
   * the review note path;
   * the exact next command I can give you to continue.

Important:

* Do not claim that human review has happened.
* Do not fill in the “Human notes” section.
* Do not push.
* Do not continue to the next major milestone until I explicitly say to continue.
