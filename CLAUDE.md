# Claude Code instructions for this repository

This is a portfolio project for reproducible ML performance engineering on ESM2 protein-language-model inference using an NVIDIA L4 GPU.

Core goals:
- Build credible benchmark and optimization infrastructure.
- Prioritize correctness, reproducibility, profiler evidence, and documentation.
- Target `facebook/esm2_t33_650M_UR50D` first.
- Optimize ESM2 embedding extraction, batching, padding efficiency, torch.compile behavior, and masked mean pooling.
- Do not overclaim. Clearly document limitations.

Operational rules:
- Commit locally after coherent working milestones.
- Do not push.
- Do not commit secrets, model weights, virtualenvs, caches, huge profiler traces, or large generated artifacts.
- For every commit, update `docs/commits/`.
- For every milestone, update `docs/milestones/`.
- Docs should start with TL;DR, then short explanation, then longer explanation.
- Always run `git status` before committing.
- Prefer quick smoke tests before heavy benchmark runs.
