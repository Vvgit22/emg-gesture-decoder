# protocol_v1 — frozen historical benchmark

This directory holds the experiment records produced by the first version of
the pipeline (commits 4885b38..725760f): 1,158 per-split run JSONs under
`runs/` and their aggregation `results.json`.

It is frozen: nothing in this project may overwrite these files. New
experiments write to `artifacts/experiments/<name>/` and readers select the
experiment with `--experiment NAME` (default `protocol_v1` only where a
historical comparison is explicitly intended).

Known defects it carries, recorded honestly rather than re-run:

- model weights were initialised BEFORE the per-split RNG seed was set, so
  init order leaked fold-to-fold dependence (fixed in audit-refinements).
- cross_subject validation subjects were a fixed slice of the fold pool,
  not resampled per fold.
- run JSONs carry no provenance block (added later; loaders default to {}).

Kept for comparison; new numbers go to new experiment dirs.
