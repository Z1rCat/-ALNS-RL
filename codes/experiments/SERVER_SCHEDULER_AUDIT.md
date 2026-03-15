# Server Scheduler Audit

This note audits the current server-facing experiment runner scripts under `codes/experiments` and recommends the right base for the final paper-grade server runner.

## Scope

There are 9 top-level server-facing runner/pipeline scripts in `codes/experiments`:

1. `run_experiments_server_unified.py`
2. `run_experiments_server_stream.py`
3. `run_experiments_server_adaptive.py`
4. `run_server_classic_baseline_suite.py`
5. `run_main_pipeline.py`
6. `run_recovery_ab_suite.py`
7. `run_v7_pipeline.py`
8. `run_experiments_server_protomem_tune.py`
9. `run_experiments_server_protomem_bo.py`

`run_experiments_common.py` is not a top-level runner, but it is the shared execution engine underneath the general schedulers. It is the real low-level foundation.

## Classification

### A. General-purpose server schedulers

| Script | Role | Strengths | Problems / Limits |
|---|---|---|---|
| `run_experiments_server_unified.py` | General server runner for arbitrary `(variant, dist, R, seed)` grids | Clean interface; directly uses `run_experiments_common`; supports `stage_mode`, checkpoint I/O, `resume_existing`, `skip_completed`, `precheck`; easiest to reason about | Variants are executed sequentially; no family-aware batching; no suite manifest/report; defaults still point to old off-grid distribution set |
| `run_experiments_server_stream.py` | Queue-style wrapper over `unified`; one subprocess per `(variant, dist, R, seed)` task | Keeps worker slots busy across variants; simple global queue; easy to resume partial large matrices | One Python subprocess per task adds overhead; weak suite semantics; no built-in family grouping or final report; effectively a wrapper on top of a wrapper |
| `run_experiments_server_adaptive.py` | Adaptive dispatcher with online duration model, autoscaling, retry, timeout, requeue | Most feature-rich scheduler; good when server pressure is unstable and task runtimes vary a lot | Too complex for a canonical paper runner; many runtime heuristics reduce transparency; defaults are heavy (`run_baseline`, `run_plots`, `run_metrics`, `cleanup_after_run` all on); harder to debug and harder to make reproducible |

### B. Suite / pipeline wrappers

| Script | Role | Strengths | Problems / Limits |
|---|---|---|---|
| `run_server_classic_baseline_suite.py` | Fixed classic comparison suite wrapper over `stream` | Reproducible one-command package; writes manifest; builds report | Hard-coded to the old 19-distribution table and fixed algorithm list; not suitable for the new symmetric transport-oriented matrix without editing code |
| `run_main_pipeline.py` | Small end-to-end pipeline: run -> probe -> phase0 summary -> plots | Convenient for quick targeted comparisons and diagnostics | Not designed for a large paper matrix; defaults are old PPO/PPO_NEW comparison cases; not family-structured |
| `run_recovery_ab_suite.py` | Three-branch suite for outer RL recovery pipeline | Useful for rollback/actor-recovery studies | Not a general baseline runner; tied to `outer_rl/run_edrl_pipeline.py` and its artifacts |
| `run_v7_pipeline.py` | Specialized V7 group protocol over mean pools | Useful for the specific pool-based V7 study | Research-line specific; not a general transport benchmark runner; uses its own group logic |

### C. Research-line-specific search / tuning scripts

| Script | Role | Strengths | Problems / Limits |
|---|---|---|---|
| `run_experiments_server_protomem_tune.py` | Fixed preset sweep for `PPO_PROTOMEM` | Good for prototype-memory preset screening | Algorithm-specific; tiny target set; not reusable as the main paper runner |
| `run_experiments_server_protomem_bo.py` | BO-style search for `PPO_PROTOMEM` | Useful for internal tuning | Algorithm-specific; objective is tuned for a very small internal target set; not suitable as the canonical experiment runner |

## What Each Script Really Does

### `run_experiments_server_unified.py`

This is the cleanest general execution entry point. It builds one `ExperimentConfig` per variant and passes it to `run_experiments_common.run_experiments(...)`. That means:

- execution, resume, lock, watchdog, baseline, metrics, and plotting all stay on the common infrastructure;
- the script is easy to inspect and easy to extend;
- it already supports the parameters that the final paper suite will need.

This is the best current base layer.

### `run_experiments_server_stream.py`

This is a higher-level global queue. It enumerates every `(variant, dist, R, seed)` cell and launches `run_experiments_server_unified.py` once per cell using a thread pool. Its value is scheduling fairness across variants and long-tail tasks. Its cost is process overhead and one more orchestration layer.

This is useful as an execution mode, but not ideal as the canonical script to keep extending.

### `run_experiments_server_adaptive.py`

This script bypasses the outer wrapper style and directly uses `run_experiments_common` planning plus its own dispatch policy, runtime prediction model, autoscaling, and requeue logic. It is powerful, but it is optimized for compute utilization rather than transparent paper execution. For a paper-grade suite, its behavior is too heuristic-heavy.

This should be treated as an advanced optional scheduler, not the main paper runner base.

## Real Run Lifecycle In `run_experiments_common.py`

No matter which top-level wrapper is used, the actual per-run execution path is mostly decided by `run_experiments_common.run_task(...)`.

The real stage order is:

1. Create / resume the `run_dir`
2. Acquire `run.lock` lease and start `heartbeat.json`
3. Start resource monitor
4. Run `Dynamic_master34959.py`
5. If baseline stage is enabled, run `run_benchmark_replay.py` once per policy:
   - `wait`
   - `reroute`
   - optional `random`
6. Run `codes/analysis/compute_metrics.py`
7. If plot stage is enabled, run `codes/plotting/plot_paper_figure.py`
8. If cleanup stage is enabled, run `codes/tools/cleanup_run.py`
9. Post-check expected artifacts
10. Write `DONE.json`
11. Write `resource_usage.json`
12. Release lease

This means the pipeline is not:

- generate one file -> test -> plot immediately

It is:

- finish master
- optionally finish all baselines
- then compute summary metrics
- then draw figures
- then optionally clean transient artifacts

## What Gets Created During A Run

The common engine plus the master/baseline scripts typically create:

- `meta.json`
- `console_output.txt`
- `rl_trace.csv`
- `rl_training.csv`
- `rl_summary.csv`
- `rl_decision.csv`
- `baseline_wait.csv`
- `baseline_reroute.csv`
- optional `baseline_random.csv`
- `run_status.json`
- `watchdog_events.jsonl`
- `heartbeat.json`
- `resource_usage.json`
- `metrics.json`
- `paper_figures/` when plot stage is on
- `DONE.json` on success
- `FAILED.json` on failure

The ALNS side also writes:

- `alns_outputs/...`
- `data/...`
- `post_stage/outer_path_map.csv`

## Watchdog / Retry / Scheduling Mechanics

The common engine already contains nontrivial protection logic:

- lease-based mutual exclusion via `run.lock`
- heartbeat refresh via `heartbeat.json`
- per-stage status tracking in `run_status.json`
- watchdog event stream in `watchdog_events.jsonl`
- output-growth watchdogs for master / baseline stages
- progress-file watchdogs for master / baseline stages
- per-stage retry budgets for:
  - master
  - baseline
  - metrics
  - plots
  - cleanup
- final post-check before `DONE.json`
- resource monitor that records CPU / RAM / GPU peaks

The scheduler layer differences are therefore mainly about:

- how tasks are enumerated
- how workers are filled
- whether heuristic dispatch is used

not about the inner run lifecycle itself.

## Current Default Matrixes In The Main Server Runners

`run_experiments_server_unified.py`

- default distributions: 16
- default seeds: 1 (`42`)
- default request numbers: 1 (`30`)
- algorithms: 1 per invocation unless multiple `--variant` are given

`run_experiments_server_stream.py`

- default distributions: 9
- default seeds: 1 (`42`)
- default request numbers: 1 (`30`)
- algorithms: any number of `--variant`, but each `(variant, dist, R, seed)` cell becomes one subprocess call to `unified`

`run_experiments_server_adaptive.py`

- default distributions: 16
- default seeds and variants depend on CLI input, but the script is designed for larger multi-cell scheduling
- includes extra adaptive dispatch heuristics beyond the common engine

`run_server_transport_main_suite.py`

- default baseline variants: 5
  - `A2C`
  - `PPO`
  - `PPO_LSTM`
  - `RARL`
  - `PLR_UED`
- default seeds: 10
  - `42-51`
- default request numbers: 1
  - `30`
- default wave: `full_main`
- default backend: `unified`

With the current `transport_main_suite_config.py`, `full_main` covers 44 distributions:

- `M`: 5
- `R`: 7
- `O`: 14
- `F1`: 6
- `F2`: 6
- `G`: 6

## Current Problems In The Scheduler Layer

1. The old default distribution lists still encode the previous off-grid story.
2. The classic reporting path is hard-coded to the old `MAIN_TABLE_DISTS`, so it cannot summarize the new symmetric matrix without code changes.
3. Several wrapper scripts mix execution concerns with report-generation concerns, which makes them brittle.
4. The project currently has no single canonical paper runner for a transport-oriented symmetric family matrix.
5. `adaptive` is operationally powerful, but it is not the right transparency/reproducibility tradeoff for the main paper suite.

## Recommendation

The final paper-grade server runner should be based on:

- `run_experiments_server_unified.py` as the execution front-end
- `run_experiments_common.py` as the real execution engine

It should borrow selected ideas from:

- `run_server_classic_baseline_suite.py`: manifest writing, explicit suite packaging
- `run_experiments_server_stream.py`: optional queue-style launch mode for very large matrices

It should **not** be based primarily on:

- `run_experiments_server_adaptive.py`
- `run_server_classic_baseline_suite.py`
- any of the ProtoMem / V7 / recovery wrappers

## Why `unified` Is The Right Base

1. It is the thinnest general wrapper over the shared execution engine.
2. It already exposes the knobs needed by the final suite: variants, distributions, `R`, seeds, stage mode, checkpoint paths, baseline/metrics/plot toggles, resume, skip-completed, and precheck.
3. It is easier to audit than `adaptive`.
4. It has fewer moving pieces than `stream`.
5. It is easier to attach a new transport-oriented suite config on top of it.

## What The Final Runner Should Add

The final runner should be a new thin wrapper, conceptually something like `run_server_transport_main_suite.py`, built on top of `unified`, with:

- an explicit family-structured distribution registry for the paper matrix;
- fixed priority waves, e.g. `smoke`, `core_shift`, `full_main`, `appendix`;
- a manifest JSON that records algorithm list, distribution families, seeds, `R`, and priority batches;
- support for leaving the main algorithm slot empty until implementation is ready;
- a report path that does not hard-code the old 19-distribution table;
- an optional `--queue-mode stream` switch if later we want cross-variant task interleaving.

## Short Conclusion

For the final paper experiments, the correct direction is:

- do not keep extending `classic_baseline_suite`;
- do not make `adaptive` the canonical runner;
- build the final runner as a new thin suite wrapper on top of `run_experiments_server_unified.py` and `run_experiments_common.py`.
