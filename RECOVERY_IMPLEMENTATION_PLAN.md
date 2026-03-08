# Recovery Mechanism Implementation Plan

## Scope

This file records the staged implementation plan for the OOD action-collapse recovery work.
It is intended to be stable reference material for later confirmation, resumption, and review.

The plan follows the current code structure that was verified in:

- `codes/Dynamic_master34959.py`
- `codes/outer_rl/run_edrl_pipeline.py`
- `codes/outer_rl/run_edrl_phase2.py`
- `codes/core/dynamic_RL34959.py`
- `codes/core/Dynamic_ALNS_RL34959.py`

## Confirmed Structure

- Phase 1:
  - launched through `run_edrl_pipeline.py`
  - executes inner RL with `Dynamic_master34959.py --stage-mode train_only`
  - currently saves one main checkpoint: `theta_phase1.zip`
- Phase 2 / Phase 3:
  - handled inside `codes/outer_rl/run_edrl_phase2.py`
  - each outer iteration generates data, runs a short inner train-only cycle, and records metrics
  - saves `theta_iterXXX.zip` checkpoints and `outer_train_round.csv`
- Phase 4:
  - launched through `run_edrl_pipeline.py`
  - runs `Dynamic_master34959.py --stage-mode eval_only`
- PPO_NEW v3:
  - built in `codes/robust_rl/ppo_new/v3_context.py`
  - uses SB3 PPO with a custom feature extractor
  - current policy structure is shared extractor + separate actor/value branches + separate `action_net` and `value_net`

## Strategy

The implementation is deliberately staged:

1. First build a unified checkpoint evaluation layer
2. Then test actor-side rollback without touching the main pipeline
3. Then add phase1 historical checkpoint infrastructure
4. Then add rollback branching as a pipeline-level fallback

This keeps high-risk changes away from the existing training path until the signal is clear.

## Milestones

### M1: Unified Evaluation Baseline

Files:

- `codes/analysis/checkpoint_eval_common.py`
- `codes/analysis/eval_checkpoint_pool.py`

Goals:

- summarize existing run directories with one consistent metric schema
- evaluate checkpoints through `Dynamic_master34959.py --stage-mode eval_only`
- output a single comparison csv for later A/B work

Status:

- implemented baseline

Current notes:

- `eval_checkpoint_pool.py` now supports:
  - `--run-dir` summarization
  - `--checkpoint` evaluation
  - `--external-data-root` to reuse a fixed dataset and skip generation
  - `--allow-partial-on-timeout` for quick checkpoint probes
- Current engineering fact:
  - full checkpoint eval is materially expensive in this repo even when generator is skipped
  - quick partial probes are useful for smoke validation, but not enough for final checkpoint ranking

### M2: Actor Rollback Minimal Prototype

Files:

- `codes/analysis/eval_actor_head_swap.py`
- optional helper: `codes/analysis/build_actor_rollback_ckpt.py`

Goals:

- build temporary checkpoints by swapping actor-side parameters
- run pure evaluation before any short retraining
- falsify or support the actor-head hypothesis quickly

Status:

- prototype implemented

Current notes:

- `eval_actor_head_swap.py` now builds mixed checkpoints for:
  - `action_head_only`
  - `actor_branch_and_head`
  - `action_head_interp`
- verified on a smoke run:
  - output checkpoint is saved successfully
  - `action_net.weight` and `action_net.bias` can be replaced independently
  - the actor MLP trunk remains untouched in `action_head_only` mode
- pipeline-level comparison-ready A branch is now implemented in `run_edrl_pipeline.py`
- current A branch behavior:
  - selects the main outer checkpoint with the same phase4 policy
  - picks an early checkpoint from explicit path or `phase1_ckpt_manifest.csv`
  - builds a mixed checkpoint via `eval_actor_head_swap.py`
  - launches one isolated outer branch from that mixed checkpoint
  - writes `actor_rollback_compare.csv`
  - selects a winner before phase4
- current guardrail:
  - A and B rollback mechanisms are forced to be mutually exclusive in one pipeline run
- validation completed:
  - static compile/import validation
  - synthetic smoke covering trigger -> early ckpt resolve -> mixed ckpt build -> branch compare -> winner selection

### M3: Phase1 Historical Checkpoint Pool

Files to inspect / modify:

- `codes/core/Dynamic_ALNS_RL34959.py`
- `codes/core/Intermodal_ALNS34959.py`
- `codes/core/dynamic_RL34959.py`

Goals:

- add periodic phase1 checkpoint saves
- add `phase1_ckpt_manifest.csv`
- make every historical checkpoint easy to batch-evaluate

Status:

- implemented

Current notes:

- periodic phase1 checkpoint saving is now implemented behind an explicit pipeline flag
- current pipeline argument:
  - `--phase1-history-every-tables`
- current outputs:
  - `post_stage/checkpoints/phase1_history/*.zip`
  - `post_stage/phase1_ckpt_manifest.csv`
- manifest currently records:
  - periodic checkpoints
  - final `theta_phase1.zip`
- helper smoke validation completed for:
  - periodic save trigger
  - final checkpoint manifest recording
- `codes/analysis/rank_phase1_checkpoints.py` is also implemented now and can:
  - optionally call `eval_checkpoint_pool.py`
  - merge manifest + eval summary
  - output `phase1_ckpt_candidates.csv`, `phase1_ckpt_ranked.csv`, `phase1_ckpt_topk.csv`

### M4: Rollback Branch Integration

Primary file:

- `codes/outer_rl/run_edrl_pipeline.py`

Supporting file:

- `codes/analysis/rank_phase1_checkpoints.py`

Goals:

- run the normal outer recovery branch first
- if recovery is insufficient, spawn one or more rollback branches from selected phase1 history points
- compare branches and choose one winner before phase4

Status:

- initial version implemented

Current notes:

- `run_edrl_pipeline.py` now supports rollback orchestration behind explicit flags:
  - `--rollback-enable`
  - `--rollback-manifest-path`
  - `--rollback-topk`
  - `--rollback-trigger-metric`
  - `--rollback-trigger-threshold`
  - `--rollback-compare-metric`
  - `--rollback-reward-floor-ratio`
  - `--rollback-eval-timeout-sec`
  - `--rollback-allow-partial-on-timeout`
- current pipeline behavior:
  - first runs the normal outer branch
  - selects the main branch checkpoint using the same phase4 policy
  - checks the configured rollback trigger metric on the selected main row
  - if below threshold, calls `rank_phase1_checkpoints.py`
  - runs top-k rollback branches as isolated outer runs under `rollback_branches/session_*`
  - writes `rollback_branch_compare.csv`
  - selects a winner branch before phase4
- current engineering guardrails:
  - rollback is default-off
  - ranking failure is best-effort and does not kill the main pipeline
  - rollback branches force `--resume-mode none`
- validation completed:
  - static compile/import validation
  - synthetic smoke covering trigger -> ranking -> branch compare -> winner selection

### M5: Recovery A/B Suite Orchestration

Files:

- `codes/experiments/run_recovery_ab_suite.py`
- `codes/analysis/summarize_recovery_ab_suite.py`

Goals:

- run three comparable branches from one suite entrypoint:
  - main
  - A / actor rollback
  - B / checkpoint rollback
- reuse the phase1 checkpoint, data root, and history manifest produced by the main branch
- generate one consolidated suite summary instead of manually comparing multiple pipeline outputs

Status:

- implemented

Current notes:

- `run_recovery_ab_suite.py` now:
  - runs `main` first with phase1 history enabled
  - reuses `theta_phase1.zip`, `data/`, and `phase1_ckpt_manifest.csv`
  - launches an `actor_rollback` pipeline run
  - launches a `rollback` pipeline run
  - writes a suite manifest
- `summarize_recovery_ab_suite.py` now:
  - reads each run's `pipeline_summary.json`
  - summarizes final run behavior using `checkpoint_eval_common.summarize_run_dir`
  - exports:
    - suite summary csv
    - branch compare rows csv
    - simple comparison plot
    - markdown summary

## Immediate Tasks

### Task A

Run one real end-to-end pipeline with:

- `--phase1-history-every-tables`
- `--rollback-enable`
- a conservative `--rollback-trigger-threshold`

Goals:

- verify real manifest density
- measure ranking latency under a reused data root
- confirm branch directory structure and final selected ckpt metadata

### Task B

Decide whether the first real rollback comparison should use:

- `action1_rate` from `outer_train_round.csv`
- or a harder eval metric from `eval_checkpoint_pool.py`

Current implementation uses `outer_train_round.csv` metrics for trigger and branch comparison.
If this is not aligned enough with the true hardest-OOD criterion, the next iteration should add an optional eval-based compare path.

### Task C

Run a first checkpoint-pool study on a real phase1 history manifest and inspect:

- how many candidates survive the reward floor
- whether top-ranked checkpoints are meaningfully earlier than the final phase1 checkpoint
- whether rollback branches actually produce better outer metrics than the main branch

## Decision Rules

- If actor-side swap evaluation shows no meaningful recovery signal, stop the A line early.
- If actor-side swap evaluation does show signal, continue with a short-train validation branch.
- The rollback branch work should still proceed because it is structurally closer to the current checkpoint-driven pipeline.

## Current Implementation Notes

- The unified evaluation layer is now in place for both existing runs and checkpoint-triggered eval.
- The first actor-side rollback constructor is now in place as an isolated analysis script.
- The phase1 history pool, manifest, ranking script, and pipeline-level rollback fallback are now all implemented.
- No PPO core changes should be made before the evaluation layer exists.
- No rollback logic should be inserted into `run_edrl_phase2.py` before phase1 checkpoint history and ranking are available.
- Current blocker for fast checkpoint ranking:
  - real eval latency is high
  - future work should prioritize either fixed reused data roots, faster eval budgets, or staged checkpoint screening
