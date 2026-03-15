# Metrics Audit For Transport Re-planning Paper

This note aligns three things:

1. the predecessor synchromodal re-planning paper,
2. the current codebase,
3. the new paper story: learning-guided transport re-planning under non-stationary / cross-distribution service regimes.

It is intentionally code-grounded. If a metric is not stably available from current outputs, it is marked as unavailable.

## 1. Metrics The Predecessor Paper Clearly Emphasized

From the predecessor paper text and tables/figures, the important reporting dimensions were:

- reward / policy quality
- delay
- waiting time
- emissions
- cost
- computation time
- action success / action proportion under transfer

This means the new paper should not stay at RL reward only. The natural metric stack is:

- transport / operations outcomes first
- cross-regime robustness second
- RL diagnostics third
- runtime / deployment fourth

## 2. What The Current Code Can Already Provide Reliably

### A. Directly available in current run outputs

These are already written to the run folder and can be read without changing the solver:

- `meta.json`
  - scenario, algorithm, seed, `R`, stage mode, paths
- `rl_trace.csv`
  - event-level RL actions, rewards, phase labels, `gt_mean`, severity, timestamps
- `rl_training.csv`
  - phase-wise reward logs, training / implementation timing, generic policy diagnostics
- `rl_summary.csv`
  - aggregate reward/action summary
- `rl_decision.csv`
  - decision-level timestamps, matched/unmatched decisions, action/reward pairs
- `baseline_wait.csv`
- `baseline_reroute.csv`
- optional `baseline_random.csv`
- `resource_usage.json`
  - wall-clock and resource peaks
- `alns_outputs/**/obj_record*.xlsx`
  - aggregate transport objective components such as:
    - `overall_cost`
    - `overall_time`
    - `overall_emission`
    - `served_requests`
    - `overall_wait_cost`
    - `overall_storage_cost`
    - `overall_delay_penalty`
    - cost decomposition terms

### B. Can be derived from existing logs without solver changes

These are not exported as a single run summary in old code, but they can be reconstructed:

- RL reward mean / std / quantiles in implementation phase
- baseline reward means
- NPS-style normalized superiority profile
- reward by `phase_label`
- reward by `gt_mean`
- implement-phase action rate
- removal / insertion action shares from trace logs
- wait share vs reroute share
- matched / unmatched decision rate from `rl_decision.csv`
- decision latency from `ts_reward - ts_decision`
- last training time / last implementation time
- scenario family / pattern / regime gap from `distribution_config.json`

### C. Computed inside ALNS but not stably exported at run-summary level before this patch

The solver computes more objective terms than the old `metrics.json` exposed. Examples from `Intermodal_ALNS34959.py`:

- `overall_number_transshipment`
- `overall_average_speed`
- `overall_average_time_ratio`
- `overall_emission_transshipment`

These exist in-memory, but they are not consistently preserved in the standard run-level metric outputs.

## 3. Important Gaps Against The New Paper Story

### A. Transport / operations metrics that are now available after the patch

The new run summary now exports, when corresponding `obj_record*.xlsx` files exist:

- train-stage transport aggregate outcome
- implement-stage transport aggregate outcome

For each stage, the summary attempts to read:

- `overall_cost`
- `overall_time`
- `overall_emission`
- `served_requests`
- `overall_request_cost`
- `overall_vehicle_cost`
- `overall_wait_cost`
- `overall_transshipment_cost`
- `overall_un_load_cost`
- `overall_emission_cost`
- `overall_storage_cost`
- `overall_delay_penalty`
- `iteration_time`

This is the main bridge from RL-only reporting toward transport / OR reporting.

### B. Still unavailable or only partially available

These matter for a TRC-style paper, but current code still does not expose them cleanly enough:

- average waiting time per request
  - current code gives `overall_wait_cost`, not direct waiting time in hours
- total delay in hours
  - current code gives `overall_delay_penalty`, not direct delay hours
- average emissions per request
  - can be approximated as `overall_emission / served_requests`, but not stored directly
- service rate / failed request count as a dedicated run metric
  - `served_requests` exists, but "requested vs failed vs late" is not exported as a clean tuple
- request-level transport outcome distribution
  - not stably exported in a simple machine-readable run summary
- mode-share outcomes / transshipment frequency in final export
  - some related quantities are computed internally, but not consistently written to final objective files

These should be treated as real gaps, not paper-writing gaps.

## 4. Metrics Priority For The New Paper

### Priority 1: Transport / operations outcomes

These should anchor the main paper tables and figures:

- served requests
- total transport cost
- total transport time
- total emissions
- waiting-related burden
- storage-related burden
- delay-related burden

### Priority 2: Robustness / cross-distribution performance

These are mainly aggregation metrics across runs, but each run summary now stores the needed ingredients:

- family (`M/R/O/F1/F2/G`)
- pattern (`ab/random_mix/aba/abba/abc`)
- regime levels (`A/B/C`)
- regime gap / span
- phase-label reward slices
- `gt_mean` reward slices

### Priority 3: RL diagnostics that remain necessary

- implementation reward mean / dispersion
- action profile
- matched decision rate
- `p_action1`
- reward conditioned on action

### Priority 4: Runtime / deployment indicators

- total wall time
- CPU / RAM / GPU peaks
- last recorded training time
- last recorded implementation time
- decision latency

## 5. Code Changes In This Patch

The run-level export is now produced by `codes/analysis/compute_metrics.py`.

For each completed run, it now writes:

- `metrics.json`
  - backward-compatible location, now richer
- `run_summary.json`
  - full machine-readable summary
- `run_summary_flat.csv`
  - one-row flat export for batch collection

It still appends a global summary row to:

- `codes/logs/summary/metrics_summary.csv`

## 6. Important Interpretation Note

The summary contains two reward views:

- `reward_metrics.overall.rl`
  - full RL implementation reward distribution
- top-level `J_*` and `reward_metrics.comparison.*`
  - comparison-oriented values aligned to the available baseline horizon

This distinction matters when baseline logs are shorter than RL event logs.

## 7. Small Consistency Repair Also Applied

`plot_paper_figure.py` previously fell back to `finish_removal` only when loading baselines. That could miss insertion-side baseline rewards.

The fallback logic now prefers:

1. `receive_reward`
2. `finish_removal` + `finish_insertion`
3. `begin_insertion`

This does not redesign the plotting pipeline, but it removes one obvious reward-accounting inconsistency.
