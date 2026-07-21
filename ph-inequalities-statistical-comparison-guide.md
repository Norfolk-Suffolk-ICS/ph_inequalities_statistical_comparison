## Overview

This module provides four functions for summarising binary/count outcomes across groups (e.g. core20, plus, alliances, PCNs, etc), each returning a per-group estimate alongside a confidence interval and a flag showing whether that group differs from the population as a whole.

| Function | Measure | Use when |
|---|---|---|
| `crude_proportion_df` | Crude proportion | You want a raw % without adjusting for population structure |
| `crude_rate_df` | Crude rate | You have row-per-unit-of-exposure data and want a raw rate |
| `directly_standardized_proportion_df` | DSP | You need to compare proportions across groups with different age/sex/etc. mixes |
| `directly_standardized_rate_df` | DSR | You need to compare rates across groups with different age/sex/etc. mixes |

All four functions share a common design: they compute a value for every group **and** append a single **"Overall" row** representing the full input dataset, then compare every group's confidence interval against that Overall value to flag whether the group sits meaningfully higher or lower.

------------------------------------------------------------------------

## 1. Data Requirements

### Row-level data expected

Every function expects **row-level tabular data** in a Polars DataFrame — not pre-summarised percentages or rates.

- `crude_proportion_df` and `directly_standardized_proportion_df` need a binary `event_col` (0/1 or boolean). Each row is one record; the proportion is `sum(event_col) / row_count`.
- `crude_rate_df` and `directly_standardized_rate_df` need an `event_col` containing event counts (any non-negative integer, not restricted to 0/1 — e.g. multiple emergency attendances for one patient in a single row is valid). Neither function accepts a denominator, exposure, or person-time argument: the denominator is always derived internally from row count within each group/stratum, treating each row as one unit of exposure present at the end of the reporting period. This is a simplifying assumption for patient-level data where true person-time is unavailable, and it is flagged on every output row (see Section 3).

### Strata columns (DSP/DSR only)

`strata_cols` define the standardisation strata (e.g. age band, sex). **Numeric strata columns are automatically binned into quartiles (Q1–Q4)** using the full dataset's distribution — you do not need to pre-bin age into bands yourself. Categorical strata columns (e.g. sex, ethnicity) are used as-is.

### Group columns

`group_cols` define what you're comparing (e.g. core20, alliance). One output row is produced per unique combination of `group_cols`, plus the Overall row. For `crude_proportion_df` and `crude_rate_df`, `group_cols` is optional — if omitted, the function returns a single row summarising the whole dataset with no group comparison. For the two standardised functions, `group_cols` is required.

### Nulls and types (strictly enforced)

All four functions validate their inputs before doing any computation, and raise informative errors rather than silently dropping or coercing bad data:

- **Missing columns** raise a clear `ValueError` naming the missing fields.
- **Nulls/NAs are rejected outright** in the numerator (`event_col`), all `strata_cols`, and all `group_cols`. The error message names every offending column and its null count in one go, e.g. `"Null/NA values are not permitted... Offending column(s): 'age' (3 null values), 'region' (1 null value)."` You must remove or impute these rows before calling the function.
- **The numerator column must be a non-negative integer column** (or boolean, cast to `Int64` automatically). Passing a float column (e.g. `1.0`, `0.0`) or a string column raises a `ValueError` naming the offending dtype. Negative integers also raise an error.
- **For the two proportion functions** (`crude_proportion_df`, `directly_standardized_proportion_df`), the numerator must contain *only* 0 or 1 values — any other integer (e.g. a stray `2`) raises an error.
- **For the two rate functions**, the numerator may be any non-negative integer.
- Zero-denominator or zero-event groups (once past validation) are handled gracefully and flagged in `notes` rather than raising errors — validation only rejects structurally invalid data, not sparse-but-valid data.

------------------------------------------------------------------------

## 2. The "Overall" Row

Every function appends a row where all `group_cols` are set to the string `"Overall"`. This row summarises the **entire input dataset** as a single group, computed the same way as any other row — with one exception for DSP and DSR (see below).

The Overall row serves two purposes: it acts as a population benchmark, letting you see the national/organisational average alongside each group's estimate without a separate calculation; and it is the fixed reference point that every other row's `significance` label is assessed against (see Section 4).

### DSP and DSR: the Overall row is crude, not standardised

For `directly_standardized_proportion_df` and `directly_standardized_rate_df`, the Overall row reports the **crude (unstandardised)** proportion or rate — not a re-run of the standardisation weighting logic. This is because the Overall row's population **is** the reference population used to build the standardisation weights in the first place; standardising the whole population against itself would just return the crude value again, so the function skips that redundant step and states this explicitly in `notes`:

> `"Overall proportion: no weighting applied as this row is itself the reference population"` (DSP)
>
> `"Overall rate: no weighting applied as this row is itself the reference population"` (DSR)

------------------------------------------------------------------------

## 3. Function-by-Function Reference

### `crude_proportion_df`

**Signature:** `crude_proportion_df(df, event_col, group_cols=None, confidence=0.95)`

**Output columns:** `[*group_cols, events, n, proportion, lower, upper, confidence, method, notes, significance]`

| Column | Description |
|---|---|
| `*group_cols` | Every column passed in `group_cols`, repeated as-is; one row per unique combination, plus a final `"Overall"` row |
| `events` | `sum(event_col)` within the group |
| `n` | Row count within the group |
| `proportion` | `events / n`; `NaN` when `n == 0` |
| `lower`, `upper` | Wilson score confidence interval bounds |
| `confidence` | The confidence level parameter (default 0.95) |
| `method` | Always `"Wilson score"` — the only CI method this function uses |
| `notes` | Pipe-separated quality flags; empty string when none apply |
| `significance` | `"Higher"`, `"Lower"`, `"Not significant"`, `"Not tested"`, or `"Reference"` (Overall row only) — see Section 4 |

Example output for three alliances against an Overall proportion of 19.5%:

| alliance | events | n | proportion | lower | upper | significance |
|---|---|---|---|---|---|---|
| A | 90 | 300 | 0.300 | 0.257 | 0.347 | Higher |
| B | 30 | 300 | 0.100 | 0.071 | 0.140 | Lower |
| C | 0 | 15 | 0.000 | 0.000 | 0.206 | Not significant |
| Overall | 120 | 615 | 0.195 | 0.163 | 0.231 | Reference |

### `crude_rate_df`

**Signature:** `crude_rate_df(df, event_col, group_cols=None, multiplier=100_000.0, confidence=0.95)`

**Output columns:** `[*group_cols, events, denominator, rate, lower, upper, multiplier, confidence, method, notes, significance]`

| Column | Description |
|---|---|
| `*group_cols` | As above |
| `events` | `sum(event_col)` within the group |
| `denominator` | Row count within the group — this is always how the denominator is derived; there is no way to supply an external person-time column |
| `rate` | `(events / denominator) * multiplier`; `NaN` if `denominator == 0` |
| `lower`, `upper` | Confidence interval bounds, scaled by `multiplier` |
| `multiplier` | The rate scaling factor passed to the function (default 100,000) |
| `confidence` | The confidence level parameter |
| `method` | `"Byar"` when `events >= 10`, `"Exact chi-square"` when `events < 10`, or `"undefined"` if the denominator is zero |
| `notes` | Always includes the end-of-period denominator caveat, plus any sparsity flags |
| `significance` | As above |

### `directly_standardized_proportion_df`

**Signature:** `directly_standardized_proportion_df(df, event_col, strata_cols, group_cols, confidence=0.95)`

**Output columns:** `[*group_cols, events, n, dsp, dsp_lower, dsp_upper, notes, significance]`

| Column | Description |
|---|---|
| `*group_cols` | As above |
| `events` | Crude (unweighted) event count for the group |
| `n` | Crude (unweighted) row count for the group |
| `dsp` | The directly standardised proportion, weighted by the full dataset's stratum composition |
| `dsp_lower`, `dsp_upper` | Wilson-Dobson confidence interval bounds around `dsp` |
| `notes` | Quality flags — see Section 5 |
| `significance` | As above; for the Overall row, this compares against itself and is always `"Reference"` |

### `directly_standardized_rate_df`

**Signature:** `directly_standardized_rate_df(df, event_col, strata_cols, group_cols, multiplier=100_000.0, confidence=0.95)`

**Output columns:** `[*group_cols, events, denominator, dsr, dsr_lower, dsr_upper, multiplier, notes, significance]`

| Column | Description |
|---|---|
| `*group_cols` | As above |
| `events` | Crude (unweighted) event count for the group |
| `denominator` | Crude (unweighted) row count for the group |
| `dsr` | The directly standardised rate, scaled by `multiplier` |
| `dsr_lower`, `dsr_upper` | Dobson-Byar confidence interval bounds around `dsr`, scaled by `multiplier` |
| `multiplier` | The rate scaling factor passed to the function |
| `notes` | Quality flags — see Section 5 |
| `significance` | As above |

------------------------------------------------------------------------

## 4. How to Interpret the `significance` Column

None of the four functions run a formal statistical hypothesis test (no p-values are calculated or reported). Instead, every non-Overall row's `significance` label is derived purely by checking whether the Overall value — treated as a **fixed benchmark**, not as a quantity with its own sampling uncertainty — falls inside or outside that row's own confidence interval:

| Label | Meaning |
|---|---|
| `"Higher"` | The Overall/reference value falls below this group's `lower` bound — the group's estimate sits above what would be expected if it matched the population |
| `"Lower"` | The Overall/reference value falls above this group's `upper` bound — the group's estimate sits below the population figure |
| `"Not significant"` | The Overall/reference value falls within this group's confidence interval — the group is not distinguishable from the population at the stated confidence level |
| `"Not tested"` | No comparison was possible (e.g. the group's confidence interval could not be computed, typically because the denominator or count was zero, or no `group_cols` were supplied) |
| `"Reference"` | Reserved for the Overall row itself, which is never compared against itself |

Because this is a confidence-interval-overlap check rather than a two-sample hypothesis test, it is intentionally simple and conservative: it does not correct for multiple comparisons across many groups, and it does not account for the fact that each group's data is itself a subset of the Overall figure it is being compared against. Wider confidence intervals (typically arising from smaller group sizes) make a `"Not significant"` result more likely purely because the interval is wide enough to contain the reference value, not necessarily because the underlying rates are truly similar — this should be borne in mind when interpreting results for small groups.

------------------------------------------------------------------------

## 5. How to Interpret the `notes` Column

The `notes` column is a single string containing zero or more flags, separated by `" | "` when more than one applies. An empty string means no flags were triggered for that row. The full vocabulary of flags, and what each one means, is as follows.

### Flags common to the crude functions

| Flag | Appears when | What it means |
|---|---|---|
| `Zero denominator` | The group's row count (or denominator) is zero | No proportion/rate could be calculated; `proportion`/`rate` will be `NaN` |
| `Zero events` | The group has no events at all | The proportion/rate is exactly 0; the confidence interval is still calculable and informative |
| `Low event count (<10)` | Fewer than 10 events in the group | The confidence interval relies on a small-sample method (Wilson score, or exact chi-square rather than Byar) and should be interpreted cautiously |
| `All events (proportion = 1)` | Every row in the group is an event (proportion functions only) | The proportion is exactly 1; interpret alongside the group's sample size |
| `Low non-event count (<10)` | Fewer than 10 non-events in the group (proportion functions only) | The opposite boundary case to a low event count — the CI is still valid but reflects a small effective sample on one side |
| `End-of-period denominator: row count used as exposure (...)` | Always present on every row for the two rate functions | A reminder that the denominator was derived by counting rows (treating each row as one patient present at the end of the reporting period) rather than from true person-time data; this does not account for patients who only had partial exposure during the period |
| `consider suppression` (appended to the low event count flag in `crude_rate_df`) | Fewer than 10 events | A standard public-health-reporting convention: rates based on very small counts are often suppressed in published outputs to avoid potentially identifying individuals or misleading small-number rates |

### Flags specific to the standardised functions (DSP/DSR)

| Flag | Appears when | What it means |
|---|---|---|
| `Haldane correction applied` | Any stratum within the group had zero events (DSR), or zero events/all events (DSP) | A small continuity correction (adding 0.5 to the stratum's event count and 1 to its denominator) was applied to that stratum before it was combined into the standardised estimate, preventing that stratum from contributing a degenerate (zero) variance to the overall calculation |
| `Unreliable: stratum n<10` (DSP) / `Unreliable: stratum denominator <10` (DSR) | Any non-empty stratum within the group has fewer than 10 records | At least one of the strata feeding into the standardised estimate is thin enough that its contribution to the standardised value may be unstable, even if the group's total sample size looks reasonable |
| `Unreliable: stratum non-event count <10` (DSP only) | Any non-empty stratum has fewer than 10 non-events | The mirror-image boundary case to the above, specific to proportions |
| `Low event count (<10)` (DSP) / `Low event count (<10): DSR should generally not be reported` (DSR) | The group's crude (unweighted) event count is below 10 | The standardised estimate is being built from a small overall event count; for DSR specifically, this is flagged as a case where the standardised rate is not generally considered reliable enough for routine reporting |
| `Zero events` | The group's crude event count is zero | The standardised value will be low or zero depending on stratum weighting |
| `All events (proportion = 1)` (DSP only) | Every record in the group is an event | As with the crude proportion function, this is the upper boundary case |
| `Zero denominator` (DSR only) | The group's crude row count is zero | No standardised rate could be calculated for this group |
| `Overall proportion: no weighting applied as this row is itself the reference population` (DSP) / `Overall rate: no weighting applied as this row is itself the reference population` (DSR) | Always present on the Overall row only | Explains why the Overall row shows the crude, not standardised, value — see Section 2 |

As with the crude functions, the end-of-period denominator caveat also appears on every row of `directly_standardized_rate_df`, since it uses the same row-count-based denominator logic as `crude_rate_df`.

------------------------------------------------------------------------

## 6. Worked Example

Given a dataset of patient-level records with an `alliance` group column, an `age` and `sex` stratum column, and a binary `event` column, calling `directly_standardized_proportion_df(df, "event", ["age", "sex"], ["alliance"])` on three alliances (A, B, C) against a population event rate of 19.5% might produce:

| alliance | events | n | dsp | dsp_lower | dsp_upper | notes | significance |
|---|---|---|---|---|---|---|---|
| A | 90 | 300 | 0.302 | 0.253 | 0.357 | | Higher |
| B | 30 | 300 | 0.101 | 0.072 | 0.140 | | Lower |
| C | 0 | 15 | 0.163 | 0.163 | 0.163 | Haldane correction applied \| Zero events | Lower |
| Overall | 120 | 615 | 0.195 | 0.166 | 0.228 | Overall proportion: no weighting applied as this row is itself the reference population | Reference |

Reading this table: Alliance A's confidence interval (0.253–0.357) sits entirely above the Overall proportion of 0.195, so it is flagged `"Higher"`. Alliance B's interval (0.072–0.140) sits entirely below the Overall figure, so it is flagged `"Lower"`. Alliance C has zero raw events, but because one of its strata triggered the Haldane correction, its standardised estimate is not exactly zero — its resulting interval still sits below the Overall figure, earning a `"Lower"` label, but the `Zero events` flag in `notes` is an important caveat: this group's estimate should be treated with more caution than Alliance A or B's, given it is built from a very small, entirely event-free sample.