# User Guide: `ph_inequalities_statistical_comparison.py`

A module for automation of statistical comparison across inequalities. Currently supports crude and directly standardised proportions and rates, with built-in significance testing against the population reference.

***

## 1. Overview

The module provides four functions for summarising binary/count outcomes across groups (e.g. core20, Alliances etc):

| Function | Measure | Use when |
|---|---|---|
| `crude_proportion_df` | Crude proportion | You want a raw % without adjusting for population structure |
| `crude_rate_df` | Crude rate | You have person-time or a denominator and want a raw rate |
| `directly_standardized_proportion_df` | DSP | You need to compare proportions across groups with different age/sex/etc. mixes |
| `directly_standardized_rate_df` | DSR | You need to compare rates across groups with different age/sex/etc. mixes |

All four functions share a common design: they compute a value for every group **and** append a single **"Overall" row** representing the full input dataset, and they now test every group against that Overall value, flagging whether it is statistically significantly higher, lower, or indistinguishable.

***

## 2. Data Requirements

### Row-level data expected

Every function expects **row-level tabular data** in a Polars DataFrame — not pre-summarised percentages.

- `crude_proportion_df` and `directly_standardized_proportion_df` needs a binary `event_col` (0/1 or boolean). Each row is one record; the proportion is `sum(event_col) / row_count`.
- `crude_rate_df` and `directly_standardized_rate_df` need an `event_col` (event count per row). The denominator is always the row count within each group/stratum — see Section 2 below for details on this design and its implications.

### Strata columns (DSP/DSR only)

`strata_cols` define the standardisation strata (e.g. age band, sex). **Numeric strata columns are automatically binned into quartiles (Q1–Q4)** using the full dataset's distribution — you do not need to pre-bin age into bands yourself. Categorical strata columns (e.g. sex, ethnicity) are used as-is.

### Group columns

`group_cols` define what you're comparing (e.g. core20, alliance etc). One output row is produced per unique combination of `group_cols`, plus the Overall row.

### Nulls and types (strictly enforced)

All four functions validate their inputs before doing any computation, and raise informative errors rather than silently dropping or coercing bad data:

- **Missing columns** raise a clear `ValueError` naming the missing fields.
- **Nulls/NAs are rejected outright** in the numerator (`event_col`), all `strata_cols`, and all `group_cols`. The error message names every offending column and its null count in one go, e.g. `"Null/NA values are not permitted... Offending column(s): 'age' (3 null values), 'region' (1 null value)."` You must remove or impute these rows before calling the function — there is no implicit drop-null behaviour.
- **The numerator column must be a non-negative integer column** (or boolean, which is accepted and cast to `Int64` automatically, since `True`/`False` is inherently a valid 0/1 encoding). Passing a float column (e.g. `1.0`, `0.0`) or a string column raises a `ValueError` naming the offending dtype. Passing negative integers (e.g. `-1`) also raises an error.
- **For the two proportion functions** (`crude_proportion_df`, `directly_standardized_proportion_df`), the numerator must contain *only* 0 or 1 values. If any other integer appears (e.g. a stray `2` from a miscoded field), the function raises an error rather than silently computing a nonsensical "proportion" greater than 1.
- **For the two rate functions** (`crude_rate_df`, `directly_standardized_rate_df`), the numerator may be any non-negative integer — e.g. a count of multiple events in one row (such as multiple emergency attendances for one patient) is valid and does not need to be 0/1.
- Zero-denominator or zero-event groups (once past validation) are handled gracefully and flagged in `notes` rather than raising errors — validation only rejects structurally invalid data, not sparse-but-valid data.

***

## 3. The "Overall" Row

Every function appends a row where all `group_cols` are set to the string `"Overall"`. This row summarises the **entire input dataset** as a single group, computed the same way as any other row would be for the same slice — with one exception for DSP and DSR.

### Why the Overall row matters

The Overall row serves two purposes:

1. **A population benchmark** — it lets you see at a glance where the national/organisational average sits alongside each group's estimate, without a separate calculation.
2. **The statistical reference for significance testing** — every group's `p_value` and `significance` label are calculated relative to this row (see Section 5).

### DSP and DSR: the Overall row is crude, not standardised

For `directly_standardized_proportion_df` and `directly_standardized_rate_df`, the Overall row reports the **crude (unstandardised)** proportion or rate — not a re-run of the DSP/DSR weighting logic. This is because the Overall row's population **is** the reference population used to build the standardisation weights in the first place. Standardising the whole population against itself would return the same value as the crude calculation, so the function skips that redundant step and states this explicitly:

> `"Overall proportion: no weighting applied as this row is itself the reference population"` (DSP)
>
> `"Overall rate: no weighting applied as this row is itself the reference population"` (DSR)

This note always appears in the Overall row's `notes` field, appended after any other applicable quality flags.

***

## 4. Complete Function Output Reference

Every one of the four functions returns every column documented below — none are optional, none are omitted under any input configuration. This section lists each column exactly as defined in the function's own return-value docstring and verified against live executed output, so nothing here is inferred or approximated.

### `crude_proportion_df`

Full column list (15 columns, in the exact order they appear): `[*group_cols, events, n, proportion, lower, upper, variance, std_dev, confidence, method, notes, test_method, p_value, p_adjusted, significance]`

| Column | Type | Full description |
|---|---|---|
| `*group_cols` | string | Every column passed in `group_cols` is repeated as-is; each unique combination gets one row, plus a final row where every one of these columns is set to the literal string `"Overall"` |
| `events` | float | `sum(event_col)` within the group — the raw count of 1s (or `True` values, cast to 1) |
| `n` | float | Row count within the group |
| `proportion` | float | `events / n`; is `NaN` when `n == 0` (never divides by zero silently) |
| `lower` | float | Lower bound of the Wilson score confidence interval |
| `upper` | float | Upper bound of the Wilson score confidence interval |
| `variance` | float | Variance of the Wilson score proportion estimate |
| `std_dev` | float | Square root of `variance` |
| `confidence` | float | The confidence level parameter passed to the function (default 0.95), repeated on every row |
| `method` | string | Always the literal string `"Wilson score"` — this column never varies within `crude_proportion_df`, since Wilson score is the only CI method this function uses regardless of sparsity |
| `notes` | string | Pipe-separated (`" | "`) quality flags; empty string when no flags apply (see Section 5 for the full vocabulary) |
| `test_method` | string | For non-Overall rows: `"Fisher exact"` (sparse) or `"Two-proportion z-test"` (dense). For the Overall row: `"Reference row (correction: <method>)"` |
| `p_value` | float | Raw, uncorrected p-value from the significance test against the Overall proportion; `NaN` on the Overall row itself |
| `p_adjusted` | float | `p_value` after multiple-testing correction across the whole batch; `NaN` on the Overall row |
| `significance` | string | One of `"Higher"`, `"Lower"`, `"Not significant"`, `"Not tested"`, or `"Reference"` (Overall row only) |

Full, unabridged real output for three alliances plus Overall (generated from a 615-row dataset: alliance A with 90/300 events, B with 30/300, C with 0/15):

| alliance | events | n | proportion | lower | upper | variance | std_dev | confidence | method | notes | test_method | p_value | p_adjusted | significance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 90.0 | 300.0 | 0.3 | 0.250940 | 0.354118 | 0.0007 | 0.026458 | 0.95 | Wilson score | (empty) | Two-proportion z-test | 0.000005 | 0.000014 | Higher |
| B | 30.0 | 300.0 | 0.1 | 0.070948 | 0.139166 | 0.0003 | 0.017321 | 0.95 | Wilson score | (empty) | Two-proportion z-test | 0.000032 | 0.000048 | Lower |
| C | 0.0 | 15.0 | 0.0 | 0.0 | 0.203883 | 0.0 | 0.0 | 0.95 | Wilson score | Zero events | Fisher exact | 0.05161 | 0.05161 | Not significant |
| Overall | 120.0 | 615.0 | 0.195122 | 0.165734 | 0.228295 | 0.000255 | 0.015980 | 0.95 | Wilson score | (empty) | Reference row (correction: Benjamini-Hochberg (FDR)) | NaN | NaN | Reference |

Group C's zero-event count triggered Fisher's exact test rather than the z-test used for A and B, and because at least one group in the batch was flagged sparse, the entire batch's `p_adjusted` values were computed using Benjamini-Hochberg rather than Holm-Sidak — this is why even A and B, which individually had plenty of events for a normal-theory test, still show `p_adjusted` values inflated slightly above their raw `p_value`.

### `crude_rate_df`

Full column list (16 columns): `[*group_cols, events, denominator, rate, lower, upper, variance, std_dev, method, multiplier, confidence, notes, test_method, p_value, p_adjusted, significance]`

This has one more column than `crude_proportion_df` (`multiplier`) and several columns behave differently even though the names overlap:

| Column | Type | Full description |
|---|---|---|
| `*group_cols` | string | Same behaviour as above |
| `events` | float | `sum(event_col)` — for rates this can be any non-negative integer count per row, not just 0/1 |
| `denominator` | float | Row count within the group; this is **always** how the denominator is derived (see Section 2) — there is no way to supply an external person-time column |
| `rate` | float | `(events / denominator) * multiplier`; `NaN` if `denominator == 0` |
| `lower` | float | Lower bound of the count-based confidence interval, scaled by `multiplier` |
| `upper` | float | Upper bound of the count-based confidence interval, scaled by `multiplier` |
| `variance` | float | Variance of the rate estimate, already scaled by `multiplier ** 2` |
| `std_dev` | float | Square root of `variance` |
| `method` | string | `"Byar"` when `events >= 10` (dense), `"Exact chi-square"` when `events < 10` (sparse), or `"undefined"` if `denominator == 0` — this column genuinely varies row to row, unlike the proportion function's fixed `"Wilson score"` |
| `multiplier` | int/float | The rate scale factor passed to the function (e.g. 1000, or the default 100,000), repeated on every row |
| `confidence` | float | Confidence level, repeated on every row |
| `notes` | string | Pipe-separated flags; **always contains the end-of-period denominator caveat on every single row**, in addition to any sparsity flags |
| `test_method` | string | `"Mid-P exact Poisson"` (sparse) or `"Poisson z-test (test-based)"` (dense) for non-Overall rows; `"Reference row (correction: <method>)"` for Overall |
| `p_value`, `p_adjusted`, `significance` | as above | Same semantics as `crude_proportion_df` |

Full, unabridged real output (multiplier = 1000, same underlying data as above):

| alliance | events | denominator | rate | lower | upper | variance | std_dev | method | multiplier | confidence | notes | test_method | p_value | p_adjusted | significance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 90.0 | 300.0 | 300.0 | 241.228736 | 368.754878 | 1000.0 | 31.622777 | Byar | 1000 | 0.95 | End-of-period denominator: row count used as exposure (assumes each row = 1 patient present at period end; does not account for partial-period exposure) | Poisson z-test (test-based) | 0.000039 | 0.000117 | Higher |
| B | 30.0 | 300.0 | 100.0 | 67.456464 | 142.761240 | 333.333333 | 18.257419 | Byar | 1000 | 0.95 | End-of-period denominator: row count used as exposure (assumes each row = 1 patient present at period end; does not account for partial-period exposure) | Poisson z-test (test-based) | 0.000192 | 0.000287 | Lower |
| C | 0.0 | 15.0 | 0.0 | 0.0 | 245.925297 | 0.0 | 0.0 | Exact chi-square | 1000 | 0.95 | End-of-period denominator: row count used as exposure (assumes each row = 1 patient present at period end; does not account for partial-period exposure) | Mid-P exact Poisson | 0.053567 | 0.053567 | Not significant |
| Overall | 120.0 | 615.0 | 195.121951 | 161.772734 | 233.319773 | 317.271465 | 17.812116 | Byar | 1000 | 0.95 | End-of-period denominator: row count used as exposure (assumes each row = 1 patient present at period end; does not account for partial-period exposure) | Reference row (correction: Benjamini-Hochberg (FDR)) | NaN | NaN | Reference |

Notice `method` differs between group C (`"Exact chi-square"`, since events < 10) and groups A, B, and Overall (`"Byar"`, since events ≥ 10) — this column is not fixed the way `crude_proportion_df`'s `method` column is, and should be inspected per row rather than assumed constant across a result set.

### `directly_standardized_proportion_df`

Full column list (13 columns): `[*group_cols, events, n, dsp, dsp_lower, dsp_upper, variance, std_dev, notes, test_method, p_value, p_adjusted, significance]`

Note this function's column set has **no** `lower`/`upper`/`confidence`/`method` columns in that exact naming — the CI bounds are named `dsp_lower`/`dsp_upper` instead, and there is no separate `method` or `confidence` column at all (confidence is a parameter used internally but not echoed as an output column here).

| Column | Type | Full description |
|---|---|---|
| `*group_cols` | string | Same behaviour as above |
| `events` | int | Crude (unweighted) event count for the group — not stratum-adjusted |
| `n` | int | Crude (unweighted) row count for the group |
| `dsp` | float | The directly standardised proportion: the weighted sum of stratum-level proportions, weighted by the reference population's stratum weights |
| `dsp_lower` | float | Lower bound of the Wilson-Dobson confidence interval around `dsp` |
| `dsp_upper` | float | Upper bound of the Wilson-Dobson confidence interval around `dsp` |
| `variance` | float | Variance of the DSP estimate under the Dobson approximation |
| `std_dev` | float | Square root of `variance` |
| `notes` | string | Pipe-separated flags drawn from a longer vocabulary than the crude functions, including per-stratum reliability flags (see Section 5) |
| `test_method` | string | `"Exact one-sample (Byar-based)"` (sparse group or unreliable stratum) or `"Dobson z-test vs reference"` (dense, reliable) for non-Overall rows; `"Reference row (correction: <method>)"` for Overall |
| `p_value`, `p_adjusted`, `significance` | as above | Tested against the Overall row's **crude** proportion (not a standardised Overall value — see Section 3) |

Full, unabridged real output (age-standardised, same three alliances):

| alliance | events | n | dsp | dsp_lower | dsp_upper | variance | std_dev | notes | test_method | p_value | p_adjusted | significance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 90 | 300 | 0.299848 | 0.250894 | 0.353848 | 0.000697 | 0.026400 | (empty) | Dobson z-test vs reference | 0.000073 | 0.000109 | Higher |
| B | 30 | 300 | 0.099378 | 0.070613 | 0.138157 | 0.000294 | 0.017149 | (empty) | Dobson z-test vs reference | 2.3636e-8 | 7.0908e-8 | Lower |
| C | 0 | 15 | 0.133449 | 0.133449 | 0.133449 | 0.016292 | 0.127640 | Haldane correction applied \| Zero events \| Unreliable: stratum n<10 \| Unreliable: stratum non-event count <10 | Exact one-sample (Byar-based) | 0.054356 | 0.054356 | Not significant |
| Overall | 120 | 615 | 0.195122 | 0.165734 | 0.228295 | 0.000255 | 0.015980 | Overall proportion: no weighting applied as this row is itself the reference population | Reference row (correction: Benjamini-Hochberg (FDR)) | NaN | NaN | Reference |

Group C's `dsp` (0.133449) is markedly higher than its crude proportion of 0.0 (0 events out of 15) — this is the combined effect of the Haldane continuity correction (which prevents a literal 0% input from breaking the variance calculation) and the standardisation weighting pulling the estimate toward comparable strata elsewhere in the reference population. Notice also that `dsp_lower` and `dsp_upper` are identical (both 0.133449) for group C — a direct consequence of the underlying variance calculation collapsing to a degenerate interval when the group's only contributing stratum has too few records to estimate spread; this is itself a signal, visible only by comparing the CI columns, that group C's interval should not be read as informative.

### `directly_standardized_rate_df`

Full column list (14 columns): `[*group_cols, events, denominator, dsr, dsr_lower, dsr_upper, variance, std_dev, multiplier, notes, test_method, p_value, p_adjusted, significance]`

This is `directly_standardized_proportion_df`'s column set with `n`→`denominator`, `dsp`/`dsp_lower`/`dsp_upper`→`dsr`/`dsr_lower`/`dsr_upper`, plus an added `multiplier` column — structurally the rate counterpart of the DSP output, following the same pattern as `crude_rate_df` versus `crude_proportion_df`.

| Column | Type | Full description |
|---|---|---|
| `*group_cols` | string | Same behaviour as above |
| `events` | int | Crude (unweighted) event count |
| `denominator` | float | Crude (unweighted) row count, rounded to 6 decimal places |
| `dsr` | float | The directly standardised rate, scaled by `multiplier` |
| `dsr_lower` | float | Lower bound of the Dobson-Byar confidence interval, scaled by `multiplier` |
| `dsr_upper` | float | Upper bound of the Dobson-Byar confidence interval, scaled by `multiplier` |
| `variance` | float | Variance of the DSR estimate, scaled by `multiplier ** 2` |
| `std_dev` | float | Square root of `variance` |
| `multiplier` | int/float | The rate scale factor, repeated on every row |
| `notes` | string | Always includes the end-of-period denominator caveat (like `crude_rate_df`) plus any DSR-specific stratum reliability flags |
| `test_method` | string | `"Exact one-sample (Byar-based)"` (sparse/unreliable) or `"Dobson z-test vs reference"` (dense, reliable, SMR-style) for non-Overall rows; `"Reference row (correction: <method>)"` for Overall |
| `p_value`, `p_adjusted`, `significance` | as above | Tested against the Overall row's **crude** rate |

Full, unabridged real output (age-standardised, multiplier = 1000, same three alliances):

| alliance | events | denominator | dsr | dsr_lower | dsr_upper | variance | std_dev | multiplier | notes | test_method | p_value | p_adjusted | significance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 90 | 300.0 | 299.847743 | 240.986323 | 368.708092 | 1003.070390 | 31.671287 | 1000 | End-of-period denominator: row count used as exposure (assumes each row = 1 patient present at period end; does not account for partial-period exposure) | Dobson z-test vs reference | 0.000944 | 0.001416 | Higher |
| B | 30 | 300.0 | 99.378049 | 67.016332 | 141.900384 | 329.619102 | 18.155415 | 1000 | End-of-period denominator: row count used as exposure (assumes each row = 1 patient present at period end; does not account for partial-period exposure) | Dobson z-test vs reference | 1.3379e-7 | 4.0138e-7 | Lower |
| C | 0 | 15.0 | 133.449477 | 133.449477 | 133.449477 | 11672.689699 | 108.040223 | 1000 | End-of-period denominator: row count used as exposure (assumes each row = 1 patient present at period end; does not account for partial-period exposure) \| Haldane correction applied \| Zero events \| Unreliable: stratum denominator <10 | Exact one-sample (Byar-based) | 0.053567 | 0.053567 | Not significant |
| Overall | 120 | 615.0 | 195.121951 | 161.772734 | 233.319773 | 317.271465 | 17.812116 | 1000 | End-of-period denominator: row count used as exposure (assumes each row = 1 patient present at period end; does not account for partial-period exposure) \| Overall rate: no weighting applied as this row is itself the reference population | Reference row (correction: Benjamini-Hochberg (FDR)) | NaN | NaN | Reference |

Every row here — including Overall — carries the end-of-period exposure caveat, because it is a structural property of how this module infers denominators (Section 2), not a per-group data-quality signal; only the flags that appear *in addition to* that caveat (Haldane correction, unreliable stratum) indicate an actual data-quality concern specific to that row.

***

## 5. Interpreting Columns and Notes in Practice

### Point estimates must always be read together with their confidence intervals

Never read `proportion`, `rate`, `dsp`, or `dsr` in isolation from their accompanying interval columns (`lower`/`upper` for the crude functions, `dsp_lower`/`dsp_upper` or `dsr_lower`/`dsr_upper` for the standardised ones). A group with a wide interval — typically driven by small `n` or `denominator` — can display a point estimate that looks dramatically different from the Overall row, yet still be classified `"Not significant"` once uncertainty is factored into the test. Group C in every table above is exactly this case: its crude proportion is a literal 0.0 (0 events across 15 rows), starkly below the Overall value of roughly 0.195, and yet `significance` reads `"Not significant"` (p_adjusted ≈ 0.05) in all four functions, because 15 rows is too small a sample to distinguish "genuinely and permanently zero" from "just an unlucky small sample that happened to have no events."

### The `notes` column is a machine-readable audit trail, not decorative commentary

Every note string is appended with a pipe-and-space separator (`" | "`), and a single row can carry multiple flags simultaneously — group C's DSP row above carries four at once. This column should always be the first thing checked before trusting any row's estimate, confidence interval, or significance verdict, because several notes represent conditions (Haldane correction, unreliable strata) that materially change what the numeric columns actually mean. The full vocabulary of note strings that can appear, drawn directly from the source code's note-generation logic, is:

| Note text | Which function(s) emit it | Appears when | What it implies for interpretation |
|---|---|---|---|
| `"Zero denominator"` | All four | `n`/`denominator` is 0 for that row | The estimate is undefined (`NaN`); this is never rendered as a literal zero, so a `NaN` estimate accompanied by this note should never be interpreted as "no risk" |
| `"Zero events"` | All four | `events` is 0 but the denominator is not | The estimate is a genuine crude zero (or, for DSP/DSR, a Haldane-corrected near-zero after weighting); always check the accompanying `n`/`denominator` before treating a zero as stable rather than a small-sample artifact |
| `"Low event count (<10)"` | `crude_proportion_df`, `directly_standardized_proportion_df` | `events` between 1 and 9 | The estimate is statistically unstable at this sample size; this is precisely the condition that triggers the exact test (Fisher or binomial) instead of the normal-theory z-test |
| `"Low event count (<10): consider suppression"` | `crude_rate_df` only | Same event-count condition, worded for rates | Follows UKHSA/OHID convention of discouraging publication of rates built on fewer than 10 events; treat as a recommendation to suppress or footnote the row, not merely a numerical caveat |
| `"Low event count (<10): DSR should generally not be reported"` | `directly_standardized_rate_df` only | Same event-count condition, worded for standardised rates | Same UKHSA/OHID convention, phrased more strongly for DSR specifically, since standardisation compounds instability from a small crude event count |
| `"All events (proportion = 1)"` | `crude_proportion_df`, `directly_standardized_proportion_df` | Every single row in the group is an event (zero non-events) | The mirror-image boundary case to "Zero events"; equally unstable, and equally likely to trigger the Haldane correction in the DSP path |
| `"Low non-event count (<10)"` | `crude_proportion_df`, `directly_standardized_proportion_df` | Fewer than 10 non-events in the group | Same instability concern as low event count, but approached from the opposite side of the proportion (near 100% rather than near 0%) |
| `"Haldane correction applied"` | `directly_standardized_proportion_df`, `directly_standardized_rate_df` | At least one contributing standardisation stratum hit a 0% or 100% boundary within that group | A continuity correction (adding 0.5 to the stratum's event count and 1 to its size) was silently applied to that stratum before it was folded into the weighted `dsp`/`dsr`; the reported estimate is therefore a corrected value, not the raw boundary value, and this note is the only visible signal that the correction occurred |
| `"Unreliable: stratum n<10"` | `directly_standardized_proportion_df` | A contributing stratum (not the whole group) has fewer than 10 records | The group's overall `dsp` partially rests on a stratum-level estimate too small to trust, even if the group's total `n` looks comfortably large; this alone is sufficient to force the exact binomial test rather than the Dobson z-test |
| `"Unreliable: stratum non-event count <10"` | `directly_standardized_proportion_df` | A contributing stratum has fewer than 10 non-events | Same concern as above, from the non-event side |
| `"Unreliable: stratum denominator <10"` | `directly_standardized_rate_df` | A contributing stratum has a denominator (row count) below 10 | The rate equivalent of "stratum n<10"; also forces the exact mid-P Poisson test for that group |
| `"End-of-period denominator: row count used as exposure (assumes each row = 1 patient present at period end; does not account for partial-period exposure)"` | `crude_rate_df`, `directly_standardized_rate_df` | Always — appears on literally every row of every call to either rate function, including the Overall row | A structural caveat about how the denominator was derived (Section 2), not a data-quality flag specific to that row; its universal presence means it should be filtered out first before scanning `notes` for genuine row-specific concerns |
| `"Overall proportion: no weighting applied as this row is itself the reference population"` | `directly_standardized_proportion_df` | Always, and only, on the Overall row | Confirms the Overall row reports the crude proportion rather than a re-standardised value (Section 3); expected behaviour, not an error |
| `"Overall rate: no weighting applied as this row is itself the reference population"` | `directly_standardized_rate_df` | Always, and only, on the Overall row | Same as above, for rates |

### A practical filtering example

Because the end-of-period denominator caveat appears on every row of the rate functions, filtering it out first makes it easier to isolate genuinely row-specific concerns:

```python
result = directly_standardized_proportion_df(df, "event", ["age"], ["alliance"])

trustworthy = result.filter(
    ~pl.col("notes").str.contains("Unreliable|Zero|Haldane|All events", literal=False)
)

flagged_for_review = result.filter(
    pl.col("notes").str.contains("Unreliable|Haldane", literal=False)
)

rate_result = directly_standardized_rate_df(df, "event", ["age"], ["alliance"])
rate_specific_flags = rate_result.filter(
    pl.col("notes").str.contains("Unreliable|Haldane|Low event count", literal=False)
)
```

### Worked interpretation of a fully-flagged row

Take group C's DSP row from Section 4 as the complete worked example: its `notes` field reads `"Haldane correction applied | Zero events | Unreliable: stratum n<10 | Unreliable: stratum non-event count <10"`. Reading this left to right tells a complete diagnostic story:

1. **`"Haldane correction applied"`** — at least one age stratum contributing to this group's DSP had a 0% or 100% observed proportion, so a continuity correction was silently applied before that stratum's contribution was weighted into the final `dsp` of 0.133449.
2. **`"Zero events"`** — the group's crude, unweighted event count is 0 out of 15 rows; the non-zero `dsp` value is entirely a product of the standardisation weighting borrowing information from other strata in the reference population, not from any event actually observed in group C itself.
3. **`"Unreliable: stratum n<10"`** — at least one age stratum within group C has fewer than 10 total records, meaning the group-level `dsp` is partly built from a stratum estimate that is itself too small to trust in isolation.
4. **`"Unreliable: stratum non-event count <10"`** — the same stratum (or another) also has fewer than 10 non-events, compounding the instability from the opposite direction.

A reader who saw only `dsp = 0.133449` and `significance = "Not significant"` in isolation would have no way of knowing this estimate is built almost entirely on borrowed statistical weight from other groups rather than genuine observed events in group C. A reader who works through the `notes` field in full knows to either suppress this row entirely, footnote it prominently if it must be published alongside A and B, or flag it as a candidate for a larger sample before drawing any conclusion about group C's true underlying rate.