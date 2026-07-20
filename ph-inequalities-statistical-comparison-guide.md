# User Guide: `ph_inequalities_statistical_comparison.py`

A guide for data professionals using the public health standardisation module for crude and directly standardised proportions and rates, with built-in significance testing against the population reference.

---

## 1. Overview

The module provides four functions for summarising binary/count outcomes across groups (e.g. NHS regions, ICBs, tumour sites):

| Function | Measure | Use when |
|---|---|---|
| `crude_proportion_df` | Crude proportion | You want a raw % without adjusting for population structure |
| `crude_rate_df` | Crude rate (per multiplier, e.g. per 100,000) | You have person-time or a denominator and want a raw rate |
| `directly_standardized_proportion_df` | DSP | You need to compare proportions across groups with different age/sex/etc. mixes |
| `directly_standardized_rate_df` | DSR | You need to compare rates across groups with different age/sex/etc. mixes |

All four functions share a common design: they compute a value for every group **and** append a single **"Overall" row** representing the full input dataset, and they now test every group against that Overall value, flagging whether it is statistically significantly higher, lower, or indistinguishable.

---

## 2. Data Requirements

### Row-level data expected

Every function expects **row-level or pre-aggregated tabular data** in a Polars DataFrame — not pre-summarised percentages.

- `crude_proportion_df` needs a binary `event_col` (0/1 or boolean). Each row is one record; the proportion is `sum(event_col) / row_count`.
- `crude_rate_df` and `directly_standardized_rate_df` need an `event_col` (event count per row). The denominator is always the row count within each group/stratum — see Section 2 below for details on this design and its implications.

### Strata columns (DSP/DSR only)

`strata_cols` define the standardisation strata (e.g. age band, sex). **Numeric strata columns are automatically binned into quartiles (Q1–Q4)** using the full dataset's distribution — you do not need to pre-bin age into bands yourself. Categorical strata columns (e.g. sex, ethnicity) are used as-is.

### Group columns

`group_cols` define what you're comparing (e.g. region, ICB, tumour site). One output row is produced per unique combination of `group_cols`, plus the Overall row.

### Rate denominators: always inferred from row count

For `crude_rate_df` and `directly_standardized_rate_df`, there is no `denom_col` parameter. The denominator is **always** inferred from row count — each row is treated as one unit of exposure, interpreted as the population/headcount **at the end of the reporting period**. This is the only supported denominator method in the module.

This is a deliberate simplifying assumption designed for patient-level dataframes where one row represents one patient. It assumes every patient contributed a full unit of exposure, even if in reality they entered, exited, or died partway through the period. Every output row (including the Overall row) is flagged in `notes` to make this assumption explicit and auditable:

> `"End-of-period denominator: row count used as exposure (assumes each row = 1 patient present at period end; does not account for partial-period exposure)"`

**What this means for your input data:**
- Your dataframe must be **row-per-unit-of-exposure**: typically one row per patient (or one row per patient per stratum/group combination if you're pre-aggregating). Do not pass pre-aggregated summary tables with a separate person-time column — the function has no way to consume one.
- If you need true person-time weighting (e.g. because your cohort has substantial partial-year follow-up, mid-period entries/exits, or multiple rows per patient), you must expand or restructure your data so that row count in each group is *already* proportional to the exposure you want represented, since the module will not accept an external exposure column.
- This is a conscious trade-off: simplicity and consistency with `crude_proportion_df`/`directly_standardized_proportion_df` (which also use row count) in exchange for giving up support for fractional or externally-supplied person-time.

### Nulls and types (strictly enforced)

All four functions validate their inputs before doing any computation, and raise informative errors rather than silently dropping or coercing bad data:

- **Missing columns** raise a clear `ValueError` naming the missing fields.
- **Nulls/NAs are rejected outright** in the numerator (`event_col`), all `strata_cols`, and all `group_cols`. The error message names every offending column and its null count in one go, e.g. `"Null/NA values are not permitted... Offending column(s): 'age' (3 null values), 'region' (1 null value)."` You must remove or impute these rows before calling the function — there is no implicit drop-null behaviour.
- **The numerator column must be a non-negative integer column** (or boolean, which is accepted and cast to `Int64` automatically, since `True`/`False` is inherently a valid 0/1 encoding). Passing a float column (e.g. `1.0`, `0.0`) or a string column raises a `ValueError` naming the offending dtype. Passing negative integers (e.g. `-1`) also raises an error.
- **For the two proportion functions** (`crude_proportion_df`, `directly_standardized_proportion_df`), the numerator must contain *only* 0 or 1 values. If any other integer appears (e.g. a stray `2` from a miscoded field), the function raises an error rather than silently computing a nonsensical "proportion" greater than 1.
- **For the two rate functions** (`crude_rate_df`, `directly_standardized_rate_df`), the numerator may be any non-negative integer — e.g. a count of multiple events in one row (such as multiple emergency attendances for one patient) is valid and does not need to be 0/1.
- Zero-denominator or zero-event groups (once past validation) are handled gracefully and flagged in `notes` rather than raising errors — validation only rejects structurally invalid data, not sparse-but-valid data.

---

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

---

## 4. New Output Columns

All four functions now return four additional columns beyond the core estimate and confidence interval:

| Column | Type | Meaning |
|---|---|---|
| `test_method` | string | Which statistical test was actually used for that row |
| `p_value` | float | Raw (unadjusted) p-value comparing the group to the Overall row |
| `p_adjusted` | float | p-value after multiple-testing correction across all groups in that call |
| `significance` | string | Plain-language verdict: `"Higher"`, `"Lower"`, `"Not significant"`, `"Not tested"`, or `"Reference"` |

### Interpreting `significance`

- **`"Higher"`** — the group's estimate is statistically significantly above the Overall value after correction (p_adjusted < 0.05)
- **`"Lower"`** — statistically significantly below the Overall value
- **`"Not significant"`** — no detectable difference from the Overall value after correction
- **`"Not tested"`** — no valid test could be run (e.g. zero denominator, or `group_cols` was omitted entirely)
- **`"Reference"`** — reserved exclusively for the Overall row, which is never tested against itself

### Interpreting `test_method`

For non-Overall rows, `test_method` names the actual test applied (see Section 5 for the selection rules). For the Overall row, it instead records which multiple-testing correction was applied across the batch, e.g.:

> `"Reference row (correction: Benjamini-Hochberg (FDR))"`

This means every group's `p_adjusted` in that output was corrected using Benjamini-Hochberg, because at least one group was flagged as sparse.

### Practical use in a Polars pipeline

```python
significant_high = result.filter(pl.col("significance") == "Higher")
sparse_groups = result.filter(pl.col("test_method").str.contains("Exact", literal=False))
```

---

## 5. Statistical Testing Approach

### The comparison problem

Each group is compared against the same shared reference — the Overall row — rather than against every other group pairwise. This is a **many-to-one comparison**, not a many-to-many one, which shapes both the test choice and the correction method.

### Adaptive test selection

Rather than applying one test uniformly, each function inspects the group's event count (and, for DSP/DSR, whether any contributing stratum is unreliable) and selects the most defensible test automatically:

| Function | Dense groups (≥10 events, reliable strata) | Sparse groups (<10 events, or unreliable stratum) |
|---|---|---|
| Crude proportion | Two-proportion z-test | Fisher's exact test |
| Crude rate | Poisson z-test (test-based, SMR-style) | Mid-P exact Poisson test |
| DSP | Dobson-variance z-test vs the Overall proportion | Exact one-sample binomial test |
| DSR | Dobson-variance z-test vs the Overall rate (SMR-style) | Exact one-sample mid-P Poisson test |

Sparse-count situations break the normal approximations that z-tests rely on, so the module falls back to exact tests in exactly the same circumstances it already flags via the `notes` column (e.g. `"Low event count (<10)"`, `"Unreliable: stratum n<10"`). This keeps the statistical testing consistent with the reliability signals already surfaced elsewhere in the output.

### Why DSP and DSR share test-selection logic but not test execution

Both DSP and DSR use the same threshold rule to decide *whether* to use an exact or normal-theory test, because the same Dobson-variance approximation underlies both and breaks down under the same conditions regardless of whether the underlying measure is a proportion or a rate. What differs is *which* exact or normal test is actually run — a binomial test for DSP versus a mid-P Poisson test for DSR — because a proportion and a rate are different probability models with different natural test statistics.

### Multiple testing correction

Because every group shares the same reference row, this is precisely the situation **Dunnett's test** is designed for, which is more powerful than a blanket Bonferroni correction for many-to-one designs. However, exact Dunnett adjustment requires the full multivariate-t machinery and joint correlation structure across comparisons, which cannot be derived from a plain p-value vector — and Dunnett's normal-theory assumptions break entirely if any comparison in the batch used an exact test instead.

The module resolves this with two labelled correction methods, neither of which claims to be exact Dunnett:

- **If every group in the batch is dense** (no exact tests triggered): a **Holm-Sidak** step-down procedure is applied. This is a robust, assumption-light approximation to Dunnett-style many-to-one correction, but it is *not* the classical Dunnett test — it tends to be more conservative than true Dunnett when group comparisons are positively correlated (as they are here, since every group shares the same Overall reference). The `test_method` field reports this plainly as `"Holm-Sidak"` rather than `"Dunnett"`, so report consumers are not misled into thinking exact Dunnett quantiles were used.
- **If any group in the batch is sparse**: the whole batch switches to **Benjamini-Hochberg (FDR)** correction instead, since mixing exact-test and normal-theory p-values under Holm-Sidak's assumptions would not be statistically coherent.

This decision is made once per function call (not per row), and the chosen method is recorded in the Overall row's `test_method` field (e.g. `"Reference row (correction: Holm-Sidak)"`) for full auditability.

### Interpretation caveat

Because the Overall row is not statistically independent of each group — every group is a subset of the Overall population — these p-values approximate the UKHSA/ONS convention for testing a standardised rate against a reference/national rate (an SMR-style test), rather than a textbook two-independent-sample hypothesis test. This is a standard and accepted simplification in public health reporting, but analysts should not treat these p-values as fully independent significance tests when writing up results for publication.

---

## 6. Worked Example: Row-Count Denominator

```python
import polars as pl
from ph_inequalities_statistical_comparison import crude_rate_df

patient_df = pl.DataFrame({
    "icb": ["A"] * 60000 + ["B"] * 40000,
    "diagnosed": [1] * 800 + [0] * 59200 + [1] * 300 + [0] * 39700,
})

result = crude_rate_df(patient_df, "diagnosed", ["icb"], multiplier=100_000)
```

Here, the function automatically uses row count as exposure — 60,000 and 40,000 patients respectively — interpreted as the end-of-period population for each ICB. There is no `denom_col` argument to supply; the `notes` column on every row confirms this assumption was applied, so downstream consumers of the output know the rate was calculated from row count rather than true person-time.

## 7. Worked Example: Significance Testing

```python
import polars as pl
from ph_inequalities_statistical_comparison import crude_proportion_df

df = pl.DataFrame({
    "region": ["N"]*100 + ["S"]*100 + ["E"]*100,
    "event":  [1]*80 + [0]*20 + [1]*45 + [0]*55 + [1]*3 + [0]*97,
})

result = crude_proportion_df(df, "event", ["region"])
```

Output:

| region | events | n | proportion | p_value | p_adjusted | significance | test_method |
|---|---|---|---|---|---|---|---|
| E | 3 | 100 | 0.030 | 7.9e-27 | 2.4e-26 | Lower | Fisher exact |
| N | 80 | 100 | 0.800 | 4.4e-14 | 6.6e-14 | Higher | Two-proportion z-test |
| S | 45 | 100 | 0.450 | 0.637 | 0.637 | Not significant | Two-proportion z-test |
| Overall | 128 | 300 | 0.427 | NaN | NaN | Reference | Reference row (correction: Benjamini-Hochberg (FDR)) |

Region E's low event count triggered Fisher's exact test and the batch-wide switch to Benjamini-Hochberg, since at least one group was sparse. Region S, close to the Overall proportion of 42.7%, is correctly flagged as not significantly different, while N and E are both flagged in the expected direction.
