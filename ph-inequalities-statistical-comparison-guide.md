# User guide

## Public API

The package exposes four functions from `ph_inequalities_statistical_comparison`:

| Function | Use when | Event column | Result |
|---|---|---|---|
| `crude_proportion_df()` | The outcome is binary and no adjustment is required | Boolean or integer 0/1 | Crude proportion with Wilson limits |
| `crude_rate_df()` | Rows represent end-of-period exposure and may contain multiple events | Non-negative integer count | Crude rate with exact/Byar Poisson limits |
| `directly_standardized_proportion_df()` | A binary outcome should be adjusted for influential characteristics (e.g age) | Boolean or integer 0/1 | DSP with Wilson-MOVER limits |
| `directly_standardized_rate_df()` | A count-based rate should be adjusted for influential characteristics (e.g age) | Non-negative integer count | DSR with Poisson-MOVER limits |


## Installation and import

Python 3.12 or later is required.

```bash
pip install git+https://github.com/Norfolk-Suffolk-ICS/ph_inequalities_statistical_comparison.git
```

```python
from ph_inequalities_statistical_comparison import (
    crude_proportion_df,
    crude_rate_df,
    directly_standardized_proportion_df,
    directly_standardized_rate_df,
)
```

## Input requirements

### Dataframe

- `df` must be a non-empty `polars.DataFrame`; pandas dataframes are not accepted.
- Required columns must exist.
- Nulls, `NaN`, and infinite values are not permitted in the event, stratum, inequality, or organisational columns used in an analysis.
- Output dimension columns are cast to strings so inactive dimensions can contain `all_label`.

### Event column

- All functions require `event_col` to be a non-empty column name.
- Boolean event columns are accepted and converted to integers.
- Proportion functions require an integer column containing only 0 and 1.
- Rate functions require a non-negative integer column; values greater than 1 are allowed.
- Floating-point event columns are rejected even when all values are mathematically whole numbers; cast them to an integer type first.

### Standardisation strata

- `strata_cols` is required by the DSP and DSR functions.
- Supply a non-empty sequence such as `["age_band", "sex"]`, not a single string such as `"age_band"`.
- A stratum column cannot also be the event column, an inequality dimension, or an organisational dimension.
- Numeric strata are automatically converted to quartile labels `Q1`-`Q4` using cut points calculated from the full input dataframe.
- With multiple strata, weights are based on the observed joint combinations.

### Dimension columns

- `inequalities_cols` must be `None` or a sequence of unique column names.
- `organisational_cols` must be `None` or a mapping from a hierarchy name to an ordered sequence of unique columns, from highest to lowest level.
- A column cannot appear in more than one organisational hierarchy or in both `inequalities_cols` and `organisational_cols`.
- Within each organisational hierarchy, every child value must map to exactly one value of its immediate parent.
- The event, strata, inequality, and organisational roles cannot overlap.
- Values in dimension columns must not equal `all_label`; choose another label or recode the source value if a collision exists.

The following names are reserved for generated or internal analysis columns and cannot be used for strata, inequalities, or organisational dimensions:

```text
events, n, denominator, proportion, rate, dsp, dsr,
lower, upper, dsp_lower, dsp_upper, dsr_lower, dsr_upper,
confidence, multiplier, method, notes, significance,
ref_count, ref_weight, _parents, _denom_end_of_period_n
```

Rate inputs must also not already contain the internal column `_denom_end_of_period_n`.

## Function signatures

```python
crude_proportion_df(
    df,
    event_col,
    inequalities_cols=None,
    organisational_cols=None,
    *,
    organisational_mode="separate",
    all_label="All",
    confidence=0.95,
)
```

```python
crude_rate_df(
    df,
    event_col,
    inequalities_cols=None,
    organisational_cols=None,
    *,
    organisational_mode="separate",
    all_label="All",
    multiplier=100_000.0,
    confidence=0.95,
)
```

```python
directly_standardized_proportion_df(
    df,
    event_col,
    strata_cols,
    inequalities_cols=None,
    organisational_cols=None,
    *,
    organisational_mode="separate",
    all_label="All",
    confidence=0.95,
)
```

```python
directly_standardized_rate_df(
    df,
    event_col,
    strata_cols,
    inequalities_cols=None,
    organisational_cols=None,
    *,
    organisational_mode="separate",
    all_label="All",
    multiplier=100_000.0,
    confidence=0.95,
)
```

## Arguments

| Argument | Functions | Requirement and effect |
|---|---|---|
| `df` | All | Non-empty Polars dataframe containing row-level analytical data. |
| `event_col` | All | Name of the event column. Binary for proportions; non-negative integer count for rates. |
| `strata_cols` | DSP, DSR | Non-empty sequence defining the joint standardisation strata. Numeric columns are binned into quartiles. |
| `inequalities_cols` | All | Optional sequence of inequality dimensions. The function returns the complete marginal cube over these columns. |
| `organisational_cols` | All | Optional named mapping of top-to-bottom organisational hierarchies, for example `{"commissioning": ["region", "icb", "practice"]}`. |
| `organisational_mode` | All | `"separate"` (default) rolls multiple hierarchies up independently; `"cross"` generates valid cross-hierarchy combinations. |
| `all_label` | All | Non-empty string used for inactive dimensions and the overall reference row; default `"All"`. It must not already occur in a dimension column. |
| `multiplier` | Rate, DSR | Finite positive number used to scale rates; default 100,000. |
| `confidence` | All | Finite real number strictly between 0 and 1; default 0.95. Booleans are rejected. |

Arguments after `*` are keyword-only.

## Grouping behaviour

### Inequality cube

Every subset of `inequalities_cols` is generated. For `inequalities_cols=["sex", "deprivation"]`, this includes:

- `sex × deprivation` combinations;
- sex margins with deprivation set to `all_label`;
- deprivation margins with sex set to `all_label`;
- the overall row with both set to `all_label`.

### Organisational hierarchies

For a hierarchy declared as:

```python
{
    "commissioning": ["region", "icb", "practice"]
}
```

only valid top-down prefixes are generated:

- region × ICB × practice;
- region × ICB;
- region;
- overall.

Lower levels are never grouped without their parents. Each organisational state is combined with the complete inequality cube.

With multiple hierarchies, `organisational_mode="separate"` generates each hierarchy's prefixes independently plus overall. `organisational_mode="cross"` takes the Cartesian product of valid prefixes, including states in which one hierarchy is active and another is overall. Duplicate grouping sets are removed.

## Worked input

The examples below use the same dataframe so the four outputs can be compared directly.

```python
import polars as pl

example = pl.DataFrame(
    {
        "event": [
            1, 0, 0, 0, 1, 1, 0, 0,
            1, 1, 1, 0, 1, 1, 0, 0,
        ],
        "count": [
            1, 1, 1, 1, 3, 3, 3, 3,
            0, 0, 0, 0, 2, 2, 2, 2,
        ],
        "age_band": ["A"] * 4 + ["B"] * 4 + ["A"] * 4 + ["B"] * 4,
        "group": ["G1"] * 8 + ["G2"] * 8,
    }
)
```

The displayed tables omit `confidence` and `notes` only to keep the worked output compact. The returned dataframes contain every column listed in the output schemas below.

## `crude_proportion_df()`

```python
crude_proportions = crude_proportion_df(
    example,
    event_col="event",
    inequalities_cols=["group"],
)
```

| group | events | n | proportion | lower | upper | method | significance |
|---|---:|---:|---:|---:|---:|---|---|
| G1 | 3 | 8 | 0.375000 | 0.136844 | 0.694258 | Wilson score | Not significant |
| G2 | 5 | 8 | 0.625000 | 0.305742 | 0.863156 | Wilson score | Not significant |
| All | 8 | 16 | 0.500000 | 0.279996 | 0.720004 | Wilson score | Reference |

Output columns, after any dimension columns:

```text
events, n, proportion, lower, upper,
confidence, method, notes, significance
```

- `events`: sum of the binary event column.
- `n`: row count.
- `proportion`: `events / n`.
- `lower`, `upper`: Wilson score confidence limits.
- `method`: always `Wilson score`.

## `crude_rate_df()`

```python
crude_rates = crude_rate_df(
    example,
    event_col="count",
    inequalities_cols=["group"],
    multiplier=1_000,
)
```

| group | events | denominator | rate | lower | upper | method | significance |
|---|---:|---:|---:|---:|---:|---|---|
| G1 | 16 | 8 | 2000.000000 | 1142.438615 | 3248.054774 | Byar | Not significant |
| G2 | 8 | 8 | 1000.000000 | 431.729022 | 1970.398653 | Exact chi-square | Not significant |
| All | 24 | 16 | 1500.000000 | 960.795056 | 2231.976027 | Byar | Reference |

Output columns, after any dimension columns:

```text
events, denominator, rate, lower, upper,
multiplier, confidence, method, notes, significance
```

- `events`: sum of the non-negative integer count column.
- `denominator`: row count, not an externally supplied person-time denominator.
- `rate`: `events / denominator * multiplier`.
- `lower`, `upper`: scaled Poisson confidence limits.
- `method`: `Exact chi-square` when total events are below 10; otherwise `Byar`.

## `directly_standardized_proportion_df()`

```python
dsp = directly_standardized_proportion_df(
    example,
    event_col="event",
    strata_cols=["age_band"],
    inequalities_cols=["group"],
)
```

| group | events | n | dsp | dsp_lower | dsp_upper | method | significance |
|---|---:|---:|---:|---:|---:|---|---|
| G1 | 3 | 8 | 0.375000 | 0.172357 | 0.659779 | Wilson-MOVER | Not significant |
| G2 | 5 | 8 | 0.625000 | 0.340221 | 0.827643 | Wilson-MOVER | Not significant |
| All | 8 | 16 | 0.500000 | 0.298627 | 0.701373 | Wilson-MOVER | Reference |

Output columns, after any dimension columns:

```text
events, n, dsp, dsp_lower, dsp_upper,
confidence, method, notes, significance
```

- `events`, `n`: unstandardised totals for the displayed group.
- `dsp`: reference-weighted sum of uncorrected stratum proportions.
- `dsp_lower`, `dsp_upper`: Wilson-MOVER limits.
- `method`: always `Wilson-MOVER`.
- DSP estimates and limits are rounded to six decimal places.

## `directly_standardized_rate_df()`

```python
dsr = directly_standardized_rate_df(
    example,
    event_col="count",
    strata_cols=["age_band"],
    inequalities_cols=["group"],
    multiplier=1_000,
)
```

| group | events | denominator | dsr | dsr_lower | dsr_upper | method | significance |
|---|---:|---:|---:|---:|---:|---|---|
| G1 | 16 | 8 | 2000.000000 | 1188.132828 | 3365.250971 | Poisson-MOVER using exact and Byar stratum intervals | Not significant |
| G2 | 8 | 8 | 1000.000000 | 431.729022 | 2074.381643 | Poisson-MOVER using exact stratum intervals | Not significant |
| All | 24 | 16 | 1500.000000 | 980.344449 | 2284.485609 | Poisson-MOVER using exact and Byar stratum intervals | Reference |

Output columns, after any dimension columns:

```text
events, denominator, dsr, dsr_lower, dsr_upper,
multiplier, confidence, method, notes, significance
```

- `events`: summed count for the displayed group.
- `denominator`: row count.
- `dsr`: reference-weighted sum of uncorrected stratum rates, scaled by `multiplier`.
- `dsr_lower`, `dsr_upper`: scaled Poisson-MOVER limits.
- `method`: states whether exact, Byar, or both types of stratum interval were used.
- DSR estimates and limits are rounded to six decimal places.

## Significance values

| Value | Meaning |
|---|---|
| `Reference` | The overall row: all generated dimensions equal `all_label`. |
| `Higher` | The fixed overall estimate is below the result's lower confidence limit. |
| `Lower` | The fixed overall estimate is above the result's upper confidence limit. |
| `Not significant` | The fixed overall estimate is inside the result's confidence interval, including its limits. |
| `Not tested` | The benchmark or confidence interval is non-finite. |

These values are descriptive labels, not p-value-based tests, and there is no multiple-comparison correction.

## Notes column

`notes` is a pipe-separated string. Several fragments can occur in one row, in the order documented below. An empty string means that no note condition was triggered. The package does not suppress results automatically; where a note says a result **must** be flagged or suppressed, the user must apply that action in downstream tables and visual outputs.

### Crude proportion notes

| Exact note text | Trigger | Interpretation/action |
|---|---|---|
| `Zero denominator - suppress the result and reconsider the organisational hierarchy or inequality dimensions.` | `n == 0` | The estimate is undefined. Validated public calls generate only non-empty groups, so this is defensive logic rather than a normally reachable public output. |
| `No events (proportion = 0) - must either flag as a boundary estimate or suppress in visual outputs.` | `events == 0` and `n > 0` | The point estimate is zero. It must be visibly flagged as a boundary estimate or suppressed downstream. |
| `Low event count (<10) - must flag as unstable or suppress in visual outputs.` | `0 < events < 10` | The result must be marked as unstable or suppressed downstream. |
| `All events (proportion = 1) - flag as a boundary estimate or suppress in visual outputs.` | `events == n` and `n > 0` | The point estimate is one. It should be flagged as a boundary estimate or suppressed downstream. |
| `Low non-event count (<10) - must flag as unstable or suppress in visual outputs.` | `0 < n - events < 10` | The non-event complement is sparse. The result must be marked as unstable or suppressed downstream. |
| `Low sample size (<40) - flag the proportion as unstable.` | `0 < n < 40` | The proportion should be marked as unstable because its denominator is small. |

`No events` and `Low event count` are mutually exclusive. `All events` and `Low non-event count` are also mutually exclusive. The sample-size note can accompany either an event-side or non-event-side warning.

### Crude rate notes

| Exact note text | Trigger | Interpretation/action |
|---|---|---|
| `End-of-period denominator: row count used as exposure (assumes each row represents one patient present at the period end and does not account for partial-period exposure).` | Every crude-rate row | Confirms the denominator construction and its exposure limitation. |
| `Zero denominator.` | `denominator == 0` | The rate is undefined. This is defensive logic and is not normally reachable from a validated non-empty public result group. |
| `Zero event count - flag as unstable or suppress in visual outputs.` | `events == 0` and denominator is positive | The result should be marked as unstable or suppressed downstream. The exact Poisson upper limit remains positive. |
| `Low event count (<10) - flag as unstable or suppress in visual outputs.` | `0 < events < 10` | The exact interval is used; flag the result as unstable or suppress it downstream. |
| `Low denominator (<40) - flag the rate as unstable.` | `0 < denominator < 40` | The rate should be marked as unstable because its row-count exposure is small. |

The end-of-period note is always first. `Zero event count` and `Low event count` are mutually exclusive.

### DSP notes

| Exact note text | Trigger | Interpretation/action |
|---|---|---|
| `Reference population: full standardisation weights used; the standardised estimate equals the overall observed proportion.` | Overall reference row | All empirical reference strata are present; the reference DSP equals the full-data crude proportion. |
| `Calculated with missing standardisation stratum: 1 stratum omitted, representing {missing%} of the reference population; the remaining reference weights ({coverage%} coverage) were renormalised to sum to 1 Strongly consider coarsening strata, inequalities or dimensions` | One positively weighted reference stratum is absent from a result group | The effective standard population excludes that stratum. Percentages are displayed to one decimal place. Strongly consider reducing the granularity of the analysis. |
| `Calculated with missing standardisation strata: {count} strata omitted, representing {missing%} of the reference population; the remaining reference weights ({coverage%} coverage) were renormalised to sum to 1 Strongly consider coarsening strata, inequalities or dimensions` | More than one positively weighted reference stratum is absent from a result group | Same as above, with plural wording and the number of omitted strata inserted. |
| `Unreliable: at least one stratum has n < 10 flag as unstable or suppress in visual outputs and strongly consider coarsening strata, inequalities or dimensions` | At least one observed stratum has `0 < n_i < 10` | Flag or suppress the result and strongly consider reducing the granularity of the analysis. |
| `Unreliable: stratum non-event count < 10 flag as unstable or suppress in visual outputs and strongly consider coarsening strata, inequalities or dimensions` | At least one observed stratum has `0 < n_i - events_i < 10` | Flag or suppress the result and strongly consider reducing the granularity of the analysis. A stratum with zero non-events does not trigger this fragment. |
| `Zero total events flag as unstable or suppress in visual outputs` | Total group events equal zero | The DSP point estimate is zero. Flag the result as unstable or suppress it downstream. |
| `low total event count (<10) flag as unstable or suppress in visual outputs` | Total group events are from 1 to 9 | Flag the result as unstable or suppress it downstream. The initial `low` is lowercase in the generated text. |
| `All events (proportion = 1) flag as unstable or suppress in visual outputs` | Every row in the group is an event | The DSP point estimate is one. Flag the result as unstable or suppress it downstream. |
| `Low non-event count (<10) flag as unstable or suppress in visual outputs` | Total group non-events are from 1 to 9 | Flag the result as unstable or suppress it downstream. |

The reference note is first when present, followed by missing-strata, stratum-level, and total-count notes. No Haldane-Anscombe correction is used or reported. A well-populated, complete non-reference group can have an empty `notes` value.

### DSR notes

| Exact note text | Trigger | Interpretation/action |
|---|---|---|
| `End-of-period denominator: row count used as exposure (assumes each row represents one patient present at the period end and does not account for partial-period exposure).` | Every DSR row | Confirms that each row contributes one unit of exposure. |
| `Reference population: full standardisation weights used.` | Overall reference row | All empirical reference weights are used. |
| `Calculated with missing standardisation stratum: 1 stratum omitted, representing {missing%} of the reference population; the remaining reference weights ({coverage%} coverage) were renormalised to sum to 1 Strongly consider coarsening strata, inequalities or dimensions` | One positively weighted reference stratum is absent from a result group | The effective standard population excludes that stratum. Percentages are displayed to one decimal place. Strongly consider reducing the granularity of the analysis. |
| `Calculated with missing standardisation strata: {count} strata omitted, representing {missing%} of the reference population; the remaining reference weights ({coverage%} coverage) were renormalised to sum to 1 Strongly consider coarsening strata, inequalities or dimensions` | More than one positively weighted reference stratum is absent from a result group | Same as above, with plural wording and the number of omitted strata inserted. |
| `Unreliable: stratum denominator < 10 flag as unstable or suppress in visual outputs and strongly consider coarsening strata, inequalities or dimensions` | At least one observed stratum has `0 < denominator_i < 10` | Flag or suppress the result and strongly consider reducing the granularity of the analysis. |
| `Zero total events flag as unstable or suppress in visual outputs` | Total group events equal zero | The DSR point estimate and lower limit are zero, while the Poisson-MOVER upper limit remains positive. Flag or suppress the result downstream. |
| `Low total event count (<10) flag as unstable or suppress in visual outputs` | Total group events are below 10, including zero | Flag the calculated DSR as unstable or suppress it downstream. |

The end-of-period note is always first. A zero-event DSR receives both the `Zero total events` and `Low total event count (<10)` fragments. There is no separate DSR note for a low event count within an individual stratum; exact versus Byar stratum treatment is recorded in `method`.

## Missing-strata example

Suppose the full input has reference weights 66.7% for stratum A and 33.3% for stratum B, but subgroup G1 has no rows in B. The standardised functions omit B and renormalise A's retained weight to 1. The subgroup note is:

```text
Calculated with missing standardisation stratum: 1 stratum omitted,
representing 33.3% of the reference population; the remaining reference
weights (66.7% coverage) were renormalised to sum to 1 Strongly consider
coarsening strata, inequalities or dimensions
```

This result should not be interpreted as though G1 had a zero outcome in B. It is an estimate for the retained standard-population coverage and may not be directly comparable when different groups omit different or substantial reference shares. The revised note explicitly recommends coarsening the strata, inequality categories, or dimensions.

## Publication checks

Before publishing output:

- review every non-empty `notes` value and apply the organisation's disclosure-control and suppression rules;
- confirm that row count is a defensible exposure denominator for rate outputs;
- inspect missing-reference-weight coverage for standardised outputs;
- avoid treating `significance` as a formal hypothesis test;
- consider multiplicity when many cube cells are compared;
- retain `method`, `confidence`, and `multiplier` with extracts so the estimates remain interpretable.
