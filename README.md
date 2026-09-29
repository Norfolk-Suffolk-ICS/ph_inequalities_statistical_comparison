# ph-inequalities-statistical-comparison

A Polars-based Python package for producing crude and directly standardised proportions and rates across inequality dimensions and organisational hierarchies.

The package exposes four public functions:

| Function | Measure | Confidence interval |
|---|---|---|
| `crude_proportion_df()` | Crude proportion from a binary event column | Wilson score |
| `crude_rate_df()` | Crude rate using row count as end-of-period exposure | Exact Poisson below 10 events; Byar from 10 events |
| `directly_standardized_proportion_df()` | Directly standardised proportion (DSP) | Wilson-MOVER |
| `directly_standardized_rate_df()` | Directly standardised rate (DSR), using row count as end-of-period exposure | Poisson-MOVER with exact or Byar stratum intervals |

All four functions:

- accept a non-empty `polars.DataFrame`;
- create a complete marginal cube over the requested inequality dimensions;
- support one or more named organisational hierarchies;
- return estimates, confidence limits, method metadata, quality notes, and a descriptive comparison with the overall reference row;
- return results rather than automatically suppressing low-count estimates.

The package currently supports proportions and rates only. It does not calculate means or accept externally supplied person-time denominators.

## Installation

The project requires Python 3.12 or later.

```bash
pip install git+https://github.com/Norfolk-Suffolk-ICS/ph_inequalities_statistical_comparison.git
```

Import the public API with:

```python
from ph_inequalities_statistical_comparison import (
    crude_proportion_df,
    crude_rate_df,
    directly_standardized_proportion_df,
    directly_standardized_rate_df,
)
```

## Minimal example

```python
import polars as pl

from ph_inequalities_statistical_comparison import (
    directly_standardized_proportion_df,
)

patients = pl.DataFrame(
    {
        "event": [1, 0, 1, 0, 1, 0],
        "age_band": ["18-39", "18-39", "40-64", "40-64", "65+", "65+"],
        "sex": ["Female", "Male", "Female", "Male", "Female", "Male"],
        "region": ["East"] * 6,
        "icb": ["ICB A"] * 6,
    }
)

result = directly_standardized_proportion_df(
    patients,
    event_col="event",
    strata_cols=["age_band"],
    inequalities_cols=["sex"],
    organisational_cols={"commissioning": ["region", "icb"]},
)
```

Reference weights for directly standardised measures are derived from the stratum distribution of the full input dataframe. Numeric strata are converted to quartiles before those weights are calculated. Rates use the number of rows in each result group as the denominator, so each row is assumed to represent one person present at the end of the period; partial-period exposure is not represented.

## Documentation

- [User guide](./ph-inequalities-statistical-comparison-guide.md): input requirements, arguments, grouping behaviour, worked outputs, output schemas, and a complete guide to `notes`.
- [Statistical documentation](./statistical-documentation.md): formulas, interval construction, standardisation, missing strata, and significance labels.
- [Snowflake integration issues](./further-issues.md): implementation considerations retained by the project.

## Development

The source package is under `src/ph_inequalities_statistical_comparison/`, and the tests are under `tests/`.

```bash
pytest
```

Helper functions are private and begin with `_`. Only the four functions listed above form the supported public API.
