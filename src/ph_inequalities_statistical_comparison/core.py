"""
ph_inequalities_statistical_comparison.py
=========================================
Public health standardisation functions using UKHSA/OHID-recommended methods.

Input validation
----------------
All four public functions validate their numerator, strata, inequality, and
organisational columns before computation:

- Null/NA and non-finite numeric values are rejected outright, with an
  informative error listing every offending column and value type.
- The numerator column must be a non-negative integer column or boolean.
- For proportion functions, the numerator must contain only 0 or 1.
- For rate functions, the numerator may contain any non-negative integer.

Public API
----------
- crude_proportion_df
- crude_rate_df
- directly_standardized_proportion_df
- directly_standardized_rate_df

The overall/reference value is treated as a fixed benchmark when assigning
the descriptive Higher, Lower, or Not significant labels. No formal
hypothesis test or multiple-testing correction is performed.

References
----------
UKHSA Fingertips technical guidance:
https://fingertips.phe.org.uk/static-reports/public-health-technical-guidance

Dependencies: polars, numpy, scipy
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations, pairwise, product
from numbers import Real
from typing import Literal

import numpy as np
import polars as pl
from scipy import stats


_INTEGER_DTYPES = (
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
)

_FLOAT_DTYPES = (
    pl.Float32,
    pl.Float64,
)

_NUMERIC_DTYPES = (
    *_INTEGER_DTYPES,
    *_FLOAT_DTYPES,
)

_DENOM_COL = "_denom_end_of_period_n"

_RESERVED_ANALYSIS_COLUMNS = frozenset(
    {
        "events",
        "n",
        "denominator",
        "proportion",
        "rate",
        "dsp",
        "dsr",
        "lower",
        "upper",
        "dsp_lower",
        "dsp_upper",
        "dsr_lower",
        "dsr_upper",
        "confidence",
        "multiplier",
        "method",
        "notes",
        "significance",
        "ref_count",
        "ref_weight",
        "_parents",
        _DENOM_COL,
    }
)


# ===========================================================================
# Shared helpers
# ===========================================================================


def _flatten_organisational_cols(
    organisational_cols: Mapping[str, Sequence[str]],
) -> list[str]:
    """Flatten named organisational hierarchies in declaration order."""
    return [
        col
        for levels in organisational_cols.values()
        for col in levels
    ]


def _dimension_cols(
    inequalities_cols: Sequence[str],
    organisational_cols: Mapping[str, Sequence[str]],
) -> list[str]:
    """
    Return all output dimensions.

    Organisational columns are placed first, followed by inequality columns,
    matching the ordering used in output tables and grouping sets.
    """
    return (
        _flatten_organisational_cols(organisational_cols)
        + list(inequalities_cols)
    )


def _cast_dimension_cols_to_utf8(
    df: pl.DataFrame,
    inequalities_cols: Sequence[str],
    organisational_cols: Mapping[str, Sequence[str]],
) -> pl.DataFrame:
    """
    Cast inequality and organisational dimensions to a common output dtype.

    This permits the configured all_label string to be inserted into rolled-up
    rows even where the original grouping column was numeric.
    """
    dimensions = _dimension_cols(
        inequalities_cols,
        organisational_cols,
    )

    if not dimensions:
        return df

    return df.with_columns(
        pl.col(col).cast(pl.Utf8)
        for col in dimensions
    )


def _bin_numeric_to_quartiles(
    series: pl.Series,
) -> pl.Series:
    """Bin a finite numeric Polars Series into quartile labels Q1-Q4."""
    q1 = series.quantile(
        0.25,
        interpolation="linear",
    )
    q2 = series.quantile(
        0.50,
        interpolation="linear",
    )
    q3 = series.quantile(
        0.75,
        interpolation="linear",
    )

    if q1 is None or q2 is None or q3 is None:
        raise ValueError(
            f"Unable to calculate quartiles for numeric stratum "
            f"'{series.name}'."
        )

    def label(value: float) -> str:
        if value <= q1:
            return "Q1"
        if value <= q2:
            return "Q2"
        if value <= q3:
            return "Q3"
        return "Q4"

    return pl.Series(
        name=series.name,
        values=[
            label(value)
            for value in series.to_list()
        ],
        dtype=pl.Utf8,
    )


def _build_reference_weights(
    df: pl.DataFrame,
    strata_cols: list[str],
) -> pl.DataFrame:
    """Derive standard-population weights from the full input dataset."""
    total = len(df)

    if total == 0:
        raise ValueError(
            "The input dataframe contains no rows, so reference weights "
            "cannot be calculated."
        )

    return (
        df.group_by(strata_cols)
        .agg(
            pl.len().alias("ref_count"),
        )
        .with_columns(
            (
                pl.col("ref_count") / total
            ).alias("ref_weight"),
        )
    )


def _bin_strata(
    df: pl.DataFrame,
    strata_cols: list[str],
) -> pl.DataFrame:
    """
    Auto-bin numeric strata columns into quartiles.

    Categorical strata are left unchanged.
    """
    for col in strata_cols:
        if df.schema[col] in _NUMERIC_DTYPES:
            df = df.with_columns(
                _bin_numeric_to_quartiles(
                    df[col],
                ).alias(col),
            )

    return df


def _check_columns(
    df: pl.DataFrame,
    cols: Sequence[str],
) -> None:
    """Raise an error if required columns are absent."""
    missing = sorted(
        set(cols) - set(df.columns),
    )

    if missing:
        raise ValueError(
            f"Columns not found in dataframe: {missing}"
        )


def _check_reserved_column_names(
    strata_cols: Sequence[str],
    dimension_cols: Sequence[str],
) -> None:
    """Reject analytical columns that collide with generated columns."""
    conflicts = sorted(
        (
            set(strata_cols)
            | set(dimension_cols)
        )
        & _RESERVED_ANALYSIS_COLUMNS
    )

    if conflicts:
        raise ValueError(
            "Strata, inequality, and organisational columns cannot use "
            "reserved analysis column names. Conflicting column(s): "
            f"{conflicts}."
        )


def _check_no_missing_or_nonfinite(
    df: pl.DataFrame,
    cols: Sequence[str],
) -> None:
    """
    Reject null, NaN, and infinite values in analytical columns.

    Polars treats NaN as distinct from null, so both conditions must be
    checked explicitly for floating-point columns.
    """
    offending: dict[str, dict[str, int]] = {}

    for col in dict.fromkeys(cols):
        series = df[col]

        counts = {
            "null": int(series.null_count()),
            "NaN": 0,
            "infinite": 0,
        }

        if series.dtype in _FLOAT_DTYPES:
            counts["NaN"] = int(
                series
                .is_nan()
                .fill_null(False)
                .sum()
            )

            counts["infinite"] = int(
                series
                .is_infinite()
                .fill_null(False)
                .sum()
            )

        if any(counts.values()):
            offending[col] = counts

    if not offending:
        return

    detail = ", ".join(
        (
            f"'{col}' ("
            + ", ".join(
                (
                    f"{count} {kind} "
                    f"value{'s' if count != 1 else ''}"
                )
                for kind, count in counts.items()
                if count
            )
            + ")"
        )
        for col, counts in offending.items()
    )

    raise ValueError(
        "Missing or non-finite values are not permitted in event, "
        "strata, inequality, or organisational columns. Offending "
        f"column(s): {detail}. Remove, recode, or impute these values "
        "before calling this function."
    )


def _validate_numerator_col(
    df: pl.DataFrame,
    event_col: str,
    binary: bool,
) -> pl.DataFrame:
    """
    Validate that event_col is a non-negative integer column.

    Boolean columns are accepted and cast to Int64. Other non-integer
    dtypes raise an error.

    When binary=True, values must be exactly 0 or 1.
    """
    dtype = df.schema[event_col]

    if dtype == pl.Boolean:
        return df.with_columns(
            pl.col(event_col).cast(pl.Int64),
        )

    if dtype not in _INTEGER_DTYPES:
        raise ValueError(
            f"Column '{event_col}' must be a non-negative integer "
            f"column or boolean, but has dtype {dtype}. Cast it to an "
            "integer type before calling this function."
        )

    min_value = df[event_col].min()

    if min_value is not None and min_value < 0:
        raise ValueError(
            f"Column '{event_col}' must contain only non-negative "
            f"integers, but found a minimum value of {min_value}."
        )

    if binary:
        distinct_values = set(
            df[event_col]
            .unique()
            .to_list(),
        )

        invalid = distinct_values - {0, 1}

        if invalid:
            raise ValueError(
                f"Column '{event_col}' must contain only 0 or 1 "
                "values for proportion functions, but found other "
                f"value(s): {sorted(invalid)}."
            )

    return df


def _prepare_dataframe_proportion(
    df: pl.DataFrame,
    event_col: str,
    strata_cols: Sequence[str],
    inequalities_cols: Sequence[str],
    organisational_cols: Mapping[str, Sequence[str]],
) -> pl.DataFrame:
    """
    Validate and prepare inputs for directly standardised proportions.

    The inequality/organisational distinction is preserved in the function
    interface. The columns are flattened only for common validation.
    """
    dimensions = _dimension_cols(
        inequalities_cols,
        organisational_cols,
    )

    required = list(
        dict.fromkeys(
            [
                event_col,
                *strata_cols,
                *dimensions,
            ],
        ),
    )

    _check_columns(
        df,
        required,
    )

    _check_no_missing_or_nonfinite(
        df,
        required,
    )

    df = _validate_numerator_col(
        df,
        event_col,
        binary=True,
    )

    return _bin_strata(
        df,
        list(strata_cols),
    )


def _prepare_dataframe_rate(
    df: pl.DataFrame,
    event_col: str,
    strata_cols: Sequence[str],
    inequalities_cols: Sequence[str],
    organisational_cols: Mapping[str, Sequence[str]],
) -> pl.DataFrame:
    """
    Validate and prepare inputs for directly standardised rates.

    The inequality/organisational distinction is preserved in the function
    interface. The columns are flattened only for common validation.
    """
    dimensions = _dimension_cols(
        inequalities_cols,
        organisational_cols,
    )

    required = list(
        dict.fromkeys(
            [
                event_col,
                *strata_cols,
                *dimensions,
            ],
        ),
    )

    _check_columns(
        df,
        required,
    )

    _check_no_missing_or_nonfinite(
        df,
        required,
    )

    df = _validate_numerator_col(
        df,
        event_col,
        binary=False,
    )

    return _bin_strata(
        df,
        list(strata_cols),
    )


def _add_end_of_period_denominator(
    df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Add an internally derived denominator column containing ones.

    Summing the column within a group is equivalent to counting rows. This
    treats each row as one unit of exposure present at the end of the
    reporting period.
    """
    if _DENOM_COL in df.columns:
        raise ValueError(
            "The input dataframe already contains the reserved internal "
            f"column '{_DENOM_COL}'. Rename or remove it before calling "
            "a rate function."
        )

    return df.with_columns(
        pl.lit(1.0).alias(_DENOM_COL),
    )


def _filter_group(
    df: pl.DataFrame,
    row: Mapping[str, object],
) -> pl.DataFrame:
    """Filter a dataframe to the supplied active grouping values."""
    mask = pl.lit(True)

    for col, value in row.items():
        mask = mask & (
            pl.col(col) == value
        )

    return df.filter(mask)


@dataclass(frozen=True)
class _GroupingSet:
    """Columns that are active for one grouping-set calculation."""

    active_cols: tuple[str, ...]


def _normalise_hierarchies(
    organisational_cols: Mapping[str, Sequence[str]] | None,
) -> dict[str, tuple[str, ...]]:
    """
    Validate and normalise named organisational hierarchies.

    Each hierarchy must be ordered from its highest to lowest level.
    """
    if organisational_cols is None:
        return {}

    if not isinstance(
        organisational_cols,
        Mapping,
    ):
        raise TypeError(
            "organisational_cols must be a named mapping, for example "
            "{'commissioning': ['region', 'icb', 'practice']}."
        )

    result: dict[str, tuple[str, ...]] = {}

    for name, levels in organisational_cols.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                "Hierarchy names must be non-empty strings."
            )

        if (
            isinstance(levels, (str, bytes))
            or not isinstance(levels, Sequence)
        ):
            raise TypeError(
                f"Hierarchy '{name}' must be an ordered sequence of "
                "columns from highest to lowest level."
            )

        normalised_levels = tuple(levels)

        if not normalised_levels:
            raise ValueError(
                f"Hierarchy '{name}' must contain at least one column."
            )

        if any(
            not isinstance(col, str) or not col
            for col in normalised_levels
        ):
            raise ValueError(
                f"Hierarchy '{name}' contains an invalid column name."
            )

        if len(normalised_levels) != len(
            set(normalised_levels),
        ):
            raise ValueError(
                f"Hierarchy '{name}' contains duplicate columns."
            )

        result[name] = normalised_levels

    return result


def _prepare_dimensions(
    df: pl.DataFrame,
    event_col: str,
    strata_cols: Sequence[str],
    inequalities_cols: Sequence[str] | None,
    organisational_cols: Mapping[str, Sequence[str]] | None,
    organisational_mode: Literal["separate", "cross"],
    all_label: str,
) -> tuple[
    pl.DataFrame,
    list[str],
    dict[str, tuple[str, ...]],
    list[str],
]:
    """
    Validate and prepare inequality and organisational dimensions.

    Returns
    -------
    tuple
        Prepared dataframe, inequality columns, organisational hierarchy
        mapping, and flattened output dimensions.
    """
    if not isinstance(df, pl.DataFrame):
        raise TypeError(
            "df must be a Polars DataFrame."
        )

    if df.height == 0:
        raise ValueError(
            "df must contain at least one row."
        )

    if not isinstance(event_col, str) or not event_col:
        raise ValueError(
            "event_col must be a non-empty string."
        )

    if organisational_mode not in {
        "separate",
        "cross",
    }:
        raise ValueError(
            "organisational_mode must be 'separate' or 'cross'."
        )

    if not isinstance(all_label, str) or not all_label:
        raise ValueError(
            "all_label must be a non-empty string."
        )

    if inequalities_cols is None:
        inequalities: list[str] = []

    elif (
        isinstance(inequalities_cols, (str, bytes))
        or not isinstance(inequalities_cols, Sequence)
    ):
        raise TypeError(
            "inequalities_cols must be a sequence or None."
        )

    else:
        inequalities = list(inequalities_cols)

    if any(
        not isinstance(col, str) or not col
        for col in inequalities
    ):
        raise ValueError(
            "inequalities_cols contains an invalid column name."
        )

    if len(inequalities) != len(set(inequalities)):
        raise ValueError(
            "inequalities_cols contains duplicate columns."
        )

    hierarchies = _normalise_hierarchies(
        organisational_cols,
    )

    organisational = _flatten_organisational_cols(
        hierarchies,
    )

    repeated = sorted(
        {
            col
            for col in organisational
            if organisational.count(col) > 1
        },
    )

    if repeated:
        raise ValueError(
            "Organisational columns occur in multiple hierarchies: "
            f"{repeated}."
        )

    inequality_organisational_overlap = sorted(
        set(inequalities) & set(organisational),
    )

    if inequality_organisational_overlap:
        raise ValueError(
            "Columns cannot be both inequality and organisational: "
            f"{inequality_organisational_overlap}."
        )

    if (
        isinstance(strata_cols, (str, bytes))
        or not isinstance(strata_cols, Sequence)
    ):
        raise TypeError(
            "strata_cols must be a sequence of column names, not a "
            "single string."
        )

    strata = list(strata_cols)

    if any(
        not isinstance(col, str) or not col
        for col in strata
    ):
        raise ValueError(
            "strata_cols contains an invalid column name."
        )

    if len(strata) != len(set(strata)):
        raise ValueError(
            "strata_cols contains duplicate columns."
        )

    dimensions = _dimension_cols(
        inequalities,
        hierarchies,
    )

    _check_reserved_column_names(
        strata_cols=strata,
        dimension_cols=dimensions,
    )

    strata_dimension_overlap = sorted(
        set(strata) & set(dimensions),
    )

    if strata_dimension_overlap:
        raise ValueError(
            "Strata cannot also be inequality or organisational "
            f"dimensions: {strata_dimension_overlap}."
        )

    if (
        event_col in dimensions
        or event_col in strata
    ):
        raise ValueError(
            "event_col cannot also be a stratum, inequality, or "
            "organisational dimension."
        )

    required = list(
        dict.fromkeys(
            [
                event_col,
                *strata,
                *dimensions,
            ],
        ),
    )

    _check_columns(
        df,
        required,
    )

    _check_no_missing_or_nonfinite(
        df,
        required,
    )

    conflicts = [
        col
        for col in dimensions
        if df.select(
            pl.col(col)
            .cast(pl.Utf8)
            .eq(all_label)
            .any(),
        ).item()
    ]

    if conflicts:
        raise ValueError(
            f"Reserved all_label {all_label!r} occurs in columns "
            f"{conflicts}; choose another all_label or recode the "
            "source values."
        )

    df = _cast_dimension_cols_to_utf8(
        df,
        inequalities,
        hierarchies,
    )

    for name, levels in hierarchies.items():
        for parent, child in pairwise(levels):
            invalid = (
                df.select(
                    parent,
                    child,
                )
                .unique()
                .group_by(child)
                .agg(
                    pl.col(parent)
                    .n_unique()
                    .alias("_parents"),
                )
                .filter(
                    pl.col("_parents") != 1,
                )
            )

            if invalid.height:
                examples = (
                    invalid[child]
                    .head(5)
                    .to_list()
                )

                raise ValueError(
                    f"Invalid hierarchy '{name}': '{child}' does not "
                    f"map to exactly one '{parent}'. Example values: "
                    f"{examples}. Hierarchies must be ordered highest "
                    "to lowest and cannot be many-to-many."
                )

    return (
        df,
        inequalities,
        hierarchies,
        dimensions,
    )


def _inequality_states(
    cols: list[str],
) -> list[tuple[str, ...]]:
    """Return the full marginal cube for inequality dimensions."""
    return [
        tuple(active)
        for size in range(
            len(cols),
            -1,
            -1,
        )
        for active in combinations(
            cols,
            size,
        )
    ]


def _organisational_states(
    hierarchies: Mapping[str, tuple[str, ...]],
    mode: Literal["separate", "cross"],
) -> list[tuple[str, ...]]:
    """
    Return hierarchy-valid organisational grouping states.

    In separate mode, each hierarchy is rolled up separately. In cross mode,
    valid levels from separate hierarchies are crossed.
    """
    if not hierarchies:
        return [()]

    if mode == "separate":
        states = [
            tuple(levels[:depth])
            for levels in hierarchies.values()
            for depth in range(
                len(levels),
                0,
                -1,
            )
        ]

        return [
            *states,
            (),
        ]

    choices = [
        [
            tuple(levels[:depth])
            for depth in range(
                len(levels),
                -1,
                -1,
            )
        ]
        for levels in hierarchies.values()
    ]

    return [
        tuple(
            col
            for state in states
            for col in state
        )
        for states in product(*choices)
    ]


def _build_grouping_sets(
    inequalities: list[str],
    hierarchies: Mapping[str, tuple[str, ...]],
    mode: Literal["separate", "cross"],
) -> list[_GroupingSet]:
    """Combine valid organisational states with the inequality cube."""
    candidates = [
        organisational_state + inequality_state
        for organisational_state in _organisational_states(
            hierarchies,
            mode,
        )
        for inequality_state in _inequality_states(
            inequalities,
        )
    ]

    seen: set[tuple[str, ...]] = set()
    result: list[_GroupingSet] = []

    for active in candidates:
        if active in seen:
            continue

        seen.add(active)

        result.append(
            _GroupingSet(
                active_cols=active,
            ),
        )

    return result


def _iter_group_slices(
    df: pl.DataFrame,
    dimensions: list[str],
    grouping_sets: Sequence[_GroupingSet],
    all_label: str,
):
    """Yield display keys, data slices, and the reference-row flag."""
    for grouping_set in grouping_sets:
        active = list(
            grouping_set.active_cols,
        )

        if active:
            keys = (
                df.select(active)
                .unique(
                    maintain_order=True,
                )
                .sort(active)
                .iter_rows(
                    named=True,
                )
            )

        else:
            keys = iter([{}])

        for key in keys:
            group_data = (
                _filter_group(
                    df,
                    key,
                )
                if active
                else df
            )

            display = {
                col: (
                    key[col]
                    if col in active
                    else all_label
                )
                for col in dimensions
            }

            yield (
                display,
                group_data,
                not active,
            )


def _validate_options(
    confidence: float,
    multiplier: float | None = None,
) -> None:
    """Validate common statistical options with explicit type errors."""
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, Real)
    ):
        raise TypeError(
            "confidence must be a real numeric value, not a boolean."
        )

    if (
        not np.isfinite(confidence)
        or not 0 < confidence < 1
    ):
        raise ValueError(
            "confidence must be finite and strictly between 0 and 1."
        )

    if multiplier is None:
        return

    if (
        isinstance(multiplier, bool)
        or not isinstance(multiplier, Real)
    ):
        raise TypeError(
            "multiplier must be a real numeric value, not a boolean."
        )

    if (
        not np.isfinite(multiplier)
        or multiplier <= 0
    ):
        raise ValueError(
            "multiplier must be finite and positive."
        )


# ===========================================================================
# Wilson score CI
# ===========================================================================


def _wilson_proportion_ci(
    events: float,
    n: float,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Calculate a Wilson score interval for one proportion."""
    if n == 0:
        return (
            np.nan,
            np.nan,
            np.nan,
        )

    proportion = events / n

    z_value = stats.norm.ppf(
        1 - (1 - confidence) / 2,
    )

    denominator = (
        1
        + z_value**2 / n
    )

    centre = (
        proportion
        + z_value**2 / (2 * n)
    ) / denominator

    half_width = (
        z_value
        * np.sqrt(
            proportion
            * (1 - proportion)
            / n
            + z_value**2
            / (4 * n**2),
        )
        / denominator
    )

    lower = max(
        0.0,
        centre - half_width,
    )

    upper = min(
        1.0,
        centre + half_width,
    )

    variance = (
        proportion
        * (1 - proportion)
        / n
    )

    return (
        float(lower),
        float(upper),
        float(variance),
    )


# ===========================================================================
# Byar and exact Poisson intervals
# ===========================================================================


def _byar_count_ci(
    count: float,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Calculate Byar's approximation for a Poisson count."""
    if count < 0:
        raise ValueError(
            "count must be non-negative."
        )

    if count == 0:
        return (
            0.0,
            float(
                -np.log(
                    1.0 - confidence,
                ),
            ),
        )

    alpha = 1.0 - confidence

    z_value = stats.norm.ppf(
        1.0 - alpha / 2.0,
    )

    lower = count * (
        1.0
        - 1.0 / (9.0 * count)
        - z_value / (
            3.0 * np.sqrt(count)
        )
    ) ** 3

    upper = (count + 1.0) * (
        1.0
        - 1.0 / (
            9.0 * (count + 1.0)
        )
        + z_value / (
            3.0 * np.sqrt(count + 1.0)
        )
    ) ** 3

    return (
        float(
            max(
                lower,
                0.0,
            ),
        ),
        float(
            max(
                upper,
                0.0,
            ),
        ),
    )


def _exact_poisson_count_ci(
    count: float,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Calculate an exact chi-square Poisson count interval."""
    alpha = 1.0 - confidence

    if count == 0:
        lower = 0.0

    else:
        lower = 0.5 * stats.chi2.ppf(
            alpha / 2.0,
            2 * count,
        )

    upper = 0.5 * stats.chi2.ppf(
        1 - alpha / 2.0,
        2 * (count + 1),
    )

    return (
        float(lower),
        float(upper),
    )


# ===========================================================================
# DSP helpers
# ===========================================================================


def _haldane_proportion_correction(
    events: float,
    n: float,
) -> tuple[float, float, float]:
    """Apply a Haldane-Anscombe correction to a proportion."""
    corrected_events = events + 0.5
    corrected_n = n + 1.0

    return (
        float(
            corrected_events / corrected_n,
        ),
        float(corrected_events),
        float(corrected_n),
    )


def _wilson_dobson_proportion_ci(
    dsp: float,
    crude_events: float,
    crude_n: float,
    stratum_weights: np.ndarray,
    stratum_props: np.ndarray,
    stratum_ns: np.ndarray,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """
    Calculate a Wilson-Dobson interval for a standardised proportion.

    The supplied stratum proportions and denominators determine the DSP
    variance. The crude Wilson interval supplies the interval shape.
    """
    if (
        crude_n <= 0
        or not np.isfinite(dsp)
    ):
        return (
            np.nan,
            np.nan,
            np.nan,
        )

    valid = stratum_ns > 0

    weights = stratum_weights[valid]
    proportions = stratum_props[valid]
    denominators = stratum_ns[valid]

    weight_sum = float(
        np.sum(stratum_weights),
    )

    if (
        weight_sum == 0
        or len(weights) == 0
    ):
        return (
            np.nan,
            np.nan,
            np.nan,
        )

    variance_dsp = float(
        np.sum(
            weights**2
            * proportions
            * (1 - proportions)
            / denominators,
        )
        / weight_sum**2,
    )

    crude_proportion = (
        crude_events / crude_n
    )

    (
        crude_lower,
        crude_upper,
        variance_crude,
    ) = _wilson_proportion_ci(
        crude_events,
        crude_n,
        confidence,
    )

    if (
        variance_crude == 0
        or not np.isfinite(variance_crude)
    ):
        (
            corrected_proportion,
            _,
            corrected_n,
        ) = _haldane_proportion_correction(
            crude_events,
            crude_n,
        )

        variance_crude = (
            corrected_proportion
            * (1 - corrected_proportion)
            / corrected_n
        )

    if (
        variance_crude == 0
        or not np.isfinite(variance_crude)
    ):
        return (
            float(dsp),
            float(dsp),
            variance_dsp,
        )

    scale = float(
        np.sqrt(
            variance_dsp / variance_crude,
        ),
    )

    lower = (
        dsp
        + scale
        * (
            crude_lower
            - crude_proportion
        )
    )

    upper = (
        dsp
        + scale
        * (
            crude_upper
            - crude_proportion
        )
    )

    return (
        float(
            max(
                lower,
                0.0,
            ),
        ),
        float(
            min(
                upper,
                1.0,
            ),
        ),
        variance_dsp,
    )


def _compute_proportion_stratum_stats(
    group_data: pl.DataFrame,
    all_strata: pl.DataFrame,
    event_col: str,
    strata_cols: list[str],
    reference_weights: pl.DataFrame,
) -> dict[str, object]:
    """
    Calculate proportion statistics across the standardisation strata.

    Boundary strata currently receive the existing Haldane correction.
    Empty positively weighted strata remain represented with n=0.
    """
    aggregation = (
        group_data.group_by(strata_cols)
        .agg(
            pl.col(event_col)
            .sum()
            .alias("events"),
            pl.len().alias("n"),
        )
    )

    scaffold = (
        all_strata
        .join(
            aggregation,
            on=strata_cols,
            how="left",
        )
        .join(
            reference_weights,
            on=strata_cols,
            how="left",
        )
        .with_columns(
            [
                pl.col("events")
                .fill_null(0)
                .cast(pl.Float64),
                pl.col("n")
                .fill_null(0)
                .cast(pl.Float64),
                pl.col("ref_weight")
                .fill_null(0)
                .cast(pl.Float64),
            ],
        )
    )

    raw_events = (
        scaffold["events"]
        .to_numpy()
        .astype(float)
    )

    raw_ns = (
        scaffold["n"]
        .to_numpy()
        .astype(float)
    )

    weights = (
        scaffold["ref_weight"]
        .to_numpy()
        .astype(float)
    )

    proportions = np.zeros(
        len(raw_events),
    )

    haldane_applied = False

    for index, (
        events,
        denominator,
    ) in enumerate(
        zip(
            raw_events,
            raw_ns,
        ),
    ):
        if denominator == 0:
            proportions[index] = 0.0

        elif (
            events == 0
            or events == denominator
        ):
            (
                corrected_proportion,
                _,
                _,
            ) = _haldane_proportion_correction(
                events,
                denominator,
            )

            proportions[index] = corrected_proportion
            haldane_applied = True

        else:
            proportions[index] = (
                events / denominator
            )

    return {
        "stratum_props": proportions,
        "stratum_ns": raw_ns,
        "stratum_weights": weights,
        "raw_stratum_ns": raw_ns,
        "raw_stratum_events": raw_events,
        "haldane_applied": haldane_applied,
    }


# ===========================================================================
# DSR helpers
# ===========================================================================


def _haldane_rate_correction(
    events: float,
    denominator: float,
) -> tuple[float, float, float]:
    """Apply a Haldane-style correction to a rate stratum."""
    corrected_events = events + 0.5
    corrected_denominator = denominator + 1.0

    return (
        float(
            corrected_events
            / corrected_denominator,
        ),
        float(corrected_events),
        float(corrected_denominator),
    )


def _dobson_byar_rate_ci(
    dsr_unscaled: float,
    crude_events: float,
    stratum_weights: np.ndarray,
    stratum_events: np.ndarray,
    stratum_denominators: np.ndarray,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Calculate a Dobson-Byar interval for an unscaled DSR."""
    if crude_events < 0:
        return (
            np.nan,
            np.nan,
            np.nan,
        )

    if crude_events < 10:
        (
            count_lower,
            count_upper,
        ) = _exact_poisson_count_ci(
            crude_events,
            confidence,
        )

    else:
        (
            count_lower,
            count_upper,
        ) = _byar_count_ci(
            crude_events,
            confidence,
        )

    valid = (
        stratum_denominators > 0
    )

    weights = stratum_weights[valid]
    events = stratum_events[valid]
    denominators = stratum_denominators[valid]

    weight_sum = float(
        np.sum(stratum_weights),
    )

    if (
        weight_sum == 0
        or len(weights) == 0
    ):
        return (
            np.nan,
            np.nan,
            np.nan,
        )

    variance_dsr = float(
        np.sum(
            weights**2
            * events
            / denominators**2,
        )
        / weight_sum**2,
    )

    variance_events = float(
        crude_events,
    )

    if variance_events == 0.0:
        (
            _,
            corrected_events,
            _,
        ) = _haldane_rate_correction(
            crude_events,
            1.0,
        )

        variance_events = corrected_events

    if (
        variance_events == 0.0
        or variance_dsr == 0.0
    ):
        return (
            float(dsr_unscaled),
            float(dsr_unscaled),
            variance_dsr,
        )

    scale = float(
        np.sqrt(
            variance_dsr
            / variance_events,
        ),
    )

    lower = (
        dsr_unscaled
        + scale
        * (
            count_lower
            - crude_events
        )
    )

    upper = (
        dsr_unscaled
        + scale
        * (
            count_upper
            - crude_events
        )
    )

    return (
        float(
            max(
                lower,
                0.0,
            ),
        ),
        float(
            max(
                upper,
                0.0,
            ),
        ),
        variance_dsr,
    )


def _compute_rate_stratum_stats(
    group_data: pl.DataFrame,
    all_strata: pl.DataFrame,
    event_col: str,
    denominator_col: str,
    strata_cols: list[str],
    reference_weights: pl.DataFrame,
) -> dict[str, object]:
    """Calculate rate statistics across standardisation strata."""
    aggregation = (
        group_data.group_by(strata_cols)
        .agg(
            pl.col(event_col)
            .sum()
            .alias("events"),
            pl.col(denominator_col)
            .sum()
            .alias("denominator"),
        )
    )

    scaffold = (
        all_strata
        .join(
            aggregation,
            on=strata_cols,
            how="left",
        )
        .join(
            reference_weights,
            on=strata_cols,
            how="left",
        )
        .with_columns(
            [
                pl.col("events")
                .fill_null(0)
                .cast(pl.Float64),
                pl.col("denominator")
                .fill_null(0)
                .cast(pl.Float64),
                pl.col("ref_weight")
                .fill_null(0)
                .cast(pl.Float64),
            ],
        )
    )

    raw_events = (
        scaffold["events"]
        .to_numpy()
        .astype(float)
    )

    raw_denominators = (
        scaffold["denominator"]
        .to_numpy()
        .astype(float)
    )

    weights = (
        scaffold["ref_weight"]
        .to_numpy()
        .astype(float)
    )

    rates = np.zeros(
        len(raw_events),
    )

    corrected_events = raw_events.copy()
    corrected_denominators = raw_denominators.copy()

    haldane_applied = False

    for index, (
        events,
        denominator,
    ) in enumerate(
        zip(
            raw_events,
            raw_denominators,
        ),
    ):
        if denominator == 0:
            rates[index] = 0.0

        elif events == 0:
            (
                corrected_rate,
                corrected_event_count,
                corrected_denominator,
            ) = _haldane_rate_correction(
                events,
                denominator,
            )

            rates[index] = corrected_rate
            corrected_events[index] = corrected_event_count
            corrected_denominators[index] = corrected_denominator
            haldane_applied = True

        else:
            rates[index] = (
                events / denominator
            )

    return {
        "stratum_rates": rates,
        "stratum_events": corrected_events,
        "stratum_denoms": corrected_denominators,
        "stratum_weights": weights,
        "raw_stratum_denoms": raw_denominators,
        "raw_stratum_events": raw_events,
        "haldane_applied": haldane_applied,
    }


# ===========================================================================
# Quality notes
# ===========================================================================


def _crude_proportion_notes(
    events: float,
    n: float,
) -> str:
    """Create quality notes for a crude proportion."""
    notes: list[str] = []

    if n == 0:
        notes.append(
            "Zero denominator - suppress the result and reconsider the "
            "organisational hierarchy or inequality dimensions."
        )

    else:
        if events == 0:
            notes.append(
                "Zero events - confidence intervals may be unstable."
            )

        elif events < 10:
            notes.append(
                "Low event count (<10) - flag as unstable or suppress "
                "in visual outputs."
            )

        non_events = n - events

        if non_events == 0:
            notes.append(
                "All events (proportion = 1) - flag as a boundary "
                "estimate or suppress in visual outputs."
            )

        elif non_events < 10:
            notes.append(
                "Low non-event count (<10) - confidence intervals may "
                "be unstable."
            )

    if 0 < n < 40:
        notes.append(
            "Low sample size (<40) - flag the proportion as unstable."
        )

    return " | ".join(notes)


def _crude_rate_notes(
    events: float,
    denominator: float,
    end_of_period_denom: bool = False,
) -> str:
    """Create quality notes for a crude rate."""
    notes: list[str] = []

    if end_of_period_denom:
        notes.append(
            "End-of-period denominator: row count used as exposure "
            "(assumes each row represents one patient present at the "
            "period end and does not account for partial-period "
            "exposure)."
        )

    if denominator == 0:
        notes.append(
            "Zero denominator."
        )

    else:
        if events == 0:
            notes.append(
                "Zero events - confidence intervals may be unstable."
            )

        elif events < 10:
            notes.append(
                "Low event count (<10) - flag as unstable or suppress "
                "in visual outputs."
            )

    if 0 < denominator < 40:
        notes.append(
            "Low denominator (<40) - flag the rate as unstable."
        )

    return " | ".join(notes)


# ===========================================================================
# Significance assessment
# ===========================================================================


def _significance_from_ci(
    reference: float,
    lower: float,
    upper: float,
) -> str:
    """
    Classify an estimate relative to a fixed reference value.

    This is a descriptive confidence-interval comparison, not a formal
    hypothesis test.
    """
    if (
        not np.isfinite(reference)
        or not np.isfinite(lower)
        or not np.isfinite(upper)
    ):
        return "Not tested"

    if reference < lower:
        return "Higher"

    if reference > upper:
        return "Lower"

    return "Not significant"


# ===========================================================================
# Public functions
# ===========================================================================


def crude_proportion_df(
    df: pl.DataFrame,
    event_col: str,
    inequalities_cols: Sequence[str] | None = None,
    organisational_cols: Mapping[str, Sequence[str]] | None = None,
    *,
    organisational_mode: Literal["separate", "cross"] = "separate",
    all_label: str = "All",
    confidence: float = 0.95,
) -> pl.DataFrame:
    """Calculate crude proportions across configured grouping sets."""
    _validate_options(
        confidence,
    )

    (
        df,
        inequalities,
        hierarchies,
        dimensions,
    ) = _prepare_dimensions(
        df=df,
        event_col=event_col,
        strata_cols=[],
        inequalities_cols=inequalities_cols,
        organisational_cols=organisational_cols,
        organisational_mode=organisational_mode,
        all_label=all_label,
    )

    df = _validate_numerator_col(
        df,
        event_col,
        binary=True,
    )

    grouping_sets = _build_grouping_sets(
        inequalities,
        hierarchies,
        organisational_mode,
    )

    overall_events = float(
        df[event_col].sum(),
    )

    overall_n = float(
        len(df),
    )

    reference = (
        overall_events / overall_n
        if overall_n
        else np.nan
    )

    records: list[dict[str, object]] = []

    for (
        key,
        group_data,
        is_reference,
    ) in _iter_group_slices(
        df,
        dimensions,
        grouping_sets,
        all_label,
    ):
        events = float(
            group_data[event_col].sum(),
        )

        n = float(
            len(group_data),
        )

        (
            lower,
            upper,
            _,
        ) = _wilson_proportion_ci(
            events,
            n,
            confidence,
        )

        estimate = (
            events / n
            if n
            else np.nan
        )

        records.append(
            {
                **key,
                "events": events,
                "n": n,
                "proportion": estimate,
                "lower": lower,
                "upper": upper,
                "confidence": confidence,
                "method": "Wilson score",
                "notes": _crude_proportion_notes(
                    events,
                    n,
                ),
                "significance": (
                    "Reference"
                    if is_reference
                    else _significance_from_ci(
                        reference,
                        lower,
                        upper,
                    )
                ),
            },
        )

    return pl.from_dicts(
        records,
    )


def crude_rate_df(
    df: pl.DataFrame,
    event_col: str,
    inequalities_cols: Sequence[str] | None = None,
    organisational_cols: Mapping[str, Sequence[str]] | None = None,
    *,
    organisational_mode: Literal["separate", "cross"] = "separate",
    all_label: str = "All",
    multiplier: float = 100_000.0,
    confidence: float = 0.95,
) -> pl.DataFrame:
    """Calculate crude rates across configured grouping sets."""
    _validate_options(
        confidence,
        multiplier,
    )

    (
        df,
        inequalities,
        hierarchies,
        dimensions,
    ) = _prepare_dimensions(
        df=df,
        event_col=event_col,
        strata_cols=[],
        inequalities_cols=inequalities_cols,
        organisational_cols=organisational_cols,
        organisational_mode=organisational_mode,
        all_label=all_label,
    )

    df = _validate_numerator_col(
        df,
        event_col,
        binary=False,
    )

    df = _add_end_of_period_denominator(
        df,
    )

    grouping_sets = _build_grouping_sets(
        inequalities,
        hierarchies,
        organisational_mode,
    )

    overall_events = float(
        df[event_col].sum(),
    )

    overall_denominator = float(
        df[_DENOM_COL].sum(),
    )

    reference = (
        overall_events
        / overall_denominator
        * multiplier
        if overall_denominator
        else np.nan
    )

    def calculate(
        events: float,
        denominator: float,
    ) -> tuple[float, float, float, str]:
        if not denominator:
            return (
                np.nan,
                np.nan,
                np.nan,
                "undefined",
            )

        if events < 10:
            (
                count_lower,
                count_upper,
            ) = _exact_poisson_count_ci(
                events,
                confidence,
            )

            method = "Exact chi-square"

        else:
            (
                count_lower,
                count_upper,
            ) = _byar_count_ci(
                events,
                confidence,
            )

            method = "Byar"

        return (
            events / denominator * multiplier,
            count_lower / denominator * multiplier,
            count_upper / denominator * multiplier,
            method,
        )

    records: list[dict[str, object]] = []

    for (
        key,
        group_data,
        is_reference,
    ) in _iter_group_slices(
        df,
        dimensions,
        grouping_sets,
        all_label,
    ):
        events = float(
            group_data[event_col].sum(),
        )

        denominator = float(
            group_data[_DENOM_COL].sum(),
        )

        (
            estimate,
            lower,
            upper,
            method,
        ) = calculate(
            events,
            denominator,
        )

        records.append(
            {
                **key,
                "events": events,
                "denominator": denominator,
                "rate": estimate,
                "lower": lower,
                "upper": upper,
                "multiplier": multiplier,
                "confidence": confidence,
                "method": method,
                "notes": _crude_rate_notes(
                    events,
                    denominator,
                    True,
                ),
                "significance": (
                    "Reference"
                    if is_reference
                    else _significance_from_ci(
                        reference,
                        lower,
                        upper,
                    )
                ),
            },
        )

    return pl.from_dicts(
        records,
    )


def directly_standardized_proportion_df(
    df: pl.DataFrame,
    event_col: str,
    strata_cols: Sequence[str],
    inequalities_cols: Sequence[str] | None = None,
    organisational_cols: Mapping[str, Sequence[str]] | None = None,
    *,
    organisational_mode: Literal["separate", "cross"] = "separate",
    all_label: str = "All",
    confidence: float = 0.95,
) -> pl.DataFrame:
    """Calculate DSPs across configured grouping sets."""
    _validate_options(
        confidence,
    )

    if (
        isinstance(strata_cols, (str, bytes))
        or not isinstance(strata_cols, Sequence)
    ):
        raise TypeError(
            "strata_cols must be a sequence of column names, not a "
            "single string."
        )

    strata = list(
        strata_cols,
    )

    if not strata:
        raise ValueError(
            "strata_cols must contain at least one column."
        )

    (
        df,
        inequalities,
        hierarchies,
        dimensions,
    ) = _prepare_dimensions(
        df=df,
        event_col=event_col,
        strata_cols=strata,
        inequalities_cols=inequalities_cols,
        organisational_cols=organisational_cols,
        organisational_mode=organisational_mode,
        all_label=all_label,
    )

    work = _prepare_dataframe_proportion(
        df=df,
        event_col=event_col,
        strata_cols=strata,
        inequalities_cols=inequalities,
        organisational_cols=hierarchies,
    )

    grouping_sets = _build_grouping_sets(
        inequalities,
        hierarchies,
        organisational_mode,
    )

    reference_weights = _build_reference_weights(
        work,
        strata,
    )

    all_strata = reference_weights.select(
        strata,
    )

    overall_events = float(
        work[event_col].sum(),
    )

    overall_n = float(
        len(work),
    )

    reference = (
        overall_events / overall_n
        if overall_n
        else np.nan
    )

    records: list[dict[str, object]] = []

    for (
        key,
        group_data,
        is_reference,
    ) in _iter_group_slices(
        work,
        dimensions,
        grouping_sets,
        all_label,
    ):
        events = float(
            group_data[event_col].sum(),
        )

        n = float(
            len(group_data),
        )

        notes: list[str] = []

        if is_reference:
            estimate = reference

            (
                lower,
                upper,
                _,
            ) = _wilson_dobson_proportion_ci(
                dsp=reference,
                crude_events=overall_events,
                crude_n=overall_n,
                stratum_weights=np.array([1.0]),
                stratum_props=np.array([reference]),
                stratum_ns=np.array([overall_n]),
                confidence=confidence,
            )

            notes.append(
                "Overall proportion: no weighting applied because this "
                "row is the reference population."
            )

        else:
            stratum_statistics = (
                _compute_proportion_stratum_stats(
                    group_data=group_data,
                    all_strata=all_strata,
                    event_col=event_col,
                    strata_cols=strata,
                    reference_weights=reference_weights,
                )
            )

            estimate = float(
                np.sum(
                    stratum_statistics["stratum_weights"]
                    * stratum_statistics["stratum_props"],
                ),
            )

            (
                lower,
                upper,
                _,
            ) = _wilson_dobson_proportion_ci(
                dsp=estimate,
                crude_events=events,
                crude_n=n,
                stratum_weights=stratum_statistics[
                    "stratum_weights"
                ],
                stratum_props=stratum_statistics[
                    "stratum_props"
                ],
                stratum_ns=stratum_statistics[
                    "stratum_ns"
                ],
                confidence=confidence,
            )

            if stratum_statistics["haldane_applied"]:
                notes.append(
                    "Haldane correction applied."
                )

            if any(
                0 < denominator < 10
                for denominator in stratum_statistics[
                    "raw_stratum_ns"
                ]
            ):
                notes.append(
                    "Unreliable: stratum n < 10."
                )

            non_events = [
                float(denominator)
                - float(stratum_events)
                for (
                    denominator,
                    stratum_events,
                ) in zip(
                    stratum_statistics[
                        "raw_stratum_ns"
                    ],
                    stratum_statistics[
                        "raw_stratum_events"
                    ],
                )
                if denominator > 0
            ]

            if any(
                0 < count < 10
                for count in non_events
            ):
                notes.append(
                    "Unreliable: stratum non-event count < 10."
                )

        non_events = n - events

        if events == 0:
            notes.append(
                "Zero events."
            )

        elif events < 10:
            notes.append(
                "Low event count (<10)."
            )

        if n > 0 and non_events == 0:
            notes.append(
                "All events (proportion = 1)."
            )

        elif 0 < non_events < 10:
            notes.append(
                "Low non-event count (<10)."
            )

        records.append(
            {
                **key,
                "events": int(events),
                "n": int(n),
                "dsp": (
                    round(
                        estimate,
                        6,
                    )
                    if np.isfinite(estimate)
                    else np.nan
                ),
                "dsp_lower": (
                    round(
                        lower,
                        6,
                    )
                    if np.isfinite(lower)
                    else np.nan
                ),
                "dsp_upper": (
                    round(
                        upper,
                        6,
                    )
                    if np.isfinite(upper)
                    else np.nan
                ),
                "notes": " | ".join(notes),
                "significance": (
                    "Reference"
                    if is_reference
                    else _significance_from_ci(
                        reference,
                        lower,
                        upper,
                    )
                ),
            },
        )

    return pl.from_dicts(
        records,
    )


def directly_standardized_rate_df(
    df: pl.DataFrame,
    event_col: str,
    strata_cols: Sequence[str],
    inequalities_cols: Sequence[str] | None = None,
    organisational_cols: Mapping[str, Sequence[str]] | None = None,
    *,
    organisational_mode: Literal["separate", "cross"] = "separate",
    all_label: str = "All",
    multiplier: float = 100_000.0,
    confidence: float = 0.95,
) -> pl.DataFrame:
    """Calculate DSRs across configured grouping sets."""
    _validate_options(
        confidence,
        multiplier,
    )

    if (
        isinstance(strata_cols, (str, bytes))
        or not isinstance(strata_cols, Sequence)
    ):
        raise TypeError(
            "strata_cols must be a sequence of column names, not a "
            "single string."
        )

    strata = list(
        strata_cols,
    )

    if not strata:
        raise ValueError(
            "strata_cols must contain at least one column."
        )

    (
        df,
        inequalities,
        hierarchies,
        dimensions,
    ) = _prepare_dimensions(
        df=df,
        event_col=event_col,
        strata_cols=strata,
        inequalities_cols=inequalities_cols,
        organisational_cols=organisational_cols,
        organisational_mode=organisational_mode,
        all_label=all_label,
    )

    work = _add_end_of_period_denominator(
        df,
    )

    work = _prepare_dataframe_rate(
        df=work,
        event_col=event_col,
        strata_cols=strata,
        inequalities_cols=inequalities,
        organisational_cols=hierarchies,
    )

    grouping_sets = _build_grouping_sets(
        inequalities,
        hierarchies,
        organisational_mode,
    )

    reference_weights = _build_reference_weights(
        work,
        strata,
    )

    all_strata = reference_weights.select(
        strata,
    )

    overall_events = float(
        work[event_col].sum(),
    )

    overall_denominator = float(
        work[_DENOM_COL].sum(),
    )

    reference_unscaled = (
        overall_events / overall_denominator
        if overall_denominator
        else np.nan
    )

    def scale(value: float) -> float:
        if not np.isfinite(value):
            return np.nan

        return round(
            value * multiplier,
            6,
        )

    records: list[dict[str, object]] = []

    for (
        key,
        group_data,
        is_reference,
    ) in _iter_group_slices(
        work,
        dimensions,
        grouping_sets,
        all_label,
    ):
        events = float(
            group_data[event_col].sum(),
        )

        denominator = float(
            group_data[_DENOM_COL].sum(),
        )

        notes = [
            (
                "End-of-period denominator: row count used as exposure "
                "(assumes each row represents one patient present at the "
                "period end and does not account for partial-period "
                "exposure)."
            ),
        ]

        if is_reference:
            estimate_unscaled = reference_unscaled

            if events < 10:
                (
                    count_lower,
                    count_upper,
                ) = _exact_poisson_count_ci(
                    events,
                    confidence,
                )

            else:
                (
                    count_lower,
                    count_upper,
                ) = _byar_count_ci(
                    events,
                    confidence,
                )

            lower_unscaled = (
                count_lower / denominator
            )

            upper_unscaled = (
                count_upper / denominator
            )

            notes.append(
                "Overall rate: no weighting applied because this row "
                "is the reference population."
            )

        else:
            stratum_statistics = (
                _compute_rate_stratum_stats(
                    group_data=group_data,
                    all_strata=all_strata,
                    event_col=event_col,
                    denominator_col=_DENOM_COL,
                    strata_cols=strata,
                    reference_weights=reference_weights,
                )
            )

            weight_sum = float(
                np.sum(
                    stratum_statistics[
                        "stratum_weights"
                    ],
                ),
            )

            estimate_unscaled = (
                float(
                    np.sum(
                        stratum_statistics[
                            "stratum_weights"
                        ]
                        * stratum_statistics[
                            "stratum_rates"
                        ],
                    )
                    / weight_sum,
                )
                if weight_sum
                else np.nan
            )

            (
                lower_unscaled,
                upper_unscaled,
                _,
            ) = _dobson_byar_rate_ci(
                dsr_unscaled=estimate_unscaled,
                crude_events=events,
                stratum_weights=stratum_statistics[
                    "stratum_weights"
                ],
                stratum_events=stratum_statistics[
                    "stratum_events"
                ],
                stratum_denominators=stratum_statistics[
                    "stratum_denoms"
                ],
                confidence=confidence,
            )

            if stratum_statistics["haldane_applied"]:
                notes.append(
                    "Haldane correction applied."
                )

            if any(
                0 < stratum_denominator < 10
                for stratum_denominator in stratum_statistics[
                    "raw_stratum_denoms"
                ]
            ):
                notes.append(
                    "Unreliable: stratum denominator < 10."
                )

        if events == 0:
            notes.append(
                "Zero events."
            )

        elif events < 10:
            notes.append(
                "Low event count (<10): DSR should generally not be "
                "reported."
            )

        estimate = scale(
            estimate_unscaled,
        )

        lower = scale(
            lower_unscaled,
        )

        upper = scale(
            upper_unscaled,
        )

        records.append(
            {
                **key,
                "events": int(events),
                "denominator": round(
                    denominator,
                    6,
                ),
                "dsr": estimate,
                "dsr_lower": lower,
                "dsr_upper": upper,
                "multiplier": multiplier,
                "notes": " | ".join(notes),
                "significance": (
                    "Reference"
                    if is_reference
                    else _significance_from_ci(
                        scale(reference_unscaled),
                        lower,
                        upper,
                    )
                ),
            },
        )

    return pl.from_dicts(
        records,
    )