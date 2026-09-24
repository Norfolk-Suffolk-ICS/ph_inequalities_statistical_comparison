"""
ph_inequalities_statistical_comparison.py
===========================================
Public health standardisation functions using UKHSA/OHID-recommended methods.

Input validation
-----------------
All four public functions validate their numerator, strata, and group
columns before computation:

- Null/NA values are rejected outright in the numerator, strata, and group
  columns, with an informative error listing every offending column and its
  null count.
- The numerator column must be a non-negative integer column (or boolean,
  which is treated as a valid 0/1 encoding and cast to Int64). Float,
  string, or other non-integer dtypes raise an error, as does any negative
  value.
- For the two proportion functions (crude_proportion_df,
  directly_standardized_proportion_df), the numerator column must contain
  only 0 or 1 values -- any other integer value (e.g. 2) raises an error,
  since a proportion numerator must be a binary event indicator.
- For the two rate functions (crude_rate_df, directly_standardized_rate_df),
  the numerator may be any non-negative integer (e.g. a count of multiple
  events per row), since rates are not constrained to a 0/1 encoding.

Public API
----------
- crude_proportion_df  : crude proportion per group with Wilson score CI
- crude_rate_df        : crude rate per group with exact / Byar CI
- directly_standardized_proportion_df : DSP per group with Wilson-Dobson CI
- directly_standardized_rate_df       : DSR per group with Dobson-Byar CI

All four functions append an "Overall" row summarising the full input
dataset. When group_cols are supplied, every non-Overall row's confidence
interval is checked against the Overall/reference value: the reference
value is treated as a FIXED benchmark (not a random variable with its own
sampling error), and a group is flagged "Higher" or "Lower" whenever that
fixed reference value falls entirely outside the group's own confidence
interval. No formal hypothesis test (no p-values, no multiple-testing
correction) is performed -- significance is assessed purely via
confidence-interval overlap against the fixed reference, consistent with
the simplified approach commonly used in UKHSA/OHID public-facing
dashboards (e.g. Fingertips).

CI method summary
------------------
+--------------------+-------------------+----------------------------------+
| Measure            | Stratum CI        | Group-level CI                   |
+--------------------+-------------------+----------------------------------+
| Crude proportion   | -                 | Wilson score                     |
| Crude rate         | -                 | Exact chi-sq (O<10) / Byar (>=10)|
| DSP                | Haldane + Wilson  | PHE Wilson-Dobson                |
| DSR                | Haldane (sparse)  | PHE Dobson-Byar                  |
+--------------------+-------------------+----------------------------------+

Significance assessment
------------------------
For every non-Overall row, the Overall/reference value is compared against
that row's own confidence interval bounds:
  - reference < lower  -> "Higher" (group's estimate exceeds the reference)
  - reference > upper  -> "Lower"  (group's estimate is below the reference)
  - otherwise          -> "Not significant" (reference falls within the CI)
The Overall row itself is always labelled "Reference".

References
----------
- UKHSA Fingertips technical guidance:
  https://fingertips.phe.org.uk/static-reports/public-health-technical-guidance

Dependencies: polars, numpy, scipy
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations, product
from typing import Literal

import numpy as np
import polars as pl
from scipy import stats


# ===========================================================================
# Shared helpers
# ===========================================================================

def _cast_group_cols_to_utf8(df: pl.DataFrame, group_cols: list[str]) -> pl.DataFrame:
    if not group_cols:
        return df
    return df.with_columns([pl.col(c).cast(pl.Utf8) for c in group_cols])

def _bin_numeric_to_quartiles(series: pl.Series) -> pl.Series:
    """Bin a numeric Polars series into quartile labels Q1-Q4."""
    try:
        q1 = series.quantile(0.25, interpolation="linear")
        q2 = series.quantile(0.50, interpolation="linear")
        q3 = series.quantile(0.75, interpolation="linear")
    except Exception:
        return series

    def _label(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        if v <= q1:
            return "Q1"
        elif v <= q2:
            return "Q2"
        elif v <= q3:
            return "Q3"
        return "Q4"

    return pl.Series(series.name, [_label(v) for v in series.to_list()])


def _build_reference_weights(df: pl.DataFrame, strata_cols: list[str]) -> pl.DataFrame:
    """Derive reference (standard population) weights from the full dataset."""
    total = len(df)
    return (
        df.group_by(strata_cols)
        .agg(pl.len().alias("ref_count"))
        .with_columns((pl.col("ref_count") / total).alias("ref_weight"))
    )


def _bin_strata(df: pl.DataFrame, strata_cols: list[str]) -> pl.DataFrame:
    """Auto-bin numeric strata columns into quartiles; leave categorical columns unchanged."""
    for col in strata_cols:
        if df[col].dtype in (pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32,
                              pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64):
            df = df.with_columns(_bin_numeric_to_quartiles(df[col]).alias(col))
    return df


def _check_columns(df: pl.DataFrame, cols: list[str]) -> None:
    missing = set(cols) - set(df.columns)
    if missing:
        raise ValueError(f"Columns not found in dataframe: {missing}")


_INTEGER_DTYPES = (
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
)


def _check_no_nulls(df: pl.DataFrame, cols: list[str]) -> None:
    """
    Raise an informative error if any of the given columns contain nulls/NAs.

    Reports every offending column and its null count in a single error,
    rather than failing on the first one found.
    """
    offending = {}
    for col in cols:
        n_null = df[col].null_count()
        if n_null > 0:
            offending[col] = int(n_null)
    if offending:
        detail = ", ".join(f"'{c}' ({n} null value{'s' if n != 1 else ''})" for c, n in offending.items())
        raise ValueError(
            f"Null/NA values are not permitted in the numerator, strata, or group "
            f"columns. Offending column(s): {detail}. Remove or impute these rows "
            f"before calling this function."
        )


def _validate_numerator_col(df: pl.DataFrame, event_col: str, binary: bool) -> pl.DataFrame:
    """
    Validate that event_col is a non-negative integer column.

    Boolean columns are accepted and cast to Int64 (a boolean is inherently
    a valid 0/1 encoding). Any other non-integer dtype (e.g. float, string)
    raises an informative error. Negative integer values raise an error.
    When binary=True (proportion functions), values must be exactly 0 or 1.
    """
    dtype = df[event_col].dtype

    if dtype == pl.Boolean:
        return df.with_columns(pl.col(event_col).cast(pl.Int64))

    if dtype not in _INTEGER_DTYPES:
        raise ValueError(
            f"Column '{event_col}' must be a non-negative integer column (or boolean), "
            f"but has dtype {dtype}. Cast it to an integer type before calling this function."
        )

    min_val = df[event_col].min()
    if min_val is not None and min_val < 0:
        raise ValueError(
            f"Column '{event_col}' must contain only non-negative integers, "
            f"but found a minimum value of {min_val}."
        )

    if binary:
        distinct_vals = set(df[event_col].unique().to_list())
        distinct_vals.discard(None)
        invalid = distinct_vals - {0, 1}
        if invalid:
            raise ValueError(
                f"Column '{event_col}' must contain only 0 or 1 values for proportion "
                f"functions, but found other value(s): {sorted(invalid)}."
            )

    return df


def _prepare_dataframe_proportion(
    df: pl.DataFrame, event_col: str, strata_cols: list[str], group_cols: list[str]
) -> pl.DataFrame:
    _check_columns(df, [event_col] + strata_cols + group_cols)
    _check_no_nulls(df, [event_col] + strata_cols + group_cols)
    df = _validate_numerator_col(df, event_col, binary=True)
    return _bin_strata(df, strata_cols)


def _prepare_dataframe_rate(
    df: pl.DataFrame, event_col: str, strata_cols: list[str], group_cols: list[str]
) -> pl.DataFrame:
    required = [event_col] + strata_cols + group_cols
    _check_columns(df, required)
    _check_no_nulls(df, required)
    df = _validate_numerator_col(df, event_col, binary=False)
    return _bin_strata(df, strata_cols)


_DENOM_COL = "_denom_end_of_period_n"


def _add_end_of_period_denominator(df: pl.DataFrame) -> pl.DataFrame:
    """
    Synthesise a denominator column of all 1s.

    Summing this column per group is equivalent to counting rows, which is
    interpreted as the end-of-period population/headcount snapshot (i.e. a
    patient-level dataframe where one row = one patient present at the end
    of the reporting period). This is a simplifying assumption -- it does
    not account for partial-period exposure, entries, or exits -- but is a
    widely used practical convention when true person-time data is
    unavailable. This is the only supported denominator method for rate
    functions in this module.
    """
    return df.with_columns(pl.lit(1.0).alias(_DENOM_COL))


def _filter_group(df: pl.DataFrame, row: dict) -> pl.DataFrame:
    mask = pl.lit(True)
    for col, val in row.items():
        mask = mask & (pl.col(col) == val)
    return df.filter(mask)


@dataclass(frozen=True)
class _GroupingSet:
    active_cols: tuple[str, ...]


def _normalise_hierarchies(
    organisational_cols: Mapping[str, Sequence[str]] | None,
) -> dict[str, tuple[str, ...]]:
    if organisational_cols is None:
        return {}
    if not isinstance(organisational_cols, Mapping):
        raise TypeError(
            "organisational_cols must be a named mapping, for example "
            "{'commissioning': ['region', 'icb', 'practice']}."
        )
    result = {}
    for name, levels in organisational_cols.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Hierarchy names must be non-empty strings.")
        if isinstance(levels, (str, bytes)) or not isinstance(levels, Sequence):
            raise TypeError(
                f"Hierarchy '{name}' must be an ordered sequence of columns "
                "from highest to lowest level."
            )
        levels = tuple(levels)
        if not levels:
            raise ValueError(f"Hierarchy '{name}' must contain at least one column.")
        if any(not isinstance(col, str) or not col for col in levels):
            raise ValueError(f"Hierarchy '{name}' contains an invalid column name.")
        if len(levels) != len(set(levels)):
            raise ValueError(f"Hierarchy '{name}' contains duplicate columns.")
        result[name] = levels
    return result


def _prepare_dimensions(
    df: pl.DataFrame,
    event_col: str,
    strata_cols: Sequence[str],
    inequalities_cols: Sequence[str] | None,
    organisational_cols: Mapping[str, Sequence[str]] | None,
    organisational_mode: Literal["separate", "cross"],
    all_label: str,
) -> tuple[pl.DataFrame, list[str], dict[str, tuple[str, ...]], list[str]]:
    if organisational_mode not in {"separate", "cross"}:
        raise ValueError("organisational_mode must be 'separate' or 'cross'.")
    if not isinstance(all_label, str) or not all_label:
        raise ValueError("all_label must be a non-empty string.")
    if inequalities_cols is None:
        inequalities = []
    elif isinstance(inequalities_cols, (str, bytes)) or not isinstance(inequalities_cols, Sequence):
        raise TypeError("inequalities_cols must be a sequence or None.")
    else:
        inequalities = list(inequalities_cols)
    if any(not isinstance(col, str) or not col for col in inequalities):
        raise ValueError("inequalities_cols contains an invalid column name.")
    if len(inequalities) != len(set(inequalities)):
        raise ValueError("inequalities_cols contains duplicate columns.")

    hierarchies = _normalise_hierarchies(organisational_cols)
    organisational = [col for levels in hierarchies.values() for col in levels]
    repeated = sorted({col for col in organisational if organisational.count(col) > 1})
    if repeated:
        raise ValueError(f"Organisational columns occur in multiple hierarchies: {repeated}.")
    overlap = sorted(set(inequalities) & set(organisational))
    if overlap:
        raise ValueError(f"Columns cannot be both inequality and organisational: {overlap}.")

    strata = list(strata_cols)
    if len(strata) != len(set(strata)):
        raise ValueError("strata_cols contains duplicate columns.")
    dimensions = organisational + inequalities
    overlap = sorted(set(strata) & set(dimensions))
    if overlap:
        raise ValueError(f"Strata cannot also be grouping dimensions: {overlap}.")
    if event_col in dimensions or event_col in strata:
        raise ValueError("event_col cannot also be a stratum or grouping dimension.")

    required = [event_col] + strata + dimensions
    _check_columns(df, required)
    _check_no_nulls(df, required)
    conflicts = [
        col for col in dimensions
        if df.select(pl.col(col).cast(pl.Utf8).eq(all_label).any()).item()
    ]
    if conflicts:
        raise ValueError(
            f"Reserved all_label {all_label!r} occurs in columns {conflicts}; "
            "choose another all_label or recode the source values."
        )
    if dimensions:
        df = df.with_columns(pl.col(col).cast(pl.Utf8) for col in dimensions)

    for name, levels in hierarchies.items():
        for parent, child in zip(levels, levels[1:]):
            invalid = (
                df.select(parent, child).unique()
                .group_by(child)
                .agg(pl.col(parent).n_unique().alias("_parents"))
                .filter(pl.col("_parents") != 1)
            )
            if invalid.height:
                examples = invalid[child].head(5).to_list()
                raise ValueError(
                    f"Invalid hierarchy '{name}': '{child}' does not map to exactly "
                    f"one '{parent}'. Example values: {examples}. Hierarchies must be "
                    "ordered highest to lowest and cannot be many-to-many."
                )
    return df, inequalities, hierarchies, dimensions


def _inequality_states(cols: list[str]) -> list[tuple[str, ...]]:
    return [
        tuple(active)
        for size in range(len(cols), -1, -1)
        for active in combinations(cols, size)
    ]


def _organisational_states(
    hierarchies: Mapping[str, tuple[str, ...]],
    mode: Literal["separate", "cross"],
) -> list[tuple[str, ...]]:
    if not hierarchies:
        return [tuple()]
    if mode == "separate":
        states = [
            tuple(levels[:depth])
            for levels in hierarchies.values()
            for depth in range(len(levels), 0, -1)
        ]
        return states + [tuple()]
    choices = [
        [tuple(levels[:depth]) for depth in range(len(levels), -1, -1)]
        for levels in hierarchies.values()
    ]
    return [tuple(col for state in states for col in state) for states in product(*choices)]


def _build_grouping_sets(
    inequalities: list[str],
    hierarchies: Mapping[str, tuple[str, ...]],
    mode: Literal["separate", "cross"],
) -> list[_GroupingSet]:
    candidates = [
        org + inequality
        for org in _organisational_states(hierarchies, mode)
        for inequality in _inequality_states(inequalities)
    ]
    seen = set()
    result = []
    for active in candidates:
        if active not in seen:
            seen.add(active)
            result.append(_GroupingSet(active))
    return result


def _iter_group_slices(
    df: pl.DataFrame,
    dimensions: list[str],
    grouping_sets: Sequence[_GroupingSet],
    all_label: str,
):
    for grouping_set in grouping_sets:
        active = list(grouping_set.active_cols)
        keys = (
            df.select(active).unique(maintain_order=True).sort(active).iter_rows(named=True)
            if active else iter([{}])
        )
        for key in keys:
            group_data = _filter_group(df, key) if active else df
            display = {col: key[col] if col in active else all_label for col in dimensions}
            yield display, group_data, not active


def _validate_options(confidence: float, multiplier: float | None = None) -> None:
    if not 0 < confidence < 1:
        raise ValueError("confidence must be strictly between 0 and 1.")
    if multiplier is not None and (not np.isfinite(multiplier) or multiplier <= 0):
        raise ValueError("multiplier must be finite and positive.")


# ===========================================================================
# Wilson score CI (crude proportion)
# ===========================================================================

def _wilson_proportion_ci(events: float, n: float, confidence: float = 0.95) -> tuple[float, float, float]:
    """Wilson score CI for a single proportion. Returns (lower, upper, variance)."""
    if n == 0:
        return np.nan, np.nan, np.nan
    p = events / n
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    denom = 1 + (z ** 2) / n
    center = (p + (z ** 2) / (2 * n)) / denom
    half_width = (z * np.sqrt((p * (1 - p) / n) + (z ** 2) / (4 * n ** 2))) / denom
    lower = max(0.0, center - half_width)
    upper = min(1.0, center + half_width)
    variance = p * (1 - p) / n
    return float(lower), float(upper), float(variance)


# ===========================================================================
# Byar / exact Poisson CI (crude rate)
# ===========================================================================

def _byar_count_ci(count: float, confidence: float = 0.95) -> tuple[float, float]:
    """Byar's approximation for a Poisson count CI."""
    if count < 0:
        raise ValueError("count must be non-negative")
    if count == 0:
        return 0.0, float(-np.log(1.0 - confidence))
    alpha = 1.0 - confidence
    z = stats.norm.ppf(1.0 - alpha / 2.0)
    lower = count * (1.0 - 1.0 / (9.0 * count) - z / (3.0 * np.sqrt(count))) ** 3
    upper = (count + 1.0) * (1.0 - 1.0 / (9.0 * (count + 1.0)) + z / (3.0 * np.sqrt(count + 1.0))) ** 3
    return float(max(lower, 0.0)), float(max(upper, 0.0))


def _exact_poisson_count_ci(count: float, confidence: float = 0.95) -> tuple[float, float]:
    """Exact chi-square-based Poisson count CI, for sparse counts (<10)."""
    alpha = 1.0 - confidence
    if count == 0:
        lower = 0.0
    else:
        lower = 0.5 * stats.chi2.ppf(alpha / 2.0, 2 * count)
    upper = 0.5 * stats.chi2.ppf(1 - alpha / 2.0, 2 * (count + 1))
    return float(lower), float(upper)


# ===========================================================================
# DSP: Haldane + Wilson-Dobson
# ===========================================================================

def _haldane_proportion_correction(events: float, n: float) -> tuple[float, float, float]:
    """Haldane-Anscombe continuity correction for a proportion. Returns (rate, events, n)."""
    corrected_events = events + 0.5
    corrected_n = n + 1.0
    return float(corrected_events / corrected_n), float(corrected_events), float(corrected_n)


def _wilson_dobson_proportion_ci(
    dsp: float, crude_events: float, crude_n: float,
    stratum_weights: np.ndarray, stratum_props: np.ndarray, stratum_ns: np.ndarray,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """
    PHE Wilson-Dobson CI for a directly standardised proportion.
    Returns (lower, upper, variance).
    """
    if crude_n <= 0 or not np.isfinite(dsp):
        return np.nan, np.nan, np.nan

    valid = stratum_ns > 0
    w = stratum_weights[valid]
    p = stratum_props[valid]
    n = stratum_ns[valid]
    w_sum = float(np.sum(stratum_weights))
    if w_sum == 0 or len(w) == 0:
        return np.nan, np.nan, np.nan

    var_dsp = float(np.sum((w ** 2) * p * (1 - p) / n) / (w_sum ** 2))

    crude_p = crude_events / crude_n
    p_lo, p_hi, var_crude = _wilson_proportion_ci(crude_events, crude_n, confidence)

    if var_crude == 0 or not np.isfinite(var_crude):
        corrected_p, corrected_e, corrected_n = _haldane_proportion_correction(crude_events, crude_n)
        var_crude = corrected_p * (1 - corrected_p) / corrected_n

    if var_crude == 0 or not np.isfinite(var_crude):
        return float(dsp), float(dsp), var_dsp

    scale = float(np.sqrt(var_dsp / var_crude))
    lower = dsp + scale * (p_lo - crude_p)
    upper = dsp + scale * (p_hi - crude_p)
    return float(max(lower, 0.0)), float(min(upper, 1.0)), var_dsp


def _compute_proportion_stratum_stats(
    group_data: pl.DataFrame, all_strata: pl.DataFrame,
    event_col: str, strata_cols: list[str], ref_weights: pl.DataFrame,
) -> dict:
    """Aggregate stratum-level events/n, apply Haldane where needed, join to reference weights."""
    agg = (
        group_data.group_by(strata_cols)
        .agg(pl.col(event_col).sum().alias("events"), pl.len().alias("n"))
    )
    scaffold = (
        all_strata
        .join(agg, on=strata_cols, how="left")
        .join(ref_weights, on=strata_cols, how="left")
        .with_columns([
            pl.col("events").fill_null(0).cast(pl.Float64),
            pl.col("n").fill_null(0).cast(pl.Float64),
            pl.col("ref_weight").fill_null(0).cast(pl.Float64),
        ])
    )
    raw_events = scaffold["events"].to_numpy().astype(float)
    raw_ns = scaffold["n"].to_numpy().astype(float)
    weights = scaffold["ref_weight"].to_numpy().astype(float)

    props = np.zeros(len(raw_events))
    haldane_applied = False
    for i, (e, n) in enumerate(zip(raw_events, raw_ns)):
        if n == 0:
            props[i] = 0.0
        elif e == 0 or e == n:
            p_c, _, _ = _haldane_proportion_correction(e, n)
            props[i] = p_c
            haldane_applied = True
        else:
            props[i] = e / n

    return {
        "stratum_props": props,
        "stratum_ns": raw_ns,
        "stratum_weights": weights,
        "raw_stratum_ns": raw_ns,
        "raw_stratum_events": raw_events,
        "haldane_applied": haldane_applied,
    }


# ===========================================================================
# DSR: Haldane + Dobson-Byar
# ===========================================================================

def _haldane_rate_correction(events: float, denom: float) -> tuple[float, float, float]:
    """Haldane-style continuity correction for a rate stratum. Returns (rate, events, denom)."""
    corrected_events = events + 0.5
    corrected_denom = denom + 1.0
    return float(corrected_events / corrected_denom), float(corrected_events), float(corrected_denom)


def _dobson_byar_rate_ci(
    dsr_unscaled: float, crude_events: float,
    stratum_weights: np.ndarray, stratum_events: np.ndarray, stratum_denoms: np.ndarray,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """
    PHE Dobson-Byar CI for a directly standardised rate (unscaled).
    Returns (lower, upper, variance).
    """
    if crude_events < 0:
        return np.nan, np.nan, np.nan

    if crude_events < 10:
        O_lo, O_hi = _exact_poisson_count_ci(crude_events, confidence)
    else:
        O_lo, O_hi = _byar_count_ci(crude_events, confidence)

    valid = stratum_denoms > 0
    w = stratum_weights[valid]
    Oi = stratum_events[valid]
    ni = stratum_denoms[valid]
    w_sum = float(np.sum(stratum_weights))
    if w_sum == 0 or len(w) == 0:
        return np.nan, np.nan, np.nan

    var_dsr = float(np.sum((w ** 2) * Oi / (ni ** 2)) / (w_sum ** 2))
    var_O = float(crude_events)

    if var_O == 0.0:
        _, corrected_events, _ = _haldane_rate_correction(crude_events, 1.0)
        var_O = corrected_events

    if var_O == 0.0 or var_dsr == 0.0:
        return float(dsr_unscaled), float(dsr_unscaled), var_dsr

    scale = float(np.sqrt(var_dsr / var_O))
    lower = dsr_unscaled + scale * (O_lo - crude_events)
    upper = dsr_unscaled + scale * (O_hi - crude_events)
    return float(max(lower, 0.0)), float(max(upper, 0.0)), var_dsr

def _compute_rate_stratum_stats(
    group_data: pl.DataFrame, all_strata: pl.DataFrame,
    event_col: str, denom_col: str, strata_cols: list[str], ref_weights: pl.DataFrame,
) -> dict:
    """Aggregate stratum-level events/denom, apply Haldane to zero-event strata, join to weights."""
    agg = (
        group_data.group_by(strata_cols)
        .agg(pl.col(event_col).sum().alias("events"), pl.col(denom_col).sum().alias("denom"))
    )
    scaffold = (
        all_strata
        .join(agg, on=strata_cols, how="left")
        .join(ref_weights, on=strata_cols, how="left")
        .with_columns([
            pl.col("events").fill_null(0).cast(pl.Float64),
            pl.col("denom").fill_null(0).cast(pl.Float64),
            pl.col("ref_weight").fill_null(0).cast(pl.Float64),
        ])
    )
    raw_events = scaffold["events"].to_numpy().astype(float)
    raw_denoms = scaffold["denom"].to_numpy().astype(float)
    weights = scaffold["ref_weight"].to_numpy().astype(float)

    rates = np.zeros(len(raw_events))
    corrected_events = raw_events.copy()
    corrected_denoms = raw_denoms.copy()
    haldane_applied = False
    for i, (e, d) in enumerate(zip(raw_events, raw_denoms)):
        if d == 0:
            rates[i] = 0.0
        elif e == 0:
            r_c, e_c, d_c = _haldane_rate_correction(e, d)
            rates[i] = r_c
            corrected_events[i] = e_c
            corrected_denoms[i] = d_c
            haldane_applied = True
        else:
            rates[i] = e / d

    return {
        "stratum_rates": rates,
        "stratum_events": corrected_events,
        "stratum_denoms": corrected_denoms,
        "stratum_weights": weights,
        "raw_stratum_denoms": raw_denoms,
        "raw_stratum_events": raw_events,
        "haldane_applied": haldane_applied,
    }


# ===========================================================================
# Quality-flag notes
# ===========================================================================

def _crude_proportion_notes(events: float, n: float) -> str:
    notes = []
    if n == 0:
        notes.append("Zero denominator - must suppress and consider changing oganisational heirachy and inequalities being passed to functions")
    else:
        if events == 0:
            notes.append("Zero events - must flag that confidence intervals are not robust")
        elif events < 10:
            notes.append("Low event count (<10) - must flag as unstable or suppress in visual")
        non_events = n - events
        if non_events == 0:
            notes.append("All events (proportion = 1) - must flag as degenerate or suppress in visual")
        elif non_events < 10:
            notes.append("Zero events - must flag that confidence intervals are not robust")

    if n < 40:
        notes.append("Low sample size - must flag that rate is unstable")


    return " | ".join(notes)


def _crude_rate_notes(events: float, denominator: float, end_of_period_denom: bool = False) -> str:
    notes = []
    if end_of_period_denom:
        notes.append(
            "End-of-period denominator: row count used as exposure "
            "(assumes each row = 1 patient present at period end; "
            "does not account for partial-period exposure)"
        )
    if denominator == 0:
        notes.append("Zero denominator")
    else:
        if events == 0:
            notes.append("Zero events - must flag that confidence intervals are not robust")
        elif events < 10:
            notes.append("Low event count (<10) - must flag as unstable or suppress in visual")

    if denominator < 40:
        notes.append("Low sample size - must flag that rate is unstable")



# ===========================================================================
# Significance assessment vs the overall/population reference
# ===========================================================================
#
# The Overall/reference value is treated as a FIXED benchmark (not a random
# variable). No formal hypothesis test is performed: a group's estimate is
# simply classified according to whether the fixed reference value falls
# inside or outside that group's own confidence interval.

def _significance_from_ci(reference: float, lower: float, upper: float) -> str:
    """
    Classify a group's estimate relative to a fixed reference value, based
    purely on whether the reference falls inside or outside the group's own
    confidence interval (no formal hypothesis test is performed).

    Returns one of: "Higher", "Lower", "Not significant", "Not tested".
    """
    if not np.isfinite(reference) or not np.isfinite(lower) or not np.isfinite(upper):
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
    """Crude proportions over hierarchy roll-ups and the full inequality cube."""
    _validate_options(confidence)
    df, inequalities, hierarchies, dimensions = _prepare_dimensions(
        df, event_col, [], inequalities_cols, organisational_cols,
        organisational_mode, all_label,
    )
    df = _validate_numerator_col(df, event_col, binary=True)
    sets = _build_grouping_sets(inequalities, hierarchies, organisational_mode)
    overall_e, overall_n = float(df[event_col].sum()), float(len(df))
    reference = overall_e / overall_n if overall_n else np.nan
    records = []
    for key, data, is_reference in _iter_group_slices(df, dimensions, sets, all_label):
        events, n = float(data[event_col].sum()), float(len(data))
        lower, upper, _ = _wilson_proportion_ci(events, n, confidence)
        estimate = events / n if n else np.nan
        records.append({
            **key, "events": events, "n": n, "proportion": estimate,
            "lower": lower, "upper": upper, "confidence": confidence,
            "method": "Wilson score", "notes": _crude_proportion_notes(events, n),
            "significance": "Reference" if is_reference else _significance_from_ci(reference, lower, upper),
        })
    return pl.from_dicts(records)


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
    """Crude rates over hierarchy roll-ups and the full inequality cube."""
    _validate_options(confidence, multiplier)
    df, inequalities, hierarchies, dimensions = _prepare_dimensions(
        df, event_col, [], inequalities_cols, organisational_cols,
        organisational_mode, all_label,
    )
    df = _add_end_of_period_denominator(_validate_numerator_col(df, event_col, binary=False))
    sets = _build_grouping_sets(inequalities, hierarchies, organisational_mode)
    overall_e, overall_d = float(df[event_col].sum()), float(df[_DENOM_COL].sum())
    reference = overall_e / overall_d * multiplier if overall_d else np.nan

    def calculate(events, denominator):
        if not denominator:
            return np.nan, np.nan, np.nan, "undefined"
        if events < 10:
            low, high = _exact_poisson_count_ci(events, confidence)
            method = "Exact chi-square"
        else:
            low, high = _byar_count_ci(events, confidence)
            method = "Byar"
        return events / denominator * multiplier, low / denominator * multiplier, high / denominator * multiplier, method

    records = []
    for key, data, is_reference in _iter_group_slices(df, dimensions, sets, all_label):
        events = float(data[event_col].sum())
        denominator = float(data[_DENOM_COL].sum())
        estimate, lower, upper, method = calculate(events, denominator)
        records.append({
            **key, "events": events, "denominator": denominator, "rate": estimate,
            "lower": lower, "upper": upper, "multiplier": multiplier,
            "confidence": confidence, "method": method,
            "notes": _crude_rate_notes(events, denominator, True),
            "significance": "Reference" if is_reference else _significance_from_ci(reference, lower, upper),
        })
    return pl.from_dicts(records)


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
    """DSPs over hierarchy roll-ups and the full inequality cube."""
    _validate_options(confidence)
    strata = list(strata_cols)
    if not strata:
        raise ValueError("strata_cols must contain at least one column.")
    df, inequalities, hierarchies, dimensions = _prepare_dimensions(
        df, event_col, strata, inequalities_cols, organisational_cols,
        organisational_mode, all_label,
    )
    work = _prepare_dataframe_proportion(df, event_col, strata, dimensions)
    sets = _build_grouping_sets(inequalities, hierarchies, organisational_mode)
    weights = _build_reference_weights(work, strata)
    all_strata = weights.select(strata)
    overall_e, overall_n = float(work[event_col].sum()), float(len(work))
    reference = overall_e / overall_n if overall_n else np.nan
    records = []
    for key, data, is_reference in _iter_group_slices(work, dimensions, sets, all_label):
        events, n = float(data[event_col].sum()), float(len(data))
        notes = []
        if is_reference:
            estimate = reference
            lower, upper, _ = _wilson_dobson_proportion_ci(
                reference, overall_e, overall_n, np.array([1.0]),
                np.array([reference]), np.array([overall_n]), confidence,
            ) if overall_n else (np.nan, np.nan, np.nan)
            notes.append("Overall proportion: no weighting applied as this row is itself the reference population")
        else:
            s = _compute_proportion_stratum_stats(data, all_strata, event_col, strata, weights)
            estimate = float(np.sum(s["stratum_weights"] * s["stratum_props"]))
            lower, upper, _ = _wilson_dobson_proportion_ci(
                estimate, events, n, s["stratum_weights"], s["stratum_props"],
                s["stratum_ns"], confidence,
            )
            if s["haldane_applied"]:
                notes.append("Haldane correction applied")
            if any(0 < x < 10 for x in s["raw_stratum_ns"]):
                notes.append("Unreliable: stratum n<10")
            non_events = [float(ni) - float(ei) for ni, ei in zip(s["raw_stratum_ns"], s["raw_stratum_events"]) if ni > 0]
            if any(0 < x < 10 for x in non_events):
                notes.append("Unreliable: stratum non-event count <10")
        non_events = n - events
        if events == 0: notes.append("Zero events")
        elif events < 10: notes.append("Low event count (<10)")
        if non_events == 0: notes.append("All events (proportion = 1)")
        elif non_events < 10: notes.append("Low non-event count (<10)")
        records.append({
            **key, "events": int(events), "n": int(n),
            "dsp": round(estimate, 6) if np.isfinite(estimate) else np.nan,
            "dsp_lower": round(lower, 6) if np.isfinite(lower) else np.nan,
            "dsp_upper": round(upper, 6) if np.isfinite(upper) else np.nan,
            "notes": " | ".join(notes),
            "significance": "Reference" if is_reference else _significance_from_ci(reference, lower, upper),
        })
    return pl.from_dicts(records)


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
    """DSRs over hierarchy roll-ups and the full inequality cube."""
    _validate_options(confidence, multiplier)
    strata = list(strata_cols)
    if not strata:
        raise ValueError("strata_cols must contain at least one column.")
    df, inequalities, hierarchies, dimensions = _prepare_dimensions(
        df, event_col, strata, inequalities_cols, organisational_cols,
        organisational_mode, all_label,
    )
    work = _add_end_of_period_denominator(df)
    work = _prepare_dataframe_rate(work, event_col, strata, dimensions)
    sets = _build_grouping_sets(inequalities, hierarchies, organisational_mode)
    weights = _build_reference_weights(work, strata)
    all_strata = weights.select(strata)
    overall_e, overall_d = float(work[event_col].sum()), float(work[_DENOM_COL].sum())
    reference_u = overall_e / overall_d if overall_d else np.nan
    scale = lambda x: round(x * multiplier, 6) if np.isfinite(x) else np.nan
    records = []
    for key, data, is_reference in _iter_group_slices(work, dimensions, sets, all_label):
        events, denominator = float(data[event_col].sum()), float(data[_DENOM_COL].sum())
        notes = ["End-of-period denominator: row count used as exposure (assumes each row = 1 patient present at period end; does not account for partial-period exposure)"]
        if is_reference:
            estimate_u = reference_u
            if denominator:
                low_count, high_count = (_exact_poisson_count_ci(events, confidence) if events < 10 else _byar_count_ci(events, confidence))
                lower_u, upper_u = low_count / denominator, high_count / denominator
            else:
                lower_u = upper_u = np.nan
            notes.append("Overall rate: no weighting applied as this row is itself the reference population")
        else:
            s = _compute_rate_stratum_stats(data, all_strata, event_col, _DENOM_COL, strata, weights)
            weight_sum = float(np.sum(s["stratum_weights"]))
            estimate_u = float(np.sum(s["stratum_weights"] * s["stratum_rates"]) / weight_sum) if weight_sum else np.nan
            lower_u, upper_u, _ = _dobson_byar_rate_ci(
                estimate_u, events, s["stratum_weights"], s["stratum_events"],
                s["stratum_denoms"], confidence,
            )
            if s["haldane_applied"]: notes.append("Haldane correction applied")
            if any(0 < x < 10 for x in s["raw_stratum_denoms"]): notes.append("Unreliable: stratum denominator <10")
        if events == 0: notes.append("Zero events")
        elif events < 10: notes.append("Low event count (<10): DSR should generally not be reported")
        if denominator == 0: notes.append("Zero denominator")
        estimate, lower, upper = scale(estimate_u), scale(lower_u), scale(upper_u)
        records.append({
            **key, "events": int(events), "denominator": round(denominator, 6),
            "dsr": estimate, "dsr_lower": lower, "dsr_upper": upper,
            "multiplier": multiplier, "notes": " | ".join(notes),
            "significance": "Reference" if is_reference else _significance_from_ci(scale(reference_u), lower, upper),
        })
    return pl.from_dicts(records)