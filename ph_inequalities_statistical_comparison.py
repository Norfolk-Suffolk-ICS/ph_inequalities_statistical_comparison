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

from typing import Sequence

import numpy as np
import polars as pl
from scipy import stats


# ===========================================================================
# Shared helpers
# ===========================================================================

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
        notes.append("Zero denominator")
    else:
        if events == 0:
            notes.append("Zero events")
        elif events < 10:
            notes.append("Low event count (<10)")
        non_events = n - events
        if non_events == 0:
            notes.append("All events (proportion = 1)")
        elif non_events < 10:
            notes.append("Low non-event count (<10)")
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
            notes.append("Zero events")
        elif events < 10:
            notes.append("Low event count (<10): consider suppression")
    return " | ".join(notes)


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
    group_cols: Sequence[str] | None = None,
    confidence: float = 0.95,
) -> pl.DataFrame:
    """
    Compute crude proportions with Wilson score confidence intervals.

    Parameters
    ----------
    df : pl.DataFrame
        Input dataset containing event_col and any group_cols.
    event_col : str
        Binary event indicator column (0/1 or boolean).
        The proportion is computed as sum(event_col) / row_count.
    group_cols : Sequence[str] | None
        Columns to group by.  One output row per unique combination.
        If None or empty, a single overall row is returned.
    confidence : float, default 0.95
        Confidence level for the Wilson interval.

    Returns
    -------
    pl.DataFrame
        Columns: [*group_cols, events, n, proportion, lower, upper,
                  confidence, method, notes, significance]
        An additional "Overall" row is appended (group_cols set to "Overall")
        summarising the crude proportion across the full input dataset.
        Each non-Overall row's significance is assessed by checking whether
        the fixed Overall proportion falls inside or outside that row's own
        Wilson score confidence interval -- no formal hypothesis test is
        performed. The Overall row itself is always labelled "Reference".
    """
    group_cols = list(group_cols or [])
    _check_columns(df, [event_col] + group_cols)
    _check_no_nulls(df, [event_col] + group_cols)
    df = _validate_numerator_col(df, event_col, binary=True)

    if not group_cols:
        events = float(df[event_col].sum())
        n = float(len(df))
        lo, hi, _ = _wilson_proportion_ci(events, n, confidence)
        proportion = np.nan if n == 0 else events / n
        return pl.DataFrame([{
            "events": events, "n": n,
            "proportion": float(proportion) if np.isfinite(proportion) else np.nan,
            "lower": lo, "upper": hi,
            "confidence": confidence, "method": "Wilson score",
            "notes": _crude_proportion_notes(events, n),
            "significance": "Not tested",
        }])

    grouped = (
        df.group_by(group_cols)
        .agg([pl.col(event_col).sum().alias("events"), pl.len().alias("n")])
        .sort(group_cols)
    )

    overall_e = float(df[event_col].sum())
    overall_n = float(len(df))
    overall_prop = np.nan if overall_n == 0 else overall_e / overall_n

    records = []
    for row in grouped.iter_rows(named=True):
        e, n = float(row["events"]), float(row["n"])
        lo, hi, _ = _wilson_proportion_ci(e, n, confidence)
        proportion = np.nan if n == 0 else e / n
        records.append({
            **{c: row[c] for c in group_cols},
            "events": e, "n": n,
            "proportion": float(proportion) if np.isfinite(proportion) else np.nan,
            "lower": lo, "upper": hi,
            "confidence": confidence, "method": "Wilson score",
            "notes": _crude_proportion_notes(e, n),
            "significance": _significance_from_ci(overall_prop, lo, hi),
        })

    o_lo, o_hi, _ = _wilson_proportion_ci(overall_e, overall_n, confidence)
    records.append({
        **{c: "Overall" for c in group_cols},
        "events": overall_e, "n": overall_n,
        "proportion": float(overall_prop) if np.isfinite(overall_prop) else np.nan,
        "lower": o_lo, "upper": o_hi,
        "confidence": confidence, "method": "Wilson score",
        "notes": _crude_proportion_notes(overall_e, overall_n),
        "significance": "Reference",
    })
    return pl.from_dicts(records)


def crude_rate_df(
    df: pl.DataFrame,
    event_col: str,
    group_cols: Sequence[str] | None = None,
    multiplier: float = 100_000.0,
    confidence: float = 0.95,
) -> pl.DataFrame:
    """
    Compute crude rates with exact chi-square (count < 10) or Byar (count >= 10) CIs.

    The denominator (exposure) is ALWAYS inferred from row count: each row is
    treated as one unit of exposure (1 patient = 1 row), interpreted as the
    population/headcount at the END of the reporting period. This is the
    only supported denominator method in this module -- it is a simplifying
    assumption for patient-level dataframes where true person-time is
    unavailable: patients who did not experience the full exposure period
    are still counted as a full unit. This is flagged explicitly via the
    "End-of-period denominator" entry in notes on every row.

    Parameters
    ----------
    df : pl.DataFrame
        Input dataset. One row is assumed to represent one unit of exposure.
    event_col : str
        Column containing observed event counts.
    group_cols : Sequence[str] | None
        Grouping columns.  One output row per unique combination.
        If None or empty, a single overall row is returned.
    multiplier : float, default 100_000
        Rate scaling factor (e.g. 100_000 for per 100,000).
    confidence : float, default 0.95
        Confidence level.

    Returns
    -------
    pl.DataFrame
        Columns: [*group_cols, events, denominator, rate, lower, upper,
                  multiplier, confidence, method, notes, significance]
        An additional "Overall" row is appended (group_cols set to "Overall")
        summarising the crude rate across the full input dataset.
        Each non-Overall row's significance is assessed by checking whether
        the fixed Overall rate falls inside or outside that row's own
        confidence interval -- no formal hypothesis test is performed.
        The Overall row itself is always labelled "Reference".
    """
    group_cols = list(group_cols or [])
    _check_columns(df, [event_col] + group_cols)
    _check_no_nulls(df, [event_col] + group_cols)
    df = _validate_numerator_col(df, event_col, binary=False)

    df = _add_end_of_period_denominator(df)
    denom_col = _DENOM_COL
    used_end_of_period_denom = True

    def _compute_row(e: float, d: float) -> dict:
        if d == 0:
            return {"rate": np.nan, "lower": np.nan, "upper": np.nan, "method": "undefined"}
        if e < 10:
            cnt_lo, cnt_hi = _exact_poisson_count_ci(e, confidence)
            method = "Exact chi-square"
        else:
            cnt_lo, cnt_hi = _byar_count_ci(e, confidence)
            method = "Byar"
        return {
            "rate": (e / d) * multiplier,
            "lower": (cnt_lo / d) * multiplier,
            "upper": (cnt_hi / d) * multiplier,
            "method": method,
        }

    if not group_cols:
        e = float(df[event_col].sum())
        d = float(df[denom_col].sum())
        r = _compute_row(e, d)
        return pl.DataFrame([{
            "events": e, "denominator": d,
            **r, "multiplier": multiplier, "confidence": confidence,
            "notes": _crude_rate_notes(e, d, used_end_of_period_denom),
            "significance": "Not tested",
        }])

    grouped = (
        df.group_by(group_cols)
        .agg([pl.col(event_col).sum().alias("events"), pl.col(denom_col).sum().alias("denominator")])
        .sort(group_cols)
    )

    overall_e = float(df[event_col].sum())
    overall_d = float(df[denom_col].sum())
    overall_rate = np.nan if overall_d == 0 else (overall_e / overall_d) * multiplier

    records = []
    for row in grouped.iter_rows(named=True):
        e, d = float(row["events"]), float(row["denominator"])
        r = _compute_row(e, d)
        records.append({
            **{c: row[c] for c in group_cols},
            "events": e, "denominator": d,
            **r, "multiplier": multiplier, "confidence": confidence,
            "notes": _crude_rate_notes(e, d, used_end_of_period_denom),
            "significance": _significance_from_ci(overall_rate, r["lower"], r["upper"]),
        })

    r_overall = _compute_row(overall_e, overall_d)
    records.append({
        **{c: "Overall" for c in group_cols},
        "events": overall_e, "denominator": overall_d,
        **r_overall, "multiplier": multiplier, "confidence": confidence,
        "notes": _crude_rate_notes(overall_e, overall_d, used_end_of_period_denom),
        "significance": "Reference",
    })
    return pl.from_dicts(records)


def directly_standardized_proportion_df(
    df: pl.DataFrame,
    event_col: str,
    strata_cols: Sequence[str],
    group_cols: Sequence[str],
    confidence: float = 0.95,
) -> pl.DataFrame:
    """
    Compute directly standardised proportions (DSP) with Wilson-Dobson CIs.

    Reference weights are derived from the full dataset.  Numeric strata columns
    are automatically binned into quartile categories.  The Wilson-Dobson method
    produces asymmetric CIs that naturally approach the [0, 1] boundary.

    Parameters
    ----------
    df : pl.DataFrame
        Input dataset.
    event_col : str
        Binary event indicator (0/1 or boolean).
    strata_cols : Sequence[str]
        Standardisation strata (numeric strata auto-binned to quartiles).
    group_cols : Sequence[str]
        Grouping columns.  One output row per unique combination.
    confidence : float, default 0.95
        Confidence level.

    Returns
    -------
    pl.DataFrame
        Columns: [*group_cols, events, n, dsp, dsp_lower, dsp_upper, notes,
                  significance]
        An additional "Overall" row is appended (group_cols set to "Overall")
        giving the crude (unstandardised) proportion across the full input
        dataset, with its own Wilson-Dobson CI. Because this row IS the
        reference population used to build the standardisation weights, it
        requires no weighting -- this is stated explicitly in its notes.
        Each non-Overall row's significance is assessed by checking whether
        the fixed Overall proportion falls inside or outside that row's own
        dsp_lower/dsp_upper interval -- no formal hypothesis test is
        performed. The Overall row itself is always labelled "Reference".

    Notes
    -----
    notes flags:
        "Haldane correction applied"              - boundary stratum (0% or 100%) detected.
        "Zero events"                             - group has no events.
        "Low event count (<10)"                   - raw group events < 10.
        "All events (proportion = 1)"             - all records are events.
        "Low non-event count (<10)"               - group has fewer than 10 non-events.
        "Unreliable: stratum n<10"                - non-empty stratum with < 10 records.
        "Unreliable: stratum non-event count <10" - non-empty stratum with < 10 non-events.
        "Overall proportion: no weighting applied as this row is itself the
         reference population"                    - appears only on the "Overall" row.
    """
    strata_cols = list(strata_cols)
    group_cols = list(group_cols)

    work_df = _prepare_dataframe_proportion(df, event_col, strata_cols, group_cols)
    ref_weights = _build_reference_weights(work_df, strata_cols)
    all_strata = ref_weights.select(strata_cols)
    group_combinations = work_df.select(group_cols).unique(maintain_order=True)

    overall_events = float(work_df[event_col].sum())
    overall_n = float(len(work_df))
    overall_proportion = overall_events / overall_n if overall_n > 0 else np.nan

    records = []
    for row in group_combinations.iter_rows(named=True):
        group_data = _filter_group(work_df, row)
        crude_events = float(group_data[event_col].sum())
        crude_n = float(len(group_data))

        s = _compute_proportion_stratum_stats(
            group_data, all_strata, event_col, strata_cols, ref_weights
        )
        dsp = float(np.sum(s["stratum_weights"] * s["stratum_props"]))
        dsp_lower, dsp_upper, _ = _wilson_dobson_proportion_ci(
            dsp, crude_events, crude_n,
            s["stratum_weights"], s["stratum_props"], s["stratum_ns"], confidence,
        )

        crude_non_events = crude_n - crude_events
        notes = []
        if s["haldane_applied"]:
            notes.append("Haldane correction applied")
        if crude_events == 0:
            notes.append("Zero events")
        elif crude_events < 10:
            notes.append("Low event count (<10)")
        if crude_non_events == 0:
            notes.append("All events (proportion = 1)")
        elif crude_non_events < 10:
            notes.append("Low non-event count (<10)")
        has_unreliable_stratum = any(0 < v < 10 for v in s["raw_stratum_ns"])
        if has_unreliable_stratum:
            notes.append("Unreliable: stratum n<10")
        raw_non_events = [
            float(n_i) - float(e_i)
            for n_i, e_i in zip(s["raw_stratum_ns"], s["raw_stratum_events"])
            if float(n_i) > 0
        ]
        has_unreliable_non_events = any(0 < v < 10 for v in raw_non_events)
        if has_unreliable_non_events:
            notes.append("Unreliable: stratum non-event count <10")

        records.append({
            **row,
            "events": int(crude_events), "n": int(crude_n),
            "dsp": round(dsp, 6),
            "dsp_lower": round(dsp_lower, 6),
            "dsp_upper": round(dsp_upper, 6),
            "notes": " | ".join(notes),
            "significance": _significance_from_ci(overall_proportion, dsp_lower, dsp_upper),
        })

    o_lo, o_hi, _ = _wilson_dobson_proportion_ci(
        overall_proportion,
        overall_events, overall_n,
        np.array([1.0]), np.array([overall_proportion]),
        np.array([overall_n]), confidence,
    ) if overall_n > 0 else (np.nan, np.nan, np.nan)
    overall_non_events = overall_n - overall_events

    overall_notes = []
    if overall_events == 0:
        overall_notes.append("Zero events")
    elif overall_events < 10:
        overall_notes.append("Low event count (<10)")
    if overall_non_events == 0:
        overall_notes.append("All events (proportion = 1)")
    elif overall_non_events < 10:
        overall_notes.append("Low non-event count (<10)")
    overall_notes.append(
        "Overall proportion: no weighting applied as this row is itself the reference population"
    )

    records.append({
        **{c: "Overall" for c in group_cols},
        "events": int(overall_events), "n": int(overall_n),
        "dsp": round(overall_proportion, 6) if np.isfinite(overall_proportion) else np.nan,
        "dsp_lower": round(o_lo, 6) if np.isfinite(o_lo) else np.nan,
        "dsp_upper": round(o_hi, 6) if np.isfinite(o_hi) else np.nan,
        "notes": " | ".join(overall_notes),
        "significance": "Reference",
    })

    if not records:
        schema = {c: pl.Utf8 for c in group_cols}
        schema.update({"events": pl.Int64, "n": pl.Int64,
                        "dsp": pl.Float64, "dsp_lower": pl.Float64,
                        "dsp_upper": pl.Float64, "notes": pl.Utf8,
                        "significance": pl.Utf8})
        return pl.DataFrame(schema=schema)
    return pl.from_dicts(records)


def directly_standardized_rate_df(
    df: pl.DataFrame,
    event_col: str,
    strata_cols: Sequence[str],
    group_cols: Sequence[str],
    multiplier: float = 100_000.0,
    confidence: float = 0.95,
) -> pl.DataFrame:
    """
    Compute directly standardised rates (DSR) with Dobson-Byar CIs.

    Reference weights are derived from the full dataset.  Numeric strata columns
    are automatically binned into quartile categories.

    The denominator (exposure) is ALWAYS inferred from row count within each
    stratum/group combination: one row = one unit of exposure, interpreted
    as the population/headcount at the END of the reporting period. This is
    the only supported denominator method in this module -- a simplifying
    assumption for patient-level dataframes where true person-time is
    unavailable: patients who did not experience the full exposure period
    are still counted as a full unit. Flagged explicitly via the
    "End-of-period denominator" entry in notes on every row.

    Parameters
    ----------
    df : pl.DataFrame
        Input dataset. One row is assumed to represent one unit of exposure.
    event_col : str
        Column containing observed event counts.
    strata_cols : Sequence[str]
        Standardisation strata (numeric strata auto-binned to quartiles).
    group_cols : Sequence[str]
        Grouping columns.  One output row per unique combination.
    multiplier : float, default 100_000
        Rate scaling factor.
    confidence : float, default 0.95
        Confidence level.

    Returns
    -------
    pl.DataFrame
        Columns: [*group_cols, events, denominator, dsr, dsr_lower, dsr_upper,
                  multiplier, notes, significance]
        An additional "Overall" row is appended (group_cols set to "Overall")
        giving the crude (unstandardised) rate across the full input dataset,
        with its own Dobson-Byar/exact CI. Because this row IS the reference
        population used to build the standardisation weights, it requires no
        weighting -- this is stated explicitly in its notes.
        Each non-Overall row's significance is assessed by checking whether
        the fixed Overall rate falls inside or outside that row's own
        dsr_lower/dsr_upper interval -- no formal hypothesis test is
        performed. The Overall row itself is always labelled "Reference".

    Notes
    -----
    notes flags:
        "End-of-period denominator: row count used as exposure"
                                                               - appears on every row.
        "Haldane correction applied"                          - zero-event non-empty stratum.
        "Zero events"                                         - group has no events.
        "Low event count (<10): DSR should generally not be reported"
        "Zero denominator"                                    - group denominator is zero.
        "Unreliable: stratum denominator <10"                 - non-empty stratum denominator < 10.
        "Overall rate: no weighting applied as this row is itself the
         reference population"                                - appears only on the "Overall" row.
    """
    strata_cols = list(strata_cols)
    group_cols = list(group_cols)

    work_df = _add_end_of_period_denominator(df)
    denom_col = _DENOM_COL
    used_end_of_period_denom = True
    work_df = _prepare_dataframe_rate(work_df, event_col, strata_cols, group_cols)
    ref_weights = _build_reference_weights(work_df, strata_cols)
    all_strata = ref_weights.select(strata_cols)
    group_combinations = work_df.select(group_cols).unique(maintain_order=True)

    overall_events = float(work_df[event_col].sum())
    overall_denom = float(work_df[denom_col].sum())
    overall_rate_u = overall_events / overall_denom if overall_denom > 0 else np.nan

    def _scale(v):
        return round(v * multiplier, 6) if np.isfinite(v) else np.nan

    records = []
    for row in group_combinations.iter_rows(named=True):
        group_data = _filter_group(work_df, row)
        crude_events = float(group_data[event_col].sum())
        crude_denom = float(group_data[denom_col].sum())

        s = _compute_rate_stratum_stats(
            group_data, all_strata, event_col, denom_col, strata_cols, ref_weights
        )
        w_sum = float(np.sum(s["stratum_weights"]))
        dsr_u = (
            float(np.sum(s["stratum_weights"] * s["stratum_rates"]) / w_sum)
            if w_sum > 0 else np.nan
        )
        lo_u, hi_u, _ = _dobson_byar_rate_ci(
            dsr_u, crude_events,
            s["stratum_weights"], s["stratum_events"], s["stratum_denoms"], confidence,
        )

        notes = []
        if used_end_of_period_denom:
            notes.append(
                "End-of-period denominator: row count used as exposure "
                "(assumes each row = 1 patient present at period end; "
                "does not account for partial-period exposure)"
            )
        if s["haldane_applied"]:
            notes.append("Haldane correction applied")
        if crude_events == 0:
            notes.append("Zero events")
        elif crude_events < 10:
            notes.append("Low event count (<10): DSR should generally not be reported")
        if crude_denom == 0:
            notes.append("Zero denominator")
        has_unreliable_stratum = any(0 < v < 10 for v in s["raw_stratum_denoms"])
        if has_unreliable_stratum:
            notes.append("Unreliable: stratum denominator <10")

        dsr_scaled = _scale(dsr_u)
        lo_scaled = _scale(lo_u)
        hi_scaled = _scale(hi_u)
        records.append({
            **row,
            "events": int(crude_events), "denominator": round(crude_denom, 6),
            "dsr": dsr_scaled,
            "dsr_lower": lo_scaled,
            "dsr_upper": hi_scaled,
            "multiplier": multiplier,
            "notes": " | ".join(notes),
            "significance": _significance_from_ci(
                overall_rate_u * multiplier if np.isfinite(overall_rate_u) else np.nan,
                lo_scaled, hi_scaled,
            ),
        })

    if overall_denom == 0:
        o_lo_u, o_hi_u = np.nan, np.nan
    else:
        if overall_events < 10:
            cnt_lo, cnt_hi = _exact_poisson_count_ci(overall_events, confidence)
        else:
            cnt_lo, cnt_hi = _byar_count_ci(overall_events, confidence)
        o_lo_u = cnt_lo / overall_denom
        o_hi_u = cnt_hi / overall_denom

    overall_notes = []
    if used_end_of_period_denom:
        overall_notes.append(
            "End-of-period denominator: row count used as exposure "
            "(assumes each row = 1 patient present at period end; "
            "does not account for partial-period exposure)"
        )
    if overall_events == 0:
        overall_notes.append("Zero events")
    elif overall_events < 10:
        overall_notes.append("Low event count (<10): DSR should generally not be reported")
    if overall_denom == 0:
        overall_notes.append("Zero denominator")
    overall_notes.append(
        "Overall rate: no weighting applied as this row is itself the reference population"
    )

    overall_rate_scaled = _scale(overall_rate_u)
    records.append({
        **{c: "Overall" for c in group_cols},
        "events": int(overall_events), "denominator": round(overall_denom, 6),
        "dsr": overall_rate_scaled,
        "dsr_lower": _scale(o_lo_u),
        "dsr_upper": _scale(o_hi_u),
        "multiplier": multiplier,
        "notes": " | ".join(overall_notes),
        "significance": "Reference",
    })

    if not records:
        schema = {c: pl.Utf8 for c in group_cols}
        schema.update({"events": pl.Int64, "denominator": pl.Float64,
                        "dsr": pl.Float64, "dsr_lower": pl.Float64,
                        "dsr_upper": pl.Float64, "multiplier": pl.Float64,
                        "notes": pl.Utf8, "significance": pl.Utf8})
        return pl.DataFrame(schema=schema)
    return pl.from_dicts(records)