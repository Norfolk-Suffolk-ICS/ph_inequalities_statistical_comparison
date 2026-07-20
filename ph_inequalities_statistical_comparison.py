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
dataset, and (when group_cols are supplied) test every non-Overall row
against that Overall/reference value, adaptively selecting the appropriate
statistical test and multiple-testing correction based on group sparsity.

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

Significance testing summary
-----------------------------
+---------------------+---------------------------+---------------------------+
| Measure             | Dense-group test          | Sparse-group test         |
+---------------------+---------------------------+---------------------------+
| Crude proportion    | Two-proportion z-test     | Fisher exact              |
| Crude rate          | Poisson z-test (SMR-style)| Mid-P exact Poisson       |
| DSP                 | Dobson z-test vs reference| Exact one-sample binomial |
| DSR                 | Dobson z-test vs reference| Exact one-sample mid-P    |
+---------------------+---------------------------+---------------------------+
Multiple testing correction: Benjamini-Hochberg (FDR) if any group in the
batch is sparse, otherwise a Holm-Sidak step-down correction (an
assumption-light approximation to Dunnett's many-to-one comparison-to-
reference procedure; labelled "Holm-Sidak" in the test_method / notes,
since it is not the exact classical Dunnett test).

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

def _wilson_proportion_ci(events: float, n: float, confidence: float = 0.95) -> tuple[float, float, float, float]:
    """Wilson score CI for a single proportion. Returns (lower, upper, variance, std_dev)."""
    if n == 0:
        return np.nan, np.nan, np.nan, np.nan
    p = events / n
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    denom = 1 + (z ** 2) / n
    center = (p + (z ** 2) / (2 * n)) / denom
    half_width = (z * np.sqrt((p * (1 - p) / n) + (z ** 2) / (4 * n ** 2))) / denom
    lower = max(0.0, center - half_width)
    upper = min(1.0, center + half_width)
    variance = p * (1 - p) / n
    std_dev = float(np.sqrt(variance)) if variance >= 0 else np.nan
    return float(lower), float(upper), float(variance), std_dev


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
) -> tuple[float, float, float, float, float]:
    """
    PHE Wilson-Dobson CI for a directly standardised proportion.
    Returns (lower, upper, scale, variance, std_dev).
    """
    if crude_n <= 0 or not np.isfinite(dsp):
        return np.nan, np.nan, np.nan, np.nan, np.nan

    valid = stratum_ns > 0
    w = stratum_weights[valid]
    p = stratum_props[valid]
    n = stratum_ns[valid]
    w_sum = float(np.sum(stratum_weights))
    if w_sum == 0 or len(w) == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    var_dsp = float(np.sum((w ** 2) * p * (1 - p) / n) / (w_sum ** 2))
    std_dev = float(np.sqrt(var_dsp)) if var_dsp >= 0 else np.nan

    crude_p = crude_events / crude_n
    p_lo, p_hi, var_crude, _ = _wilson_proportion_ci(crude_events, crude_n, confidence)

    if var_crude == 0 or not np.isfinite(var_crude):
        return float(dsp), float(dsp), 0.0, var_dsp, std_dev

    scale = float(np.sqrt(var_dsp / var_crude))
    lower = dsp + scale * (p_lo - crude_p)
    upper = dsp + scale * (p_hi - crude_p)
    return float(max(lower, 0.0)), float(min(upper, 1.0)), scale, var_dsp, std_dev


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
) -> tuple[float, float, float, float, float]:
    """
    PHE Dobson-Byar CI for a directly standardised rate (unscaled).
    Returns (lower, upper, scale, variance, std_dev).
    """
    if crude_events < 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan

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
        return np.nan, np.nan, np.nan, np.nan, np.nan

    var_dsr = float(np.sum((w ** 2) * Oi / (ni ** 2)) / (w_sum ** 2))
    std_dev = float(np.sqrt(var_dsr)) if var_dsr >= 0 else np.nan
    var_O = float(crude_events)

    if var_O == 0.0 or var_dsr == 0.0:
        return float(dsr_unscaled), float(dsr_unscaled), 0.0, var_dsr, std_dev

    scale = float(np.sqrt(var_dsr / var_O))
    lower = dsr_unscaled + scale * (O_lo - crude_events)
    upper = dsr_unscaled + scale * (O_hi - crude_events)
    return float(max(lower, 0.0)), float(max(upper, 0.0)), scale, var_dsr, std_dev


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
# Significance testing vs the overall/population reference
# ===========================================================================

def _select_proportion_test(events: float, n: float) -> str:
    """
    Choose the appropriate test for comparing a group proportion against the
    overall population proportion.

    Fisher's exact test is used whenever either cell count is small
    (events < 10 or non-events < 10), matching the sparsity threshold used
    elsewhere in this module.  Otherwise a two-proportion z-test is used.
    """
    if n == 0 or events > n:
        return "undefined"
    non_events = n - events
    if events < 10 or non_events < 10:
        return "Fisher exact"
    return "Two-proportion z-test"


def _test_proportion_vs_overall(
    events: float, n: float, overall_events: float, overall_n: float
) -> tuple[float, str]:
    """Test a group's proportion against the overall/population proportion."""
    method = _select_proportion_test(events, n)
    if method == "undefined" or overall_n == 0:
        return np.nan, method

    if method == "Fisher exact":
        group_non_events = n - events
        rest_events = max(overall_events - events, 0.0)
        rest_non_events = max((overall_n - overall_events) - group_non_events, 0.0)
        table = [[events, group_non_events], [rest_events, rest_non_events]]
        _, p = stats.fisher_exact(table)
        return float(p), method

    p_group = events / n
    p_pool = overall_events / overall_n
    se = np.sqrt(p_pool * (1.0 - p_pool) * (1.0 / n))
    if se == 0:
        return np.nan, method
    z = (p_group - p_pool) / se
    p = float(2.0 * (1.0 - stats.norm.cdf(abs(z))))
    return p, method


def _select_rate_test(events: float) -> str:
    """
    Choose the appropriate test for comparing a group rate against the
    overall population rate.

    A mid-P exact Poisson test is used for sparse groups (events < 10);
    otherwise a Poisson (SMR-style) z-test is used.
    """
    if events < 10:
        return "Mid-P exact Poisson"
    return "Poisson z-test (test-based)"


def _test_rate_vs_overall(
    events: float, denom: float, overall_events: float, overall_denom: float
) -> tuple[float, str]:
    """
    Test a group's rate against the overall/population rate (SMR-style).
    The overall rate is the fixed reference; expected = overall_rate * denom.
    """
    method = _select_rate_test(events)
    if denom == 0 or overall_denom == 0:
        return np.nan, method

    overall_rate = overall_events / overall_denom
    expected = overall_rate * denom

    if expected == 0:
        return (np.nan if events == 0 else 0.0), method

    if method == "Mid-P exact Poisson":
        lower_tail = stats.poisson.cdf(events, expected) - 0.5 * stats.poisson.pmf(events, expected)
        upper_tail = 1.0 - stats.poisson.cdf(events - 1, expected) - 0.5 * stats.poisson.pmf(events, expected)
        p = 2.0 * min(lower_tail, upper_tail, 0.5)
        return float(np.clip(p, 0.0, 1.0)), method

    z = (events - expected) / np.sqrt(expected)
    p = float(2.0 * (1.0 - stats.norm.cdf(abs(z))))
    return p, method


def _select_standardized_test(events: float, has_unreliable_stratum: bool) -> str:
    """
    Choose the appropriate test for comparing a standardised proportion/rate
    against the overall (reference) population value.

    An exact one-sample test (binomial for proportions, mid-P Poisson for
    rates) is used when the group is sparse or contains an unreliable
    stratum; otherwise a Dobson-variance z-test is used.
    """
    if events < 10 or has_unreliable_stratum:
        return "Exact one-sample (Byar-based)"
    return "Dobson z-test vs reference"


def _test_dsp_vs_overall(
    dsp: float, var_dsp: float, overall_proportion: float,
    crude_events: float, crude_n: float, has_unreliable_stratum: bool,
) -> tuple[float, str]:
    """
    Test a group's DSP against the overall crude proportion.

    Normal-theory path: z = (DSP - overall_proportion) / sqrt(Var(DSP))
    Exact path: binomial test of crude_events out of crude_n against the
                overall_proportion as the null probability.
    """
    method = _select_standardized_test(crude_events, has_unreliable_stratum)
    if not np.isfinite(overall_proportion) or crude_n == 0:
        return np.nan, method

    if method == "Exact one-sample (Byar-based)":
        result = stats.binomtest(int(round(crude_events)), int(round(crude_n)), overall_proportion)
        return float(result.pvalue), method

    if not np.isfinite(var_dsp) or var_dsp <= 0:
        return np.nan, method
    z = (dsp - overall_proportion) / np.sqrt(var_dsp)
    p = float(2.0 * (1.0 - stats.norm.cdf(abs(z))))
    return p, method


def _test_dsr_vs_overall(
    dsr_unscaled: float, var_dsr_unscaled: float, overall_rate_unscaled: float,
    crude_events: float, crude_denom: float, has_unreliable_stratum: bool,
) -> tuple[float, str]:
    """
    Test a group's DSR against the overall crude rate (SMR-style).

    Normal-theory path: z = (DSR - overall_rate) / sqrt(Var(DSR))
    Exact path: mid-P Poisson test of crude_events against the "expected"
                count implied by applying the overall rate to crude_denom.
    """
    method = _select_standardized_test(crude_events, has_unreliable_stratum)
    if not np.isfinite(overall_rate_unscaled) or crude_denom == 0:
        return np.nan, method

    expected = overall_rate_unscaled * crude_denom

    if method == "Exact one-sample (Byar-based)":
        if expected == 0:
            return (np.nan if crude_events == 0 else 0.0), method
        lower_tail = stats.poisson.cdf(crude_events, expected) - 0.5 * stats.poisson.pmf(crude_events, expected)
        upper_tail = 1.0 - stats.poisson.cdf(crude_events - 1, expected) - 0.5 * stats.poisson.pmf(crude_events, expected)
        p = 2.0 * min(lower_tail, upper_tail, 0.5)
        return float(np.clip(p, 0.0, 1.0)), method

    if not np.isfinite(var_dsr_unscaled) or var_dsr_unscaled <= 0:
        return np.nan, method
    z = (dsr_unscaled - overall_rate_unscaled) / np.sqrt(var_dsr_unscaled)
    p = float(2.0 * (1.0 - stats.norm.cdf(abs(z))))
    return p, method


def _select_correction_method(any_sparse: bool) -> str:
    """
    Choose the multiple-testing correction method.

    Dunnett's test is the natural choice for many-to-one comparisons against
    a shared reference, but its normal-theory assumptions do not hold when
    exact tests are mixed in for sparse groups, and exact Dunnett adjustment
    requires the full multivariate-t machinery and joint correlation
    structure rather than a raw p-value vector. A Holm-Sidak step-down
    procedure is used instead as a robust, assumption-light approximation
    to Dunnett-style many-to-one correction. Benjamini-Hochberg (FDR) is
    used as a fallback whenever any group in the batch is sparse.
    """
    return "Benjamini-Hochberg (FDR)" if any_sparse else "Holm-Sidak"


def _apply_multiple_testing_correction(
    p_values: list[float], method: str, alpha: float = 0.05
) -> list[float]:
    """
    Apply a multiple-testing correction to a list of p-values.

    Benjamini-Hochberg (FDR) is applied directly.  Where "Holm-Sidak" is
    selected, this applies the Holm-Sidak step-down procedure -- a robust,
    assumption-light approximation to Dunnett-style many-to-one correction,
    since exact Dunnett adjustment requires the full multivariate-t
    machinery and joint correlation structure rather than a raw p-value
    vector.
    """
    valid_idx = [i for i, p in enumerate(p_values) if np.isfinite(p)]
    if not valid_idx:
        return list(p_values)

    valid_p = np.array([p_values[i] for i in valid_idx])
    m = len(valid_p)

    if method == "Benjamini-Hochberg (FDR)":
        order = np.argsort(valid_p)
        ranked = valid_p[order]
        adjusted = ranked * m / (np.arange(m) + 1)
        adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
        adjusted = np.clip(adjusted, 0, 1)
        out = np.empty(m)
        out[order] = adjusted
    else:
        order = np.argsort(valid_p)
        ranked = valid_p[order]
        adjusted = 1.0 - (1.0 - ranked) ** (m - np.arange(m))
        adjusted = np.maximum.accumulate(adjusted)
        adjusted = np.clip(adjusted, 0, 1)
        out = np.empty(m)
        out[order] = adjusted

    result = list(p_values)
    for idx, val in zip(valid_idx, out):
        result[idx] = float(val)
    return result


def _significance_label(p_adjusted: float, estimate: float, reference: float, alpha: float = 0.05) -> str:
    """
    Classify a group's estimate relative to the overall/reference value.

    Returns one of: "Higher", "Lower", "Not significant", "Not tested".
    """
    if not np.isfinite(p_adjusted):
        return "Not tested"
    if p_adjusted >= alpha:
        return "Not significant"
    if not np.isfinite(estimate) or not np.isfinite(reference):
        return "Not tested"
    return "Higher" if estimate > reference else "Lower"


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
        Columns: [*group_cols, events, n, proportion, lower, upper, variance,
                  std_dev, confidence, method, notes, test_method, p_value,
                  p_adjusted, significance]
        An additional "Overall" row is appended (group_cols set to "Overall")
        summarising the crude proportion across the full input dataset.
        Each non-Overall row is tested against the Overall proportion using
        a Fisher exact test (sparse groups) or a two-proportion z-test
        (otherwise), with p-values corrected for multiple testing across
        all groups using Benjamini-Hochberg (if any group is sparse) or a
        Holm-Sidak step-down correction (an approximation to Dunnett's
        many-to-one comparison-to-reference procedure).
        The Overall row itself is never tested (it is the reference).
    """
    group_cols = list(group_cols or [])
    _check_columns(df, [event_col] + group_cols)
    _check_no_nulls(df, [event_col] + group_cols)
    df = _validate_numerator_col(df, event_col, binary=True)

    if not group_cols:
        events = float(df[event_col].sum())
        n = float(len(df))
        lo, hi, variance, std_dev = _wilson_proportion_ci(events, n, confidence)
        proportion = np.nan if n == 0 else events / n
        return pl.DataFrame([{
            "events": events, "n": n,
            "proportion": float(proportion) if np.isfinite(proportion) else np.nan,
            "lower": lo, "upper": hi,
            "variance": variance, "std_dev": std_dev,
            "confidence": confidence, "method": "Wilson score",
            "notes": _crude_proportion_notes(events, n),
            "test_method": "Not tested (no groups)", "p_value": np.nan,
            "p_adjusted": np.nan, "significance": "Not tested",
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
    raw_p_values = []
    any_sparse = False
    for row in grouped.iter_rows(named=True):
        e, n = float(row["events"]), float(row["n"])
        lo, hi, variance, std_dev = _wilson_proportion_ci(e, n, confidence)
        proportion = np.nan if n == 0 else e / n
        p_value, test_method = _test_proportion_vs_overall(e, n, overall_e, overall_n)
        if test_method == "Fisher exact":
            any_sparse = True
        raw_p_values.append(p_value)
        records.append({
            **{c: row[c] for c in group_cols},
            "events": e, "n": n,
            "proportion": float(proportion) if np.isfinite(proportion) else np.nan,
            "lower": lo, "upper": hi,
            "variance": variance, "std_dev": std_dev,
            "confidence": confidence, "method": "Wilson score",
            "notes": _crude_proportion_notes(e, n),
            "test_method": test_method, "p_value": p_value,
        })

    correction_method = _select_correction_method(any_sparse)
    p_adjusted_list = _apply_multiple_testing_correction(raw_p_values, correction_method)
    for rec, p_adj in zip(records, p_adjusted_list):
        rec["p_adjusted"] = p_adj
        rec["significance"] = _significance_label(p_adj, rec["proportion"], overall_prop)

    o_lo, o_hi, o_var, o_sd = _wilson_proportion_ci(overall_e, overall_n, confidence)
    records.append({
        **{c: "Overall" for c in group_cols},
        "events": overall_e, "n": overall_n,
        "proportion": float(overall_prop) if np.isfinite(overall_prop) else np.nan,
        "lower": o_lo, "upper": o_hi,
        "variance": o_var, "std_dev": o_sd,
        "confidence": confidence, "method": "Wilson score",
        "notes": _crude_proportion_notes(overall_e, overall_n),
        "test_method": f"Reference row (correction: {correction_method})",
        "p_value": np.nan, "p_adjusted": np.nan, "significance": "Reference",
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
                  variance, std_dev, multiplier, confidence, method, notes,
                  test_method, p_value, p_adjusted, significance]
        An additional "Overall" row is appended (group_cols set to "Overall")
        summarising the crude rate across the full input dataset.
        Each non-Overall row is tested against the Overall rate using a
        mid-P exact Poisson test (sparse groups) or a Poisson (SMR-style)
        z-test (otherwise), with p-values corrected for multiple testing
        using Benjamini-Hochberg (if any group is sparse) or a Holm-Sidak
        step-down correction (a Dunnett approximation).  The Overall row
        itself is never tested (it is the reference).
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
            return {"rate": np.nan, "lower": np.nan, "upper": np.nan,
                    "variance": np.nan, "std_dev": np.nan, "method": "undefined"}
        if e < 10:
            cnt_lo, cnt_hi = _exact_poisson_count_ci(e, confidence)
            method = "Exact chi-square"
        else:
            cnt_lo, cnt_hi = _byar_count_ci(e, confidence)
            method = "Byar"
        variance = float(e / (d ** 2)) * (multiplier ** 2)
        std_dev = float(np.sqrt(variance))
        return {
            "rate": (e / d) * multiplier,
            "lower": (cnt_lo / d) * multiplier,
            "upper": (cnt_hi / d) * multiplier,
            "variance": variance,
            "std_dev": std_dev,
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
            "test_method": "Not tested (no groups)", "p_value": np.nan,
            "p_adjusted": np.nan, "significance": "Not tested",
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
    raw_p_values = []
    any_sparse = False
    for row in grouped.iter_rows(named=True):
        e, d = float(row["events"]), float(row["denominator"])
        r = _compute_row(e, d)
        p_value, test_method = _test_rate_vs_overall(e, d, overall_e, overall_d)
        if test_method == "Mid-P exact Poisson":
            any_sparse = True
        raw_p_values.append(p_value)
        records.append({
            **{c: row[c] for c in group_cols},
            "events": e, "denominator": d,
            **r, "multiplier": multiplier, "confidence": confidence,
            "notes": _crude_rate_notes(e, d, used_end_of_period_denom),
            "test_method": test_method, "p_value": p_value,
        })

    correction_method = _select_correction_method(any_sparse)
    p_adjusted_list = _apply_multiple_testing_correction(raw_p_values, correction_method)
    for rec, p_adj in zip(records, p_adjusted_list):
        rec["p_adjusted"] = p_adj
        rec["significance"] = _significance_label(p_adj, rec["rate"], overall_rate)

    r_overall = _compute_row(overall_e, overall_d)
    records.append({
        **{c: "Overall" for c in group_cols},
        "events": overall_e, "denominator": overall_d,
        **r_overall, "multiplier": multiplier, "confidence": confidence,
        "notes": _crude_rate_notes(overall_e, overall_d, used_end_of_period_denom),
        "test_method": f"Reference row (correction: {correction_method})",
        "p_value": np.nan, "p_adjusted": np.nan, "significance": "Reference",
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
        Columns: [*group_cols, events, n, dsp, dsp_lower, dsp_upper, variance,
                  std_dev, notes, test_method, p_value, p_adjusted, significance]
        An additional "Overall" row is appended (group_cols set to "Overall")
        giving the crude (unstandardised) proportion across the full input
        dataset, with its own Wilson-Dobson CI. Because this row IS the
        reference population used to build the standardisation weights, it
        requires no weighting -- this is stated explicitly in its notes.
        Each non-Overall row is tested against the Overall proportion using
        an exact one-sample binomial test (sparse groups or unreliable
        strata) or a Dobson-variance z-test (otherwise), with p-values
        corrected for multiple testing using Benjamini-Hochberg (if any
        group is sparse) or a Holm-Sidak step-down correction (a Dunnett
        approximation) otherwise.  The Overall row itself is never tested.

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
    raw_p_values = []
    any_sparse = False
    for row in group_combinations.iter_rows(named=True):
        group_data = _filter_group(work_df, row)
        crude_events = float(group_data[event_col].sum())
        crude_n = float(len(group_data))

        s = _compute_proportion_stratum_stats(
            group_data, all_strata, event_col, strata_cols, ref_weights
        )
        dsp = float(np.sum(s["stratum_weights"] * s["stratum_props"]))
        dsp_lower, dsp_upper, _, var_dsp, sd_dsp = _wilson_dobson_proportion_ci(
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

        p_value, test_method = _test_dsp_vs_overall(
            dsp, var_dsp, overall_proportion, crude_events, crude_n,
            has_unreliable_stratum or has_unreliable_non_events,
        )
        if test_method == "Exact one-sample (Byar-based)":
            any_sparse = True
        raw_p_values.append(p_value)

        records.append({
            **row,
            "events": int(crude_events), "n": int(crude_n),
            "dsp": round(dsp, 6),
            "dsp_lower": round(dsp_lower, 6),
            "dsp_upper": round(dsp_upper, 6),
            "variance": round(var_dsp, 10) if np.isfinite(var_dsp) else np.nan,
            "std_dev": round(sd_dsp, 10) if np.isfinite(sd_dsp) else np.nan,
            "notes": " | ".join(notes),
            "test_method": test_method, "p_value": p_value,
        })

    correction_method = _select_correction_method(any_sparse)
    p_adjusted_list = _apply_multiple_testing_correction(raw_p_values, correction_method)
    for rec, p_adj in zip(records, p_adjusted_list):
        rec["p_adjusted"] = p_adj
        rec["significance"] = _significance_label(p_adj, rec["dsp"], overall_proportion)

    o_lo, o_hi, _, o_var, o_sd = _wilson_dobson_proportion_ci(
        overall_proportion,
        overall_events, overall_n,
        np.array([1.0]), np.array([overall_proportion]),
        np.array([overall_n]), confidence,
    ) if overall_n > 0 else (np.nan, np.nan, np.nan, np.nan, np.nan)
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
        "variance": round(o_var, 10) if np.isfinite(o_var) else np.nan,
        "std_dev": round(o_sd, 10) if np.isfinite(o_sd) else np.nan,
        "notes": " | ".join(overall_notes),
        "test_method": f"Reference row (correction: {correction_method})",
        "p_value": np.nan, "p_adjusted": np.nan, "significance": "Reference",
    })

    if not records:
        schema = {c: pl.Utf8 for c in group_cols}
        schema.update({"events": pl.Int64, "n": pl.Int64,
                        "dsp": pl.Float64, "dsp_lower": pl.Float64,
                        "dsp_upper": pl.Float64, "variance": pl.Float64,
                        "std_dev": pl.Float64, "notes": pl.Utf8,
                        "test_method": pl.Utf8, "p_value": pl.Float64,
                        "p_adjusted": pl.Float64, "significance": pl.Utf8})
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
                  variance, std_dev, multiplier, notes, test_method, p_value,
                  p_adjusted, significance]
        An additional "Overall" row is appended (group_cols set to "Overall")
        giving the crude (unstandardised) rate across the full input dataset,
        with its own Dobson-Byar/exact CI. Because this row IS the reference
        population used to build the standardisation weights, it requires no
        weighting -- this is stated explicitly in its notes.
        Each non-Overall row is tested against the Overall rate using an
        exact one-sample mid-P Poisson test (sparse groups or unreliable
        strata) or a Dobson-variance z-test (otherwise, SMR-style), with
        p-values corrected for multiple testing using Benjamini-Hochberg (if
        any group is sparse) or a Holm-Sidak step-down correction (a Dunnett
        approximation) otherwise.  The Overall row itself is never tested.

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
    raw_p_values = []
    any_sparse = False
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
        lo_u, hi_u, _, var_dsr_u, sd_dsr_u = _dobson_byar_rate_ci(
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

        p_value, test_method = _test_dsr_vs_overall(
            dsr_u, var_dsr_u, overall_rate_u, crude_events, crude_denom,
            has_unreliable_stratum,
        )
        if test_method == "Exact one-sample (Byar-based)":
            any_sparse = True
        raw_p_values.append(p_value)

        var_dsr_scaled = float(var_dsr_u * (multiplier ** 2)) if np.isfinite(var_dsr_u) else np.nan
        sd_dsr_scaled = float(np.sqrt(var_dsr_scaled)) if np.isfinite(var_dsr_scaled) else np.nan
        records.append({
            **row,
            "events": int(crude_events), "denominator": round(crude_denom, 6),
            "dsr": _scale(dsr_u),
            "dsr_lower": _scale(lo_u),
            "dsr_upper": _scale(hi_u),
            "variance": round(var_dsr_scaled, 6) if np.isfinite(var_dsr_scaled) else np.nan,
            "std_dev": round(sd_dsr_scaled, 6) if np.isfinite(sd_dsr_scaled) else np.nan,
            "multiplier": multiplier,
            "notes": " | ".join(notes),
            "test_method": test_method, "p_value": p_value,
        })

    correction_method = _select_correction_method(any_sparse)
    p_adjusted_list = _apply_multiple_testing_correction(raw_p_values, correction_method)
    overall_rate_scaled = _scale(overall_rate_u)
    for rec, p_adj in zip(records, p_adjusted_list):
        rec["p_adjusted"] = p_adj
        rec["significance"] = _significance_label(p_adj, rec["dsr"], overall_rate_scaled)

    if overall_denom == 0:
        o_lo_u, o_hi_u, o_var_u = np.nan, np.nan, np.nan
    else:
        if overall_events < 10:
            cnt_lo, cnt_hi = _exact_poisson_count_ci(overall_events, confidence)
        else:
            cnt_lo, cnt_hi = _byar_count_ci(overall_events, confidence)
        o_lo_u = cnt_lo / overall_denom
        o_hi_u = cnt_hi / overall_denom
        o_var_u = overall_events / (overall_denom ** 2)

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

    var_overall_scaled = float(o_var_u * (multiplier ** 2)) if np.isfinite(o_var_u) else np.nan
    sd_overall_scaled = float(np.sqrt(var_overall_scaled)) if np.isfinite(var_overall_scaled) else np.nan

    records.append({
        **{c: "Overall" for c in group_cols},
        "events": int(overall_events), "denominator": round(overall_denom, 6),
        "dsr": overall_rate_scaled,
        "dsr_lower": _scale(o_lo_u),
        "dsr_upper": _scale(o_hi_u),
        "variance": round(var_overall_scaled, 6) if np.isfinite(var_overall_scaled) else np.nan,
        "std_dev": round(sd_overall_scaled, 6) if np.isfinite(sd_overall_scaled) else np.nan,
        "multiplier": multiplier,
        "notes": " | ".join(overall_notes),
        "test_method": f"Reference row (correction: {correction_method})",
        "p_value": np.nan, "p_adjusted": np.nan, "significance": "Reference",
    })

    if not records:
        schema = {c: pl.Utf8 for c in group_cols}
        schema.update({"events": pl.Int64, "denominator": pl.Float64,
                        "dsr": pl.Float64, "dsr_lower": pl.Float64,
                        "dsr_upper": pl.Float64, "variance": pl.Float64,
                        "std_dev": pl.Float64, "multiplier": pl.Float64,
                        "notes": pl.Utf8, "test_method": pl.Utf8,
                        "p_value": pl.Float64, "p_adjusted": pl.Float64,
                        "significance": pl.Utf8})
        return pl.DataFrame(schema=schema)
    return pl.from_dicts(records)