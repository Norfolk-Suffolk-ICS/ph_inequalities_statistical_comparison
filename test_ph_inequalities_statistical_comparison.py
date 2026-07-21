"""
test_ph_inequalities_statistical_comparison.py
===========================
Pytest suite for ph_inequalities_statistical_comparison.py
"""

import numpy as np
import polars as pl
import pytest

from ph_inequalities_statistical_comparison import (
    _bin_numeric_to_quartiles,
    _wilson_proportion_ci,
    _byar_count_ci,
    _exact_poisson_count_ci,
    _haldane_proportion_correction,
    _haldane_rate_correction,
    _wilson_dobson_proportion_ci,
    _dobson_byar_rate_ci,
    _build_reference_weights,
    _prepare_dataframe_proportion,
    _prepare_dataframe_rate,
    _compute_proportion_stratum_stats,
    _compute_rate_stratum_stats,
    _significance_from_ci,
    _check_no_nulls,
    _validate_numerator_col,
    crude_proportion_df,
    crude_rate_df,
    directly_standardized_proportion_df,
    directly_standardized_rate_df,
)


@pytest.fixture
def simple_prop_df():
    np.random.seed(42)
    n = 400
    return pl.DataFrame({
        "region": np.random.choice(["North", "South", "East", "West"], n).tolist(),
        "sex": np.random.choice(["M", "F"], n).tolist(),
        "age": np.random.randint(20, 80, n).tolist(),
        "event": np.random.binomial(1, 0.4, n).tolist(),
    })


@pytest.fixture
def simple_rate_df():
    np.random.seed(42)
    n = 400
    return pl.DataFrame({
        "region": np.random.choice(["North", "South", "East", "West"], n).tolist(),
        "sex": np.random.choice(["M", "F"], n).tolist(),
        "age": np.random.randint(20, 80, n).tolist(),
        "events": np.random.poisson(0.8, n).tolist(),
    })


@pytest.fixture
def sparse_prop_df():
    np.random.seed(7)
    n = 60
    return pl.DataFrame({
        "icb": np.random.choice(["A", "B"], n).tolist(),
        "band": np.random.choice(["1", "2", "3"], n).tolist(),
        "event": np.random.binomial(1, 0.05, n).tolist(),
    })


@pytest.fixture
def sparse_rate_df():
    np.random.seed(7)
    n = 60
    return pl.DataFrame({
        "icb": np.random.choice(["A", "B"], n).tolist(),
        "band": np.random.choice(["1", "2", "3"], n).tolist(),
        "events": np.random.poisson(0.05, n).tolist(),
    })


# ============================================================
# Low-level statistical helpers
# ============================================================

class TestBinNumericToQuartiles:
    def test_returns_four_labels(self):
        s = pl.Series("x", list(range(100)))
        out = _bin_numeric_to_quartiles(s)
        assert set(out.to_list()) == {"Q1", "Q2", "Q3", "Q4"}

    def test_none_preserved(self):
        s = pl.Series("x", [1.0, 2.0, None, 4.0])
        out = _bin_numeric_to_quartiles(s)
        assert out.to_list()[2] is None


class TestWilsonProportionCI:
    def test_bounds_ordered(self):
        lo, hi, var = _wilson_proportion_ci(50, 100)
        assert 0 <= lo <= hi <= 1

    def test_zero_n_returns_nan(self):
        lo, hi, var = _wilson_proportion_ci(0, 0)
        assert np.isnan(lo)

    def test_wider_with_higher_confidence(self):
        lo95, hi95, _ = _wilson_proportion_ci(50, 100, 0.95)
        lo99, hi99, _ = _wilson_proportion_ci(50, 100, 0.99)
        assert (hi99 - lo99) > (hi95 - lo95)


class TestByarCountCI:
    def test_zero_count(self):
        lo, hi = _byar_count_ci(0)
        assert lo == 0
        assert hi > 0

    def test_bounds_ordered(self):
        lo, hi = _byar_count_ci(20)
        assert 0 <= lo <= hi


class TestExactPoissonCountCI:
    def test_zero_count_lower_zero(self):
        lo, hi = _exact_poisson_count_ci(0)
        assert lo == 0
        assert hi > 0

    def test_bounds_ordered(self):
        lo, hi = _exact_poisson_count_ci(5)
        assert 0 <= lo <= hi


class TestHaldaneProportionCorrection:
    def test_zero_events(self):
        p, e, n = _haldane_proportion_correction(0, 100)
        assert p > 0
        assert e == 0.5
        assert n == 101.0

    def test_all_events(self):
        p, e, n = _haldane_proportion_correction(100, 100)
        assert p < 1.0


class TestHaldaneRateCorrection:
    def test_zero_events_positive_rate(self):
        r, e, d = _haldane_rate_correction(0, 100)
        assert r > 0
        assert e == 0.5
        assert d == 101.0


class TestWilsonDobsonProportionCI:
    def test_bounds_ordered(self):
        w = np.array([0.5, 0.5])
        p = np.array([0.3, 0.5])
        n = np.array([50.0, 50.0])
        dsp = float(np.sum(w * p))
        lo, hi, var = _wilson_dobson_proportion_ci(dsp, 40, 100, w, p, n)
        assert 0 <= lo <= hi <= 1


class TestDobsonByarRateCI:
    def test_bounds_ordered(self):
        w = np.array([0.5, 0.5])
        Oi = np.array([10.0, 15.0])
        ni = np.array([1000.0, 800.0])
        dsr = np.sum(w * (Oi / ni)) / np.sum(w)
        lo, hi, var = _dobson_byar_rate_ci(dsr, 25.0, w, Oi, ni)
        assert 0 <= lo <= hi


class TestReferenceWeights:
    def test_sum_to_one(self, simple_prop_df):
        work = _prepare_dataframe_proportion(simple_prop_df, "event", ["sex"], ["region"])
        ref = _build_reference_weights(work, ["sex"])
        assert abs(ref["ref_weight"].sum() - 1.0) < 1e-9


class TestPrepareDataFrame:
    def test_missing_column_raises_prop(self, simple_prop_df):
        with pytest.raises(ValueError):
            _prepare_dataframe_proportion(simple_prop_df, "event", ["missing"], ["region"])

    def test_numeric_strata_binned_prop(self, simple_prop_df):
        out = _prepare_dataframe_proportion(simple_prop_df, "event", ["age"], ["region"])
        assert out["age"].dtype == pl.Utf8

    def test_missing_column_raises_rate(self, simple_rate_df):
        with pytest.raises(ValueError):
            _prepare_dataframe_rate(simple_rate_df, "missing", ["sex"], ["region"])


# ============================================================
# Significance-from-CI helper
# ============================================================

class TestSignificanceFromCi:
    def test_not_significant_when_reference_inside_ci(self):
        assert _significance_from_ci(0.5, 0.4, 0.6) == "Not significant"

    def test_higher_when_reference_below_lower(self):
        assert _significance_from_ci(0.3, 0.4, 0.6) == "Higher"

    def test_lower_when_reference_above_upper(self):
        assert _significance_from_ci(0.7, 0.4, 0.6) == "Lower"

    def test_not_tested_when_reference_nan(self):
        assert _significance_from_ci(np.nan, 0.4, 0.6) == "Not tested"

    def test_not_tested_when_bounds_nan(self):
        assert _significance_from_ci(0.5, np.nan, np.nan) == "Not tested"

    def test_boundary_reference_equals_lower_not_significant(self):
        assert _significance_from_ci(0.4, 0.4, 0.6) == "Not significant"

    def test_boundary_reference_equals_upper_not_significant(self):
        assert _significance_from_ci(0.6, 0.4, 0.6) == "Not significant"


# ============================================================
# crude_proportion_df
# ============================================================

class TestCrudeProportionDf:
    def test_output_columns(self, simple_prop_df):
        out = crude_proportion_df(simple_prop_df, "event", ["region"])
        assert set(out.columns) == {
            "region", "events", "n", "proportion", "lower", "upper",
            "confidence", "method", "notes", "significance"
        }

    def test_one_row_per_group_plus_overall(self, simple_prop_df):
        out = crude_proportion_df(simple_prop_df, "event", ["region"])
        assert out.shape[0] == 5  # 4 regions + Overall

    def test_bounds_valid(self, simple_prop_df):
        out = crude_proportion_df(simple_prop_df, "event", ["region"])
        assert (out["lower"] <= out["proportion"]).all()
        assert (out["proportion"] <= out["upper"]).all()

    def test_no_group_cols_single_row(self, simple_prop_df):
        out = crude_proportion_df(simple_prop_df, "event")
        assert out.shape[0] == 1
        assert out["significance"][0] == "Not tested"

    def test_missing_column_raises(self, simple_prop_df):
        with pytest.raises(ValueError):
            crude_proportion_df(simple_prop_df, "missing", ["region"])

    def test_boolean_event_col(self):
        df = pl.DataFrame({
            "grp": ["A"] * 20 + ["B"] * 20,
            "event": [True] * 10 + [False] * 10 + [True] * 5 + [False] * 15,
        })
        out = crude_proportion_df(df, "event", ["grp"])
        assert out.shape[0] == 3


class TestCrudeProportionDfSignificance:
    def test_columns_present(self, simple_prop_df):
        out = crude_proportion_df(simple_prop_df, "event", ["region"])
        assert "significance" in out.columns

    def test_overall_row_is_reference(self, simple_prop_df):
        out = crude_proportion_df(simple_prop_df, "event", ["region"])
        overall = out.filter(pl.col("region") == "Overall")
        assert overall["significance"][0] == "Reference"

    def test_extreme_group_flagged_significant(self):
        df = pl.DataFrame({
            "grp": ["A"] * 100 + ["B"] * 100,
            "event": [1] * 95 + [0] * 5 + [1] * 5 + [0] * 95,
        })
        out = crude_proportion_df(df, "event", ["grp"])
        group_a = out.filter(pl.col("grp") == "A")
        assert group_a["significance"][0] == "Higher"

    def test_similar_group_not_significant(self):
        np.random.seed(1)
        df = pl.DataFrame({
            "grp": ["A"] * 500 + ["B"] * 500,
            "event": np.random.binomial(1, 0.5, 1000).tolist(),
        })
        out = crude_proportion_df(df, "event", ["grp"])
        assert set(out["significance"].to_list()) <= {"Not significant", "Reference"}

    def test_no_group_cols_has_not_tested(self, simple_prop_df):
        out = crude_proportion_df(simple_prop_df, "event")
        assert out["significance"][0] == "Not tested"

    def test_reference_outside_ci_flagged_for_sparse_group(self, sparse_prop_df):
        out = crude_proportion_df(sparse_prop_df, "event", ["icb"])
        non_overall = out.filter(pl.col("icb") != "Overall")
        assert set(non_overall["significance"].to_list()) <= {
            "Higher", "Lower", "Not significant", "Not tested"
        }


# ============================================================
# crude_rate_df
# ============================================================

class TestCrudeRateDf:
    def test_output_columns(self, simple_rate_df):
        out = crude_rate_df(simple_rate_df, "events", ["region"])
        assert set(out.columns) == {
            "region", "events", "denominator", "rate", "lower", "upper",
            "multiplier", "confidence", "method", "notes", "significance"
        }

    def test_one_row_per_group_plus_overall(self, simple_rate_df):
        out = crude_rate_df(simple_rate_df, "events", ["region"])
        assert out.shape[0] == 5

    def test_bounds_valid(self, simple_rate_df):
        out = crude_rate_df(simple_rate_df, "events", ["region"])
        assert (out["lower"] <= out["rate"]).all()
        assert (out["rate"] <= out["upper"]).all()

    def test_multiplier_changes_scale(self, simple_rate_df):
        r1 = crude_rate_df(simple_rate_df, "events", ["region"], multiplier=1)
        r100k = crude_rate_df(simple_rate_df, "events", ["region"], multiplier=100000)
        ratio = (r100k["rate"] / r1["rate"]).mean()
        assert abs(ratio - 100000) < 5

    def test_missing_column_raises(self, simple_rate_df):
        with pytest.raises(ValueError):
            crude_rate_df(simple_rate_df, "missing", ["region"])


class TestCrudeRateDfSignificance:
    def test_columns_present(self, simple_rate_df):
        out = crude_rate_df(simple_rate_df, "events", ["region"])
        assert "significance" in out.columns

    def test_overall_row_is_reference(self, simple_rate_df):
        out = crude_rate_df(simple_rate_df, "events", ["region"])
        overall = out.filter(pl.col("region") == "Overall")
        assert overall["significance"][0] == "Reference"

    def test_extreme_group_flagged_significant(self):
        df = pl.DataFrame({
            "grp": ["A"] * 50 + ["B"] * 50,
            "events": [50] * 50 + [1] * 50,
        })
        out = crude_rate_df(df, "events", ["grp"])
        group_a = out.filter(pl.col("grp") == "A")
        assert group_a["significance"][0] == "Higher"

    def test_sparse_group_significance_within_valid_labels(self, sparse_rate_df):
        out = crude_rate_df(sparse_rate_df, "events", ["icb"])
        non_overall = out.filter(pl.col("icb") != "Overall")
        assert set(non_overall["significance"].to_list()) <= {
            "Higher", "Lower", "Not significant", "Not tested"
        }


# ============================================================
# directly_standardized_proportion_df
# ============================================================

class TestDirectlyStandardizedProportion:
    def test_output_columns(self, simple_prop_df):
        out = directly_standardized_proportion_df(
            simple_prop_df, "event", ["age", "sex"], ["region"]
        )
        assert set(out.columns) == {
            "region", "events", "n", "dsp", "dsp_lower", "dsp_upper",
            "notes", "significance"
        }

    def test_row_count_includes_overall(self, simple_prop_df):
        out = directly_standardized_proportion_df(
            simple_prop_df, "event", ["age", "sex"], ["region"]
        )
        assert out.shape[0] == 5

    def test_bounds_valid(self, simple_prop_df):
        out = directly_standardized_proportion_df(
            simple_prop_df, "event", ["age", "sex"], ["region"]
        )
        assert (out["dsp_lower"] <= out["dsp"]).all()
        assert (out["dsp"] <= out["dsp_upper"]).all()

    def test_note_haldane_applied(self):
        df = pl.DataFrame({
            "grp": ["A"] * 20,
            "strat": ["X"] * 10 + ["Y"] * 10,
            "event": [0] * 10 + [1] * 10,
        })
        out = directly_standardized_proportion_df(df, "event", ["strat"], ["grp"])
        row = out.filter(pl.col("grp") == "A")
        assert "Haldane correction applied" in row["notes"][0]

    def test_missing_column_raises(self, simple_prop_df):
        with pytest.raises(ValueError):
            directly_standardized_proportion_df(
                simple_prop_df, "event", ["missing"], ["region"]
            )


class TestDirectlyStandardizedProportionSignificance:
    def test_columns_present(self, simple_prop_df):
        out = directly_standardized_proportion_df(
            simple_prop_df, "event", ["age", "sex"], ["region"]
        )
        assert "significance" in out.columns

    def test_overall_row_is_reference_and_unweighted_note(self, simple_prop_df):
        out = directly_standardized_proportion_df(
            simple_prop_df, "event", ["age", "sex"], ["region"]
        )
        overall = out.filter(pl.col("region") == "Overall")
        assert overall["significance"][0] == "Reference"
        assert "no weighting applied" in overall["notes"][0]

    def test_non_overall_rows_have_significance_label(self, simple_prop_df):
        out = directly_standardized_proportion_df(
            simple_prop_df, "event", ["age", "sex"], ["region"]
        )
        non_overall = out.filter(pl.col("region") != "Overall")
        assert non_overall["significance"].null_count() == 0
        assert all(m != "" for m in non_overall["significance"].to_list())

    def test_sparse_group_significance_within_valid_labels(self, sparse_prop_df):
        out = directly_standardized_proportion_df(
            sparse_prop_df, "event", ["band"], ["icb"]
        )
        non_overall = out.filter(pl.col("icb") != "Overall")
        assert set(non_overall["significance"].to_list()) <= {
            "Higher", "Lower", "Not significant", "Not tested"
        }


# ============================================================
# directly_standardized_rate_df
# ============================================================

class TestDirectlyStandardizedRate:
    def test_output_columns(self, simple_rate_df):
        out = directly_standardized_rate_df(simple_rate_df, "events", ["sex", "age"], ["region"])
        assert set(out.columns) == {
            "region", "events", "denominator", "dsr", "dsr_lower", "dsr_upper",
            "multiplier", "notes", "significance"
        }

    def test_row_count_includes_overall(self, simple_rate_df):
        out = directly_standardized_rate_df(simple_rate_df, "events", ["sex", "age"], ["region"])
        assert out.shape[0] == 5

    def test_bounds_valid(self, simple_rate_df):
        out = directly_standardized_rate_df(simple_rate_df, "events", ["sex", "age"], ["region"])
        non_null = out.filter(pl.col("dsr").is_not_null())
        assert (non_null["dsr_lower"] <= non_null["dsr"]).all()
        assert (non_null["dsr"] <= non_null["dsr_upper"]).all()

    def test_denominator_equals_row_count(self):
        df = pl.DataFrame({
            "grp": ["A"] * 40,
            "strat": ["X"] * 20 + ["Y"] * 20,
            "events": [1] * 20 + [2] * 20,
        })
        out = directly_standardized_rate_df(df, "events", ["strat"], ["grp"])
        row = out.filter(pl.col("grp") == "A")
        assert row["denominator"][0] == 40.0

    def test_note_end_of_period_denominator(self):
        df = pl.DataFrame({
            "grp": ["A"] * 40,
            "strat": ["X"] * 20 + ["Y"] * 20,
            "events": [1] * 20 + [2] * 20,
        })
        out = directly_standardized_rate_df(df, "events", ["strat"], ["grp"])
        row = out.filter(pl.col("grp") == "A")
        assert "End-of-period denominator" in row["notes"][0]

    def test_note_zero_events(self):
        df = pl.DataFrame({
            "grp": ["A"] * 40,
            "strat": ["X"] * 20 + ["Y"] * 20,
            "events": [0] * 40,
        })
        out = directly_standardized_rate_df(df, "events", ["strat"], ["grp"])
        row = out.filter(pl.col("grp") == "A")
        assert "Zero events" in row["notes"][0]

    def test_note_haldane_applied(self):
        df = pl.DataFrame({
            "grp": ["A"] * 40,
            "strat": ["X"] * 20 + ["Y"] * 20,
            "events": [0] * 20 + [5] * 20,
        })
        out = directly_standardized_rate_df(df, "events", ["strat"], ["grp"])
        row = out.filter(pl.col("grp") == "A")
        assert "Haldane correction applied" in row["notes"][0]

    def test_missing_column_raises(self, simple_rate_df):
        with pytest.raises(ValueError):
            directly_standardized_rate_df(
                simple_rate_df, "missing", ["sex"], ["region"]
            )

    def test_extra_positional_arg_raises_typeerror(self, simple_rate_df):
        with pytest.raises(TypeError):
            directly_standardized_rate_df(
                simple_rate_df, "events", ["sex"], ["region"], "some_denom_col", 100_000.0, 0.95, "extra"
            )


class TestDirectlyStandardizedRateSignificance:
    def test_columns_present(self, simple_rate_df):
        out = directly_standardized_rate_df(simple_rate_df, "events", ["sex", "age"], ["region"])
        assert "significance" in out.columns

    def test_overall_row_is_reference_and_unweighted_note(self, simple_rate_df):
        out = directly_standardized_rate_df(simple_rate_df, "events", ["sex", "age"], ["region"])
        overall = out.filter(pl.col("region") == "Overall")
        assert overall["significance"][0] == "Reference"
        assert "no weighting applied" in overall["notes"][0]

    def test_overall_row_flags_end_of_period(self, simple_rate_df):
        out = directly_standardized_rate_df(simple_rate_df, "events", ["sex", "age"], ["region"])
        overall = out.filter(pl.col("region") == "Overall")
        assert "End-of-period denominator" in overall["notes"][0]

    def test_non_overall_rows_have_significance_label(self, simple_rate_df):
        out = directly_standardized_rate_df(simple_rate_df, "events", ["sex", "age"], ["region"])
        non_overall = out.filter(pl.col("region") != "Overall")
        assert non_overall["significance"].null_count() == 0
        assert all(m != "" for m in non_overall["significance"].to_list())

    def test_sparse_group_significance_within_valid_labels(self, sparse_rate_df):
        out = directly_standardized_rate_df(sparse_rate_df, "events", ["band"], ["icb"])
        non_overall = out.filter(pl.col("icb") != "Overall")
        assert set(non_overall["significance"].to_list()) <= {
            "Higher", "Lower", "Not significant", "Not tested"
        }


# ============================================================
# crude_rate_df: row-count denominator behaviour
# ============================================================

class TestCrudeRateDfRowCountDenominator:
    def test_denominator_equals_row_count(self):
        df = pl.DataFrame({
            "grp": ["A"] * 20 + ["B"] * 20,
            "events": [1] * 5 + [0] * 15 + [1] * 2 + [0] * 18,
        })
        out = crude_rate_df(df, "events", ["grp"])
        row_a = out.filter(pl.col("grp") == "A")
        assert row_a["denominator"][0] == 20.0

    def test_note_end_of_period_denominator(self):
        df = pl.DataFrame({
            "grp": ["A"] * 20,
            "events": [1] * 5 + [0] * 15,
        })
        out = crude_rate_df(df, "events", ["grp"])
        row_a = out.filter(pl.col("grp") == "A")
        assert "End-of-period denominator" in row_a["notes"][0]

    def test_overall_row_flagged_too(self):
        df = pl.DataFrame({
            "grp": ["A"] * 20,
            "events": [1] * 5 + [0] * 15,
        })
        out = crude_rate_df(df, "events", ["grp"])
        overall = out.filter(pl.col("grp") == "Overall")
        assert "End-of-period denominator" in overall["notes"][0]

    def test_extra_positional_arg_raises_typeerror(self, simple_rate_df):
        with pytest.raises(TypeError):
            crude_rate_df(simple_rate_df, "events", ["region"], 100_000.0, 0.95, "extra")

    def test_manual_person_time_replication_matches(self):
        df_rowlevel = pl.DataFrame({
            "grp": ["A"] * 20 + ["B"] * 20,
            "events": [1] * 5 + [0] * 15 + [1] * 2 + [0] * 18,
        })
        out = crude_rate_df(df_rowlevel, "events", ["grp"])
        row_a = out.filter(pl.col("grp") == "A")
        expected_rate = (5.0 / 20.0) * 100_000.0
        assert abs(row_a["rate"][0] - expected_rate) < 1e-6


# ============================================================
# Null/NA validation across numerator, strata, and group columns
# ============================================================

class TestNullValidation:
    def test_null_in_event_col_raises_crude_proportion(self):
        df = pl.DataFrame({
            "grp": ["A", "A", "B", "B"],
            "event": [1, None, 0, 1],
        })
        with pytest.raises(ValueError, match="Null/NA values"):
            crude_proportion_df(df, "event", ["grp"])

    def test_null_in_group_col_raises_crude_proportion(self):
        df = pl.DataFrame({
            "grp": ["A", None, "B", "B"],
            "event": [1, 0, 0, 1],
        })
        with pytest.raises(ValueError, match="Null/NA values"):
            crude_proportion_df(df, "event", ["grp"])

    def test_null_in_event_col_raises_crude_rate(self):
        df = pl.DataFrame({
            "grp": ["A", "A", "B", "B"],
            "events": [1, None, 0, 1],
        })
        with pytest.raises(ValueError, match="Null/NA values"):
            crude_rate_df(df, "events", ["grp"])

    def test_null_in_strata_col_raises_dsp(self):
        df = pl.DataFrame({
            "grp": ["A"] * 20 + ["B"] * 20,
            "strat": (["X"] * 10 + [None] + ["X"] * 9) * 2,
            "event": [1, 0] * 20,
        })
        with pytest.raises(ValueError, match="Null/NA values"):
            directly_standardized_proportion_df(df, "event", ["strat"], ["grp"])

    def test_null_in_strata_col_raises_dsr(self):
        df = pl.DataFrame({
            "grp": ["A"] * 20 + ["B"] * 20,
            "strat": (["X"] * 10 + [None] + ["X"] * 9) * 2,
            "events": [1, 0] * 20,
        })
        with pytest.raises(ValueError, match="Null/NA values"):
            directly_standardized_rate_df(df, "events", ["strat"], ["grp"])

    def test_null_in_group_col_raises_dsp(self):
        df = pl.DataFrame({
            "grp": ["A"] * 10 + [None] * 10 + ["B"] * 20,
            "strat": ["X"] * 40,
            "event": [1, 0] * 20,
        })
        with pytest.raises(ValueError, match="Null/NA values"):
            directly_standardized_proportion_df(df, "event", ["strat"], ["grp"])

    def test_error_lists_all_offending_columns(self):
        df = pl.DataFrame({
            "grp": ["A", None, "B", "B"],
            "event": [1, None, 0, 1],
        })
        with pytest.raises(ValueError) as exc_info:
            crude_proportion_df(df, "event", ["grp"])
        msg = str(exc_info.value)
        assert "event" in msg
        assert "grp" in msg

    def test_check_no_nulls_helper_directly(self):
        df = pl.DataFrame({"a": [1, None, 3], "b": [1, 2, 3]})
        with pytest.raises(ValueError, match="Null/NA values"):
            _check_no_nulls(df, ["a", "b"])

    def test_check_no_nulls_passes_when_clean(self):
        df = pl.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]})
        _check_no_nulls(df, ["a", "b"])  # should not raise


# ============================================================
# Numerator column type/value validation
# ============================================================

class TestNumeratorValidation:
    def test_float_event_col_raises_crude_proportion(self):
        df = pl.DataFrame({"grp": ["A", "B"], "event": [1.0, 0.0]})
        with pytest.raises(ValueError, match="non-negative integer"):
            crude_proportion_df(df, "event", ["grp"])

    def test_string_event_col_raises_crude_rate(self):
        df = pl.DataFrame({"grp": ["A", "B"], "events": ["1", "0"]})
        with pytest.raises(ValueError, match="non-negative integer"):
            crude_rate_df(df, "events", ["grp"])

    def test_negative_integer_raises(self):
        df = pl.DataFrame({"grp": ["A", "B"], "events": [-1, 2]})
        with pytest.raises(ValueError, match="non-negative"):
            crude_rate_df(df, "events", ["grp"])

    def test_proportion_col_must_be_0_or_1(self):
        df = pl.DataFrame({"grp": ["A", "B", "C"], "event": [0, 1, 2]})
        with pytest.raises(ValueError, match="0 or 1"):
            crude_proportion_df(df, "event", ["grp"])

    def test_rate_col_allows_values_above_1(self):
        df = pl.DataFrame({"grp": ["A", "B", "C"], "events": [0, 1, 5]})
        out = crude_rate_df(df, "events", ["grp"])
        assert out.shape[0] == 4

    def test_boolean_event_col_accepted_for_proportion(self):
        df = pl.DataFrame({"grp": ["A", "B"], "event": [True, False]})
        out = crude_proportion_df(df, "event", ["grp"])
        assert out.shape[0] == 3

    def test_boolean_event_col_accepted_for_rate(self):
        df = pl.DataFrame({"grp": ["A", "B"], "events": [True, False]})
        out = crude_rate_df(df, "events", ["grp"])
        assert out.shape[0] == 3

    def test_dsp_rejects_non_binary_numerator(self):
        df = pl.DataFrame({
            "grp": ["A"] * 10 + ["B"] * 10,
            "strat": ["X"] * 20,
            "event": [0, 1, 2] * 6 + [0, 1],
        })
        with pytest.raises(ValueError, match="0 or 1"):
            directly_standardized_proportion_df(df, "event", ["strat"], ["grp"])

    def test_dsr_allows_multi_valued_integer_numerator(self):
        df = pl.DataFrame({
            "grp": ["A"] * 10 + ["B"] * 10,
            "strat": ["X"] * 20,
            "events": [0, 1, 2, 3] * 5,
        })
        out = directly_standardized_rate_df(df, "events", ["strat"], ["grp"])
        assert out.shape[0] == 3

    def test_validate_numerator_col_helper_negative(self):
        df = pl.DataFrame({"events": [-5, 1, 2]})
        with pytest.raises(ValueError, match="non-negative"):
            _validate_numerator_col(df, "events", binary=False)

    def test_validate_numerator_col_helper_float_dtype(self):
        df = pl.DataFrame({"events": [1.5, 2.0, 3.0]})
        with pytest.raises(ValueError, match="non-negative integer"):
            _validate_numerator_col(df, "events", binary=True)

    def test_validate_numerator_col_helper_binary_violation(self):
        df = pl.DataFrame({"event": [0, 1, 2]})
        with pytest.raises(ValueError, match="0 or 1"):
            _validate_numerator_col(df, "event", binary=True)


class TestAdditionalCoverage:
    def test_byar_count_ci_negative_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            _byar_count_ci(-1)

    def test_bin_numeric_to_quartiles_handles_exception(self, monkeypatch):
        class BrokenSeries(pl.Series):
            def quantile(self, *a, **kw):
                raise RuntimeError("boom")
        s = BrokenSeries("x", [1, 2, 3])
        out = _bin_numeric_to_quartiles(s)
        assert out is s

    def test_wilson_dobson_proportion_ci_zero_n_returns_nan(self):
        result = _wilson_dobson_proportion_ci(
            0.5, 5, 0, np.array([1.0]), np.array([0.5]), np.array([10.0])
        )
        assert all(np.isnan(v) for v in result)

    def test_wilson_dobson_proportion_ci_zero_weight_sum_returns_nan(self):
        result = _wilson_dobson_proportion_ci(
            0.5, 5, 10, np.array([0.0]), np.array([0.5]), np.array([10.0])
        )
        assert all(np.isnan(v) for v in result)

    def test_dobson_byar_rate_ci_negative_events_returns_nan(self):
        result = _dobson_byar_rate_ci(
            0.5, -1, np.array([1.0]), np.array([5.0]), np.array([10.0])
        )
        assert all(np.isnan(v) for v in result)

    def test_dobson_byar_rate_ci_zero_weight_sum_returns_nan(self):
        result = _dobson_byar_rate_ci(
            0.5, 5, np.array([0.0]), np.array([5.0]), np.array([10.0])
        )
        assert all(np.isnan(v) for v in result)

    def test_significance_from_ci_not_tested_when_reference_nan(self):
        label = _significance_from_ci(np.nan, 0.4, 0.6)
        assert label == "Not tested"

    def test_crude_rate_df_empty_group_produces_overall_row(self):
        df = pl.DataFrame({
            "grp": pl.Series([], dtype=pl.Utf8),
            "event": pl.Series([], dtype=pl.Int64),
        })
        out = crude_rate_df(df, "event", ["grp"])
        assert out.shape[0] == 1
        for col in ["events", "denominator", "rate", "significance"]:
            assert col in out.columns
        assert out["significance"][0] == "Reference"

    def test_dsp_df_empty_input_produces_correct_schema(self):
        df = pl.DataFrame({
            "grp": pl.Series([], dtype=pl.Utf8),
            "strat": pl.Series([], dtype=pl.Utf8),
            "event": pl.Series([], dtype=pl.Int64),
        })
        out = directly_standardized_proportion_df(df, "event", ["strat"], ["grp"])
        assert out.shape[0] == 1

    def test_dsr_df_empty_input_produces_overall_row(self):
        df = pl.DataFrame({
            "grp": pl.Series([], dtype=pl.Utf8),
            "strat": pl.Series([], dtype=pl.Utf8),
            "events": pl.Series([], dtype=pl.Int64),
        })
        out = directly_standardized_rate_df(df, "events", ["strat"], ["grp"])
        assert out.shape[0] == 1