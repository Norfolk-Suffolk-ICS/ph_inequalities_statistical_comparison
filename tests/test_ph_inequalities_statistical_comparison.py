"""Tests for ph_inequalities_statistical_comparison."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import polars as pl
import pytest
from scipy import stats

import ph_inequalities_statistical_comparison as package
from ph_inequalities_statistical_comparison import (
    crude_proportion_df,
    crude_rate_df,
    directly_standardized_proportion_df,
    directly_standardized_rate_df,
)
from ph_inequalities_statistical_comparison.core import (
    _DENOM_COL,
    _add_end_of_period_denominator,
    _build_grouping_sets,
    _byar_count_ci,
    _check_no_missing_or_nonfinite,
    _crude_proportion_action,
    _crude_rate_action,
    _compute_proportion_stratum_stats,
    _compute_rate_stratum_stats,
    _ds_proportion_action,
    _ds_rate_action,
    _exact_poisson_count_ci,
    _inequality_states,
    _missing_strata_note,
    _mover_weighted_proportion_ci,
    _mover_weighted_rate_ci,
    _normalise_hierarchies,
    _organisational_states,
    _prepare_dataframe_proportion,
    _prepare_dataframe_rate,
    _significance_from_ci,
    _validate_options,
    _validate_strata_cols,
    _wilson_proportion_ci,
)

PUBLIC_NAMES = {
    "crude_proportion_df",
    "crude_rate_df",
    "directly_standardized_proportion_df",
    "directly_standardized_rate_df",
}


# ===========================================================================
# Reference calculations used by tests
# ===========================================================================


def reference_wilson(
    events: float,
    denominator: float,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Independently calculate a Wilson score interval."""
    proportion = events / denominator
    z_value = stats.norm.ppf(
        1 - (1 - confidence) / 2,
    )

    scale = 1 + z_value**2 / denominator

    centre = (proportion + z_value**2 / (2 * denominator)) / scale

    half_width = (
        z_value
        * np.sqrt(
            proportion * (1 - proportion) / denominator
            + z_value**2 / (4 * denominator**2),
        )
        / scale
    )

    return (
        max(0.0, centre - half_width),
        min(1.0, centre + half_width),
    )


def reference_exact_poisson(
    count: float,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Independently calculate an exact Poisson count interval."""
    alpha = 1 - confidence

    lower = (
        0.0
        if count == 0
        else 0.5
        * stats.chi2.ppf(
            alpha / 2,
            2 * count,
        )
    )

    upper = 0.5 * stats.chi2.ppf(
        1 - alpha / 2,
        2 * (count + 1),
    )

    return (
        float(lower),
        float(upper),
    )


def reference_byar(
    count: float,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Independently calculate Byar count limits."""
    alpha = 1 - confidence
    z_value = stats.norm.ppf(
        1 - alpha / 2,
    )

    lower = count * (1 - 1 / (9 * count) - z_value / (3 * np.sqrt(count))) ** 3

    upper = (count + 1) * (
        1 - 1 / (9 * (count + 1)) + z_value / (3 * np.sqrt(count + 1))
    ) ** 3

    return (
        max(0.0, float(lower)),
        max(0.0, float(upper)),
    )


def reference_mover_proportion(
    weights: np.ndarray,
    events: np.ndarray,
    denominators: np.ndarray,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Independently combine Wilson intervals using MOVER."""
    weights = weights / weights.sum()
    proportions = events / denominators

    intervals = [
        reference_wilson(
            event,
            denominator,
            confidence,
        )
        for event, denominator in zip(
            events,
            denominators,
        )
    ]

    lower_limits = np.array(
        [interval[0] for interval in intervals],
    )

    upper_limits = np.array(
        [interval[1] for interval in intervals],
    )

    estimate = float(
        np.sum(
            weights * proportions,
        ),
    )

    lower = max(
        0.0,
        estimate
        - float(
            np.sqrt(
                np.sum(
                    weights**2 * (proportions - lower_limits) ** 2,
                ),
            ),
        ),
    )

    upper = min(
        1.0,
        estimate
        + float(
            np.sqrt(
                np.sum(
                    weights**2 * (upper_limits - proportions) ** 2,
                ),
            ),
        ),
    )

    return (
        estimate,
        lower,
        upper,
    )


def reference_mover_rate(
    weights: np.ndarray,
    events: np.ndarray,
    denominators: np.ndarray,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Independently combine Poisson rate intervals using MOVER."""
    weights = weights / weights.sum()
    rates = events / denominators

    lower_limits: list[float] = []
    upper_limits: list[float] = []

    for count, denominator in zip(
        events,
        denominators,
    ):
        if count < 10:
            count_lower, count_upper = reference_exact_poisson(
                count,
                confidence,
            )
        else:
            count_lower, count_upper = reference_byar(
                count,
                confidence,
            )

        lower_limits.append(
            count_lower / denominator,
        )
        upper_limits.append(
            count_upper / denominator,
        )

    lower_array = np.array(lower_limits)
    upper_array = np.array(upper_limits)

    estimate = float(
        np.sum(
            weights * rates,
        ),
    )

    lower = max(
        0.0,
        estimate
        - float(
            np.sqrt(
                np.sum(
                    weights**2 * (rates - lower_array) ** 2,
                ),
            ),
        ),
    )

    upper = estimate + float(
        np.sqrt(
            np.sum(
                weights**2 * (upper_array - rates) ** 2,
            ),
        ),
    )

    return (
        estimate,
        lower,
        upper,
    )


def get_row(
    result: pl.DataFrame,
    **filters: object,
) -> dict[str, Any]:
    """Return one output row matching the supplied values."""
    filtered = result

    for column, value in filters.items():
        filtered = filtered.filter(
            pl.col(column) == value,
        )

    assert filtered.height == 1

    return filtered.row(
        0,
        named=True,
    )


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def analytical_df() -> pl.DataFrame:
    """Patient-level data containing valid inequality and hierarchy data."""
    return pl.DataFrame(
        {
            "event": [
                0,
                1,
                1,
                0,
                1,
                1,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                0,
                1,
                0,
            ],
            "count": [
                0,
                1,
                2,
                0,
                1,
                3,
                0,
                1,
                0,
                2,
                1,
                4,
                2,
                0,
                3,
                1,
            ],
            "age_band": [
                "Young",
                "Young",
                "Old",
                "Old",
                "Young",
                "Young",
                "Old",
                "Old",
                "Young",
                "Young",
                "Old",
                "Old",
                "Young",
                "Young",
                "Old",
                "Old",
            ],
            "sex": [
                "Female",
                "Male",
                "Female",
                "Male",
                "Female",
                "Male",
                "Female",
                "Male",
                "Female",
                "Male",
                "Female",
                "Male",
                "Female",
                "Male",
                "Female",
                "Male",
            ],
            "ethnicity": [
                "White",
                "White",
                "Other",
                "Other",
                "White",
                "White",
                "Other",
                "Other",
                "White",
                "White",
                "Other",
                "Other",
                "White",
                "White",
                "Other",
                "Other",
            ],
            "region": [
                *["East"] * 8,
                *["West"] * 8,
            ],
            "icb": [
                *["E1"] * 8,
                *["W1"] * 8,
            ],
            "practice": [
                *["P1"] * 4,
                *["P2"] * 4,
                *["P3"] * 4,
                *["P4"] * 4,
            ],
        },
    )


@pytest.fixture
def missing_strata_df() -> pl.DataFrame:
    """Data in which one comparison group has no observations in stratum B."""
    return pl.DataFrame(
        {
            "event": [
                0,
                0,
                1,
                0,
                1,
                1,
            ],
            "count": [
                0,
                0,
                1,
                2,
                1,
                3,
            ],
            "age_band": [
                "A",
                "A",
                "A",
                "A",
                "B",
                "B",
            ],
            "group": [
                "G1",
                "G1",
                "G2",
                "G2",
                "G2",
                "G2",
            ],
        },
    )


# ===========================================================================
# Package API
# ===========================================================================


def test_package_exposes_public_functions() -> None:
    """All documented functions are available from the package root."""
    assert PUBLIC_NAMES <= set(package.__all__)

    for name in PUBLIC_NAMES:
        assert hasattr(package, name)
        assert callable(getattr(package, name))


def test_package_functions_match_core_exports() -> None:
    """Package-level imports reference the expected functions."""
    assert package.crude_proportion_df is crude_proportion_df
    assert package.crude_rate_df is crude_rate_df

    assert (
        package.directly_standardized_proportion_df
        is directly_standardized_proportion_df
    )

    assert package.directly_standardized_rate_df is directly_standardized_rate_df


# ===========================================================================
# Statistical helper tests
# ===========================================================================


@pytest.mark.parametrize(
    ("events", "denominator"),
    [
        (0.0, 10.0),
        (1.0, 10.0),
        (5.0, 10.0),
        (10.0, 10.0),
    ],
)
def test_wilson_matches_reference(
    events: float,
    denominator: float,
) -> None:
    """Wilson limits match an independent implementation."""
    expected_lower, expected_upper = reference_wilson(
        events,
        denominator,
    )

    lower, upper, variance = _wilson_proportion_ci(
        events,
        denominator,
    )

    assert lower == pytest.approx(
        expected_lower,
    )

    assert upper == pytest.approx(
        expected_upper,
    )

    expected_proportion = events / denominator

    assert variance == pytest.approx(
        expected_proportion * (1 - expected_proportion) / denominator,
    )


def test_wilson_zero_denominator_returns_nan() -> None:
    """A zero denominator has no estimable Wilson interval."""
    lower, upper, variance = _wilson_proportion_ci(
        0,
        0,
    )

    assert np.isnan(lower)
    assert np.isnan(upper)
    assert np.isnan(variance)


@pytest.mark.parametrize(
    "count",
    [
        0.0,
        1.0,
        5.0,
        9.0,
    ],
)
def test_exact_poisson_matches_reference(
    count: float,
) -> None:
    """Exact Poisson limits match the chi-square calculation."""
    expected = reference_exact_poisson(
        count,
    )

    actual = _exact_poisson_count_ci(
        count,
    )

    assert actual[0] == pytest.approx(
        expected[0],
    )

    assert actual[1] == pytest.approx(
        expected[1],
    )


@pytest.mark.parametrize(
    "count",
    [
        10.0,
        20.0,
        100.0,
    ],
)
def test_byar_matches_reference(
    count: float,
) -> None:
    """Byar limits match an independent implementation."""
    expected = reference_byar(
        count,
    )

    actual = _byar_count_ci(
        count,
    )

    assert actual[0] == pytest.approx(
        expected[0],
    )

    assert actual[1] == pytest.approx(
        expected[1],
    )


@pytest.mark.parametrize(
    "helper",
    [
        _exact_poisson_count_ci,
        _byar_count_ci,
    ],
)
def test_poisson_helpers_reject_negative_counts(
    helper: Callable[..., tuple[float, float]],
) -> None:
    """Poisson interval helpers reject negative counts."""
    with pytest.raises(
        ValueError,
        match="non-negative",
    ):
        helper(-1)


def test_proportion_mover_matches_reference() -> None:
    """Wilson-MOVER matches an independent calculation."""
    weights = np.array(
        [0.25, 0.75],
    )

    events = np.array(
        [0.0, 8.0],
    )

    denominators = np.array(
        [10.0, 10.0],
    )

    expected = reference_mover_proportion(
        weights,
        events,
        denominators,
    )

    actual = _mover_weighted_proportion_ci(
        weights,
        events,
        denominators,
    )

    assert actual == pytest.approx(
        expected,
    )


def test_proportion_mover_zero_events_has_positive_upper_limit() -> None:
    """Zero-event strata retain a zero estimate and positive upper limit."""
    estimate, lower, upper = _mover_weighted_proportion_ci(
        stratum_weights=np.array(
            [0.5, 0.5],
        ),
        stratum_events=np.array(
            [0.0, 0.0],
        ),
        stratum_ns=np.array(
            [10.0, 20.0],
        ),
    )

    assert estimate == 0
    assert lower == 0
    assert upper > 0


def test_proportion_mover_all_events_has_nonzero_lower_limit() -> None:
    """All-event strata retain an estimate of one."""
    estimate, lower, upper = _mover_weighted_proportion_ci(
        stratum_weights=np.array(
            [0.5, 0.5],
        ),
        stratum_events=np.array(
            [10.0, 20.0],
        ),
        stratum_ns=np.array(
            [10.0, 20.0],
        ),
    )

    assert estimate == pytest.approx(1)
    assert 0 < lower < 1
    assert upper == pytest.approx(1)


def test_rate_mover_matches_reference() -> None:
    """Poisson-MOVER matches an independent calculation."""
    weights = np.array(
        [0.4, 0.6],
    )

    events = np.array(
        [0.0, 20.0],
    )

    denominators = np.array(
        [100.0, 200.0],
    )

    expected = reference_mover_rate(
        weights,
        events,
        denominators,
    )

    actual = _mover_weighted_rate_ci(
        weights,
        events,
        denominators,
    )

    assert actual == pytest.approx(
        expected,
    )


def test_rate_mover_zero_events_has_positive_upper_limit() -> None:
    """An all-zero DSR has a positive MOVER upper confidence limit."""
    estimate, lower, upper = _mover_weighted_rate_ci(
        stratum_weights=np.array(
            [0.5, 0.5],
        ),
        stratum_events=np.array(
            [0.0, 0.0],
        ),
        stratum_denominators=np.array(
            [100.0, 200.0],
        ),
    )

    assert estimate == 0
    assert lower == 0
    assert upper > 0


@pytest.mark.parametrize(
    (
        "weights",
        "events",
        "denominators",
        "message",
    ),
    [
        (
            np.array([-0.5, 1.5]),
            np.array([1.0, 1.0]),
            np.array([10.0, 10.0]),
            "weights must be non-negative",
        ),
        (
            np.array([0.5, 0.5]),
            np.array([-1.0, 1.0]),
            np.array([10.0, 10.0]),
            "events must be non-negative",
        ),
        (
            np.array([0.5, 0.5]),
            np.array([1.0, 1.0]),
            np.array([-10.0, 10.0]),
            "denominators must be non-negative",
        ),
    ],
)
def test_rate_mover_rejects_invalid_inputs(
    weights: np.ndarray,
    events: np.ndarray,
    denominators: np.ndarray,
    message: str,
) -> None:
    """Poisson-MOVER validates weights, events and denominators."""
    with pytest.raises(
        ValueError,
        match=message,
    ):
        _mover_weighted_rate_ci(
            weights,
            events,
            denominators,
        )


def test_mover_rejects_different_array_lengths() -> None:
    """MOVER inputs must have equal lengths."""
    with pytest.raises(
        ValueError,
        match="equal lengths",
    ):
        _mover_weighted_proportion_ci(
            stratum_weights=np.array(
                [1.0],
            ),
            stratum_events=np.array(
                [1.0, 2.0],
            ),
            stratum_ns=np.array(
                [10.0],
            ),
        )


def test_mover_rejects_non_vector_inputs() -> None:
    """MOVER inputs must be one-dimensional."""
    with pytest.raises(
        ValueError,
        match="one-dimensional",
    ):
        _mover_weighted_rate_ci(
            stratum_weights=np.array(
                [[1.0]],
            ),
            stratum_events=np.array(
                [[1.0]],
            ),
            stratum_denominators=np.array(
                [[10.0]],
            ),
        )


# ===========================================================================
# Grouping logic
# ===========================================================================


def test_inequality_states_form_complete_cube() -> None:
    """Two inequality dimensions produce four marginal states."""
    assert _inequality_states(
        ["sex", "ethnicity"],
    ) == [
        (
            "sex",
            "ethnicity",
        ),
        ("sex",),
        ("ethnicity",),
        (),
    ]


def test_separate_organisational_states() -> None:
    """Separate mode emits valid hierarchy prefixes."""
    hierarchies = {
        "geography": (
            "region",
            "icb",
            "practice",
        ),
    }

    assert _organisational_states(
        hierarchies,
        "separate",
    ) == [
        (
            "region",
            "icb",
            "practice",
        ),
        (
            "region",
            "icb",
        ),
        ("region",),
        (),
    ]


def test_cross_organisational_states() -> None:
    """Cross mode intersects independently declared hierarchies."""
    hierarchies = {
        "geography": ("region",),
        "provider": ("provider",),
    }

    assert _organisational_states(
        hierarchies,
        "cross",
    ) == [
        (
            "region",
            "provider",
        ),
        ("region",),
        ("provider",),
        (),
    ]


def test_grouping_sets_remove_duplicates() -> None:
    """The grouping-set builder emits unique states."""
    grouping_sets = _build_grouping_sets(
        inequalities=["sex"],
        hierarchies={
            "geography": (
                "region",
                "icb",
            ),
        },
        mode="separate",
    )

    active_states = [grouping_set.active_cols for grouping_set in grouping_sets]

    assert len(active_states) == len(
        set(active_states),
    )

    assert () in active_states

    assert (
        "region",
        "icb",
        "sex",
    ) in active_states


def test_normalise_hierarchies_accepts_valid_mapping() -> None:
    """Hierarchy declarations are converted to tuples."""
    result = _normalise_hierarchies(
        {
            "geography": [
                "region",
                "icb",
            ],
        },
    )

    assert result == {
        "geography": (
            "region",
            "icb",
        ),
    }


@pytest.mark.parametrize(
    "value",
    [
        ["region", "icb"],
        "region",
        12,
    ],
)
def test_normalise_hierarchies_rejects_non_mapping(
    value: object,
) -> None:
    """Organisational columns must be supplied as a named mapping."""
    with pytest.raises(
        TypeError,
        match="named mapping",
    ):
        _normalise_hierarchies(value)  # type: ignore[arg-type]


# ===========================================================================
# Validation
# ===========================================================================


@pytest.mark.parametrize(
    "function,kwargs",
    [
        (
            crude_proportion_df,
            {
                "event_col": "event",
            },
        ),
        (
            crude_rate_df,
            {
                "event_col": "event",
            },
        ),
        (
            directly_standardized_proportion_df,
            {
                "event_col": "event",
                "strata_cols": ["age"],
            },
        ),
        (
            directly_standardized_rate_df,
            {
                "event_col": "event",
                "strata_cols": ["age"],
            },
        ),
    ],
)
def test_public_functions_reject_empty_dataframes(
    function: Callable[..., pl.DataFrame],
    kwargs: dict[str, object],
) -> None:
    """All public functions reject empty dataframes consistently."""
    df = pl.DataFrame(
        {
            "event": pl.Series(
                [],
                dtype=pl.Int64,
            ),
            "age": pl.Series(
                [],
                dtype=pl.String,
            ),
        },
    )

    with pytest.raises(
        ValueError,
        match="at least one row",
    ):
        function(
            df,
            **kwargs,
        )


def test_public_function_requires_polars_dataframe() -> None:
    """Non-Polars inputs receive a clear type error."""
    with pytest.raises(
        TypeError,
        match="Polars DataFrame",
    ):
        crude_proportion_df(  # type: ignore[arg-type]
            {
                "event": [
                    0,
                    1,
                ],
            },
            event_col="event",
        )


def test_missing_required_column_is_rejected() -> None:
    """Missing analytical columns are reported."""
    df = pl.DataFrame(
        {
            "event": [
                0,
                1,
            ],
        },
    )

    with pytest.raises(
        ValueError,
        match="group",
    ):
        crude_proportion_df(
            df,
            event_col="event",
            inequalities_cols=["group"],
        )


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (
            [1.0, None],
            "null",
        ),
        (
            [1.0, float("nan")],
            "NaN",
        ),
        (
            [1.0, float("inf")],
            "infinite",
        ),
        (
            [1.0, float("-inf")],
            "infinite",
        ),
    ],
)
def test_missing_and_nonfinite_values_are_rejected(
    values: list[float | None],
    message: str,
) -> None:
    """Null, NaN and infinite grouping values are rejected."""
    df = pl.DataFrame(
        {
            "event": [
                0,
                1,
            ],
            "group": values,
        },
    )

    with pytest.raises(
        ValueError,
        match=message,
    ):
        crude_proportion_df(
            df,
            event_col="event",
            inequalities_cols=["group"],
        )


def test_nonfinite_stratum_is_rejected() -> None:
    """Non-finite values in standardisation strata are rejected."""
    df = pl.DataFrame(
        {
            "event": [
                0,
                1,
            ],
            "age": [
                20.0,
                float("nan"),
            ],
        },
    )

    with pytest.raises(
        ValueError,
        match="NaN",
    ):
        directly_standardized_proportion_df(
            df,
            event_col="event",
            strata_cols=["age"],
        )


def test_missing_helper_reports_multiple_problem_types() -> None:
    """The missing-value helper reports all offending value types."""
    df = pl.DataFrame(
        {
            "value": [
                None,
                float("nan"),
                float("inf"),
            ],
        },
    )

    with pytest.raises(
        ValueError,
    ) as error:
        _check_no_missing_or_nonfinite(
            df,
            ["value"],
        )

    message = str(error.value)

    assert "null" in message
    assert "NaN" in message
    assert "infinite" in message


def test_negative_rate_numerator_is_rejected() -> None:
    """Rate events must be non-negative."""
    df = pl.DataFrame(
        {
            "count": [
                1,
                -1,
            ],
        },
    )

    with pytest.raises(
        ValueError,
        match="non-negative",
    ):
        crude_rate_df(
            df,
            event_col="count",
        )


def test_nonbinary_proportion_numerator_is_rejected() -> None:
    """Proportion events must contain only zero and one."""
    df = pl.DataFrame(
        {
            "event": [
                0,
                2,
            ],
        },
    )

    with pytest.raises(
        ValueError,
        match="only 0 or 1",
    ):
        crude_proportion_df(
            df,
            event_col="event",
        )


def test_float_numerator_is_rejected() -> None:
    """Numerators must use an integer or Boolean dtype."""
    df = pl.DataFrame(
        {
            "event": [
                0.0,
                1.0,
            ],
        },
    )

    with pytest.raises(
        ValueError,
        match="integer",
    ):
        crude_proportion_df(
            df,
            event_col="event",
        )


def test_boolean_numerator_is_accepted() -> None:
    """Boolean proportion numerators are converted to integers."""
    df = pl.DataFrame(
        {
            "event": [
                True,
                False,
                True,
            ],
        },
    )

    result = crude_proportion_df(
        df,
        event_col="event",
    )

    row = result.row(
        0,
        named=True,
    )

    assert row["events"] == 2
    assert row["n"] == 3
    assert row["proportion"] == pytest.approx(
        2 / 3,
    )


@pytest.mark.parametrize(
    "confidence",
    [
        "0.95",
        True,
        complex(
            0.95,
            0,
        ),
    ],
)
def test_confidence_rejects_invalid_types(
    confidence: object,
) -> None:
    """Confidence rejects non-real and Boolean values."""
    with pytest.raises(
        TypeError,
        match="real numeric",
    ):
        _validate_options(
            confidence,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "confidence",
    [
        0.0,
        1.0,
        -0.5,
        1.5,
        float("nan"),
        float("inf"),
    ],
)
def test_confidence_rejects_invalid_values(
    confidence: float,
) -> None:
    """Confidence must be finite and strictly between zero and one."""
    with pytest.raises(
        ValueError,
        match="strictly between",
    ):
        _validate_options(
            confidence,
        )


@pytest.mark.parametrize(
    "multiplier",
    [
        "100000",
        True,
        complex(
            100_000,
            0,
        ),
    ],
)
def test_multiplier_rejects_invalid_types(
    multiplier: object,
) -> None:
    """Multiplier rejects non-real and Boolean values."""
    with pytest.raises(
        TypeError,
        match="real numeric",
    ):
        _validate_options(
            0.95,
            multiplier,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "multiplier",
    [
        0.0,
        -1.0,
        float("nan"),
        float("inf"),
    ],
)
def test_multiplier_rejects_invalid_values(
    multiplier: float,
) -> None:
    """Multiplier must be finite and positive."""
    with pytest.raises(
        ValueError,
        match="finite and positive",
    ):
        _validate_options(
            0.95,
            multiplier,
        )


def test_numpy_real_options_are_accepted() -> None:
    """NumPy real scalar values pass option validation."""
    _validate_options(
        np.float64(0.95),
        np.float64(100_000),
    )


@pytest.mark.parametrize(
    "strata",
    [
        "age",
        b"age",
    ],
)
def test_strata_reject_single_string(
    strata: object,
) -> None:
    """strata_cols must be a sequence rather than one string."""
    with pytest.raises(
        TypeError,
        match="sequence",
    ):
        _validate_strata_cols(
            strata,  # type: ignore[arg-type]
        )


def test_strata_reject_empty_sequence() -> None:
    """At least one standardisation stratum is required."""
    with pytest.raises(
        ValueError,
        match="at least one",
    ):
        _validate_strata_cols(
            [],
        )


@pytest.mark.parametrize(
    "reserved_name",
    [
        "events",
        "n",
        "notes",
        "significance",
        "ref_weight",
        "dsp",
        "dsr",
    ],
)
def test_reserved_dimension_names_are_rejected(
    reserved_name: str,
) -> None:
    """Grouping dimensions cannot collide with output columns."""
    df = pl.DataFrame(
        {
            "event": [
                0,
                1,
            ],
            reserved_name: [
                "A",
                "B",
            ],
        },
    )

    with pytest.raises(
        ValueError,
        match="reserved analysis",
    ):
        crude_proportion_df(
            df,
            event_col="event",
            inequalities_cols=[
                reserved_name,
            ],
        )


def test_reserved_stratum_name_is_rejected() -> None:
    """Standardisation strata cannot collide with generated columns."""
    df = pl.DataFrame(
        {
            "event": [
                0,
                1,
            ],
            "ref_weight": [
                "A",
                "B",
            ],
        },
    )

    with pytest.raises(
        ValueError,
        match="reserved analysis",
    ):
        directly_standardized_proportion_df(
            df,
            event_col="event",
            strata_cols=["ref_weight"],
        )


def test_internal_denominator_is_not_overwritten() -> None:
    """Rate functions protect the internal denominator column."""
    df = pl.DataFrame(
        {
            "count": [
                1,
                2,
            ],
            _DENOM_COL: [
                10.0,
                10.0,
            ],
        },
    )

    with pytest.raises(
        ValueError,
        match="already contains",
    ):
        crude_rate_df(
            df,
            event_col="count",
        )


def test_inequality_and_organisation_overlap_is_rejected() -> None:
    """A column cannot be both an inequality and organisation."""
    df = pl.DataFrame(
        {
            "event": [
                0,
                1,
            ],
            "region": [
                "A",
                "B",
            ],
        },
    )

    with pytest.raises(
        ValueError,
        match="both inequality and organisational",
    ):
        crude_proportion_df(
            df,
            event_col="event",
            inequalities_cols=["region"],
            organisational_cols={
                "geography": ["region"],
            },
        )


def test_stratum_and_dimension_overlap_is_rejected() -> None:
    """A stratum cannot also be an output grouping dimension."""
    df = pl.DataFrame(
        {
            "event": [
                0,
                1,
            ],
            "age": [
                "A",
                "B",
            ],
        },
    )

    with pytest.raises(
        ValueError,
        match="Strata cannot also",
    ):
        directly_standardized_proportion_df(
            df,
            event_col="event",
            strata_cols=["age"],
            inequalities_cols=["age"],
        )


def test_event_and_dimension_overlap_is_rejected() -> None:
    """The numerator cannot also be a grouping dimension."""
    df = pl.DataFrame(
        {
            "event": [
                0,
                1,
            ],
        },
    )

    with pytest.raises(
        ValueError,
        match="event_col cannot",
    ):
        crude_proportion_df(
            df,
            event_col="event",
            inequalities_cols=["event"],
        )


def test_all_label_collision_is_rejected() -> None:
    """Observed dimension values cannot equal the reserved all label."""
    df = pl.DataFrame(
        {
            "event": [
                0,
                1,
            ],
            "group": [
                "All",
                "B",
            ],
        },
    )

    with pytest.raises(
        ValueError,
        match="Reserved all_label",
    ):
        crude_proportion_df(
            df,
            event_col="event",
            inequalities_cols=["group"],
        )


def test_invalid_hierarchy_is_rejected() -> None:
    """A child organisation cannot map to multiple parents."""
    df = pl.DataFrame(
        {
            "event": [
                0,
                1,
            ],
            "region": [
                "R1",
                "R2",
            ],
            "practice": [
                "P1",
                "P1",
            ],
        },
    )

    with pytest.raises(
        ValueError,
        match="does not map to exactly one",
    ):
        crude_proportion_df(
            df,
            event_col="event",
            organisational_cols={
                "geography": [
                    "region",
                    "practice",
                ],
            },
        )


def test_invalid_organisational_mode_is_rejected() -> None:
    """Only separate and cross organisational modes are valid."""
    df = pl.DataFrame(
        {
            "event": [
                0,
                1,
            ],
        },
    )

    with pytest.raises(
        ValueError,
        match="separate.*cross",
    ):
        crude_proportion_df(
            df,
            event_col="event",
            organisational_mode="invalid",  # type: ignore[arg-type]
        )


# ===========================================================================
# Data preparation
# ===========================================================================


def test_numeric_strata_are_converted_to_quartiles() -> None:
    """Numeric standardisation strata are binned into Q1-Q4."""
    df = pl.DataFrame(
        {
            "event": [
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
            ],
            "age": [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
            ],
        },
    )

    prepared = _prepare_dataframe_proportion(
        df=df,
        event_col="event",
        strata_cols=["age"],
        inequalities_cols=[],
        organisational_cols={},
    )

    assert prepared.schema["age"] == pl.String

    assert set(
        prepared["age"].unique().to_list(),
    ) == {
        "Q1",
        "Q2",
        "Q3",
        "Q4",
    }


def test_rate_preparation_allows_counts_above_one() -> None:
    """Rate preparation accepts recurrent event counts."""
    df = pl.DataFrame(
        {
            "count": [
                0,
                2,
                4,
                1,
            ],
            "age": [
                "A",
                "A",
                "B",
                "B",
            ],
        },
    )

    prepared = _prepare_dataframe_rate(
        df=df,
        event_col="count",
        strata_cols=["age"],
        inequalities_cols=[],
        organisational_cols={},
    )

    assert prepared["count"].to_list() == [
        0,
        2,
        4,
        1,
    ]


def test_end_of_period_denominator_contains_ones() -> None:
    """Each input row contributes one denominator unit."""
    df = pl.DataFrame(
        {
            "count": [
                1,
                2,
                3,
            ],
        },
    )

    result = _add_end_of_period_denominator(
        df,
    )

    assert result[_DENOM_COL].to_list() == [
        1.0,
        1.0,
        1.0,
    ]


def test_numeric_dimensions_are_cast_to_strings() -> None:
    """Numeric inequality dimensions support the string all label."""
    df = pl.DataFrame(
        {
            "event": [
                0,
                1,
                0,
                1,
            ],
            "code": [
                1,
                1,
                2,
                2,
            ],
        },
    )

    result = crude_proportion_df(
        df,
        event_col="event",
        inequalities_cols=["code"],
    )

    assert result.schema["code"] == pl.String

    assert set(
        result["code"].to_list(),
    ) == {
        "1",
        "2",
        "All",
    }


# ===========================================================================
# Crude proportions
# ===========================================================================


def test_crude_proportion_point_estimate_and_interval() -> None:
    """The crude proportion uses the observed estimate and Wilson limits."""
    df = pl.DataFrame(
        {
            "event": [
                0,
                1,
                1,
                0,
            ],
        },
    )

    result = crude_proportion_df(
        df,
        event_col="event",
    )

    row = result.row(
        0,
        named=True,
    )

    expected_lower, expected_upper = reference_wilson(
        2,
        4,
    )

    assert row["events"] == 2
    assert row["n"] == 4
    assert row["proportion"] == pytest.approx(0.5)

    assert row["lower"] == pytest.approx(
        expected_lower,
    )

    assert row["upper"] == pytest.approx(
        expected_upper,
    )

    assert row["method"] == "Wilson score"
    assert row["confidence"] == pytest.approx(0.95)
    assert row["significance"] == "Reference"


@pytest.mark.parametrize(
    ("events", "expected"),
    [
        (
            [0] * 10,
            0.0,
        ),
        (
            [1] * 10,
            1.0,
        ),
    ],
)
def test_crude_proportion_boundary_estimates(
    events: list[int],
    expected: float,
) -> None:
    """Wilson intervals remain informative at zero and one."""
    result = crude_proportion_df(
        pl.DataFrame(
            {
                "event": events,
            },
        ),
        event_col="event",
    )

    row = result.row(
        0,
        named=True,
    )

    tolerance = 1e-12

    assert row["proportion"] == pytest.approx(
        expected,
        abs=tolerance,
    )

    assert row["lower"] <= expected + tolerance
    assert row["upper"] >= expected - tolerance
    assert row["upper"] > row["lower"]


def test_crude_proportion_significance_labels() -> None:
    """Groups are classified relative to the fixed overall reference."""
    df = pl.DataFrame(
        {
            "event": [
                *[1] * 20,
                *[0] * 80,
            ],
            "group": [
                *["A"] * 20,
                *["B"] * 80,
            ],
        },
    )

    result = crude_proportion_df(
        df,
        event_col="event",
        inequalities_cols=["group"],
    )

    assert (
        get_row(
            result,
            group="A",
        )["significance"]
        == "Higher"
    )

    assert (
        get_row(
            result,
            group="B",
        )["significance"]
        == "Lower"
    )

    assert (
        get_row(
            result,
            group="All",
        )["significance"]
        == "Reference"
    )


def test_crude_proportion_output_schema() -> None:
    """Crude proportion output contains its documented metadata."""
    result = crude_proportion_df(
        pl.DataFrame(
            {
                "event": [
                    0,
                    1,
                ],
            },
        ),
        event_col="event",
    )

    assert result.columns == [
        "events",
        "n",
        "proportion",
        "lower",
        "upper",
        "confidence",
        "method",
        "action",
        "notes",
        "significance",
    ]


# ===========================================================================
# Crude rates
# ===========================================================================


def test_crude_rate_uses_exact_interval_below_ten() -> None:
    """Crude rates use exact Poisson limits below ten events."""
    df = pl.DataFrame(
        {
            "count": [
                1,
                0,
                2,
            ],
        },
    )

    result = crude_rate_df(
        df,
        event_col="count",
        multiplier=1_000,
    )

    row = result.row(
        0,
        named=True,
    )

    expected_lower, expected_upper = reference_exact_poisson(
        3,
    )

    assert row["events"] == 3
    assert row["denominator"] == 3
    assert row["rate"] == pytest.approx(
        1_000,
    )

    assert row["lower"] == pytest.approx(
        expected_lower / 3 * 1_000,
    )

    assert row["upper"] == pytest.approx(
        expected_upper / 3 * 1_000,
    )

    assert row["method"] == "Exact chi-square"


def test_crude_rate_uses_byar_from_ten_events() -> None:
    """Crude rates use Byar limits from ten events."""
    df = pl.DataFrame(
        {
            "count": [
                1,
            ]
            * 10,
        },
    )

    result = crude_rate_df(
        df,
        event_col="count",
        multiplier=1_000,
    )

    row = result.row(
        0,
        named=True,
    )

    expected_lower, expected_upper = reference_byar(
        10,
    )

    assert row["rate"] == pytest.approx(
        1_000,
    )

    assert row["lower"] == pytest.approx(
        expected_lower / 10 * 1_000,
    )

    assert row["upper"] == pytest.approx(
        expected_upper / 10 * 1_000,
    )

    assert row["method"] == "Byar"


def test_crude_rate_zero_events_has_positive_upper_limit() -> None:
    """A zero crude rate retains an exact positive upper limit."""
    result = crude_rate_df(
        pl.DataFrame(
            {
                "count": [
                    0,
                    0,
                    0,
                ],
            },
        ),
        event_col="count",
    )

    row = result.row(
        0,
        named=True,
    )

    assert row["rate"] == 0
    assert row["lower"] == 0
    assert row["upper"] > 0
    assert "Zero event count" in row["notes"]


def test_crude_rate_records_denominator_assumption() -> None:
    """Rate notes document the row-count exposure assumption."""
    result = crude_rate_df(
        pl.DataFrame(
            {
                "count": [
                    0,
                    1,
                ],
            },
        ),
        event_col="count",
    )

    row = result.row(
        0,
        named=True,
    )

    assert "End-of-period denominator" in row["notes"]


def test_crude_rate_output_schema() -> None:
    """Crude rate output contains method and scaling metadata."""
    result = crude_rate_df(
        pl.DataFrame(
            {
                "count": [
                    0,
                    1,
                ],
            },
        ),
        event_col="count",
    )

    assert result.columns == [
        "events",
        "denominator",
        "rate",
        "lower",
        "upper",
        "multiplier",
        "confidence",
        "method",
        "action",
        "notes",
        "significance",
    ]


# ===========================================================================
# Organisational and inequality outputs
# ===========================================================================


def test_complete_inequality_cube(
    analytical_df: pl.DataFrame,
) -> None:
    """Two fully crossed inequalities produce nine output rows."""
    result = crude_proportion_df(
        analytical_df,
        event_col="event",
        inequalities_cols=[
            "sex",
            "ethnicity",
        ],
    )

    assert result.height == 9

    assert (
        get_row(
            result,
            sex="All",
            ethnicity="All",
        )["significance"]
        == "Reference"
    )


def test_organisational_hierarchy_rollups(
    analytical_df: pl.DataFrame,
) -> None:
    """A three-level hierarchy produces valid prefix roll-ups."""
    result = crude_proportion_df(
        analytical_df,
        event_col="event",
        organisational_cols={
            "geography": [
                "region",
                "icb",
                "practice",
            ],
        },
    )

    assert result.height == 9

    assert result.filter(pl.col("practice") != "All").height == 4

    assert (
        result.filter((pl.col("icb") != "All") & (pl.col("practice") == "All")).height
        == 2
    )

    assert (
        result.filter(
            (pl.col("region") != "All")
            & (pl.col("icb") == "All")
            & (pl.col("practice") == "All")
        ).height
        == 2
    )


def test_separate_and_cross_organisational_modes() -> None:
    """Cross mode emits intersections across independent hierarchies."""
    df = pl.DataFrame(
        {
            "event": [
                0,
                1,
                0,
                1,
            ],
            "region": [
                "R1",
                "R1",
                "R2",
                "R2",
            ],
            "provider": [
                "A",
                "B",
                "A",
                "B",
            ],
        },
    )

    hierarchies = {
        "geography": ["region"],
        "provider": ["provider"],
    }

    separate = crude_proportion_df(
        df,
        event_col="event",
        organisational_cols=hierarchies,
        organisational_mode="separate",
    )

    crossed = crude_proportion_df(
        df,
        event_col="event",
        organisational_cols=hierarchies,
        organisational_mode="cross",
    )

    assert separate.height == 5
    assert crossed.height == 9


# ===========================================================================
# Directly standardised proportions
# ===========================================================================


def test_dsp_matches_independent_mover_calculation() -> None:
    """The public DSP matches an independent Wilson-MOVER calculation."""
    df = pl.DataFrame(
        {
            "event": [
                1,
                0,
                0,
                0,
                1,
                1,
                1,
                0,
            ],
            "age_band": [
                *["A"] * 4,
                *["B"] * 4,
            ],
        },
    )

    result = directly_standardized_proportion_df(
        df,
        event_col="event",
        strata_cols=["age_band"],
    )

    row = result.row(
        0,
        named=True,
    )

    expected = reference_mover_proportion(
        weights=np.array(
            [0.5, 0.5],
        ),
        events=np.array(
            [1.0, 3.0],
        ),
        denominators=np.array(
            [4.0, 4.0],
        ),
    )

    assert row["dsp"] == pytest.approx(
        expected[0],
        abs=1e-6,
    )

    assert row["dsp_lower"] == pytest.approx(
        expected[1],
        abs=1e-6,
    )

    assert row["dsp_upper"] == pytest.approx(
        expected[2],
        abs=1e-6,
    )

    assert row["method"] == "Wilson-MOVER"
    assert row["significance"] == "Reference"


def test_dsp_missing_stratum_is_renormalised_and_noted(
    missing_strata_df: pl.DataFrame,
) -> None:
    """Missing DSP strata are omitted and remaining weights renormalised."""
    result = directly_standardized_proportion_df(
        missing_strata_df,
        event_col="event",
        strata_cols=["age_band"],
        inequalities_cols=["group"],
    )

    row = get_row(
        result,
        group="G1",
    )

    assert row["dsp"] == 0
    assert row["dsp_upper"] > 0

    assert "Calculated with missing standardisation stratum" in row["notes"]

    assert "33.3%" in row["notes"]
    assert "renormalised" in row["notes"]


def test_dsp_reference_equals_overall_observed_proportion(
    missing_strata_df: pl.DataFrame,
) -> None:
    """Internal reference weights reproduce the overall proportion."""
    result = directly_standardized_proportion_df(
        missing_strata_df,
        event_col="event",
        strata_cols=["age_band"],
        inequalities_cols=["group"],
    )

    reference = get_row(
        result,
        group="All",
    )

    assert reference["dsp"] == pytest.approx(
        3 / 6,
    )

    assert reference["significance"] == "Reference"
    assert "Reference population" in reference["notes"]


def test_dsp_zero_events_is_not_haldane_corrected() -> None:
    """A zero-event population retains a DSP point estimate of zero."""
    df = pl.DataFrame(
        {
            "event": [
                0,
                0,
                0,
                0,
            ],
            "age_band": [
                "A",
                "A",
                "B",
                "B",
            ],
        },
    )

    result = directly_standardized_proportion_df(
        df,
        event_col="event",
        strata_cols=["age_band"],
    )

    row = result.row(
        0,
        named=True,
    )

    assert row["dsp"] == 0
    assert row["dsp_lower"] == 0
    assert row["dsp_upper"] > 0
    assert "Haldane" not in row["notes"]


def test_dsp_all_events_retains_estimate_one() -> None:
    """A population containing only events retains a DSP of one."""
    df = pl.DataFrame(
        {
            "event": [
                1,
                1,
                1,
                1,
            ],
            "age_band": [
                "A",
                "A",
                "B",
                "B",
            ],
        },
    )

    result = directly_standardized_proportion_df(
        df,
        event_col="event",
        strata_cols=["age_band"],
    )

    row = result.row(
        0,
        named=True,
    )

    assert row["dsp"] == 1
    assert row["dsp_lower"] < 1
    assert row["dsp_upper"] == 1


def test_dsp_stratum_helper_reports_missing_weight(
    missing_strata_df: pl.DataFrame,
) -> None:
    """The DSP helper returns missing-stratum metadata."""
    reference_weights = pl.DataFrame(
        {
            "age_band": [
                "A",
                "B",
            ],
            "ref_count": [
                4,
                2,
            ],
            "ref_weight": [
                4 / 6,
                2 / 6,
            ],
        },
    )

    group = missing_strata_df.filter(
        pl.col("group") == "G1",
    )

    result = _compute_proportion_stratum_stats(
        group_data=group,
        all_strata=reference_weights.select(
            "age_band",
        ),
        event_col="event",
        strata_cols=["age_band"],
        reference_weights=reference_weights,
    )

    assert result["missing_strata_count"] == 1

    assert result["missing_reference_weight"] == pytest.approx(
        1 / 3,
    )

    assert result["observed_reference_weight"] == pytest.approx(
        2 / 3,
    )

    assert np.sum(
        result["normalised_weights"],
    ) == pytest.approx(1)


def test_dsp_output_schema() -> None:
    """DSP output reports interval methodology and confidence."""
    df = pl.DataFrame(
        {
            "event": [
                0,
                1,
            ],
            "age_band": [
                "A",
                "B",
            ],
        },
    )

    result = directly_standardized_proportion_df(
        df,
        event_col="event",
        strata_cols=["age_band"],
    )

    assert result.columns == [
        "events",
        "n",
        "dsp",
        "dsp_lower",
        "dsp_upper",
        "confidence",
        "method",
        "action",
        "notes",
        "significance",
    ]


# ===========================================================================
# Directly standardised rates
# ===========================================================================


def test_dsr_matches_independent_mover_calculation() -> None:
    """The public DSR matches an independent Poisson-MOVER calculation."""
    df = pl.DataFrame(
        {
            "count": [
                1,
                1,
                1,
                1,
                3,
                3,
                3,
                3,
            ],
            "age_band": [
                *["A"] * 4,
                *["B"] * 4,
            ],
        },
    )

    multiplier = 1_000.0

    result = directly_standardized_rate_df(
        df,
        event_col="count",
        strata_cols=["age_band"],
        multiplier=multiplier,
    )

    row = result.row(
        0,
        named=True,
    )

    expected = reference_mover_rate(
        weights=np.array(
            [0.5, 0.5],
        ),
        events=np.array(
            [4.0, 12.0],
        ),
        denominators=np.array(
            [4.0, 4.0],
        ),
    )

    assert row["dsr"] == pytest.approx(
        expected[0] * multiplier,
        abs=1e-6,
    )

    assert row["dsr_lower"] == pytest.approx(
        expected[1] * multiplier,
        abs=1e-6,
    )

    assert row["dsr_upper"] == pytest.approx(
        expected[2] * multiplier,
        abs=1e-6,
    )

    assert row["method"] == "Poisson-MOVER using exact and Byar stratum intervals"


def test_dsr_missing_stratum_is_renormalised_and_noted(
    missing_strata_df: pl.DataFrame,
) -> None:
    """Missing DSR strata are omitted and remaining weights renormalised."""
    result = directly_standardized_rate_df(
        missing_strata_df,
        event_col="count",
        strata_cols=["age_band"],
        inequalities_cols=["group"],
        multiplier=1_000,
    )

    row = get_row(
        result,
        group="G1",
    )

    assert row["dsr"] == 0
    assert row["dsr_lower"] == 0
    assert row["dsr_upper"] > 0

    assert "Calculated with missing standardisation stratum" in row["notes"]

    assert "33.3%" in row["notes"]
    assert "renormalised" in row["notes"]


def test_dsr_zero_events_is_not_haldane_corrected() -> None:
    """A zero-event population retains a DSR point estimate of zero."""
    df = pl.DataFrame(
        {
            "count": [
                0,
                0,
                0,
                0,
            ],
            "age_band": [
                "A",
                "A",
                "B",
                "B",
            ],
        },
    )

    result = directly_standardized_rate_df(
        df,
        event_col="count",
        strata_cols=["age_band"],
    )

    row = result.row(
        0,
        named=True,
    )

    assert row["dsr"] == 0
    assert row["dsr_lower"] == 0
    assert row["dsr_upper"] > 0
    assert "Haldane" not in row["notes"]
    assert "Zero total events" in row["notes"]


def test_dsr_below_ten_events_is_returned_but_flagged() -> None:
    """Low-count DSRs are calculated rather than suppressed."""
    df = pl.DataFrame(
        {
            "count": [
                0,
                1,
                0,
                1,
            ],
            "age_band": [
                "A",
                "A",
                "B",
                "B",
            ],
        },
    )

    result = directly_standardized_rate_df(
        df,
        event_col="count",
        strata_cols=["age_band"],
    )

    row = result.row(
        0,
        named=True,
    )

    assert np.isfinite(row["dsr"])
    assert np.isfinite(row["dsr_lower"])
    assert np.isfinite(row["dsr_upper"])

    assert "Low total event count (<10)" in row["notes"]


def test_dsr_reference_equals_overall_crude_rate(
    missing_strata_df: pl.DataFrame,
) -> None:
    """Internal weights reproduce the overall rate for the reference row."""
    result = directly_standardized_rate_df(
        missing_strata_df,
        event_col="count",
        strata_cols=["age_band"],
        inequalities_cols=["group"],
        multiplier=1_000,
    )

    reference = get_row(
        result,
        group="All",
    )

    expected = (
        sum(
            missing_strata_df["count"],
        )
        / missing_strata_df.height
        * 1_000
    )

    assert reference["dsr"] == pytest.approx(
        expected,
    )

    assert reference["significance"] == "Reference"
    assert "Reference population" in reference["notes"]


def test_dsr_stratum_helper_reports_missing_weight(
    missing_strata_df: pl.DataFrame,
) -> None:
    """The DSR helper returns missing-stratum metadata."""
    working = _add_end_of_period_denominator(
        missing_strata_df,
    )

    reference_weights = pl.DataFrame(
        {
            "age_band": [
                "A",
                "B",
            ],
            "ref_count": [
                4,
                2,
            ],
            "ref_weight": [
                4 / 6,
                2 / 6,
            ],
        },
    )

    group = working.filter(
        pl.col("group") == "G1",
    )

    result = _compute_rate_stratum_stats(
        group_data=group,
        all_strata=reference_weights.select(
            "age_band",
        ),
        event_col="count",
        denominator_col=_DENOM_COL,
        strata_cols=["age_band"],
        reference_weights=reference_weights,
    )

    assert result["missing_strata_count"] == 1

    assert result["missing_reference_weight"] == pytest.approx(
        1 / 3,
    )

    assert result["observed_reference_weight"] == pytest.approx(
        2 / 3,
    )

    assert np.sum(
        result["normalised_weights"],
    ) == pytest.approx(1)


def test_dsr_output_schema() -> None:
    """DSR output reports scaling and interval methodology."""
    df = pl.DataFrame(
        {
            "count": [
                0,
                1,
            ],
            "age_band": [
                "A",
                "B",
            ],
        },
    )

    result = directly_standardized_rate_df(
        df,
        event_col="count",
        strata_cols=["age_band"],
    )

    assert result.columns == [
        "events",
        "denominator",
        "dsr",
        "dsr_lower",
        "dsr_upper",
        "multiplier",
        "confidence",
        "method",
        "action",
        "notes",
        "significance",
    ]


# ===========================================================================
# Shared reporting helpers
# ===========================================================================


@pytest.mark.parametrize(
    (
        "reference",
        "lower",
        "upper",
        "expected",
    ),
    [
        (
            0.5,
            0.6,
            0.8,
            "Higher",
        ),
        (
            0.5,
            0.1,
            0.4,
            "Lower",
        ),
        (
            0.5,
            0.4,
            0.6,
            "Not significant",
        ),
        (
            np.nan,
            0.4,
            0.6,
            "Not tested",
        ),
        (
            0.5,
            np.nan,
            np.nan,
            "Not tested",
        ),
    ],
)
def test_significance_classification(
    reference: float,
    lower: float,
    upper: float,
    expected: str,
) -> None:
    """Fixed-reference interval classification is deterministic."""
    assert (
        _significance_from_ci(
            reference,
            lower,
            upper,
        )
        == expected
    )


def test_missing_strata_note_singular() -> None:
    """One omitted stratum uses singular wording."""
    note = _missing_strata_note(
        missing_strata_count=1,
        missing_reference_weight=0.25,
        observed_reference_weight=0.75,
    )

    assert "1 stratum omitted" in note
    assert "25.0%" in note
    assert "75.0% coverage" in note


def test_missing_strata_note_plural() -> None:
    """Multiple omitted strata use plural wording."""
    note = _missing_strata_note(
        missing_strata_count=2,
        missing_reference_weight=0.4,
        observed_reference_weight=0.6,
    )

    assert "2 strata omitted" in note
    assert "40.0%" in note
    assert "60.0% coverage" in note


# ===========================================================================
# Integration tests
# ===========================================================================


@pytest.mark.parametrize(
    (
        "function",
        "event_col",
        "extra_kwargs",
    ),
    [
        (
            crude_proportion_df,
            "event",
            {},
        ),
        (
            crude_rate_df,
            "count",
            {},
        ),
        (
            directly_standardized_proportion_df,
            "event",
            {
                "strata_cols": [
                    "age_band",
                ],
            },
        ),
        (
            directly_standardized_rate_df,
            "count",
            {
                "strata_cols": [
                    "age_band",
                ],
            },
        ),
    ],
)
def test_all_functions_support_dimensions(
    analytical_df: pl.DataFrame,
    function: Callable[..., pl.DataFrame],
    event_col: str,
    extra_kwargs: dict[str, object],
) -> None:
    """All public functions support inequalities and organisations."""
    result = function(
        analytical_df,
        event_col=event_col,
        inequalities_cols=["sex"],
        organisational_cols={
            "geography": [
                "region",
                "icb",
            ],
        },
        **extra_kwargs,
    )

    assert result.height > 1

    assert {
        "region",
        "icb",
        "sex",
        "action",
        "notes",
        "significance",
    } <= set(result.columns)

    reference = get_row(
        result,
        region="All",
        icb="All",
        sex="All",
    )

    assert reference["significance"] == "Reference"


@pytest.mark.parametrize(
    (
        "function",
        "event_col",
        "strata_cols",
        "estimate_col",
        "lower_col",
        "upper_col",
    ),
    [
        (
            crude_proportion_df,
            "event",
            None,
            "proportion",
            "lower",
            "upper",
        ),
        (
            crude_rate_df,
            "count",
            None,
            "rate",
            "lower",
            "upper",
        ),
        (
            directly_standardized_proportion_df,
            "event",
            ["age_band"],
            "dsp",
            "dsp_lower",
            "dsp_upper",
        ),
        (
            directly_standardized_rate_df,
            "count",
            ["age_band"],
            "dsr",
            "dsr_lower",
            "dsr_upper",
        ),
    ],
)
def test_estimates_lie_within_confidence_intervals(
    analytical_df: pl.DataFrame,
    function: Callable[..., pl.DataFrame],
    event_col: str,
    strata_cols: list[str] | None,
    estimate_col: str,
    lower_col: str,
    upper_col: str,
) -> None:
    """Every finite point estimate lies within its confidence interval."""
    kwargs: dict[str, object] = {
        "event_col": event_col,
        "inequalities_cols": ["sex"],
    }

    if strata_cols is not None:
        kwargs["strata_cols"] = strata_cols

    result = function(
        analytical_df,
        **kwargs,
    )

    finite = result.filter(
        pl.col(estimate_col).is_finite()
        & pl.col(lower_col).is_finite()
        & pl.col(upper_col).is_finite(),
    )

    assert finite.height > 0

    assert finite.select(
        (pl.col(lower_col) <= pl.col(estimate_col)).all(),
    ).item()

    assert finite.select(
        (pl.col(estimate_col) <= pl.col(upper_col)).all(),
    ).item()


# ===========================================================================
# Action column
# ===========================================================================

VALID_ACTIONS = {
    "Show",
    "Flag",
    "Suppress",
}


@pytest.mark.parametrize(
    ("events", "n", "expected"),
    [
        # one case per row of the approved rules table (CP01-CP14)
        (0, 0, "Suppress"),  # CP01 empty group
        (20, 100, "Show"),  # CP02 clean
        (0, 100, "Flag"),  # CP03 zero events, large n
        (5, 100, "Flag"),  # CP04 few events, large n
        (100, 100, "Flag"),  # CP05 all events, large n
        (95, 100, "Flag"),  # CP06 few non-events, large n
        (15, 30, "Flag"),  # CP07 small n only
        (0, 20, "Suppress"),  # CP08 small n, zero events
        (5, 30, "Suppress"),  # CP09 small n, few events
        (20, 20, "Suppress"),  # CP10 small n, all events
        (25, 30, "Suppress"),  # CP11 small n, few non-events
        (0, 5, "Suppress"),  # CP12 tiny n, zero events
        (5, 5, "Suppress"),  # CP13 tiny n, all events
        (4, 9, "Suppress"),  # CP14 tiny n, both counts low
        # boundaries
        (5, 40, "Flag"),  # n = 40 counts as an adequate sample
        (20, 40, "Show"),
        (10, 39, "Flag"),  # events = 10 is not low
        (9, 39, "Suppress"),
    ],
)
def test_crude_proportion_action_rules(
    events: float,
    n: float,
    expected: str,
) -> None:
    """Crude proportion actions follow the approved rules table."""
    assert _crude_proportion_action(events, n) == expected


@pytest.mark.parametrize(
    ("events", "denominator", "expected"),
    [
        # one case per row of the approved rules table (CR01-CR07)
        (0, 0, "Suppress"),  # CR01 zero denominator
        (20, 1000, "Show"),  # CR02 clean
        (0, 1000, "Flag"),  # CR03 zero events, large denominator
        (5, 1000, "Flag"),  # CR04 few events, large denominator
        (12, 30, "Flag"),  # CR05 small denominator only
        (0, 30, "Suppress"),  # CR06 small denominator, zero events
        (5, 30, "Suppress"),  # CR07 small denominator, few events
        # boundaries
        (10, 40, "Show"),
        (9, 40, "Flag"),
        (10, 39, "Flag"),
        (9, 39, "Suppress"),
    ],
)
def test_crude_rate_action_rules(
    events: float,
    denominator: float,
    expected: str,
) -> None:
    """Crude rate actions follow the approved rules table."""
    assert _crude_rate_action(events, denominator) == expected


@pytest.mark.parametrize(
    ("events", "n", "missing_strata_count", "stratum_ns", "coverage", "expected"),
    [
        (30, 100, 0, [50, 50], 1.0, "Show"),  # no warnings
        (5, 100, 0, [50, 50], 1.0, "Flag"),  # low events only
        (95, 100, 0, [50, 50], 1.0, "Flag"),  # low non-events only
        (15, 30, 0, [15, 15], 1.0, "Flag"),  # n < 40 only
        (30, 100, 1, [0, 100], 0.95, "Flag"),  # missing strata only
        (30, 100, 0, [95, 5], 1.0, "Flag"),  # one small stratum only
        (5, 30, 0, [15, 15], 1.0, "Suppress"),  # n < 40 and low events
        (30, 100, 1, [0, 95, 5], 0.95, "Suppress"),  # missing + small stratum
        (30, 100, 1, [0, 100], 0.80, "Suppress"),  # coverage < 90%
        (12, 19, 0, [10, 9], 1.0, "Suppress"),  # n < 20
        (0, 0, 0, [], 1.0, "Suppress"),  # empty group
        # boundaries
        (30, 100, 1, [0, 100], 0.90, "Flag"),  # coverage of exactly 90% is allowed
        (10, 20, 0, [10, 10], 1.0, "Flag"),  # n = 20 is not < 20, so only the n < 40 warning
        (10, 40, 0, [20, 20], 1.0, "Show"),  # n = 40 and counts of 10 are not low
    ],
)
def test_ds_proportion_action_rules(
    events: float,
    n: float,
    missing_strata_count: int,
    stratum_ns: list[float],
    coverage: float,
    expected: str,
) -> None:
    """DSP actions: hard suppression first, then count the warnings."""
    action = _ds_proportion_action(
        events=events,
        n=n,
        missing_strata_count=missing_strata_count,
        raw_stratum_ns=stratum_ns,
        observed_reference_weight=coverage,
    )

    assert action == expected


@pytest.mark.parametrize(
    ("events", "denominator", "missing_strata_count", "stratum_denominators", "coverage", "expected"),
    [
        (50, 1000, 0, [500, 500], 1.0, "Show"),  # no warnings
        (5, 1000, 0, [500, 500], 1.0, "Flag"),  # low events only
        (50, 30, 0, [15, 15], 1.0, "Flag"),  # denominator < 40 only
        (50, 1000, 1, [0, 1000], 0.80, "Flag"),  # missing strata only (low coverage alone does not suppress)
        (50, 1000, 0, [995, 5], 1.0, "Flag"),  # one small stratum only
        (5, 30, 0, [15, 15], 1.0, "Suppress"),  # low events and denominator < 40
        (50, 1000, 1, [0, 995, 5], 0.95, "Suppress"),  # missing + small stratum
        (50, 15, 1, [0, 15], 0.80, "Suppress"),  # denominator < 20 and coverage < 90%
        (50, 15, 0, [15], 1.0, "Flag"),  # denominator < 20 on its own is one warning
        # boundaries
        (10, 40, 0, [20, 20], 1.0, "Show"),
        (9, 40, 0, [20, 20], 1.0, "Flag"),
    ],
)
def test_ds_rate_action_rules(
    events: float,
    denominator: float,
    missing_strata_count: int,
    stratum_denominators: list[float],
    coverage: float,
    expected: str,
) -> None:
    """DSR actions: hard suppression first, then count the warnings."""
    action = _ds_rate_action(
        events=events,
        denominator=denominator,
        missing_strata_count=missing_strata_count,
        raw_stratum_denominators=stratum_denominators,
        observed_reference_weight=coverage,
    )

    assert action == expected


@pytest.mark.parametrize(
    ("events", "expected"),
    [
        ([1] * 20 + [0] * 80, "Show"),  # 20 / 100
        ([1] * 5 + [0] * 95, "Flag"),  # 5 / 100
        ([1] * 2 + [0] * 2, "Suppress"),  # 2 / 4
    ],
)
def test_crude_proportion_df_returns_action(
    events: list[int],
    expected: str,
) -> None:
    """The public crude proportion function fills the action column."""
    result = crude_proportion_df(
        pl.DataFrame({"event": events}),
        event_col="event",
    )

    assert result.row(0, named=True)["action"] == expected


@pytest.mark.parametrize(
    ("counts", "expected"),
    [
        ([1] * 50, "Show"),  # 50 events / 50 rows
        ([1] * 5 + [0] * 95, "Flag"),  # 5 events / 100 rows
        ([1, 0, 2], "Suppress"),  # 3 events / 3 rows
    ],
)
def test_crude_rate_df_returns_action(
    counts: list[int],
    expected: str,
) -> None:
    """The public crude rate function fills the action column."""
    result = crude_rate_df(
        pl.DataFrame({"count": counts}),
        event_col="count",
    )

    assert result.row(0, named=True)["action"] == expected


@pytest.mark.parametrize(
    ("events", "expected"),
    [
        # stratum A: 50 rows, stratum B: 50 rows, 40 events -> no warnings
        ([1] * 20 + [0] * 30 + [1] * 20 + [0] * 30, "Show"),
        # 100 rows but only 5 events -> one warning
        ([1] * 5 + [0] * 45 + [0] * 50, "Flag"),
    ],
)
def test_dsp_df_returns_action(
    events: list[int],
    expected: str,
) -> None:
    """The public DSP function fills the action column."""
    df = pl.DataFrame(
        {
            "event": events,
            "age_band": ["A"] * 50 + ["B"] * 50,
        },
    )

    result = directly_standardized_proportion_df(
        df,
        event_col="event",
        strata_cols=["age_band"],
    )

    assert result.row(0, named=True)["action"] == expected


def test_dsp_df_suppresses_small_missing_strata_group(
    missing_strata_df: pl.DataFrame,
) -> None:
    """A tiny group with 67% coverage is suppressed."""
    result = directly_standardized_proportion_df(
        missing_strata_df,
        event_col="event",
        strata_cols=["age_band"],
        inequalities_cols=["group"],
    )

    assert get_row(result, group="G1")["action"] == "Suppress"


@pytest.mark.parametrize(
    ("counts", "expected"),
    [
        # 100 rows, 100 events -> no warnings
        ([1] * 100, "Show"),
        # 100 rows but only 5 events -> one warning
        ([1] * 5 + [0] * 95, "Flag"),
    ],
)
def test_dsr_df_returns_action(
    counts: list[int],
    expected: str,
) -> None:
    """The public DSR function fills the action column."""
    df = pl.DataFrame(
        {
            "count": counts,
            "age_band": ["A"] * 50 + ["B"] * 50,
        },
    )

    result = directly_standardized_rate_df(
        df,
        event_col="count",
        strata_cols=["age_band"],
    )

    assert result.row(0, named=True)["action"] == expected


def test_dsr_df_suppresses_small_missing_strata_group(
    missing_strata_df: pl.DataFrame,
) -> None:
    """A tiny group with 67% coverage is suppressed."""
    result = directly_standardized_rate_df(
        missing_strata_df,
        event_col="count",
        strata_cols=["age_band"],
        inequalities_cols=["group"],
    )

    assert get_row(result, group="G1")["action"] == "Suppress"


@pytest.mark.parametrize(
    ("function", "event_col", "extra_kwargs"),
    [
        (crude_proportion_df, "event", {}),
        (crude_rate_df, "count", {}),
        (directly_standardized_proportion_df, "event", {"strata_cols": ["age_band"]}),
        (directly_standardized_rate_df, "count", {"strata_cols": ["age_band"]}),
    ],
)
def test_all_functions_return_valid_actions(
    analytical_df: pl.DataFrame,
    function: Callable[..., pl.DataFrame],
    event_col: str,
    extra_kwargs: dict[str, object],
) -> None:
    """Every row of every function has an action of Show, Flag or Suppress."""
    result = function(
        analytical_df,
        event_col=event_col,
        inequalities_cols=["sex"],
        **extra_kwargs,
    )

    assert set(result["action"].to_list()) <= VALID_ACTIONS
