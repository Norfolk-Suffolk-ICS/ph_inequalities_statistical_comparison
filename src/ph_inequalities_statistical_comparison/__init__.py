"""
Public API for ph_inequalities_statistical_comparison.

The package exposes functions for calculating crude and directly
standardised proportions and rates across inequality and organisational
dimensions.
"""

from .core import (
    crude_proportion_df,
    crude_rate_df,
    directly_standardized_proportion_df,
    directly_standardized_rate_df,
)

__all__ = [
    "crude_proportion_df",
    "crude_rate_df",
    "directly_standardized_proportion_df",
    "directly_standardized_rate_df",
]
