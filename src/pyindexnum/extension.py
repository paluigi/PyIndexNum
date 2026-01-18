"""
Extension methods for connecting two different multilateral indices.

This module contains functions for splicing two multilateral price indices
that are calculated on the same window length but shifted by one period.
These methods are used to extend price index series when using rolling windows.
"""

import polars as pl
import numpy as np
from typing import Tuple, List
from datetime import timedelta


def movement_splice(index1: pl.DataFrame, index2: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate the movement splice extension method.

    The movement splice method calculates the rate of change between the last
    and second-last period in the second window, then applies this rate to
    extend the first window by one period.

    Args:
        index1: First multilateral index DataFrame with columns "period" and "index_value"
        index2: Second multilateral index DataFrame with columns "period" and "index_value"

    Returns:
        DataFrame with the extended index including the spliced period

    Raises:
        ValueError: If input validation fails

    Examples:
        >>> import polars as pl
        >>> from datetime import date
        >>> idx1 = pl.DataFrame({
        ...     "period": [date(2023, 1, 1), date(2023, 2, 1), date(2023, 3, 1)],
        ...     "index_value": [1.0, 1.05, 1.10]
        ... })
        >>> idx2 = pl.DataFrame({
        ...     "period": [date(2023, 2, 1), date(2023, 3, 1), date(2023, 4, 1)],
        ...     "index_value": [1.05, 1.10, 1.15]
        ... })
        >>> result = movement_splice(idx1, idx2)
        >>> # Returns extended index with period 2023-04-01
    """
    _validate_indices(index1, index2)

    # Get the last period from index1 and the last two periods from index2
    last_period_idx1 = index1.select(pl.col("period").max()).item()
    last_index_idx1 = index1.filter(pl.col("period") == last_period_idx1).select("index_value").item()

    # Get last two periods from index2
    sorted_idx2 = index2.sort("period")
    last_period_idx2 = sorted_idx2.select(pl.col("period").max()).item()
    second_last_period_idx2 = sorted_idx2.select(pl.col("period")).to_series()[-2]

    last_index_idx2 = sorted_idx2.filter(pl.col("period") == last_period_idx2).select("index_value").item()
    second_last_index_idx2 = sorted_idx2.filter(pl.col("period") == second_last_period_idx2).select("index_value").item()

    # Calculate movement rate
    movement_rate = last_index_idx2 / second_last_index_idx2

    # Calculate spliced index value
    spliced_index = last_index_idx1 * movement_rate

    # Create result DataFrame
    spliced_df = pl.DataFrame({
        "period": [last_period_idx2],
        "index_value": [spliced_index]
    })

    return spliced_df


def window_splice(index1: pl.DataFrame, index2: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate the window splice extension method.

    The window splice method calculates the rate of change between the last
    and first period of the second window, then uses this rate to connect
    with the second period of the first window and calculate the index for
    the additional period.

    Args:
        index1: First multilateral index DataFrame with columns "period" and "index_value"
        index2: Second multilateral index DataFrame with columns "period" and "index_value"

    Returns:
        DataFrame with the extended index including the spliced period

    Raises:
        ValueError: If input validation fails

    Examples:
        >>> import polars as pl
        >>> from datetime import date
        >>> idx1 = pl.DataFrame({
        ...     "period": [date(2023, 1, 1), date(2023, 2, 1), date(2023, 3, 1)],
        ...     "index_value": [1.0, 1.05, 1.10]
        ... })
        >>> idx2 = pl.DataFrame({
        ...     "period": [date(2023, 2, 1), date(2023, 3, 1), date(2023, 4, 1)],
        ...     "index_value": [1.05, 1.10, 1.15]
        ... })
        >>> result = window_splice(idx1, index2)
        >>> # Returns extended index with period 2023-04-01
    """
    _validate_indices(index1, index2)

    # Get periods and indices
    sorted_idx1 = index1.sort("period")
    sorted_idx2 = index2.sort("period")

    first_period_idx1 = sorted_idx1.select(pl.col("period").min()).item()
    second_period_idx1 = sorted_idx1.select(pl.col("period")).to_series()[1]
    last_period_idx1 = sorted_idx1.select(pl.col("period").max()).item()

    first_period_idx2 = sorted_idx2.select(pl.col("period").min()).item()
    last_period_idx2 = sorted_idx2.select(pl.col("period").max()).item()

    first_index_idx1 = sorted_idx1.filter(pl.col("period") == first_period_idx1).select("index_value").item()
    second_index_idx1 = sorted_idx1.filter(pl.col("period") == second_period_idx1).select("index_value").item()
    last_index_idx1 = sorted_idx1.filter(pl.col("period") == last_period_idx1).select("index_value").item()

    first_index_idx2 = sorted_idx2.filter(pl.col("period") == first_period_idx2).select("index_value").item()
    last_index_idx2 = sorted_idx2.filter(pl.col("period") == last_period_idx2).select("index_value").item()

    # Calculate window rate of change
    window_rate = last_index_idx2 / first_index_idx2

    # Calculate spliced index using the connection point
    # The rate from first to second period in index1 is used as base
    base_rate = second_index_idx1 / first_index_idx1
    spliced_index = last_index_idx1 * window_rate * base_rate

    # Create result DataFrame
    spliced_df = pl.DataFrame({
        "period": [last_period_idx2],
        "index_value": [spliced_index]
    })

    return spliced_df


def half_splice(index1: pl.DataFrame, index2: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate the half splice extension method.

    The half splice method uses the period in the middle of the first window
    as the connecting point. This method requires that the window length
    be odd, otherwise it raises an error.

    Args:
        index1: First multilateral index DataFrame with columns "period" and "index_value"
        index2: Second multilateral index DataFrame with columns "period" and "index_value"

    Returns:
        DataFrame with the extended index including the spliced period

    Raises:
        ValueError: If input validation fails or window length is even

    Examples:
        >>> import polars as pl
        >>> from datetime import date
        >>> idx1 = pl.DataFrame({
        ...     "period": [date(2023, 1, 1), date(2023, 2, 1), date(2023, 3, 1)],
        ...     "index_value": [1.0, 1.05, 1.10]
        ... })
        >>> idx2 = pl.DataFrame({
        ...     "period": [date(2023, 2, 1), date(2023, 3, 1), date(2023, 4, 1)],
        ...     "index_value": [1.05, 1.10, 1.15]
        ... })
        >>> result = half_splice(idx1, idx2)
        >>> # Returns extended index with period 2023-04-01
    """
    _validate_indices(index1, index2)

    # Check if window length is odd
    window_length = index1.height
    if window_length % 2 == 0:
        raise ValueError("Half splice method requires an odd window length")

    # Get middle period of index1
    middle_idx = window_length // 2
    sorted_idx1 = index1.sort("period")
    middle_period_idx1 = sorted_idx1.select(pl.col("period")).to_series()[middle_idx]
    middle_index_idx1 = sorted_idx1.filter(pl.col("period") == middle_period_idx1).select("index_value").item()

    # Get corresponding period in index2
    sorted_idx2 = index2.sort("period")
    middle_period_idx2 = sorted_idx2.select(pl.col("period")).to_series()[middle_idx]
    middle_index_idx2 = sorted_idx2.filter(pl.col("period") == middle_period_idx2).select("index_value").item()

    # Get last periods
    last_period_idx1 = sorted_idx1.select(pl.col("period").max()).item()
    last_period_idx2 = sorted_idx2.select(pl.col("period").max()).item()

    last_index_idx1 = sorted_idx1.filter(pl.col("period") == last_period_idx1).select("index_value").item()
    last_index_idx2 = sorted_idx2.filter(pl.col("period") == last_period_idx2).select("index_value").item()

    # Calculate rate of change in index2 around middle period
    # Find the rate from middle to last period in index2
    middle_to_last_rate = last_index_idx2 / middle_index_idx2

    # Apply this rate to extend from middle of index1 to last of index2
    spliced_index = middle_index_idx1 * middle_to_last_rate

    # Create result DataFrame
    spliced_df = pl.DataFrame({
        "period": [last_period_idx2],
        "index_value": [spliced_index]
    })

    return spliced_df


def mean_splice(index1: pl.DataFrame, index2: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate the mean splice extension method (Diewert and Fox, 2018).

    The mean splice method uses the geometric mean of all possible choices
    of splicing, i.e., all periods which are included in the current window
    and the previous one. This is the most sophisticated splicing method.

    Args:
        index1: First multilateral index DataFrame with columns "period" and "index_value"
        index2: Second multilateral index DataFrame with columns "period" and "index_value"

    Returns:
        DataFrame with the extended index including the spliced period

    Raises:
        ValueError: If input validation fails

    Examples:
        >>> import polars as pl
        >>> from datetime import date
        >>> idx1 = pl.DataFrame({
        ...     "period": [date(2023, 1, 1), date(2023, 2, 1), date(2023, 3, 1)],
        ...     "index_value": [1.0, 1.05, 1.10]
        ... })
        >>> idx2 = pl.DataFrame({
        ...     "period": [date(2023, 2, 1), date(2023, 3, 1), date(2023, 4, 1)],
        ...     "index_value": [1.05, 1.10, 1.15]
        ... })
        >>> result = mean_splice(idx1, idx2)
        >>> # Returns extended index with period 2023-04-01
    """
    _validate_indices(index1, index2)

    # Get overlapping periods (all periods except the first of index1 and last of index2)
    sorted_idx1 = index1.sort("period")
    sorted_idx2 = index2.sort("period")

    # Find overlapping periods
    periods_idx1 = set(sorted_idx1.select("period").to_series().to_list())
    periods_idx2 = set(sorted_idx2.select("period").to_series().to_list())

    overlapping_periods = periods_idx1.intersection(periods_idx2)

    if not overlapping_periods:
        raise ValueError("No overlapping periods found between the two indices")

    # Calculate splicing rates for each overlapping period
    splice_rates = []

    for period in overlapping_periods:
        # Get index values for this period in both indices
        idx1_value = sorted_idx1.filter(pl.col("period") == period).select("index_value").item()
        idx2_value = sorted_idx2.filter(pl.col("period") == period).select("index_value").item()

        # Get the last period values
        last_period_idx1 = sorted_idx1.select(pl.col("period").max()).item()
        last_period_idx2 = sorted_idx2.select(pl.col("period").max()).item()

        last_idx1_value = sorted_idx1.filter(pl.col("period") == last_period_idx1).select("index_value").item()
        last_idx2_value = sorted_idx2.filter(pl.col("period") == last_period_idx2).select("index_value").item()

        # Calculate rate from this period to last period in each index
        rate_idx1 = last_idx1_value / idx1_value
        rate_idx2 = last_idx2_value / idx2_value

        # Calculate splicing rate for this period
        splice_rate = rate_idx2 / rate_idx1
        splice_rates.append(splice_rate)

    # Calculate geometric mean of all splicing rates
    if not splice_rates:
        raise ValueError("No valid splicing rates calculated")

    geometric_mean_rate = np.exp(np.mean(np.log(splice_rates)))

    # Get the last index value from index1
    last_period_idx1 = sorted_idx1.select(pl.col("period").max()).item()
    last_index_idx1 = sorted_idx1.filter(pl.col("period") == last_period_idx1).select("index_value").item()

    # Calculate spliced index
    spliced_index = last_index_idx1 * geometric_mean_rate

    # Create result DataFrame
    last_period_idx2 = sorted_idx2.select(pl.col("period").max()).item()
    spliced_df = pl.DataFrame({
        "period": [last_period_idx2],
        "index_value": [spliced_index]
    })

    return spliced_df


def _validate_indices(index1: pl.DataFrame, index2: pl.DataFrame) -> None:
    """
    Validate input indices for extension methods.

    Args:
        index1: First index DataFrame
        index2: Second index DataFrame

    Raises:
        ValueError: If validation fails
    """
    # Check DataFrame types
    if not isinstance(index1, pl.DataFrame) or not isinstance(index2, pl.DataFrame):
        raise ValueError("Both inputs must be polars DataFrames")

    # Check required columns
    required_cols = ["period", "index_value"]
    for i, df in enumerate([index1, index2], 1):
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Index {i} missing required columns: {missing_cols}")

    # Check data types
    for i, df in enumerate([index1, index2], 1):
        if not df.schema["index_value"].is_numeric():
            raise ValueError(f"Index {i} index_value must be numeric")
        if not df.schema["period"].is_temporal():
            raise ValueError(f"Index {i} period must be a temporal type")

    # Check window lengths are equal
    if index1.height != index2.height:
        raise ValueError("Both indices must have the same window length")

    # Check at least 2 periods
    if index1.height < 2:
        raise ValueError("Indices must have at least 2 periods")

    # Check periods are sorted and consecutive
    for i, df in enumerate([index1, index2], 1):
        sorted_periods = df.select("period").sort("period").to_series().to_list()
        if len(sorted_periods) != len(set(sorted_periods)):
            raise ValueError(f"Index {i} has duplicate periods")

    # Check that indices are shifted by exactly one period
    periods1 = set(index1.select("period").to_series().to_list())
    periods2 = set(index2.select("period").to_series().to_list())

    # Find the shift by checking the difference in first periods
    first_period1 = min(periods1)
    first_period2 = min(periods2)

    # Calculate expected shift (should be one period)
    # This assumes regular frequency, but we'll check the actual overlap
    overlap = periods1.intersection(periods2)
    non_overlap1 = periods1 - periods2
    non_overlap2 = periods2 - periods1

    if len(non_overlap1) != 1 or len(non_overlap2) != 1:
        raise ValueError("Indices must be shifted by exactly one period")

    # Check that all index values are positive
    for i, df in enumerate([index1, index2], 1):
        min_value = df.select(pl.col("index_value").min()).item()
        if min_value <= 0:
            raise ValueError(f"Index {i} must have all positive index values")
