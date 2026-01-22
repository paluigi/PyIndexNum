"""
Extension methods for connecting two different multilateral indices.

This module contains functions for splicing two multilateral price indices
that are calculated on the same window length but shifted by one period.
These methods are used to extend price index series when using rolling windows.
"""

import polars as pl
import numpy as np
from typing import Tuple, List
from datetime import datetime, timedelta


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
        DataFrame with the full extended index series including all periods from index1 plus the spliced period

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
        >>> # Returns the full extended index series including periods 2023-01-01, 2023-02-01, 2023-03-01, and 2023-04-01
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

    # Return the full extended series
    return pl.concat([index1, spliced_df])


def window_splice(index1: pl.DataFrame, index2: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate the window splice extension method.

    The window splice method calculates the rate of change between the last
    and first period of the second window, then uses this rate to connect
    the second period of the first window to the last period of the second window.

    Args:
        index1: First multilateral index DataFrame with columns "period" and "index_value"
        index2: Second multilateral index DataFrame with columns "period" and "index_value"

    Returns:
        DataFrame with the full extended index series including all periods from index1 plus the spliced period

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
        >>> result = window_splice(idx1, idx2)
        >>> # Returns the full extended index series including periods 2023-01-01, 2023-02-01, 2023-03-01, and 2023-04-01
    """
    _validate_indices(index1, index2)

    # Get periods and indices
    sorted_idx1 = index1.sort("period")
    sorted_idx2 = index2.sort("period")

    second_period_idx1 = sorted_idx1.select(pl.col("period")).to_series()[1]
    last_period_idx2 = sorted_idx2.select(pl.col("period").max()).item()

    second_index_idx1 = sorted_idx1.filter(pl.col("period") == second_period_idx1).select("index_value").item()
    first_index_idx2 = sorted_idx2.select(pl.col("index_value")).to_series()[0]
    last_index_idx2 = sorted_idx2.filter(pl.col("period") == last_period_idx2).select("index_value").item()

    # Calculate window rate of change (from first to last period in index2)
    window_rate = last_index_idx2 / first_index_idx2

    # Calculate spliced index by applying the window rate to the second period of index1
    spliced_index = second_index_idx1 * window_rate

    # Create result DataFrame
    spliced_df = pl.DataFrame({
        "period": [last_period_idx2],
        "index_value": [spliced_index]
    })

    # Return the full extended series
    return pl.concat([index1, spliced_df])


def half_splice(index1: pl.DataFrame, index2: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate the half splice extension method.

    The half splice method uses the period in the middle of the first window
    (T/2 if the window is even, T/2+1 if the window is odd) as the connecting point.

    Args:
        index1: First multilateral index DataFrame with columns "period" and "index_value"
        index2: Second multilateral index DataFrame with columns "period" and "index_value"

    Returns:
        DataFrame with the full extended index series including all periods from index1 plus the spliced period

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
        >>> # Returns the full extended index series including periods 2023-01-01, 2023-02-01, 2023-03-01, and 2023-04-01
    """
    _validate_indices(index1, index2)

    # Get middle period of index1
    window_length = index1.height
    if window_length % 2 == 0:
        # Python is zero-based for indexing
        middle_idx = (window_length // 2) - 1
    else:
        middle_idx = window_length // 2


    sorted_idx1 = index1.sort("period")
    middle_period_idx1 = sorted_idx1.select(pl.col("period")).to_series()[middle_idx]
    middle_index_idx1 = sorted_idx1.filter(pl.col("period") == middle_period_idx1).select("index_value").item()

    # Get the same period from index2 (should be the overlapping middle period)
    sorted_idx2 = index2.sort("period")
    middle_index_idx2 = sorted_idx2.filter(pl.col("period") == middle_period_idx1).select("index_value").item()

    # Get last period for index2
    last_period_idx2 = sorted_idx2.select(pl.col("period").max()).item()
    last_index_idx2 = sorted_idx2.filter(pl.col("period") == last_period_idx2).select("index_value").item()

    # Calculate rate of change in index2 from middle to last period
    middle_to_last_rate = last_index_idx2 / middle_index_idx2

    # Apply this rate to the middle period of index1 to get the spliced index
    # The spliced index = middle_index_idx1 * middle_to_last_rate
    spliced_index = middle_index_idx1 * middle_to_last_rate

    # Create result DataFrame
    spliced_df = pl.DataFrame({
        "period": [last_period_idx2],
        "index_value": [spliced_index]
    })

    # Return the full extended series
    return pl.concat([index1, spliced_df])


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
        DataFrame with the full extended index series including all periods from index1 plus the spliced period

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
        >>> # Returns the full extended index series including periods 2023-01-01, 2023-02-01, 2023-03-01, and 2023-04-01
    """
    _validate_indices(index1, index2)

    # Get overlapping periods (all periods except the first of index1 and last of index2)
    sorted_idx1 = index1.sort("period")
    sorted_idx2 = index2.sort("period")

    # Find overlapping periods
    periods_idx1 = set(sorted_idx1.select("period").to_series().to_list())
    periods_idx2 = set(sorted_idx2.select("period").to_series().to_list())

    overlapping_periods = periods_idx1.intersection(periods_idx2)

    # Get last period for index2
    last_period_idx2 = sorted_idx2.select(pl.col("period").max()).item()
    last_index_idx2 = sorted_idx2.filter(pl.col("period") == last_period_idx2).select("index_value").item()

    # Calculate splicing indices for each overlapping period
    splice_list = []

    for period in overlapping_periods:
        # Get index values for this period in both indices
        idx1_value = sorted_idx1.filter(pl.col("period") == period).select("index_value").item()
        idx2_value = sorted_idx2.filter(pl.col("period") == period).select("index_value").item()

        # Calculate rate of change in index2 from this to last period
        current_to_last_rate = last_index_idx2 / idx2_value

        # Apply this rate to the current period of index1 to get the current spliced index
        spliced_current = idx1_value * current_to_last_rate
        splice_list.append(spliced_current)


    # Calculate geometric mean of all splicing indices
    if not splice_list:
        raise ValueError("No valid splicing index calculated")

    spliced_index = np.exp(np.mean(np.log(splice_list)))

    # Create result DataFrame
    last_period_idx2 = sorted_idx2.select(pl.col("period").max()).item()
    spliced_df = pl.DataFrame({
        "period": [last_period_idx2],
        "index_value": [spliced_index]
    })

    # Return the full extended series
    return pl.concat([index1, spliced_df])


def fixed_base_rolling_window(index1: pl.DataFrame, index2: pl.DataFrame, base_period: str) -> pl.DataFrame:
    """
    Calculate the fixed base rolling window extension method.

    The fixed base rolling method calculates the rate of change between the last
    period of the second window and a reference period common to the first and 
    second window, then uses this rate to connect the base period of the first 
    window to the last period of the second window.

    Args:
        index1: First multilateral index DataFrame with columns "period" and "index_value"
        index2: Second multilateral index DataFrame with columns "period" and "index_value"
        base_period: string indicating the date of the base period in YYYY-MM-DD format

    Returns:
        DataFrame with the full extended index series including all periods from index1 plus the spliced period

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
        >>> result = fixed_base_rolling_window(idx1, idx2, "2023-02-01")
        >>> # Returns the full extended index series including periods 2023-01-01, 2023-02-01, 2023-03-01, and 2023-04-01
    """
    _validate_indices(index1, index2)

    # Get periods and indices
    sorted_idx1 = index1.sort("period")
    sorted_idx2 = index2.sort("period")

    base_period_date = datetime.strptime(base_period, "%Y-%m-%d").date()
    last_period_idx2 = sorted_idx2.select(pl.col("period").max()).item()

    base_index_idx1 = sorted_idx1.filter(pl.col("period") == base_period_date).select("index_value").item()
    base_index_idx2 = sorted_idx2.filter(pl.col("period") == base_period_date).select("index_value").item()
    last_index_idx2 = sorted_idx2.filter(pl.col("period") == last_period_idx2).select("index_value").item()

    # Calculate link rate of change (from base period to last period in index2)
    link_rate = last_index_idx2 / base_index_idx1

    # Calculate spliced index by applying the window rate to the second period of index1
    spliced_index = base_index_idx2 * link_rate

    # Create result DataFrame
    spliced_df = pl.DataFrame({
        "period": [last_period_idx2],
        "index_value": [spliced_index]
    })

    # Return the full extended series
    return pl.concat([index1, spliced_df])


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

    # Calculate expected shift (should be one period)
    overlap = periods1.intersection(periods2)
    non_overlap1 = periods1 - periods2
    non_overlap2 = periods2 - periods1

    # Check that the non overlapping periods are the begin of th e first and end of the second
    first_period1 = min(periods1)
    last_period2 = max(periods2)
    if first_period1 not in non_overlap1 or last_period2 not in non_overlap2:
        raise ValueError("The non-overlappting periods must be the first of index1 and last of index2")

    # Check that exactly one period is unique to each index (shifted by one period)
    # For some edge cases (like no overlapping periods), raise a different error
    if len(non_overlap1) != 1 or len(non_overlap2) != 1:
        # Check if this is a case where there are no overlapping periods
        if len(overlap) == 0:
            raise ValueError("Indices must be overlapped")
        else:
            raise ValueError("Indices must be shifted by exactly one period")

    # Check that all index values are positive
    for i, df in enumerate([index1, index2], 1):
        min_value = df.select(pl.col("index_value").min()).item()
        if min_value <= 0:
            raise ValueError(f"Index {i} must have all positive index values")
