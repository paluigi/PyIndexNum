"""
Utility functions for the PyIndexNum library.

This module contains helper functions for data processing and calculations
used throughout the library.
"""

import polars as pl
from typing import Optional, Literal


def standardize_columns(
    df: pl.DataFrame,
    date_col: str = "date",
    price_col: str = "price",
    id_col: str = "product_id",
    quantity_col: Optional[str] = None,
    date_format: str = "%Y-%m-%d"
) -> pl.DataFrame:
    """
    Standardize column names and types for price index calculations.

    This function selects specified columns, renames them to standard nomenclature,
    converts the date column to Date type, validates numeric types, and filters
    out rows where quantity is zero if quantity column is provided.

    Args:
        df: Input polars DataFrame.
        date_col: Name of the date column in input DataFrame (default "date").
        price_col: Name of the price column in input DataFrame (default "price").
        id_col: Name of the product ID column in input DataFrame (default "product_id").
        quantity_col: Name of the quantity column in input DataFrame (default None).
        date_format: Format string for parsing date column (default "%Y-%m-%d").

    Returns:
        DataFrame with standardized columns: "date" (Date), "price" (numeric),
        "product_id", "quantity" (numeric, if provided).

    Raises:
        ValueError: If required columns are missing or have invalid types.
    """
    # Check required columns exist
    required_cols = [date_col, price_col, id_col]
    if quantity_col:
        required_cols.append(quantity_col)

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Select columns
    df = df.select(required_cols)

    # Rename to standard names
    rename_dict = {
        date_col: "date",
        price_col: "price",
        id_col: "product_id"
    }
    if quantity_col:
        rename_dict[quantity_col] = "quantity"
    df = df.rename(rename_dict)

    # Convert date column
    df = df.with_columns(
        pl.col("date").str.strptime(pl.Date, date_format).alias("date")
    )

    # Validate price is numeric
    if not df.schema["price"].is_numeric():
        raise ValueError("Price column must be numeric")

    # Handle quantity column if provided
    if quantity_col:
        if not df.schema["quantity"].is_numeric():
            raise ValueError("Quantity column must be numeric")
        # Filter out rows where quantity == 0
        df = df.filter(pl.col("quantity") != 0)

    return df


def geometric_mean_expr(col: str) -> pl.Expr:
    """
    Compute geometric mean of a column using polars expressions.

    Handles zero and negative values by excluding them from calculation.
    If any invalid values, returns null.

    Args:
        col: Column name to compute geometric mean for.

    Returns:
        Polars expression for geometric mean.
    """
    return (
        pl.when((pl.col(col) <= 0).any() | pl.col(col).is_null().any())
        .then(None)
        .otherwise(pl.col(col).log().mean().exp())
    )


def harmonic_mean_expr(col: str) -> pl.Expr:
    """
    Compute harmonic mean of a column using polars expressions.

    Handles zero and negative values by excluding them from calculation.
    If any invalid values, returns null.

    Args:
        col: Column name to compute harmonic mean for.

    Returns:
        Polars expression for harmonic mean.
    """
    return (
        pl.when((pl.col(col) <= 0).any() | pl.col(col).is_null().any())
        .then(None)
        .otherwise(1.0 / (1.0 / pl.col(col)).mean())
    )


def weighted_arithmetic_mean_expr(price_col: str, weight_col: str) -> pl.Expr:
    """
    Compute weighted arithmetic mean using polars expressions.

    Args:
        price_col: Column name for values to average.
        weight_col: Column name for weights.

    Returns:
        Polars expression for weighted arithmetic mean.
    """
    return (pl.col(price_col) * pl.col(weight_col)).sum() / pl.col(weight_col).sum()


def weighted_geometric_mean_expr(price_col: str, weight_col: str) -> pl.Expr:
    """
    Compute weighted geometric mean using polars expressions.

    Handles zero and negative values by excluding them from calculation.
    If any invalid values, returns null.

    Args:
        price_col: Column name for values to average.
        weight_col: Column name for weights.

    Returns:
        Polars expression for weighted geometric mean.
    """
    total_weight = pl.col(weight_col).sum()
    weight_share = pl.col(weight_col) / total_weight
    return (
        pl.when((pl.col(price_col) <= 0).any() | pl.col(price_col).is_null().any() | pl.col(weight_col).is_null().any())
        .then(None)
        .otherwise((pl.col(price_col).log() * weight_share).sum().exp())
    )


def weighted_harmonic_mean_expr(price_col: str, weight_col: str) -> pl.Expr:
    """
    Compute weighted harmonic mean using polars expressions.

    Handles zero and negative values by excluding them from calculation.
    If any invalid values, returns null.

    Args:
        price_col: Column name for values to average.
        weight_col: Column name for weights.

    Returns:
        Polars expression for weighted harmonic mean.
    """
    return (
        pl.when((pl.col(price_col) <= 0).any() | pl.col(price_col).is_null().any() | pl.col(weight_col).is_null().any())
        .then(None)
        .otherwise(pl.col(weight_col).sum() / (pl.col(weight_col) / pl.col(price_col)).sum())
    )


def aggregate_time(
    df: pl.DataFrame,
    date_col: str = "date",
    price_col: str = "price",
    quantity_col: Optional[str] = None,
    id_col: str = "product_id",
    agg_type: Literal[
        "arithmetic", "geometric", "harmonic",
        "weighted_arithmetic", "weighted_geometric", "weighted_harmonic"
    ] = "arithmetic",
    freq: str = "1mo"
) -> pl.DataFrame:
    """
    Aggregate time series data to a specified frequency.

    This function aggregates price and quantity data by grouping on the id column
    and truncated date periods. Prices are aggregated according to the specified
    aggregation type, while quantities are always summed if provided.

    Args:
        df: Input polars DataFrame containing the data.
        date_col: Name of the column containing dates. Will be parsed to datetime if string.
        price_col: Name of the column containing prices. Must be numeric.
        quantity_col: Name of the column containing quantities. If None, quantity aggregation is skipped.
        id_col: Name of the column containing unique identifiers (e.g., product IDs).
        agg_type: Type of aggregation for prices. Options:
            - 'arithmetic': Arithmetic mean
            - 'geometric': Geometric mean
            - 'harmonic': Harmonic mean
            - 'weighted_arithmetic': Weighted arithmetic mean (requires quantity_col)
            - 'weighted_geometric': Weighted geometric mean (requires quantity_col)
            - 'weighted_harmonic': Weighted harmonic mean (requires quantity_col)
        freq: Frequency for aggregation (e.g., '1d', '1w', '1mo', '1q', '1y').

    Returns:
        Aggregated DataFrame with columns: id_col, period (truncated date), aggregated_price, aggregated_quantity (if quantity_col provided).

    Raises:
        ValueError: If weighted aggregation type is selected but quantity_col is None.
        ValueError: If required columns are missing or have wrong types.

    Examples:
        >>> df = pl.DataFrame({
        ...     "date": ["2023-01-01", "2023-01-15", "2023-02-01"],
        ...     "product": ["A", "A", "A"],
        ...     "price": [100, 110, 120],
        ...     "quantity": [10, 12, 15]
        ... })
        >>> result = aggregate_time(df, "date", "price", "quantity", "product", "arithmetic", "1mo")
    """
    # Validate inputs
    if not isinstance(df, pl.DataFrame):
        raise ValueError("df must be a polars DataFrame")

    required_cols = [date_col, price_col, id_col]
    if quantity_col:
        required_cols.append(quantity_col)

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Validate weighted aggregation requires quantity_col
    if agg_type.startswith("weighted_") and not quantity_col:
        raise ValueError(f"Weighted aggregation '{agg_type}' requires quantity_col to be specified")

    # Parse date column if needed
    if df.schema[date_col] == pl.Utf8:
        df = df.with_columns(pl.col(date_col).str.strptime(pl.Date, "%Y-%m-%d").alias(date_col))
    elif not df.schema[date_col].is_temporal():
        raise ValueError(f"date_col '{date_col}' must be datetime or parseable string date")

    # Ensure price_col is numeric
    if not df.schema[price_col].is_numeric():
        raise ValueError(f"price_col '{price_col}' must be numeric")

    # Ensure quantity_col is numeric if provided
    if quantity_col and not df.schema[quantity_col].is_numeric():
        raise ValueError(f"quantity_col '{quantity_col}' must be numeric")

    # Create period column by truncating date
    df_agg = df.with_columns(
        pl.col(date_col).dt.truncate(freq).alias("period")
    )

    # Define aggregation expressions
    agg_exprs = []

    # Always aggregate by id and period
    group_cols = [id_col, "period"]

    # Price aggregation
    if agg_type == "arithmetic":
        agg_exprs.append(pl.col(price_col).mean().alias("aggregated_price"))
    elif agg_type == "geometric":
        agg_exprs.append(geometric_mean_expr(price_col).alias("aggregated_price"))
    elif agg_type == "harmonic":
        agg_exprs.append(harmonic_mean_expr(price_col).alias("aggregated_price"))
    elif agg_type == "weighted_arithmetic":
        agg_exprs.append(weighted_arithmetic_mean_expr(price_col, quantity_col).alias("aggregated_price"))
    elif agg_type == "weighted_geometric":
        agg_exprs.append(weighted_geometric_mean_expr(price_col, quantity_col).alias("aggregated_price"))
    elif agg_type == "weighted_harmonic":
        agg_exprs.append(weighted_harmonic_mean_expr(price_col, quantity_col).alias("aggregated_price"))

    # Quantity aggregation (always sum if provided)
    if quantity_col:
        agg_exprs.append(pl.col(quantity_col).sum().alias("aggregated_quantity"))

    # Perform aggregation
    result = df_agg.group_by(group_cols).agg(agg_exprs)

    return result


def remove_unbalanced(df: pl.DataFrame) -> pl.DataFrame:
    """
    Remove products that are not present in all time periods.

    This function filters out any product_id that does not appear in every
    unique time period in the dataset, resulting in a balanced panel dataset.
    It assumes the DataFrame has been processed by standardize_columns and
    aggregate_time, with columns "product_id" and "period".

    Args:
        df: Polars DataFrame with columns "product_id" and "period" (from aggregate_time).

    Returns:
        Filtered DataFrame containing only products present in all periods.

    Raises:
        ValueError: If required columns "product_id" or "period" are missing.

    Examples:
        >>> df = pl.DataFrame({
        ...     "product_id": ["A", "A", "B", "B", "C"],
        ...     "period": [pl.date(2023, 1, 1), pl.date(2023, 2, 1), pl.date(2023, 1, 1), pl.date(2023, 2, 1), pl.date(2023, 1, 1)],
        ...     "aggregated_price": [100, 110, 200, 210, 300]
        ... })
        >>> result = remove_unbalanced(df)
        >>> # Only product "C" is removed as it's missing period 2023-02-01
    """
    # Validate required columns
    required_cols = ["product_id", "period"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Get total number of unique periods
    total_periods = df.select("period").unique().height

    # Count periods per product
    product_period_counts = (
        df.group_by("product_id")
        .agg(pl.col("period").n_unique().alias("period_count"))
    )

    # Get balanced products (present in all periods)
    balanced_product_list = (
        product_period_counts
        .filter(pl.col("period_count") == total_periods)
        .select("product_id")
        .to_series()
        .to_list()
    )

    # Filter original DataFrame
    return df.filter(pl.col("product_id").is_in(balanced_product_list))


def get_summary(df: pl.DataFrame) -> dict:
    """
    Get summary information about a standardized price index DataFrame.

    This function provides key statistics about a DataFrame that has been
    standardized using standardize_columns().

    Args:
        df: Polars DataFrame with standardized columns (date, price, product_id, quantity).

    Returns:
        Dictionary containing:
        - n_products: Number of unique product IDs
        - start_date: Earliest date in the data
        - end_date: Latest date in the data
        - quantity: Boolean indicating if quantity column exists and has non-null values
    """
    n_products = df.select("product_id").unique().height
    start_date = df.select(pl.col("date").min()).item()
    end_date = df.select(pl.col("date").max()).item()
    quantity_present = "quantity" in df.columns and df.select(pl.col("quantity").is_not_null().any()).item()

    return {
        "n_products": n_products,
        "start_date": start_date,
        "end_date": end_date,
        "quantity": quantity_present
    }


def carry_forward_imputation(
    df: pl.DataFrame,
    value_cols: list[str],
    id_col: str = "product_id",
    time_col: str = "period"
) -> pl.DataFrame:
    """
    Create balanced panel and fill missing values using forward imputation.

    This function creates a balanced panel dataset by generating all possible
    combinations of product IDs and time periods, then fills missing values
    by carrying forward the last available observation for each product.

    Args:
        df: Input polars DataFrame with aggregated data (may be unbalanced).
        value_cols: List of column names to impute (e.g., ["aggregated_price", "aggregated_quantity"]).
        id_col: Name of the column containing unique identifiers (default "product_id").
        time_col: Name of the column containing time periods (default "period").

    Returns:
        Balanced DataFrame with all product-period combinations and nulls filled using forward imputation.

    Raises:
        ValueError: If required columns are missing from the DataFrame.

    Examples:
        >>> df = pl.DataFrame({
        ...     "product_id": ["A", "A", "B"],
        ...     "period": [pl.date(2023, 1, 1), pl.date(2023, 2, 1), pl.date(2023, 1, 1)],
        ...     "aggregated_price": [100.0, 110.0, 200.0]
        ... })
        >>> result = carry_forward_imputation(df, ["aggregated_price"])
        >>> # Creates balanced panel: A in both periods, B in both periods
        >>> # A: 100.0, 110.0; B: 200.0, 200.0 (forward filled)
    """
    # Validate required columns
    required_cols = [id_col, time_col] + value_cols
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Get all unique product IDs and periods
    unique_ids = df.select(id_col).unique()
    unique_periods = df.select(time_col).unique()

    # Create balanced grid (cross product of all IDs and periods)
    grid = unique_ids.join(unique_periods, how="cross")

    # Left join the input data to the grid
    balanced_df = grid.join(df, on=[id_col, time_col], how="left")

    # Apply forward fill for each value column
    for col in value_cols:
        balanced_df = balanced_df.with_columns(
            pl.col(col).fill_null(strategy="forward").over(id_col, order_by=time_col).alias(col)
        )

    return balanced_df


def carry_backward_imputation(
    df: pl.DataFrame,
    value_cols: list[str],
    id_col: str = "product_id",
    time_col: str = "period"
) -> pl.DataFrame:
    """
    Create balanced panel and fill missing values using backward imputation.

    This function creates a balanced panel dataset by generating all possible
    combinations of product IDs and time periods, then fills missing values
    by carrying backward the first future observation for each product.

    Args:
        df: Input polars DataFrame with aggregated data (may be unbalanced).
        value_cols: List of column names to impute (e.g., ["aggregated_price", "aggregated_quantity"]).
        id_col: Name of the column containing unique identifiers (default "product_id").
        time_col: Name of the column containing time periods (default "period").

    Returns:
        Balanced DataFrame with all product-period combinations and nulls filled using backward imputation.

    Raises:
        ValueError: If required columns are missing from the DataFrame.

    Examples:
        >>> df = pl.DataFrame({
        ...     "product_id": ["A", "A", "B"],
        ...     "period": [pl.date(2023, 1, 1), pl.date(2023, 2, 1), pl.date(2023, 1, 1)],
        ...     "aggregated_price": [100.0, 110.0, 200.0]
        ... })
        >>> result = carry_backward_imputation(df, ["aggregated_price"])
        >>> # Creates balanced panel: A in both periods, B in both periods
        >>> # A: 100.0, 110.0; B: 200.0, 200.0 (no fill needed)
    """
    # Validate required columns
    required_cols = [id_col, time_col] + value_cols
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Get all unique product IDs and periods
    unique_ids = df.select(id_col).unique()
    unique_periods = df.select(time_col).unique()

    # Create balanced grid (cross product of all IDs and periods)
    grid = unique_ids.join(unique_periods, how="cross")

    # Left join the input data to the grid
    balanced_df = grid.join(df, on=[id_col, time_col], how="left")

    # Apply backward fill for each value column
    for col in value_cols:
        balanced_df = balanced_df.with_columns(
            pl.col(col).fill_null(strategy="backward").over(id_col, order_by=time_col).alias(col)
        )

    return balanced_df
