"""
Bilateral price index functions for the PyIndexNum library.

This module contains functions for calculating unweighted bilateral price indices
that compare prices between two time periods.
"""

import polars as pl


def jevons(df: pl.DataFrame) -> float:
    """
    Compute the Jevons price index (geometric mean of price relatives).

    The Jevons index is calculated as the geometric mean of the price relatives
    (current price / base price) for each product.

    Args:
        df: Polars DataFrame with standardized columns ("date", "price", "product_id")
            containing data for exactly two periods, with each product having
            exactly one price per period.

    Returns:
        The Jevons price index as a float.

    Raises:
        ValueError: If DataFrame doesn't have exactly two unique dates,
                   or if any product has multiple prices per period,
                   or if products differ between periods,
                   or if price relatives contain zeros or negatives.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({
        ...     "date": ["2023-01-01", "2023-01-01", "2023-02-01", "2023-02-01"],
        ...     "product_id": ["A", "B", "A", "B"],
        ...     "price": [100, 200, 110, 190]
        ... })
        >>> jevons(df)  # doctest: +ELLIPSIS
        0.95...
    """
    _validate_bilateral_input(df)

    # Split into base and current periods
    dates = df.select("date").unique().sort("date")
    base_date = dates[0, "date"]
    current_date = dates[1, "date"]

    df_base = df.filter(pl.col("date") == base_date)
    df_current = df.filter(pl.col("date") == current_date)

    # Join and compute relatives
    joined = df_base.join(df_current, on="product_id", suffix="_current")
    relatives = joined.with_columns(
        (pl.col("price_current") / pl.col("price")).alias("relative")
    )

    # Check for invalid relatives
    invalid_relatives = relatives.filter(
        pl.col("relative").is_null() |
        (pl.col("relative") <= 0) |
        pl.col("relative").is_infinite()
    )
    if invalid_relatives.height > 0:
        raise ValueError("Cannot compute Jevons index with zero, negative, or infinite price relatives")

    # Geometric mean of relatives
    index = relatives.select(pl.col("relative").log().mean().exp()).item()

    if index is None:
        raise ValueError("Cannot compute Jevons index with zero, negative, or null price relatives")

    return index


def dudot(df: pl.DataFrame) -> float:
    """
    Compute the Dudot price index (arithmetic mean of price relatives).

    The Dudot index is calculated as the arithmetic mean of the price relatives
    (current price / base price) for each product.

    Args:
        df: Polars DataFrame with standardized columns ("date", "price", "product_id")
            containing data for exactly two periods, with each product having
            exactly one price per period.

    Returns:
        The Dudot price index as a float.

    Raises:
        ValueError: If DataFrame doesn't have exactly two unique dates,
                   or if any product has multiple prices per period,
                   or if products differ between periods,
                   or if price relatives contain negatives.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({
        ...     "date": ["2023-01-01", "2023-01-01", "2023-02-01", "2023-02-01"],
        ...     "product_id": ["A", "B", "A", "B"],
        ...     "price": [100, 200, 110, 190]
        ... })
        >>> dudot(df)  # doctest: +ELLIPSIS
        0.95...
    """
    _validate_bilateral_input(df)

    # Split into base and current periods
    dates = df.select("date").unique().sort("date")
    base_date = dates[0, "date"]
    current_date = dates[1, "date"]

    df_base = df.filter(pl.col("date") == base_date)
    df_current = df.filter(pl.col("date") == current_date)

    # Join and compute relatives
    joined = df_base.join(df_current, on="product_id", suffix="_current")
    relatives = joined.with_columns(
        (pl.col("price_current") / pl.col("price")).alias("relative")
    )

    # Arithmetic mean of relatives
    index = relatives.select(pl.col("relative").mean()).item()

    if index is None or index <= 0:
        raise ValueError("Cannot compute Dudot index with negative or null price relatives")

    return index


def carli(df: pl.DataFrame) -> float:
    """
    Compute the Carli price index (ratio of arithmetic means).

    The Carli index is calculated as the ratio of the arithmetic mean of prices
    in the current period to the arithmetic mean of prices in the base period.

    Args:
        df: Polars DataFrame with standardized columns ("date", "price", "product_id")
            containing data for exactly two periods, with each product having
            exactly one price per period.

    Returns:
        The Carli price index as a float.

    Raises:
        ValueError: If DataFrame doesn't have exactly two unique dates,
                   or if any product has multiple prices per period,
                   or if products differ between periods,
                   or if prices contain negatives.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({
        ...     "date": ["2023-01-01", "2023-01-01", "2023-02-01", "2023-02-01"],
        ...     "product_id": ["A", "B", "A", "B"],
        ...     "price": [100, 200, 110, 190]
        ... })
        >>> carli(df)  # doctest: +ELLIPSIS
        0.95...
    """
    _validate_bilateral_input(df)

    # Split into base and current periods
    dates = df.select("date").unique().sort("date")
    base_date = dates[0, "date"]
    current_date = dates[1, "date"]

    df_base = df.filter(pl.col("date") == base_date)
    df_current = df.filter(pl.col("date") == current_date)

    # Compute arithmetic means
    mean_base = df_base.select(pl.col("price").mean()).item()
    mean_current = df_current.select(pl.col("price").mean()).item()

    if mean_base is None or mean_current is None or mean_base <= 0:
        raise ValueError("Cannot compute Carli index with negative, zero, or null prices")

    return mean_current / mean_base


def laspeyres(df: pl.DataFrame) -> float:
    """
    Compute the Laspeyres price index.

    The Laspeyres index is calculated as the ratio of the cost of the basket
    in the current period using base period quantities to the cost of the
    basket in the base period.

    Formula: sum(p_t * q_0) / sum(p_0 * q_0)

    Args:
        df: Polars DataFrame with standardized columns ("date", "price", "product_id", "quantity")
            containing data for exactly two periods, with each product having
            exactly one price and quantity per period.

    Returns:
        The Laspeyres price index as a float.

    Raises:
        ValueError: If DataFrame doesn't have exactly two unique dates,
                   or if any product has multiple prices/quantities per period,
                   or if products differ between periods,
                   or if prices or quantities contain negatives or zeros.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({
        ...     "date": ["2023-01-01", "2023-01-01", "2023-02-01", "2023-02-01"],
        ...     "product_id": ["A", "B", "A", "B"],
        ...     "price": [100, 200, 110, 190],
        ...     "quantity": [10, 20, 10, 20]
        ... })
        >>> laspeyres(df)  # doctest: +ELLIPSIS
        0.95...
    """
    _validate_weighted_bilateral_input(df)

    # Split into base and current periods
    dates = df.select("date").unique().sort("date")
    base_date = dates[0, "date"]
    current_date = dates[1, "date"]

    df_base = df.filter(pl.col("date") == base_date)
    df_current = df.filter(pl.col("date") == current_date)

    # Join on product_id
    joined = df_base.join(df_current, on="product_id", suffix="_current")

    # Compute numerator: sum(p_t * q_0)
    numerator = joined.select((pl.col("price_current") * pl.col("quantity")).sum()).item()

    # Compute denominator: sum(p_0 * q_0)
    denominator = joined.select((pl.col("price") * pl.col("quantity")).sum()).item()

    if denominator is None or denominator <= 0:
        raise ValueError("Cannot compute Laspeyres index with zero or negative base period expenditures")

    if numerator is None:
        raise ValueError("Cannot compute Laspeyres index with null current period prices")

    return numerator / denominator


def paasche(df: pl.DataFrame) -> float:
    """
    Compute the Paasche price index.

    The Paasche index is calculated as the ratio of the cost of the basket
    in the current period using current period quantities to the cost of the
    basket in the base period using current period quantities.

    Formula: sum(p_t * q_t) / sum(p_0 * q_t)

    Args:
        df: Polars DataFrame with standardized columns ("date", "price", "product_id", "quantity")
            containing data for exactly two periods, with each product having
            exactly one price and quantity per period.

    Returns:
        The Paasche price index as a float.

    Raises:
        ValueError: If DataFrame doesn't have exactly two unique dates,
                   or if any product has multiple prices/quantities per period,
                   or if products differ between periods,
                   or if prices or quantities contain negatives or zeros.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({
        ...     "date": ["2023-01-01", "2023-01-01", "2023-02-01", "2023-02-01"],
        ...     "product_id": ["A", "B", "A", "B"],
        ...     "price": [100, 200, 110, 190],
        ...     "quantity": [10, 20, 15, 25]
        ... })
        >>> paasche(df)  # doctest: +ELLIPSIS
        0.97...
    """
    _validate_weighted_bilateral_input(df)

    # Split into base and current periods
    dates = df.select("date").unique().sort("date")
    base_date = dates[0, "date"]
    current_date = dates[1, "date"]

    df_base = df.filter(pl.col("date") == base_date)
    df_current = df.filter(pl.col("date") == current_date)

    # Join on product_id
    joined = df_base.join(df_current, on="product_id", suffix="_current")

    # Compute numerator: sum(p_t * q_t)
    numerator = joined.select((pl.col("price_current") * pl.col("quantity_current")).sum()).item()

    # Compute denominator: sum(p_0 * q_t)
    denominator = joined.select((pl.col("price") * pl.col("quantity_current")).sum()).item()

    if denominator is None or denominator <= 0:
        raise ValueError("Cannot compute Paasche index with zero or negative current period expenditures")

    if numerator is None:
        raise ValueError("Cannot compute Paasche index with null current period prices")

    return numerator / denominator


def fisher(df: pl.DataFrame) -> float:
    """
    Compute the Fisher price index.

    The Fisher index is the geometric mean of the Laspeyres and Paasche indices,
    designed to satisfy the time reversal and factor reversal tests.

    Formula: sqrt(Laspeyres * Paasche)

    Args:
        df: Polars DataFrame with standardized columns ("date", "price", "product_id", "quantity")
            containing data for exactly two periods, with each product having
            exactly one price and quantity per period.

    Returns:
        The Fisher price index as a float.

    Raises:
        ValueError: If DataFrame doesn't have exactly two unique dates,
                   or if any product has multiple prices/quantities per period,
                   or if products differ between periods,
                   or if prices or quantities contain negatives or zeros.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({
        ...     "date": ["2023-01-01", "2023-01-01", "2023-02-01", "2023-02-01"],
        ...     "product_id": ["A", "B", "A", "B"],
        ...     "price": [100, 200, 110, 190],
        ...     "quantity": [10, 20, 10, 20]
        ... })
        >>> fisher(df)  # doctest: +ELLIPSIS
        0.96...
    """
    _validate_weighted_bilateral_input(df)

    laspeyres_index = laspeyres(df)
    paasche_index = paasche(df)

    return (laspeyres_index * paasche_index) ** 0.5


def tornqvist(df: pl.DataFrame) -> float:
    """
    Compute the Törnqvist price index.

    The Törnqvist index is calculated using the geometric mean of price relatives
    weighted by the average quantity shares in the two periods.

    Formula: exp(sum((q_0 + q_t)/(2 * Q) * ln(p_t / p_0)))
    where Q = sum(q_0 + q_t)

    Args:
        df: Polars DataFrame with standardized columns ("date", "price", "product_id", "quantity")
            containing data for exactly two periods, with each product having
            exactly one price and quantity per period.

    Returns:
        The Törnqvist price index as a float.

    Raises:
        ValueError: If DataFrame doesn't have exactly two unique dates,
                   or if any product has multiple prices/quantities per period,
                   or if products differ between periods,
                   or if prices or quantities contain negatives or zeros.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({
        ...     "date": ["2023-01-01", "2023-01-01", "2023-02-01", "2023-02-01"],
        ...     "product_id": ["A", "B", "A", "B"],
        ...     "price": [100, 200, 110, 190],
        ...     "quantity": [10, 20, 15, 25]
        ... })
        >>> tornqvist(df)  # doctest: +ELLIPSIS
        0.96...
    """
    _validate_weighted_bilateral_input(df)

    # Split into base and current periods
    dates = df.select("date").unique().sort("date")
    base_date = dates[0, "date"]
    current_date = dates[1, "date"]

    df_base = df.filter(pl.col("date") == base_date)
    df_current = df.filter(pl.col("date") == current_date)

    # Join on product_id
    joined = df_base.join(df_current, on="product_id", suffix="_current")

    # Compute total quantity Q = sum(q_0 + q_t)
    total_quantity = joined.select((pl.col("quantity") + pl.col("quantity_current")).sum()).item()

    if total_quantity is None or total_quantity <= 0:
        raise ValueError("Cannot compute Törnqvist index with zero or negative total quantities")

    # Compute weighted log relatives: (q_0 + q_t)/(2 * Q) * ln(p_t / p_0)
    weighted_logs = joined.with_columns(
        (
            (pl.col("quantity") + pl.col("quantity_current")) / (2 * total_quantity) *
            (pl.col("price_current") / pl.col("price")).log()
        ).alias("weighted_log")
    )

    # Sum the weighted logs and exponentiate
    index = weighted_logs.select(pl.col("weighted_log").sum().exp()).item()

    if index is None or index <= 0:
        raise ValueError("Cannot compute Törnqvist index with invalid price relatives or quantities")

    return index


def walsh(df: pl.DataFrame) -> float:
    """
    Compute the Walsh price index.

    The Walsh index is calculated as a weighted geometric mean using base period
    quantities as weights, where the price relative for each product is the
    geometric mean of the prices in the two periods.

    Formula: sum(sqrt(p_t * p_0) * q_0) / sum(p_0 * q_0)

    Args:
        df: Polars DataFrame with standardized columns ("date", "price", "product_id", "quantity")
            containing data for exactly two periods, with each product having
            exactly one price and quantity per period.

    Returns:
        The Walsh price index as a float.

    Raises:
        ValueError: If DataFrame doesn't have exactly two unique dates,
                   or if any product has multiple prices/quantities per period,
                   or if products differ between periods,
                   or if prices or quantities contain negatives or zeros.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({
        ...     "date": ["2023-01-01", "2023-01-01", "2023-02-01", "2023-02-01"],
        ...     "product_id": ["A", "B", "A", "B"],
        ...     "price": [100, 200, 110, 190],
        ...     "quantity": [10, 20, 10, 20]
        ... })
        >>> walsh(df)  # doctest: +ELLIPSIS
        0.95...
    """
    _validate_weighted_bilateral_input(df)

    # Split into base and current periods
    dates = df.select("date").unique().sort("date")
    base_date = dates[0, "date"]
    current_date = dates[1, "date"]

    df_base = df.filter(pl.col("date") == base_date)
    df_current = df.filter(pl.col("date") == current_date)

    # Join on product_id
    joined = df_base.join(df_current, on="product_id", suffix="_current")

    # Compute numerator: sum(sqrt(p_t * p_0) * q_0)
    numerator = joined.select(
        ((pl.col("price_current") * pl.col("price")).sqrt() * pl.col("quantity")).sum()
    ).item()

    # Compute denominator: sum(p_0 * q_0)
    denominator = joined.select((pl.col("price") * pl.col("quantity")).sum()).item()

    if denominator is None or denominator <= 0:
        raise ValueError("Cannot compute Walsh index with zero or negative base period expenditures")

    if numerator is None:
        raise ValueError("Cannot compute Walsh index with null prices")

    return numerator / denominator


def _validate_weighted_bilateral_input(df: pl.DataFrame) -> None:
    """
    Validate DataFrame for weighted bilateral index computation.

    Args:
        df: DataFrame to validate

    Raises:
        ValueError: If validation fails
    """
    # Call base validation
    _validate_bilateral_input(df)

    # Check quantity column exists
    if "quantity" not in df.columns:
        raise ValueError("Missing required column: quantity")

    # Check no negative or zero quantities
    min_quantity = df.select(pl.col("quantity").min()).item()
    if min_quantity <= 0:
        raise ValueError("All quantities must be positive")


def _validate_bilateral_input(df: pl.DataFrame) -> None:
    """
    Validate DataFrame for bilateral index computation.

    Args:
        df: DataFrame to validate

    Raises:
        ValueError: If validation fails
    """
    # Check required columns
    required_cols = ["date", "price", "product_id"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Check exactly two dates
    dates = df.select("date").unique()
    if len(dates) != 2:
        raise ValueError(f"DataFrame must contain exactly 2 unique dates, found {len(dates)}")

    # Check each product has exactly one price per date
    grouped = df.group_by(["product_id", "date"]).len()
    max_count = grouped.select(pl.col("len").max()).item()
    if max_count > 1:
        raise ValueError("Each product must have exactly one price per period (use aggregate_time first)")

    # Check same products in both periods
    dates_list = dates["date"].to_list()
    products_base = df.filter(pl.col("date") == dates_list[0]).select("product_id").unique().sort("product_id")
    products_current = df.filter(pl.col("date") == dates_list[1]).select("product_id").unique().sort("product_id")

    if not products_base.equals(products_current):
        raise ValueError("Products must be identical in both periods")

    # Check no negative or zero prices
    min_price = df.select(pl.col("price").min()).item()
    if min_price <= 0:
        raise ValueError("All prices must be positive")
