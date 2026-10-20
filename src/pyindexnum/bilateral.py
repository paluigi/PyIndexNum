"""
Bilateral price index functions for the PyIndexNum library.

This module contains functions for calculating bilateral price indices
that compare prices between two time periods.

All functions accept a DataFrame in the canonical panel schema produced by
``aggregate_time``: columns "period", "product_id", "aggregated_price" and
(for weighted indices) "aggregated_quantity". Custom column names can be
passed via the ``period_col``/``price_col``/``id_col``/``quantity_col``
arguments. By default the two periods must contain exactly the same products
(``balanced=True``); pass ``balanced=False`` to compute the index on the
products matched between the two periods instead, and ``panel_warnings=True``
to be warned about how many products each comparison drops.
"""

import warnings

import polars as pl
from typing import Optional


def _to_canonical(
    df: pl.DataFrame,
    period_col: str = "period",
    price_col: str = "aggregated_price",
    id_col: str = "product_id",
    quantity_col: Optional[str] = None,
) -> pl.DataFrame:
    """
    Rename user-specified columns to the canonical panel schema.

    Args:
        df: Input DataFrame.
        period_col: Name of the period column.
        price_col: Name of the price column.
        id_col: Name of the product identifier column.
        quantity_col: Name of the quantity column, if required.

    Returns:
        DataFrame with the specified columns renamed to "period",
        "aggregated_price", "product_id" and (optionally) "aggregated_quantity".

    Raises:
        ValueError: If a specified column is missing, or a rename would
            collide with an existing column.
    """
    mapping = {
        "period": period_col,
        "aggregated_price": price_col,
        "product_id": id_col,
    }
    if quantity_col is not None:
        mapping["aggregated_quantity"] = quantity_col

    missing_cols = [col for col in mapping.values() if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    rename = {src: dst for dst, src in mapping.items() if src != dst}
    for src, dst in rename.items():
        if dst in df.columns:
            raise ValueError(
                f"Cannot rename column '{src}' to '{dst}': a column '{dst}' already exists"
            )
    return df.rename(rename)


def _split_pair(df: pl.DataFrame):
    """
    Split a two-period DataFrame into base and current period frames.

    The base period is the chronologically first of the two periods.

    Args:
        df: Canonical DataFrame containing exactly two unique periods.

    Returns:
        Tuple of (base_period, df_base, df_current).
    """
    periods = df.select("period").unique().sort("period")
    base_period = periods[0, "period"]
    current_period = periods[1, "period"]
    return (
        base_period,
        df.filter(pl.col("period") == base_period),
        df.filter(pl.col("period") == current_period),
    )


def jevons(
    df: pl.DataFrame,
    period_col: str = "period",
    price_col: str = "aggregated_price",
    id_col: str = "product_id",
    balanced: bool = True,
    panel_warnings: bool = False,
) -> float:
    """
    Compute the Jevons price index (geometric mean of price relatives).

    The Jevons index is calculated as the geometric mean of the price relatives
    (current price / base price) for each product.

    Formula: P_J = [prod_i (p_i^t / p_i^0)]^(1/N)

    Args:
        df: Polars DataFrame with the panel columns identified by period_col,
            price_col and id_col, containing data for exactly two periods,
            with each product having exactly one price per period.
        period_col: Name of the period column (default "period").
        price_col: Name of the price column (default "aggregated_price").
        id_col: Name of the product identifier column (default "product_id").
        balanced: If True (default), require both periods to contain exactly
            the same products. If False, compute the index on the products
            present in both periods.
        panel_warnings: If True, emit a UserWarning summarising how many
            products are unmatched when balanced=False.

    Returns:
        The Jevons price index as a float.

    Raises:
        ValueError: If DataFrame doesn't have exactly two unique periods,
                   or if any product has multiple prices per period,
                   or if products differ between periods (balanced=True),
                   or if no products are common to both periods (balanced=False),
                   or if price relatives contain zeros or negatives.

    Examples:
        >>> from datetime import date
        >>> import polars as pl
        >>> df = pl.DataFrame({
        ...     "period": [date(2023, 1, 1), date(2023, 1, 1), date(2023, 2, 1), date(2023, 2, 1)],
        ...     "product_id": ["A", "B", "A", "B"],
        ...     "aggregated_price": [100, 200, 110, 190]
        ... })
        >>> jevons(df)  # doctest: +ELLIPSIS
        1.022...
    """
    df = _to_canonical(df, period_col, price_col, id_col)
    _validate_bilateral_input(df, balanced=balanced, panel_warnings=panel_warnings)

    _, df_base, df_current = _split_pair(df)

    # Join and compute relatives (inner join keeps matched products only)
    joined = df_base.join(df_current, on="product_id", suffix="_current")
    relatives = joined.with_columns(
        (pl.col("aggregated_price_current") / pl.col("aggregated_price")).alias("relative")
    )

    # Check for invalid relatives
    invalid_relatives = relatives.filter(
        pl.col("relative").is_null()
        | (pl.col("relative") <= 0)
        | pl.col("relative").is_infinite()
    )
    if invalid_relatives.height > 0:
        raise ValueError("Cannot compute Jevons index with zero, negative, or infinite price relatives")

    # Geometric mean of relatives
    index = relatives.select(pl.col("relative").log().mean().exp()).item()

    if index is None:
        raise ValueError("Cannot compute Jevons index with zero, negative, or null price relatives")

    return index


def carli(
    df: pl.DataFrame,
    period_col: str = "period",
    price_col: str = "aggregated_price",
    id_col: str = "product_id",
    balanced: bool = True,
    panel_warnings: bool = False,
) -> float:
    """
    Compute the Carli price index (arithmetic mean of price relatives).

    The Carli index is calculated as the arithmetic mean of the price relatives
    (current price / base price) for each product.

    Formula: P_C = (1/N) * sum_i (p_i^t / p_i^0)

    Args:
        df: Polars DataFrame with the panel columns identified by period_col,
            price_col and id_col, containing data for exactly two periods,
            with each product having exactly one price per period.
        period_col: Name of the period column (default "period").
        price_col: Name of the price column (default "aggregated_price").
        id_col: Name of the product identifier column (default "product_id").
        balanced: If True (default), require both periods to contain exactly
            the same products. If False, compute the index on the products
            present in both periods.
        panel_warnings: If True, emit a UserWarning summarising how many
            products are unmatched when balanced=False.

    Returns:
        The Carli price index as a float.

    Raises:
        ValueError: If DataFrame doesn't have exactly two unique periods,
                   or if any product has multiple prices per period,
                   or if products differ between periods (balanced=True),
                   or if no products are common to both periods (balanced=False),
                   or if price relatives contain negatives.

    Examples:
        >>> from datetime import date
        >>> import polars as pl
        >>> df = pl.DataFrame({
        ...     "period": [date(2023, 1, 1), date(2023, 1, 1), date(2023, 2, 1), date(2023, 2, 1)],
        ...     "product_id": ["A", "B", "A", "B"],
        ...     "aggregated_price": [100, 200, 110, 190]
        ... })
        >>> carli(df)  # doctest: +ELLIPSIS
        1.025
    """
    df = _to_canonical(df, period_col, price_col, id_col)
    _validate_bilateral_input(df, balanced=balanced, panel_warnings=panel_warnings)

    _, df_base, df_current = _split_pair(df)

    joined = df_base.join(df_current, on="product_id", suffix="_current")
    relatives = joined.with_columns(
        (pl.col("aggregated_price_current") / pl.col("aggregated_price")).alias("relative")
    )

    index = relatives.select(pl.col("relative").mean()).item()

    if index is None or index <= 0:
        raise ValueError("Cannot compute Carli index with negative or null price relatives")

    return index


def dutot(
    df: pl.DataFrame,
    period_col: str = "period",
    price_col: str = "aggregated_price",
    id_col: str = "product_id",
    balanced: bool = True,
    panel_warnings: bool = False,
) -> float:
    """
    Compute the Dutot price index (ratio of arithmetic means).

    The Dutot index is calculated as the ratio of the arithmetic mean of prices
    in the current period to the arithmetic mean of prices in the base period.

    Formula: P_D = mean(p^t) / mean(p^0)

    Note:
        The Dutot index is a ratio of arithmetic means over different product
        sets, so — unlike the relative-based indices — unmatched products can
        never be made fully comparable. With ``balanced=False`` the means are
        computed over the products matched between the two periods, which is
        the standard matched-sample convention, but the result remains
        sensitive to the composition of that matched set.

    Args:
        df: Polars DataFrame with the panel columns identified by period_col,
            price_col and id_col, containing data for exactly two periods,
            with each product having exactly one price per period.
        period_col: Name of the period column (default "period").
        price_col: Name of the price column (default "aggregated_price").
        id_col: Name of the product identifier column (default "product_id").
        balanced: If True (default), require both periods to contain exactly
            the same products. If False, compute the index on the products
            present in both periods.
        panel_warnings: If True, emit a UserWarning summarising how many
            products are unmatched when balanced=False.

    Returns:
        The Dutot price index as a float.

    Raises:
        ValueError: If DataFrame doesn't have exactly two unique periods,
                   or if any product has multiple prices per period,
                   or if products differ between periods (balanced=True),
                   or if no products are common to both periods (balanced=False),
                   or if prices contain negatives.

    Examples:
        >>> from datetime import date
        >>> import polars as pl
        >>> df = pl.DataFrame({
        ...     "period": [date(2023, 1, 1), date(2023, 1, 1), date(2023, 2, 1), date(2023, 2, 1)],
        ...     "product_id": ["A", "B", "A", "B"],
        ...     "aggregated_price": [100, 200, 110, 190]
        ... })
        >>> dutot(df)  # doctest: +ELLIPSIS
        1.0
    """
    df = _to_canonical(df, period_col, price_col, id_col)
    _validate_bilateral_input(df, balanced=balanced, panel_warnings=panel_warnings)

    _, df_base, df_current = _split_pair(df)

    # Join on product_id so that, with balanced=False, the means run over the
    # matched products only
    joined = df_base.join(df_current, on="product_id", suffix="_current")

    mean_base = joined.select(pl.col("aggregated_price").mean()).item()
    mean_current = joined.select(pl.col("aggregated_price_current").mean()).item()

    if mean_base is None or mean_current is None or mean_base <= 0:
        raise ValueError("Cannot compute Dutot index with negative, zero, or null prices")

    return mean_current / mean_base


def laspeyres(
    df: pl.DataFrame,
    period_col: str = "period",
    price_col: str = "aggregated_price",
    id_col: str = "product_id",
    quantity_col: str = "aggregated_quantity",
    balanced: bool = True,
    panel_warnings: bool = False,
) -> float:
    """
    Compute the Laspeyres price index.

    The Laspeyres index is calculated as the ratio of the cost of the basket
    in the current period using base period quantities to the cost of the
    basket in the base period.

    Formula: sum(p_t * q_0) / sum(p_0 * q_0)

    Args:
        df: Polars DataFrame with the panel columns identified by period_col,
            price_col, id_col and quantity_col, containing data for exactly
            two periods, with each product having exactly one price and
            quantity per period.
        period_col: Name of the period column (default "period").
        price_col: Name of the price column (default "aggregated_price").
        id_col: Name of the product identifier column (default "product_id").
        quantity_col: Name of the quantity column (default "aggregated_quantity").
        balanced: If True (default), require both periods to contain exactly
            the same products. If False, compute the index on the products
            present in both periods.
        panel_warnings: If True, emit a UserWarning summarising how many
            products are unmatched when balanced=False.

    Returns:
        The Laspeyres price index as a float.

    Raises:
        ValueError: If DataFrame doesn't have exactly two unique periods,
                   or if any product has multiple prices/quantities per period,
                   or if products differ between periods (balanced=True),
                   or if no products are common to both periods (balanced=False),
                   or if prices or quantities contain negatives or zeros.

    Examples:
        >>> from datetime import date
        >>> import polars as pl
        >>> df = pl.DataFrame({
        ...     "period": [date(2023, 1, 1), date(2023, 1, 1), date(2023, 2, 1), date(2023, 2, 1)],
        ...     "product_id": ["A", "B", "A", "B"],
        ...     "aggregated_price": [100, 200, 110, 190],
        ...     "aggregated_quantity": [10, 20, 10, 20]
        ... })
        >>> laspeyres(df)  # doctest: +ELLIPSIS
        0.98
    """
    df = _to_canonical(df, period_col, price_col, id_col, quantity_col)
    _validate_weighted_bilateral_input(df, balanced=balanced, panel_warnings=panel_warnings)

    _, df_base, df_current = _split_pair(df)

    # Join on product_id (inner join keeps matched products only)
    joined = df_base.join(df_current, on="product_id", suffix="_current")

    # Compute numerator: sum(p_t * q_0)
    numerator = joined.select(
        (pl.col("aggregated_price_current") * pl.col("aggregated_quantity")).sum()
    ).item()

    # Compute denominator: sum(p_0 * q_0)
    denominator = joined.select(
        (pl.col("aggregated_price") * pl.col("aggregated_quantity")).sum()
    ).item()

    if denominator is None or denominator <= 0:
        raise ValueError("Cannot compute Laspeyres index with zero or negative base period expenditures")

    if numerator is None:
        raise ValueError("Cannot compute Laspeyres index with null current period prices")

    return numerator / denominator


def paasche(
    df: pl.DataFrame,
    period_col: str = "period",
    price_col: str = "aggregated_price",
    id_col: str = "product_id",
    quantity_col: str = "aggregated_quantity",
    balanced: bool = True,
    panel_warnings: bool = False,
) -> float:
    """
    Compute the Paasche price index.

    The Paasche index is calculated as the ratio of the cost of the basket
    in the current period using current period quantities to the cost of the
    basket in the base period using current period quantities.

    Formula: sum(p_t * q_t) / sum(p_0 * q_t)

    Args:
        df: Polars DataFrame with the panel columns identified by period_col,
            price_col, id_col and quantity_col, containing data for exactly
            two periods, with each product having exactly one price and
            quantity per period.
        period_col: Name of the period column (default "period").
        price_col: Name of the price column (default "aggregated_price").
        id_col: Name of the product identifier column (default "product_id").
        quantity_col: Name of the quantity column (default "aggregated_quantity").
        balanced: If True (default), require both periods to contain exactly
            the same products. If False, compute the index on the products
            present in both periods.
        panel_warnings: If True, emit a UserWarning summarising how many
            products are unmatched when balanced=False.

    Returns:
        The Paasche price index as a float.

    Raises:
        ValueError: If DataFrame doesn't have exactly two unique periods,
                   or if any product has multiple prices/quantities per period,
                   or if products differ between periods (balanced=True),
                   or if no products are common to both periods (balanced=False),
                   or if prices or quantities contain negatives or zeros.

    Examples:
        >>> from datetime import date
        >>> import polars as pl
        >>> df = pl.DataFrame({
        ...     "period": [date(2023, 1, 1), date(2023, 1, 1), date(2023, 2, 1), date(2023, 2, 1)],
        ...     "product_id": ["A", "B", "A", "B"],
        ...     "aggregated_price": [100, 200, 110, 190],
        ...     "aggregated_quantity": [10, 20, 15, 25]
        ... })
        >>> paasche(df)  # doctest: +ELLIPSIS
        0.984...
    """
    df = _to_canonical(df, period_col, price_col, id_col, quantity_col)
    _validate_weighted_bilateral_input(df, balanced=balanced, panel_warnings=panel_warnings)

    _, df_base, df_current = _split_pair(df)

    # Join on product_id (inner join keeps matched products only)
    joined = df_base.join(df_current, on="product_id", suffix="_current")

    # Compute numerator: sum(p_t * q_t)
    numerator = joined.select(
        (pl.col("aggregated_price_current") * pl.col("aggregated_quantity_current")).sum()
    ).item()

    # Compute denominator: sum(p_0 * q_t)
    denominator = joined.select(
        (pl.col("aggregated_price") * pl.col("aggregated_quantity_current")).sum()
    ).item()

    if denominator is None or denominator <= 0:
        raise ValueError("Cannot compute Paasche index with zero or negative current period expenditures")

    if numerator is None:
        raise ValueError("Cannot compute Paasche index with null current period prices")

    return numerator / denominator


def fisher(
    df: pl.DataFrame,
    period_col: str = "period",
    price_col: str = "aggregated_price",
    id_col: str = "product_id",
    quantity_col: str = "aggregated_quantity",
    balanced: bool = True,
    panel_warnings: bool = False,
) -> float:
    """
    Compute the Fisher price index.

    The Fisher index is the geometric mean of the Laspeyres and Paasche indices,
    designed to satisfy the time reversal and factor reversal tests.

    Formula: sqrt(Laspeyres * Paasche)

    Args:
        df: Polars DataFrame with the panel columns identified by period_col,
            price_col, id_col and quantity_col, containing data for exactly
            two periods, with each product having exactly one price and
            quantity per period.
        period_col: Name of the period column (default "period").
        price_col: Name of the price column (default "aggregated_price").
        id_col: Name of the product identifier column (default "product_id").
        quantity_col: Name of the quantity column (default "aggregated_quantity").
        balanced: If True (default), require both periods to contain exactly
            the same products. If False, compute the index on the products
            present in both periods.
        panel_warnings: If True, emit a UserWarning summarising how many
            products are unmatched when balanced=False.

    Returns:
        The Fisher price index as a float.

    Raises:
        ValueError: If DataFrame doesn't have exactly two unique periods,
                   or if any product has multiple prices/quantities per period,
                   or if products differ between periods (balanced=True),
                   or if no products are common to both periods (balanced=False),
                   or if prices or quantities contain negatives or zeros.

    Examples:
        >>> from datetime import date
        >>> import polars as pl
        >>> df = pl.DataFrame({
        ...     "period": [date(2023, 1, 1), date(2023, 1, 1), date(2023, 2, 1), date(2023, 2, 1)],
        ...     "product_id": ["A", "B", "A", "B"],
        ...     "aggregated_price": [100, 200, 110, 190],
        ...     "aggregated_quantity": [10, 20, 10, 20]
        ... })
        >>> fisher(df)  # doctest: +ELLIPSIS
        0.982...
    """
    laspeyres_index = laspeyres(
        df,
        period_col=period_col,
        price_col=price_col,
        id_col=id_col,
        quantity_col=quantity_col,
        balanced=balanced,
        panel_warnings=panel_warnings,
    )
    paasche_index = paasche(
        df,
        period_col=period_col,
        price_col=price_col,
        id_col=id_col,
        quantity_col=quantity_col,
        balanced=balanced,
        panel_warnings=panel_warnings,
    )

    return (laspeyres_index * paasche_index) ** 0.5


def tornqvist(
    df: pl.DataFrame,
    period_col: str = "period",
    price_col: str = "aggregated_price",
    id_col: str = "product_id",
    quantity_col: str = "aggregated_quantity",
    balanced: bool = True,
    panel_warnings: bool = False,
) -> float:
    """
    Compute the Törnqvist price index.

    The Törnqvist index is a weighted geometric average of the price relatives.
    The weights are the arithmetic average of the expenditure shares in the two periods.

    Formula: exp(sum(0.5 * (s_0 + s_t) * ln(p_t / p_0)))
    where s_i = (p_i * q_i) / sum(p_i * q_i)

    With ``balanced=False`` the expenditure shares are re-normalised over the
    products matched between the two periods (matched-sample Törnqvist); with
    balanced data this is identical to computing shares over all products.

    Args:
        df: Polars DataFrame with the panel columns identified by period_col,
            price_col, id_col and quantity_col, containing data for exactly
            two periods, with each product having exactly one price and
            quantity per period.
        period_col: Name of the period column (default "period").
        price_col: Name of the price column (default "aggregated_price").
        id_col: Name of the product identifier column (default "product_id").
        quantity_col: Name of the quantity column (default "aggregated_quantity").
        balanced: If True (default), require both periods to contain exactly
            the same products. If False, compute the index on the products
            present in both periods.
        panel_warnings: If True, emit a UserWarning summarising how many
            products are unmatched when balanced=False.

    Returns:
        The Törnqvist price index as a float.

    Raises:
        ValueError: If DataFrame doesn't have exactly two unique periods,
                   or if any product has multiple prices/quantities per period,
                   or if products differ between periods (balanced=True),
                   or if no products are common to both periods (balanced=False),
                   or if prices or quantities contain negatives or zeros.

    Examples:
        >>> from datetime import date
        >>> import polars as pl
        >>> df = pl.DataFrame({
        ...     "period": [date(2023, 1, 1), date(2023, 1, 1), date(2023, 2, 1), date(2023, 2, 1)],
        ...     "product_id": ["A", "B", "A", "B"],
        ...     "aggregated_price": [100, 200, 110, 190],
        ...     "aggregated_quantity": [10, 20, 15, 25]
        ... })
        >>> tornqvist(df)  # doctest: +ELLIPSIS
        0.982...
    """
    df = _to_canonical(df, period_col, price_col, id_col, quantity_col)
    _validate_weighted_bilateral_input(df, balanced=balanced, panel_warnings=panel_warnings)

    _, df_base, df_current = _split_pair(df)

    # Join on product_id (inner join keeps matched products only)
    joined = df_base.join(df_current, on="product_id", suffix="_current")

    # Expenditure shares are computed on the joined frame so that, when
    # balanced=False, they are re-normalised over the matched products
    joined = joined.with_columns(
        (
            (pl.col("aggregated_price") * pl.col("aggregated_quantity"))
            / (pl.col("aggregated_price") * pl.col("aggregated_quantity")).sum()
        ).alias("share"),
        (
            (pl.col("aggregated_price_current") * pl.col("aggregated_quantity_current"))
            / (pl.col("aggregated_price_current") * pl.col("aggregated_quantity_current")).sum()
        ).alias("share_current"),
    )

    # Compute weighted log relatives: 0.5 * (s_0 + s_t) * ln(p_t / p_0)
    weighted_logs = joined.with_columns(
        (
            0.5
            * (pl.col("share") + pl.col("share_current"))
            * (pl.col("aggregated_price_current") / pl.col("aggregated_price")).log()
        ).alias("weighted_log")
    )

    # Sum the weighted logs and exponentiate
    index = weighted_logs.select(pl.col("weighted_log").sum().exp()).item()

    if index is None or index <= 0:
        raise ValueError("Cannot compute Törnqvist index with invalid price relatives or quantities")

    return index


def walsh(
    df: pl.DataFrame,
    period_col: str = "period",
    price_col: str = "aggregated_price",
    id_col: str = "product_id",
    quantity_col: str = "aggregated_quantity",
    balanced: bool = True,
    panel_warnings: bool = False,
) -> float:
    """
    Compute the Walsh price index.

    The Walsh index is a pure price index that uses the geometric mean of the
    quantities from the two periods as the fixed basket.

    Formula: sum(p_t * sqrt(q_0 * q_t)) / sum(p_0 * sqrt(q_0 * q_t))

    Args:
        df: Polars DataFrame with the panel columns identified by period_col,
            price_col, id_col and quantity_col, containing data for exactly
            two periods, with each product having exactly one price and
            quantity per period.
        period_col: Name of the period column (default "period").
        price_col: Name of the price column (default "aggregated_price").
        id_col: Name of the product identifier column (default "product_id").
        quantity_col: Name of the quantity column (default "aggregated_quantity").
        balanced: If True (default), require both periods to contain exactly
            the same products. If False, compute the index on the products
            present in both periods.
        panel_warnings: If True, emit a UserWarning summarising how many
            products are unmatched when balanced=False.

    Returns:
        The Walsh price index as a float.

    Raises:
        ValueError: If DataFrame doesn't have exactly two unique periods,
                   or if any product has multiple prices/quantities per period,
                   or if products differ between periods (balanced=True),
                   or if no products are common to both periods (balanced=False),
                   or if prices or quantities contain negatives or zeros.

    Examples:
        >>> from datetime import date
        >>> import polars as pl
        >>> df = pl.DataFrame({
        ...     "period": [date(2023, 1, 1), date(2023, 1, 1), date(2023, 2, 1), date(2023, 2, 1)],
        ...     "product_id": ["A", "B", "A", "B"],
        ...     "aggregated_price": [100, 200, 110, 190],
        ...     "aggregated_quantity": [10, 20, 10, 20]
        ... })
        >>> walsh(df)  # doctest: +ELLIPSIS
        0.982...
    """
    df = _to_canonical(df, period_col, price_col, id_col, quantity_col)
    _validate_weighted_bilateral_input(df, balanced=balanced, panel_warnings=panel_warnings)

    _, df_base, df_current = _split_pair(df)

    # Join on product_id (inner join keeps matched products only)
    joined = df_base.join(df_current, on="product_id", suffix="_current")

    # Compute fixed basket: sqrt(q_0 * q_t)
    joined = joined.with_columns(
        (pl.col("aggregated_quantity") * pl.col("aggregated_quantity_current"))
        .sqrt()
        .alias("basket")
    )

    # Compute numerator: sum(p_t * basket)
    numerator = joined.select(
        (pl.col("aggregated_price_current") * pl.col("basket")).sum()
    ).item()

    # Compute denominator: sum(p_0 * basket)
    denominator = joined.select(
        (pl.col("aggregated_price") * pl.col("basket")).sum()
    ).item()

    if denominator is None or denominator <= 0:
        raise ValueError("Cannot compute Walsh index with zero or negative base period expenditures")

    if numerator is None:
        raise ValueError("Cannot compute Walsh index with null prices")

    return numerator / denominator


def _validate_weighted_bilateral_input(
    df: pl.DataFrame,
    balanced: bool = True,
    panel_warnings: bool = False,
) -> None:
    """
    Validate DataFrame for weighted bilateral index computation.

    Args:
        df: Canonical DataFrame to validate.
        balanced: Whether both periods must contain the same products.
        panel_warnings: Whether to warn about unmatched products when
            balanced=False.

    Raises:
        ValueError: If validation fails
    """
    # Call base validation
    _validate_bilateral_input(df, balanced=balanced, panel_warnings=panel_warnings)

    # Check quantity column exists
    if "aggregated_quantity" not in df.columns:
        raise ValueError("Missing required column: aggregated_quantity")

    # Check no negative or zero quantities
    min_quantity = df.select(pl.col("aggregated_quantity").min()).item()
    if min_quantity <= 0:
        raise ValueError("All quantities must be positive")


def _validate_bilateral_input(
    df: pl.DataFrame,
    balanced: bool = True,
    panel_warnings: bool = False,
) -> None:
    """
    Validate DataFrame for bilateral index computation.

    Args:
        df: Canonical DataFrame to validate (columns "period",
            "aggregated_price", "product_id").
        balanced: If True, require both periods to contain exactly the same
            products (default). If False, require at least one common product.
        panel_warnings: If True and balanced=False, emit a UserWarning
            summarising unmatched products.

    Raises:
        ValueError: If validation fails
    """
    # Check required columns
    required_cols = ["period", "aggregated_price", "product_id"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Check exactly two periods
    periods = df.select("period").unique().sort("period")
    if periods.height != 2:
        raise ValueError(f"DataFrame must contain exactly 2 unique periods, found {periods.height}")

    # Check each product has exactly one price per period
    grouped = df.group_by(["product_id", "period"]).len()
    max_count = grouped.select(pl.col("len").max()).item()
    if max_count > 1:
        raise ValueError("Each product must have exactly one price per period (use aggregate_time first)")

    # Check product overlap between the two periods
    base_period, current_period = periods["period"].to_list()
    products_base = set(
        df.filter(pl.col("period") == base_period).select("product_id").to_series().to_list()
    )
    products_current = set(
        df.filter(pl.col("period") == current_period).select("product_id").to_series().to_list()
    )

    if balanced:
        if products_base != products_current:
            raise ValueError("Products must be identical in both periods")
    else:
        common = products_base & products_current
        if not common:
            raise ValueError("No common products between the two periods")
        if panel_warnings and products_base != products_current:
            warnings.warn(
                f"Unbalanced comparison between periods {base_period} and {current_period}: "
                f"{len(products_base - products_current)} product(s) present in base only, "
                f"{len(products_current - products_base)} present in current only; "
                f"index computed on {len(common)} matched product(s).",
                UserWarning,
            )

    # Check no negative or zero prices
    min_price = df.select(pl.col("aggregated_price").min()).item()
    if min_price <= 0:
        raise ValueError("All prices must be positive")
