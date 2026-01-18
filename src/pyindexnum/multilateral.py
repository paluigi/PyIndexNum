"""
Multilateral price index functions for the PyIndexNum library.

This module contains functions for calculating multilateral price indices
that compare prices across multiple time periods simultaneously.
"""

import polars as pl
import numpy as np
from scipy.optimize import root_scalar
from typing import Optional
from .bilateral import fisher, tornqvist


def geks_fisher(df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute the GEKS-Fisher multilateral price index.

    The GEKS (Generalized EKS) method uses bilateral Fisher indices between
    all pairs of periods, then takes the geometric mean for each period.
    This produces transitive multilateral indices.

    Args:
        df: Polars DataFrame with standardized columns ("product_id", "period",
            "aggregated_price", "aggregated_quantity") containing data for
            multiple periods, with each product having exactly one price
            and quantity per period.

    Returns:
        DataFrame with columns "period" (Date) and "index_value" (float),
        where index_value represents the multilateral price index for each period
        relative to the base period (first chronological period = 1.0).

    Raises:
        ValueError: If DataFrame doesn't meet requirements (see _validate_multilateral_input).

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({
        ...     "product_id": ["A", "A", "B", "B"],
        ...     "period": [pl.date(2023, 1, 1), pl.date(2023, 2, 1), pl.date(2023, 1, 1), pl.date(2023, 2, 1)],
        ...     "aggregated_price": [100, 110, 200, 210],
        ...     "aggregated_quantity": [10, 10, 20, 20]
        ... })
        >>> result = geks_fisher(df)
        >>> # Returns DataFrame with period and index_value columns
    """
    _validate_multilateral_input(df)

    periods = df.select("period").unique().sort("period").to_series().to_list()

    if len(periods) < 2:
        raise ValueError("GEKS requires at least 2 periods")

    # Compute bilateral Fisher indices for all pairs
    bilateral_indices = {}
    for i, period_i in enumerate(periods):
        for j, period_j in enumerate(periods):
            if i < j:  # Only compute each pair once
                df_i = df.filter(pl.col("period") == period_i)
                df_j = df.filter(pl.col("period") == period_j)

                # Create bilateral dataframe
                bilateral_df = pl.concat([
                    df_i.select(["product_id", "aggregated_price", "aggregated_quantity"])
                        .rename({"aggregated_price": "price", "aggregated_quantity": "quantity"})
                        .with_columns(pl.lit(period_i).alias("date")),
                    df_j.select(["product_id", "aggregated_price", "aggregated_quantity"])
                        .rename({"aggregated_price": "price", "aggregated_quantity": "quantity"})
                        .with_columns(pl.lit(period_j).alias("date"))
                ])

                try:
                    idx = fisher(bilateral_df)
                    bilateral_indices[(i, j)] = idx
                    bilateral_indices[(j, i)] = 1.0 / idx  # Store inverse for efficiency
                except ValueError:
                    # If Fisher fails, skip this pair
                    continue

    # Compute GEKS indices
    indices = []
    for i, period in enumerate(periods):
        # Get all bilateral indices involving this period
        period_indices = []
        for j, other_period in enumerate(periods):
            if i != j:
                if (min(i, j), max(i, j)) in bilateral_indices:
                    # Get the index where this period is the base
                    if i < j:
                        period_indices.append(bilateral_indices[(i, j)])
                    else:
                        period_indices.append(bilateral_indices[(j, i)])

        if period_indices:
            # Geometric mean of bilateral indices
            geks_index = np.exp(np.mean(np.log(period_indices)))
            indices.append({"period": period, "index_value": geks_index})
        else:
            indices.append({"period": period, "index_value": 1.0})

    # Set base period (first period) to 1.0 and adjust others
    base_index = indices[0]["index_value"]
    for idx in indices:
        idx["index_value"] = idx["index_value"] / base_index

    return pl.DataFrame(indices)


def geks_tornqvist(df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute the GEKS-Törnqvist multilateral price index.

    The GEKS (Generalized EKS) method uses bilateral Törnqvist indices between
    all pairs of periods, then takes the geometric mean for each period.
    This produces transitive multilateral indices.

    Args:
        df: Polars DataFrame with standardized columns ("product_id", "period",
            "aggregated_price", "aggregated_quantity") containing data for
            multiple periods, with each product having exactly one price
            and quantity per period.

    Returns:
        DataFrame with columns "period" (Date) and "index_value" (float),
        where index_value represents the multilateral price index for each period
        relative to the base period (first chronological period = 1.0).

    Raises:
        ValueError: If DataFrame doesn't meet requirements (see _validate_multilateral_input).

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({
        ...     "product_id": ["A", "A", "B", "B"],
        ...     "period": [pl.date(2023, 1, 1), pl.date(2023, 2, 1), pl.date(2023, 1, 1), pl.date(2023, 2, 1)],
        ...     "aggregated_price": [100, 110, 200, 210],
        ...     "aggregated_quantity": [10, 10, 20, 20]
        ... })
        >>> result = geks_tornqvist(df)
        >>> # Returns DataFrame with period and index_value columns
    """
    _validate_multilateral_input(df)

    periods = df.select("period").unique().sort("period").to_series().to_list()

    if len(periods) < 2:
        raise ValueError("GEKS requires at least 2 periods")

    # Compute bilateral Törnqvist indices for all pairs
    bilateral_indices = {}
    for i, period_i in enumerate(periods):
        for j, period_j in enumerate(periods):
            if i < j:  # Only compute each pair once
                df_i = df.filter(pl.col("period") == period_i)
                df_j = df.filter(pl.col("period") == period_j)

                # Create bilateral dataframe
                bilateral_df = pl.concat([
                    df_i.select(["product_id", "aggregated_price", "aggregated_quantity"])
                        .rename({"aggregated_price": "price", "aggregated_quantity": "quantity"})
                        .with_columns(pl.lit(period_i).alias("date")),
                    df_j.select(["product_id", "aggregated_price", "aggregated_quantity"])
                        .rename({"aggregated_price": "price", "aggregated_quantity": "quantity"})
                        .with_columns(pl.lit(period_j).alias("date"))
                ])

                try:
                    idx = tornqvist(bilateral_df)
                    bilateral_indices[(i, j)] = idx
                    bilateral_indices[(j, i)] = 1.0 / idx  # Store inverse for efficiency
                except ValueError:
                    # If Törnqvist fails, skip this pair
                    continue

    # Compute GEKS indices
    indices = []
    for i, period in enumerate(periods):
        # Get all bilateral indices involving this period
        period_indices = []
        for j, other_period in enumerate(periods):
            if i != j:
                if (min(i, j), max(i, j)) in bilateral_indices:
                    # Get the index where this period is the base
                    if i < j:
                        period_indices.append(bilateral_indices[(i, j)])
                    else:
                        period_indices.append(bilateral_indices[(j, i)])

        if period_indices:
            # Geometric mean of bilateral indices
            geks_index = np.exp(np.mean(np.log(period_indices)))
            indices.append({"period": period, "index_value": geks_index})
        else:
            indices.append({"period": period, "index_value": 1.0})

    # Set base period (first period) to 1.0 and adjust others
    base_index = indices[0]["index_value"]
    for idx in indices:
        idx["index_value"] = idx["index_value"] / base_index

    return pl.DataFrame(indices)


def geary_khamis(df: pl.DataFrame, max_iter: int = 100, tol: float = 1e-8) -> pl.DataFrame:
    """
    Compute the Geary-Khamis multilateral price index.

    The Geary-Khamis method is an iterative multilateral index that solves
    for price levels and quantity weights simultaneously. It produces
    transitive indices that satisfy circularity tests.

    Args:
        df: Polars DataFrame with standardized columns ("product_id", "period",
            "aggregated_price", "aggregated_quantity") containing data for
            multiple periods, with each product having exactly one price
            and quantity per period.
        max_iter: Maximum number of iterations for convergence (default 100).
        tol: Tolerance for convergence check (default 1e-8).

    Returns:
        DataFrame with columns "period" (Date) and "index_value" (float),
        where index_value represents the multilateral price index for each period
        relative to the base period (first chronological period = 1.0).

    Raises:
        ValueError: If DataFrame doesn't meet requirements or iteration doesn't converge.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({
        ...     "product_id": ["A", "A", "B", "B"],
        ...     "period": [pl.date(2023, 1, 1), pl.date(2023, 2, 1), pl.date(2023, 1, 1), pl.date(2023, 2, 1)],
        ...     "aggregated_price": [100, 110, 200, 210],
        ...     "aggregated_quantity": [10, 10, 20, 20]
        ... })
        >>> result = geary_khamis(df)
        >>> # Returns DataFrame with period and index_value columns
    """
    _validate_multilateral_input(df)

    periods = df.select("period").unique().sort("period").to_series().to_list()
    products = df.select("product_id").unique().to_series().to_list()

    if len(periods) < 2 or len(products) < 2:
        raise ValueError("Geary-Khamis requires at least 2 periods and 2 products")

    # Initialize price levels (all start at 1.0)
    price_levels = {period: 1.0 for period in periods}

    # Initialize quantity weights (arithmetic mean of quantities across periods)
    quantity_weights = {}
    for product in products:
        product_data = df.filter(pl.col("product_id") == product)
        avg_quantity = product_data.select(pl.col("aggregated_quantity").mean()).item()
        quantity_weights[product] = avg_quantity

    # Iterative solution
    for iteration in range(max_iter):
        # Update price levels
        new_price_levels = {}
        for period in periods:
            period_data = df.filter(pl.col("period") == period)

            numerator = 0.0
            denominator = 0.0

            for product in products:
                product_row = period_data.filter(pl.col("product_id") == product)
                if product_row.height > 0:
                    price = product_row.select("aggregated_price").item()
                    quantity = quantity_weights[product]
                    numerator += price * quantity
                    denominator += quantity

            if denominator > 0:
                new_price_levels[period] = numerator / denominator
            else:
                new_price_levels[period] = 1.0

        # Update quantity weights
        new_quantity_weights = {}
        for product in products:
            product_data = df.filter(pl.col("product_id") == product)

            numerator = 0.0
            denominator = 0.0

            for period in periods:
                period_row = product_data.filter(pl.col("period") == period)
                if period_row.height > 0:
                    price = period_row.select("aggregated_price").item()
                    price_level = new_price_levels[period]
                    numerator += price / price_level
                    denominator += 1.0

            if denominator > 0:
                new_quantity_weights[product] = numerator / denominator
            else:
                new_quantity_weights[product] = quantity_weights[product]

        # Check convergence
        max_diff_levels = max(abs(price_levels[p] - new_price_levels[p]) for p in periods)
        max_diff_weights = max(abs(quantity_weights[p] - new_quantity_weights[p]) for p in products)

        price_levels = new_price_levels
        quantity_weights = new_quantity_weights

        if max_diff_levels < tol and max_diff_weights < tol:
            break

    if iteration == max_iter - 1:
        raise ValueError(f"Geary-Khamis did not converge within {max_iter} iterations")

    # Create result DataFrame, normalize to base period
    base_period = periods[0]
    base_level = price_levels[base_period]

    indices = []
    for period in periods:
        indices.append({
            "period": period,
            "index_value": price_levels[period] / base_level
        })

    return pl.DataFrame(indices)


def time_product_dummy(df: pl.DataFrame, weighted: bool = True) -> pl.DataFrame:
    """
    Compute the Time Product Dummy multilateral price index.

    The Time Product Dummy (TPD) method uses regression analysis to estimate
    price indices. Time and product dummy variables are included in the model,
    with the index values derived from the time dummy coefficients.

    Args:
        df: Polars DataFrame with standardized columns ("product_id", "period",
            "aggregated_price") and optionally "aggregated_quantity" if weighted=True.
            Contains data for multiple periods, with each product having exactly
            one price per period.
        weighted: If True, use weighted least squares with aggregated_quantity as weights.
                 If False, use unweighted OLS. If no quantity column, automatically
                 uses unweighted regardless of this parameter.

    Returns:
        DataFrame with columns "period" (Date) and "index_value" (float),
        where index_value represents the multilateral price index for each period
        relative to the base period (first chronological period = 1.0).

    Raises:
        ValueError: If DataFrame doesn't meet requirements.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({
        ...     "product_id": ["A", "A", "B", "B"],
        ...     "period": [pl.date(2023, 1, 1), pl.date(2023, 2, 1), pl.date(2023, 1, 1), pl.date(2023, 2, 1)],
        ...     "aggregated_price": [100, 110, 200, 210],
        ...     "aggregated_quantity": [10, 10, 20, 20]
        ... })
        >>> result = time_product_dummy(df, weighted=True)
        >>> # Returns DataFrame with period and index_value columns
    """
    _validate_multilateral_input(df, weighted)

    periods = df.select("period").unique().sort("period").to_series().to_list()
    products = df.select("product_id").unique().to_series().to_list()

    if len(periods) < 2 or len(products) < 2:
        raise ValueError("Time Product Dummy requires at least 2 periods and 2 products")

    # Create design matrix for regression
    # Dependent variable: log(price)
    # Independent variables: time dummies + product dummies

    # Create time dummy variables (exclude base period)
    base_period = periods[0]
    time_dummies = {}
    for period in periods[1:]:  # Skip base period
        time_dummies[period] = [1 if p == period else 0 for p in df.select("period").to_series()]

    # Create product dummy variables (exclude one product to avoid multicollinearity)
    base_product = products[0]
    product_dummies = {}
    for product in products[1:]:  # Skip base product
        product_dummies[product] = [1 if p == product else 0 for p in df.select("product_id").to_series()]

    # Prepare X matrix (design matrix)
    n_obs = df.height
    n_time_dummies = len(time_dummies)
    n_product_dummies = len(product_dummies)
    n_vars = n_time_dummies + n_product_dummies

    X = np.zeros((n_obs, n_vars))

    # Fill time dummies
    for i, (period, dummy_vals) in enumerate(time_dummies.items()):
        X[:, i] = dummy_vals

    # Fill product dummies
    for i, (product, dummy_vals) in enumerate(product_dummies.items()):
        X[:, n_time_dummies + i] = dummy_vals

    # Add intercept column (for base period and base product)
    X = np.column_stack([np.ones(n_obs), X])

    # Dependent variable: log of prices
    y = np.log(df.select("aggregated_price").to_series().to_numpy())

    # Weights for WLS (if weighted and quantities available)
    weights = None
    if weighted and "aggregated_quantity" in df.columns:
        weights = df.select("aggregated_quantity").to_series().to_numpy()
        # Normalize weights
        weights = weights / np.sum(weights) * len(weights)

    # Perform regression
    if weighted and weights is not None:
        # Weighted least squares
        W = np.diag(weights)
        beta = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ y
    else:
        # Ordinary least squares
        beta = np.linalg.inv(X.T @ X) @ X.T @ y

    # Extract time dummy coefficients
    # beta[0] is intercept (base period)
    # beta[1:n_time_dummies+1] are time dummy coefficients
    indices = [{"period": base_period, "index_value": 1.0}]  # Base period = 1.0

    for i, period in enumerate(periods[1:], 1):
        index_value = np.exp(beta[i])  # exp(time_dummy_coeff)
        indices.append({"period": period, "index_value": index_value})

    return pl.DataFrame(indices)


def _validate_multilateral_input(df: pl.DataFrame, weighted: bool = True) -> None:
    """
    Validate DataFrame for multilateral index computation.

    Args:
        df: DataFrame to validate
        weighted: Whether weighted regression is requested, defult True

    Raises:
        ValueError: If validation fails
    """
    # Check required columns
    if weighted:
        required_cols = ["product_id", "period", "aggregated_price", "aggregated_quantity"]
    else:
        required_cols = ["product_id", "period", "aggregated_price"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Check data types
    if not df.schema["aggregated_price"].is_numeric():
        raise ValueError("aggregated_price must be numeric")

    # Check no negative or zero prices
    min_price = df.select(pl.col("aggregated_price").min()).item()
    if min_price <= 0:
        raise ValueError("All aggregated_price values must be positive")

    # Check each product has exactly one observation per period
    grouped = df.group_by(["product_id", "period"]).len()
    max_count = grouped.select(pl.col("len").max()).item()
    if max_count > 1:
        raise ValueError("Each product must have exactly one observation per period")

    # Check we have at least 2 periods
    n_periods = df.select("period").unique().height
    if n_periods < 2:
        raise ValueError("Multilateral indices require at least 2 periods")

    # Check we have at least 2 products
    n_products = df.select("product_id").unique().height
    if n_products < 2:
        raise ValueError("Multilateral indices require at least 2 products")

    # Additional checks for weighted indices
    if weighted:
        # Check aggregated_quantity is numeric if weighted
        if "aggregated_quantity" in df.columns and not df.schema["aggregated_quantity"].is_numeric():
            raise ValueError("aggregated_quantity must be numeric")
        # Check quantities are positive if present
        if "aggregated_quantity" in df.columns:
            min_quantity = df.select(pl.col("aggregated_quantity").min()).item()
            if min_quantity <= 0:
                raise ValueError("All aggregated_quantity values must be positive")

    
