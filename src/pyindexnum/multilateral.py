"""
Multilateral price index functions for the PyIndexNum library.

This module contains functions for calculating multilateral price indices
that compare prices across multiple time periods simultaneously.
"""

import polars as pl
import numpy as np
from scipy.optimize import root_scalar
from typing import Optional
from .bilateral import fisher, tornqvist, jevons


def geks_fisher(df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute the GEKS-Fisher multilateral price index.

    The GEKS (Generalized EKS) method uses bilateral Fisher indices between
    all pairs of periods. The price level for period t relative to period 1
    is the geometric mean of all possible bilateral links.

    Formula: P_geks_t = product_{k=1}^T [P_F(p^k, p^t, q^k, q^t) / P_F(p^k, p^1, q^k, q^1)]^(1/T)

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
    return _geks_base(df, fisher)


def geks_tornqvist(df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute the GEKS-Törnqvist (CCDI) multilateral price index.

    The GEKS method applied using the Törnqvist index as the underlying
    bilateral formula.

    Formula: P_ccdi_t = product_{k=1}^T [P_T(p^k, p^t, s^k, s^t) / P_T(p^k, p^1, s^k, s^1)]^(1/T)

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
    return _geks_base(df, tornqvist)


def geks_jevons(df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute the GEKS-Jevons multilateral price index.

    The GEKS method applied using the Jevons (unweighted geometric mean of
    price relatives) index as the underlying bilateral formula. Since the
    Jevons index is unweighted, this method only requires price information
    (though the quantity column must still be present for API consistency).

    Formula: P_geks-J(0,t) = product_{k=0}^{T-1} [P_J(k,t) / P_J(k,0)]^(1/T)

    where P_J(a,b) is the Jevons bilateral index between periods a and b:
    P_J(a,b) = [prod_{i=1}^{N} (p_i^b / p_i^a)]^(1/N)

    GEKS-Jevons is particularly useful for web-scraped data where quantity
    information is unavailable. Despite being unweighted, it has been found
    to outperform some weighted bilateral methods in empirical studies.

    Args:
        df: Polars DataFrame with standardized columns ("product_id", "period",
            "aggregated_price", "aggregated_quantity") containing data for
            multiple periods, with each product having exactly one price
            and quantity per period. Note: aggregated_quantity is required
            for input validation consistency but is not used in the computation.

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
        >>> result = geks_jevons(df)
        >>> # Returns DataFrame with period and index_value columns
    """
    return _geks_base(df, jevons)


def _geks_base(df: pl.DataFrame, bilateral_func) -> pl.DataFrame:
    """Helper for GEKS indices computation."""
    _validate_multilateral_input(df)

    periods = df.select("period").unique().sort("period").to_series().to_list()
    T = len(periods)

    if T < 2:
        raise ValueError("GEKS requires at least 2 periods")

    # Pre-calculate all bilateral indices P(k, t)
    # Store in a matrix-like dictionary for easy access
    bilateral_matrix = {}
    for i, period_i in enumerate(periods):
        for j, period_j in enumerate(periods):
            if i == j:
                bilateral_matrix[(i, j)] = 1.0
                continue
            
            # Filter data for the two periods
            df_i = df.filter(pl.col("period") == period_i)
            df_j = df.filter(pl.col("period") == period_j)

            # Important: Ensure periods are in chronological order for the bilateral function
            # The bilateral functions expect first date as base, second as current.
            if i < j:
                base_p, curr_p = period_i, period_j
                base_df, curr_df = df_i, df_j
            else:
                base_p, curr_p = period_j, period_i
                base_df, curr_df = df_j, df_i

            # Create bilateral dataframe format expected by bilateral functions
            bilateral_df = pl.concat([
                base_df.select(["product_id", "aggregated_price", "aggregated_quantity"])
                    .rename({"aggregated_price": "price", "aggregated_quantity": "quantity"})
                    .with_columns(pl.lit(base_p).alias("date")),
                curr_df.select(["product_id", "aggregated_price", "aggregated_quantity"])
                    .rename({"aggregated_price": "price", "aggregated_quantity": "quantity"})
                    .with_columns(pl.lit(curr_p).alias("date"))
            ])

            try:
                # Calculate bilateral index
                idx = bilateral_func(bilateral_df)
                
                # If i > j, we need the inverse because bilateral_func(bilateral_df) 
                # calculated P(j, i) but we want P(i, j)
                if i < j:
                    bilateral_matrix[(i, j)] = idx
                else:
                    bilateral_matrix[(i, j)] = 1.0 / idx
            except ValueError:
                # In GEKS, we should ideally have a complete graph, 
                # but if data is missing, this is a simplified version.
                # Here we assume full overlap for simplicity as per requirements.
                raise

    # Compute GEKS: P_geks_t = product_{k=0}^{T-1} [P(k, t) / P(k, 0)]^(1/T)
    # This is equivalent to linking period t to period 0 via all intermediate periods k
    indices = []
    for t in range(T):
        log_sum = 0.0
        for k in range(T):
            # log(P(k, t) / P(k, 0))
            # Note: bilateral_matrix stores P(i, j) = Price in j relative to i
            # So P(k, t) is price in t relative to k
            # P(k, 0) is price in 0 relative to k
            # P(k, t) / P(k, 0) is the link from 0 to t via k
            log_sum += np.log(bilateral_matrix[(k, t)] / bilateral_matrix[(k, 0)])
        
        geks_t = np.exp(log_sum / T)
        indices.append({"period": periods[t], "index_value": geks_t})

    return pl.DataFrame(indices)


def geary_khamis(df: pl.DataFrame, max_iter: int = 100, tol: float = 1e-8) -> pl.DataFrame:
    """
    Compute the Geary-Khamis multilateral price index.

    The Geary-Khamis method is an iterative multilateral index that solves
    for reference prices and period price levels simultaneously.

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
    """
    _validate_multilateral_input(df)

    periods = df.select("period").unique().sort("period").to_series().to_list()
    products = df.select("product_id").unique().to_series().to_list()
    T = len(periods)
    N = len(products)

    # Convert to matrix for faster iterative computation
    # Rows: periods, Cols: products
    pivot_price = df.pivot(index="period", on="product_id", values="aggregated_price").sort("period")
    pivot_qty = df.pivot(index="period", on="product_id", values="aggregated_quantity").sort("period")
    
    # Fill nulls if any (though validation should catch this)
    product_cols = [str(p) for p in products]
    P = pivot_price.select(product_cols).to_numpy()
    Q = pivot_qty.select(product_cols).to_numpy()

    # Initialize price levels P_GK^t = 1 for all t
    price_levels = np.ones(T)

    for iteration in range(max_iter):
        prev_price_levels = price_levels.copy()

        # 1. Calculate Reference Prices (alpha_n)
        # alpha_n = [sum_t (q_n^t * p_n^t / P_GK^t)] / [sum_t q_n^t]
        numerator_alpha = np.sum((Q * P) / price_levels[:, np.newaxis], axis=0)
        denominator_alpha = np.sum(Q, axis=0)
        alpha = numerator_alpha / denominator_alpha

        # 2. Update Period Price Levels (P_GK^t)
        # P_GK^t = [sum_n p_n^t * q_n^t] / [sum_n alpha_n * q_n^t]
        numerator_P = np.sum(P * Q, axis=1)
        denominator_P = np.sum(alpha * Q, axis=1)
        price_levels = numerator_P / denominator_P

        # Normalize price_levels to make the system stable (e.g., first period = 1)
        # This doesn't change the relatives but helps convergence
        price_levels /= price_levels[0]

        # Check convergence
        if np.max(np.abs(price_levels - prev_price_levels)) < tol:
            break
    else:
        raise ValueError(f"Geary-Khamis did not converge within {max_iter} iterations")

    indices = []
    for t in range(T):
        indices.append({
            "period": periods[t],
            "index_value": price_levels[t]
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
        weighted: If True, use weighted least squares with expenditure shares
                 (p*q / sum(p*q) per period) as weights. If False, use unweighted OLS.
                 If no quantity column, automatically uses unweighted regardless of
                 this parameter.

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

    # Weights for WLS: expenditure shares per period (p*q / sum(p*q) within each period)
    weights = None
    if weighted and "aggregated_quantity" in df.columns:
        prices_arr = df.select("aggregated_price").to_series().to_numpy()
        quantities_arr = df.select("aggregated_quantity").to_series().to_numpy()
        expenditure = prices_arr * quantities_arr
        periods_arr = df.select("period").to_series()

        weights = np.empty(n_obs)
        unique_periods = periods_arr.unique(maintain_order=True).to_list()
        for p in unique_periods:
            mask = (periods_arr == p)
            period_total = expenditure[mask.to_numpy()].sum()
            weights[mask.to_numpy()] = expenditure[mask.to_numpy()] / period_total

    # Perform regression using sqrt-weights trick to avoid dense N×N diagonal matrix.
    # Use QR decomposition via np.linalg.lstsq for numerical stability with
    # large, potentially ill-conditioned design matrices.
    if weighted and weights is not None:
        sqrt_w = np.sqrt(weights)
        Xw = X * sqrt_w[:, np.newaxis]
        yw = y * sqrt_w
        beta, _, _, _ = np.linalg.lstsq(Xw, yw, rcond=None)
    else:
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

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

    
