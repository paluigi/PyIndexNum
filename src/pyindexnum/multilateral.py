"""
Multilateral price index functions for the PyIndexNum library.

This module contains functions for calculating multilateral price indices
that compare prices across multiple time periods simultaneously.

All functions accept a DataFrame in the canonical panel schema produced by
``aggregate_time``: columns "product_id", "period", "aggregated_price" and
(for weighted methods) "aggregated_quantity". Custom column names can be
passed via the ``period_col``/``price_col``/``id_col``/``quantity_col``
arguments. The GEKS and Geary-Khamis methods require a balanced panel by
default (``balanced=True``); pass ``balanced=False`` to compute each period
comparison on the products matched between the periods involved instead.
The time-product-dummy method is regression-based and naturally handles
unbalanced panels without any flag.
"""

import warnings

import numpy as np
import polars as pl

from .bilateral import _to_canonical, fisher, jevons, tornqvist


def geks_fisher(
    df: pl.DataFrame,
    period_col: str = "period",
    price_col: str = "aggregated_price",
    id_col: str = "product_id",
    quantity_col: str = "aggregated_quantity",
    balanced: bool = True,
    panel_warnings: bool = False,
) -> pl.DataFrame:
    """
    Compute the GEKS-Fisher multilateral price index.

    The GEKS (Generalized EKS) method uses bilateral Fisher indices between
    all pairs of periods. The price level for period t relative to period 1
    is the geometric mean of all possible bilateral links.

    Formula: P_geks_t = product_{k=1}^T [P_F(p^k, p^t, q^k, q^t) / P_F(p^k, p^1, q^k, q^1)]^(1/T)

    Args:
        df: Polars DataFrame with the panel columns identified by period_col,
            price_col, id_col and quantity_col, containing data for multiple
            periods, with each product having exactly one price and quantity
            per period.
        period_col: Name of the period column (default "period").
        price_col: Name of the price column (default "aggregated_price").
        id_col: Name of the product identifier column (default "product_id").
        quantity_col: Name of the quantity column (default "aggregated_quantity").
        balanced: If True (default), require every product to be present in
            every period. If False, each bilateral link is computed on the
            products matched between the two periods involved, and the GEKS
            average for a period runs over the available links.
        panel_warnings: If True, emit a UserWarning summarising unmatched
            products and skipped period pairs when balanced=False.

    Returns:
        DataFrame with columns "period" and "index_value" (float),
        where index_value represents the multilateral price index for each period
        relative to the base period (first chronological period = 1.0).

    Raises:
        ValueError: If DataFrame doesn't meet requirements (see _validate_multilateral_input).

    Examples:
        >>> from datetime import date
        >>> import polars as pl
        >>> df = pl.DataFrame({
        ...     "product_id": ["A", "A", "B", "B"],
        ...     "period": [date(2023, 1, 1), date(2023, 2, 1), date(2023, 1, 1), date(2023, 2, 1)],
        ...     "aggregated_price": [100, 110, 200, 210],
        ...     "aggregated_quantity": [10, 10, 20, 20]
        ... })
        >>> result = geks_fisher(df)
        >>> # Returns DataFrame with period and index_value columns
    """
    df = _to_canonical(df, period_col, price_col, id_col, quantity_col)
    return _geks_base(df, fisher, weighted=True, balanced=balanced, panel_warnings=panel_warnings)


def geks_tornqvist(
    df: pl.DataFrame,
    period_col: str = "period",
    price_col: str = "aggregated_price",
    id_col: str = "product_id",
    quantity_col: str = "aggregated_quantity",
    balanced: bool = True,
    panel_warnings: bool = False,
) -> pl.DataFrame:
    """
    Compute the GEKS-Törnqvist (CCDI) multilateral price index.

    The GEKS method applied using the Törnqvist index as the underlying
    bilateral formula.

    Formula: P_ccdi_t = product_{k=1}^T [P_T(p^k, p^t, s^k, s^t) / P_T(p^k, p^1, s^k, s^1)]^(1/T)

    Args:
        df: Polars DataFrame with the panel columns identified by period_col,
            price_col, id_col and quantity_col, containing data for multiple
            periods, with each product having exactly one price and quantity
            per period.
        period_col: Name of the period column (default "period").
        price_col: Name of the price column (default "aggregated_price").
        id_col: Name of the product identifier column (default "product_id").
        quantity_col: Name of the quantity column (default "aggregated_quantity").
        balanced: If True (default), require every product to be present in
            every period. If False, each bilateral link is computed on the
            products matched between the two periods involved, and the GEKS
            average for a period runs over the available links.
        panel_warnings: If True, emit a UserWarning summarising unmatched
            products and skipped period pairs when balanced=False.

    Returns:
        DataFrame with columns "period" and "index_value" (float),
        where index_value represents the multilateral price index for each period
        relative to the base period (first chronological period = 1.0).

    Raises:
        ValueError: If DataFrame doesn't meet requirements (see _validate_multilateral_input).

    Examples:
        >>> from datetime import date
        >>> import polars as pl
        >>> df = pl.DataFrame({
        ...     "product_id": ["A", "A", "B", "B"],
        ...     "period": [date(2023, 1, 1), date(2023, 2, 1), date(2023, 1, 1), date(2023, 2, 1)],
        ...     "aggregated_price": [100, 110, 200, 210],
        ...     "aggregated_quantity": [10, 10, 20, 20]
        ... })
        >>> result = geks_tornqvist(df)
        >>> # Returns DataFrame with period and index_value columns
    """
    df = _to_canonical(df, period_col, price_col, id_col, quantity_col)
    return _geks_base(df, tornqvist, weighted=True, balanced=balanced, panel_warnings=panel_warnings)


def geks_jevons(
    df: pl.DataFrame,
    period_col: str = "period",
    price_col: str = "aggregated_price",
    id_col: str = "product_id",
    balanced: bool = True,
    panel_warnings: bool = False,
) -> pl.DataFrame:
    """
    Compute the GEKS-Jevons multilateral price index.

    The GEKS method applied using the Jevons (unweighted geometric mean of
    price relatives) index as the underlying bilateral formula. Since the
    Jevons index is unweighted, only price information is required — no
    quantity column is necessary.

    Formula: P_geks-J(0,t) = product_{k=0}^{T-1} [P_J(k,t) / P_J(k,0)]^(1/T)

    where P_J(a,b) is the Jevons bilateral index between periods a and b:
    P_J(a,b) = [prod_{i=1}^{N} (p_i^b / p_i^a)]^(1/N)

    GEKS-Jevons is particularly useful for web-scraped data where quantity
    information is unavailable. Despite being unweighted, it has been found
    to outperform some weighted bilateral methods in empirical studies.

    Args:
        df: Polars DataFrame with the panel columns identified by period_col,
            price_col and id_col, containing data for multiple periods, with
            each product having exactly one price per period. A quantity
            column, if present, is ignored (Jevons is an unweighted index).
        period_col: Name of the period column (default "period").
        price_col: Name of the price column (default "aggregated_price").
        id_col: Name of the product identifier column (default "product_id").
        balanced: If True (default), require every product to be present in
            every period. If False, each bilateral link is computed on the
            products matched between the two periods involved, and the GEKS
            average for a period runs over the available links.
        panel_warnings: If True, emit a UserWarning summarising unmatched
            products and skipped period pairs when balanced=False.

    Returns:
        DataFrame with columns "period" and "index_value" (float),
        where index_value represents the multilateral price index for each period
        relative to the base period (first chronological period = 1.0).

    Raises:
        ValueError: If DataFrame doesn't meet requirements (see _validate_multilateral_input).

    Examples:
        >>> from datetime import date
        >>> import polars as pl
        >>> df = pl.DataFrame({
        ...     "product_id": ["A", "A", "B", "B"],
        ...     "period": [date(2023, 1, 1), date(2023, 2, 1), date(2023, 1, 1), date(2023, 2, 1)],
        ...     "aggregated_price": [100, 110, 200, 210],
        ... })
        >>> result = geks_jevons(df)
        >>> # Returns DataFrame with period and index_value columns
    """
    df = _to_canonical(df, period_col, price_col, id_col)
    return _geks_base(df, jevons, weighted=False, balanced=balanced, panel_warnings=panel_warnings)


def _geks_base(
    df: pl.DataFrame,
    bilateral_func,
    weighted: bool = True,
    balanced: bool = True,
    panel_warnings: bool = False,
) -> pl.DataFrame:
    """Helper for GEKS indices computation.

    Args:
        df: Input DataFrame in canonical panel schema.
        bilateral_func: Bilateral index function to use (e.g., fisher, jevons).
        weighted: If True, require aggregated_quantity column. If False, only
            require aggregated_price (suitable for unweighted indices like Jevons).
        balanced: If True, require a complete panel. If False, compute each
            bilateral link on the matched products of the pair involved.
        panel_warnings: If True and balanced=False, emit one summary warning
            about unmatched products and skipped pairs.
    """
    _validate_multilateral_input(df, weighted, balanced=balanced)

    has_quantity = "aggregated_quantity" in df.columns

    periods = df.select("period").unique().sort("period").to_series().to_list()
    T = len(periods)

    if T < 2:
        raise ValueError("GEKS requires at least 2 periods")

    # Product set observed in each period, for per-pair matching
    product_sets = {
        period: set(
            df.filter(pl.col("period") == period).select("product_id").to_series().to_list()
        )
        for period in periods
    }

    pair_cols = ["product_id", "period", "aggregated_price"]
    if has_quantity:
        pair_cols.append("aggregated_quantity")

    # Pre-calculate all bilateral indices P(i, j) for i < j (price in j
    # relative to i), stored symmetrically. Pairs with no common products
    # are stored as None and skipped in the final average.
    bilateral_matrix = {(i, i): 1.0 for i in range(T)}
    unmatched_pairs = []
    skipped_pairs = []

    for i in range(T):
        for j in range(i + 1, T):
            common = product_sets[periods[i]] & product_sets[periods[j]]
            if not common:
                bilateral_matrix[(i, j)] = None
                bilateral_matrix[(j, i)] = None
                skipped_pairs.append((periods[i], periods[j]))
                continue

            # Chronological order is guaranteed by i < j: period i is the
            # base of the bilateral comparison. The bilateral call itself
            # matches products on the common set (balanced=False); with a
            # complete panel the two modes coincide.
            pair_df = pl.concat([
                df.filter(pl.col("period") == periods[i]).select(pair_cols),
                df.filter(pl.col("period") == periods[j]).select(pair_cols),
            ])
            idx = bilateral_func(pair_df, balanced=False, panel_warnings=False)
            bilateral_matrix[(i, j)] = idx
            bilateral_matrix[(j, i)] = 1.0 / idx

            n_unmatched = len(product_sets[periods[i]] ^ product_sets[periods[j]])
            if n_unmatched:
                unmatched_pairs.append((periods[i], periods[j], n_unmatched))

    if panel_warnings and (unmatched_pairs or skipped_pairs):
        parts = []
        if unmatched_pairs:
            total_unmatched = sum(n for _, _, n in unmatched_pairs)
            parts.append(
                f"{len(unmatched_pairs)} of {T * (T - 1) // 2} period pairs have unmatched "
                f"products ({total_unmatched} unmatched product occurrences in total); "
                "each link uses the products matched between its two periods"
            )
        if skipped_pairs:
            formatted = ", ".join(f"{p} vs {q}" for p, q in skipped_pairs)
            parts.append(f"period pairs with no common products were skipped: {formatted}")
        warnings.warn(
            "Unbalanced panel: " + "; ".join(parts) + ".",
            UserWarning,
        )

    # Compute GEKS: P_geks_t = product_{k=0}^{T-1} [P(k, t) / P(k, 0)]^(1/T)
    # This is equivalent to linking period t to period 0 via all intermediate periods k.
    # With an unbalanced panel, links whose pair has no common products are
    # skipped and the average runs over the available links only.
    indices = []
    for t in range(T):
        log_sum = 0.0
        n_links = 0
        for k in range(T):
            # log(P(k, t) / P(k, 0))
            # Note: bilateral_matrix stores P(i, j) = Price in j relative to i
            # So P(k, t) is price in t relative to k
            # P(k, 0) is price in 0 relative to k
            # P(k, t) / P(k, 0) is the link from 0 to t via k
            p_kt = bilateral_matrix.get((k, t))
            p_k0 = bilateral_matrix.get((k, 0))
            if p_kt is None or p_k0 is None:
                continue
            log_sum += np.log(p_kt / p_k0)
            n_links += 1

        if n_links == 0:
            raise ValueError(
                f"No common products between period {periods[t]} and any other period; "
                "cannot compute GEKS index"
            )

        geks_t = np.exp(log_sum / n_links)
        indices.append({"period": periods[t], "index_value": geks_t})

    return pl.DataFrame(indices)


def geary_khamis(
    df: pl.DataFrame,
    max_iter: int = 100,
    tol: float = 1e-8,
    period_col: str = "period",
    price_col: str = "aggregated_price",
    id_col: str = "product_id",
    quantity_col: str = "aggregated_quantity",
    balanced: bool = True,
    panel_warnings: bool = False,
) -> pl.DataFrame:
    """
    Compute the Geary-Khamis multilateral price index.

    The Geary-Khamis method is an iterative multilateral index that solves
    for reference prices and period price levels simultaneously.

    With ``balanced=False``, missing (product, period) combinations are
    excluded from the reference-price and price-level sums (the iteration
    runs over observed cells only). Convergence is well-defined for complete
    panels; for strongly unbalanced panels it is not theoretically guaranteed,
    so ``max_iter`` and ``tol`` remain available as safeguards.

    Args:
        df: Polars DataFrame with the panel columns identified by period_col,
            price_col, id_col and quantity_col, containing data for multiple
            periods, with each product having exactly one price and quantity
            per period.
        max_iter: Maximum number of iterations for convergence (default 100).
        tol: Tolerance for convergence check (default 1e-8).
        period_col: Name of the period column (default "period").
        price_col: Name of the price column (default "aggregated_price").
        id_col: Name of the product identifier column (default "product_id").
        quantity_col: Name of the quantity column (default "aggregated_quantity").
        balanced: If True (default), require every product to be present in
            every period. If False, run the iteration over observed cells.
        panel_warnings: If True, emit a UserWarning when the panel is
            unbalanced and balanced=False.

    Returns:
        DataFrame with columns "period" and "index_value" (float),
        where index_value represents the multilateral price index for each period
        relative to the base period (first chronological period = 1.0).

    Raises:
        ValueError: If DataFrame doesn't meet requirements or iteration doesn't converge.
    """
    df = _to_canonical(df, period_col, price_col, id_col, quantity_col)
    _validate_multilateral_input(df, weighted=True, balanced=balanced)

    if panel_warnings and not balanced:
        n_unbalanced = _count_unbalanced_products(df)
        if n_unbalanced:
            warnings.warn(
                f"Unbalanced panel: {n_unbalanced} product(s) are not present in all periods; "
                "the Geary-Khamis iteration runs over observed cells only.",
                UserWarning,
            )

    periods = df.select("period").unique().sort("period").to_series().to_list()
    products = df.select("product_id").unique().to_series().to_list()
    T = len(periods)
    N = len(products)

    # Convert to matrix for faster iterative computation
    # Rows: periods, Cols: products
    pivot_price = df.pivot(index="period", on="product_id", values="aggregated_price").sort("period")
    pivot_qty = df.pivot(index="period", on="product_id", values="aggregated_quantity").sort("period")

    # Missing (product, period) combinations pivot to null -> NaN, which the
    # masked (nansum) iteration below treats as unobserved cells
    product_cols = [str(p) for p in products]
    P = pivot_price.select(product_cols).to_numpy().astype(float)
    Q = pivot_qty.select(product_cols).to_numpy().astype(float)

    # Initialize price levels P_GK^t = 1 for all t
    price_levels = np.ones(T)

    for iteration in range(max_iter):
        prev_price_levels = price_levels.copy()

        # 1. Calculate Reference Prices (alpha_n)
        # alpha_n = [sum_t (q_n^t * p_n^t / P_GK^t)] / [sum_t q_n^t]
        # (sums over the periods where product n is observed)
        numerator_alpha = np.nansum((Q * P) / price_levels[:, np.newaxis], axis=0)
        denominator_alpha = np.nansum(Q, axis=0)
        alpha = numerator_alpha / denominator_alpha

        # 2. Update Period Price Levels (P_GK^t)
        # P_GK^t = [sum_n p_n^t * q_n^t] / [sum_n alpha_n * q_n^t]
        # (sums over the products observed in period t)
        numerator_P = np.nansum(P * Q, axis=1)
        denominator_P = np.nansum(alpha * Q, axis=1)
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


def time_product_dummy(
    df: pl.DataFrame,
    weighted: bool = True,
    period_col: str = "period",
    price_col: str = "aggregated_price",
    id_col: str = "product_id",
    quantity_col: str = "aggregated_quantity",
) -> pl.DataFrame:
    """
    Compute the Time Product Dummy multilateral price index.

    The Time Product Dummy (TPD) method uses regression analysis to estimate
    price indices. Time and product dummy variables are included in the model,
    with the index values derived from the time dummy coefficients.

    The estimation is row-based (one row per observed product-period
    combination), so the method naturally handles unbalanced panels without
    imputation or product matching; expenditure-share weights are
    re-normalised within each period's observed products.

    Note:
        Products observed in only one period have their dummy variables
        collinear with the time dummies; the least-squares solver resolves
        this silently with a minimum-norm solution. Consider removing such
        products for cleaner estimates.

    Args:
        df: Polars DataFrame with the panel columns identified by period_col,
            price_col, id_col (and quantity_col if weighted=True), containing
            data for multiple periods, with each product having exactly one
            price per period.
        weighted: If True, use weighted least squares with expenditure shares
                 (p*q / sum(p*q) per period) as weights. This requires the
                 quantity column to be present. If False, use unweighted OLS.
        period_col: Name of the period column (default "period").
        price_col: Name of the price column (default "aggregated_price").
        id_col: Name of the product identifier column (default "product_id").
        quantity_col: Name of the quantity column (default "aggregated_quantity").

    Returns:
        DataFrame with columns "period" and "index_value" (float),
        where index_value represents the multilateral price index for each period
        relative to the base period (first chronological period = 1.0).

    Raises:
        ValueError: If DataFrame doesn't meet requirements.

    Examples:
        >>> from datetime import date
        >>> import polars as pl
        >>> df = pl.DataFrame({
        ...     "product_id": ["A", "A", "B", "B"],
        ...     "period": [date(2023, 1, 1), date(2023, 2, 1), date(2023, 1, 1), date(2023, 2, 1)],
        ...     "aggregated_price": [100, 110, 200, 210],
        ...     "aggregated_quantity": [10, 10, 20, 20]
        ... })
        >>> result = time_product_dummy(df, weighted=True)
        >>> # Returns DataFrame with period and index_value columns
    """
    df = _to_canonical(df, period_col, price_col, id_col, quantity_col if weighted else None)
    _validate_multilateral_input(df, weighted, balanced=False)

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


def _count_unbalanced_products(df: pl.DataFrame) -> int:
    """
    Count products that are not observed in every period.

    Args:
        df: Canonical DataFrame with columns "product_id" and "period".

    Returns:
        Number of products missing from at least one period.
    """
    total_periods = df.select("period").unique().height
    product_period_counts = df.group_by("product_id").agg(
        pl.col("period").n_unique().alias("period_count")
    )
    return product_period_counts.filter(pl.col("period_count") < total_periods).height


def _validate_multilateral_input(
    df: pl.DataFrame,
    weighted: bool = True,
    balanced: bool = True,
) -> None:
    """
    Validate DataFrame for multilateral index computation.

    Args:
        df: Canonical DataFrame to validate.
        weighted: Whether weighted regression is requested, default True
        balanced: If True (default), require every product to be present in
            every period. If False, unbalanced panels are accepted.

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

    # Check the panel is complete when a balanced panel is required
    if balanced:
        n_unbalanced = _count_unbalanced_products(df)
        if n_unbalanced:
            raise ValueError(
                f"Panel is not balanced: {n_unbalanced} product(s) are not present in all periods. "
                "Pass balanced=False to compute the index on matched product sets, "
                "or use remove_unbalanced()/carry_forward_imputation() first."
            )

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
