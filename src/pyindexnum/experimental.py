"""
Experimental multilateral price index functions for spatial comparisons.

This module hosts index methods that target cross-sectional (spatial)
comparisons across regions rather than comparisons over time. The API in
this module is experimental: it intentionally deviates from the package's
temporal conventions (an ``entity_col`` defaulting to "region" instead of
``period_col``, value columns named after the method instead of
"index_value") and may change in future releases. If the methods
stabilise, they will either migrate to the standard multilateral
conventions or the core API will generalise towards entity-style column
naming.

Spatial GEKS-Paasche and GEKS-Laspeyres
---------------------------------------

The classic GEKS combination of bilateral indices presupposes a bilateral
formula that satisfies time (country) reversal, P(j, k) * P(k, j) = 1, as
Fisher and Törnqvist do. Paasche and Laspeyres indices use asymmetric
expenditure weights and fail this test, so the traditional GEKS
implementation is not conceptually sound for them. In particular, the
symmetric least-squares regression formulation
``ln P(j, k) = pi_k - pi_j + u`` (Rao and Timmer 2003; Diewert 1999)
reproduces the GEKS-Fisher parities exactly when fed Paasche or Laspeyres
bilateral indices (van IJzeren 1987, pp. 62-65, reported in Diewert 1999,
fn. 28), which would make a dedicated implementation redundant.

This module therefore uses the one-sided bridge form: each term
``ln[P(m, k) / P(m, b)]`` is the two-hop parity estimate of entity k
against the base entity b through bridge entity m, and the index is the
average of these estimates over all bridges. The resulting indices are
transitive and base invariant without requiring country reversal, remain
distinct from GEKS-Fisher, and reduce to the bilateral Fisher index when
only two entities are compared. ``geks_laspeyres`` implements the GEKS-L
index of Białek (2022) applied over regions; ``geks_paasche`` is its
Paasche mirror image (every comparison into a region uses that region's
own basket).

Data arrive in the same panel schema as the rest of the package, with the
entity (region) column playing the role of the period column. For data
produced by the temporal pipeline (``standardize_columns`` followed by
``aggregate_time``, which emits a "period" column), rename the column
before calling these functions::

    df_region = df_agg.rename({"period": "region"})
    pin.geks_paasche(df_region)

Unbalanced panels are handled on matched product sets: each bilateral link
uses the products matched between its two entities, each entity's bridge
average runs over its available links, and entity pairs with no common
products are skipped (a summary ``UserWarning`` is emitted when
``panel_warnings=True``). An entity with no link to any other entity
raises a ``ValueError``.
"""

import warnings
from typing import Literal, Optional

import numpy as np
import polars as pl

from .bilateral import _to_canonical


def geks_paasche(
    df: pl.DataFrame,
    entity_col: str = "region",
    id_col: str = "product_id",
    price_col: str = "aggregated_price",
    quantity_col: str = "aggregated_quantity",
    base_entity: Optional[object] = None,
    panel_warnings: bool = False,
) -> pl.DataFrame:
    """
    Compute the GEKS-Paasche multilateral price index for spatial comparisons.

    Each entity's price level relative to the base entity is estimated by
    averaging bilateral Paasche comparisons over all possible bridge
    entities. Every bilateral link into an entity uses that entity's own
    expenditure basket (the Paasche weighting), which makes this index the
    mirror image of Białek's (2022) GEKS-L. The index is transitive and
    base invariant without requiring country reversal, and reduces to the
    bilateral Fisher index when only two entities are compared.

    Formula: P_GEKS-P(k) = product_{m in M} [P_P(m, k) / P_P(m, b)]^(1/M)

    where P_P(m, k) is the bilateral Paasche index with entity m as base
    and entity k as current (k's quantities as weights), computed on the
    products matched between m and k; b is the base entity and M is the
    number of entities.

    Args:
        df: Polars DataFrame with the panel columns identified by
            entity_col, price_col, id_col and quantity_col, containing
            data for multiple entities, with each product having exactly
            one price and quantity per entity. Unbalanced panels are
            supported: each bilateral link is computed on the products
            matched between the two entities involved.
        entity_col: Name of the entity column (default "region"). For
            spatial data this is typically a region column; for data from
            the temporal pipeline, rename "period" to "region" first
            (df.rename({"period": "region"})).
        id_col: Name of the product identifier column (default "product_id").
        price_col: Name of the price column (default "aggregated_price").
        quantity_col: Name of the quantity column
            (default "aggregated_quantity").
        base_entity: The reference entity whose index value is 1.0. If
            None (default), the first sorted entity is used (for string
            labels, the alphabetically first region).
        panel_warnings: If True, emit a UserWarning summarising unmatched
            products and skipped entity pairs.

    Returns:
        DataFrame with columns [entity_col, "geks_paasche"] (float), where
        the value is the multilateral price index of each entity relative
        to the base entity (base entity = 1.0). Note: this output schema
        intentionally differs from the ["period", "index_value"] schema of
        the temporal multilateral methods and does not feed the extension
        (splicing) functions.

    Raises:
        ValueError: If DataFrame doesn't meet requirements (see
            _validate_spatial_input), if base_entity is not found, or if
            an entity shares no common products with any other entity.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({
        ...     "region": ["North", "North", "South", "South"],
        ...     "product_id": ["A", "B", "A", "B"],
        ...     "aggregated_price": [1.0, 2.0, 1.2, 2.2],
        ...     "aggregated_quantity": [10.0, 5.0, 8.0, 6.0],
        ... })
        >>> result = geks_paasche(df)
        >>> # Returns DataFrame with columns ["region", "geks_paasche"]
    """
    df = _to_canonical(df, period_col=entity_col, price_col=price_col, id_col=id_col,
                       quantity_col=quantity_col)
    result = _geks_spatial_base(
        df,
        formula="paasche",
        base_entity=base_entity,
        panel_warnings=panel_warnings,
        value_name="geks_paasche",
    )
    if entity_col != "period":
        result = result.rename({"period": entity_col})
    return result


def geks_laspeyres(
    df: pl.DataFrame,
    entity_col: str = "region",
    id_col: str = "product_id",
    price_col: str = "aggregated_price",
    quantity_col: str = "aggregated_quantity",
    base_entity: Optional[object] = None,
    panel_warnings: bool = False,
) -> pl.DataFrame:
    """
    Compute the GEKS-Laspeyres multilateral price index for spatial comparisons.

    Each entity's price level relative to the base entity is estimated by
    averaging bilateral Laspeyres comparisons over all possible bridge
    entities. Every bilateral link through bridge m uses the bridge's own
    expenditure basket (the Laspeyres weighting); this is the GEKS-L index
    of Białek (2022) applied over regions instead of time periods. The
    index is transitive and base invariant without requiring country
    reversal, and reduces to the bilateral Fisher index when only two
    entities are compared.

    Formula: P_GEKS-L(k) = product_{m in M} [P_L(m, k) / P_L(m, b)]^(1/M)

    where P_L(m, k) is the bilateral Laspeyres index with entity m as base
    and entity k as current (m's quantities as weights), computed on the
    products matched between m and k; b is the base entity and M is the
    number of entities.

    Args:
        df: Polars DataFrame with the panel columns identified by
            entity_col, price_col, id_col and quantity_col, containing
            data for multiple entities, with each product having exactly
            one price and quantity per entity. Unbalanced panels are
            supported: each bilateral link is computed on the products
            matched between the two entities involved.
        entity_col: Name of the entity column (default "region"). For
            spatial data this is typically a region column; for data from
            the temporal pipeline, rename "period" to "region" first
            (df.rename({"period": "region"})).
        id_col: Name of the product identifier column (default "product_id").
        price_col: Name of the price column (default "aggregated_price").
        quantity_col: Name of the quantity column
            (default "aggregated_quantity").
        base_entity: The reference entity whose index value is 1.0. If
            None (default), the first sorted entity is used (for string
            labels, the alphabetically first region).
        panel_warnings: If True, emit a UserWarning summarising unmatched
            products and skipped entity pairs.

    Returns:
        DataFrame with columns [entity_col, "geks_laspeyres"] (float),
        where the value is the multilateral price index of each entity
        relative to the base entity (base entity = 1.0). Note: this output
        schema intentionally differs from the ["period", "index_value"]
        schema of the temporal multilateral methods and does not feed the
        extension (splicing) functions.

    Raises:
        ValueError: If DataFrame doesn't meet requirements (see
            _validate_spatial_input), if base_entity is not found, or if
            an entity shares no common products with any other entity.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({
        ...     "region": ["North", "North", "South", "South"],
        ...     "product_id": ["A", "B", "A", "B"],
        ...     "aggregated_price": [1.0, 2.0, 1.2, 2.2],
        ...     "aggregated_quantity": [10.0, 5.0, 8.0, 6.0],
        ... })
        >>> result = geks_laspeyres(df)
        >>> # Returns DataFrame with columns ["region", "geks_laspeyres"]
    """
    df = _to_canonical(df, period_col=entity_col, price_col=price_col, id_col=id_col,
                       quantity_col=quantity_col)
    result = _geks_spatial_base(
        df,
        formula="laspeyres",
        base_entity=base_entity,
        panel_warnings=panel_warnings,
        value_name="geks_laspeyres",
    )
    if entity_col != "period":
        result = result.rename({"period": entity_col})
    return result


def _geks_spatial_base(
    df: pl.DataFrame,
    formula: Literal["paasche", "laspeyres"],
    base_entity: Optional[object],
    panel_warnings: bool,
    value_name: str,
) -> pl.DataFrame:
    """
    Helper for the spatial GEKS indices (bridge form).

    Computes all directed bilateral indices in one vectorized self-join
    (no reciprocal symmetrization: Paasche and Laspeyres links in both
    directions are computed from their own weights), then averages each
    entity's bridge estimates ln B(m, k) - ln B(m, b) over the bridges m
    available for that entity.

    Args:
        df: Input DataFrame in canonical panel schema ("period" holding
            entity labels).
        formula: Bilateral formula, "paasche" or "laspeyres".
        base_entity: Reference entity, or None for the first sorted entity.
        panel_warnings: If True and links were skipped or computed on
            unmatched product sets, emit one summary warning.
        value_name: Name of the output value column.
    """
    _validate_spatial_input(df)

    entities = df.select("period").unique().sort("period").to_series().to_list()
    M = len(entities)

    if base_entity is None:
        base = entities[0]
    else:
        if base_entity not in entities:
            raise ValueError(
                f"base_entity '{base_entity}' not found in the entity column"
            )
        base = base_entity

    # Product set observed in each entity, for per-pair matching
    product_sets = {
        entity: set(
            df.filter(pl.col("period") == entity)
            .select("product_id")
            .to_series()
            .to_list()
        )
        for entity in entities
    }

    # All directed bilateral indices B(m, k) = level of k relative to m,
    # computed on the products matched between m and k via one self-join.
    # Both directions are computed from their own weights because Paasche
    # and Laspeyres do not satisfy country reversal.
    left = df.select(
        pl.col("product_id"),
        pl.col("period").alias("base_period"),
        pl.col("aggregated_price").alias("p_base"),
        pl.col("aggregated_quantity").alias("q_base"),
    )
    right = df.select(
        pl.col("product_id"),
        pl.col("period").alias("curr_period"),
        pl.col("aggregated_price").alias("p_curr"),
        pl.col("aggregated_quantity").alias("q_curr"),
    )
    pairs = (
        left.join(right, on="product_id")
        .filter(pl.col("base_period") != pl.col("curr_period"))
        .group_by("base_period", "curr_period")
        .agg(
            (pl.col("p_curr") * pl.col("q_curr")).sum().alias("pq_curr"),
            (pl.col("p_base") * pl.col("q_curr")).sum().alias("p_base_q_curr"),
            (pl.col("p_curr") * pl.col("q_base")).sum().alias("p_curr_q_base"),
            (pl.col("p_base") * pl.col("q_base")).sum().alias("pq_base"),
        )
    )
    # Paasche: sum(p_k q_k) / sum(p_m q_k); Laspeyres: sum(p_k q_m) / sum(p_m q_m)
    if formula == "paasche":
        pairs = pairs.select(
            "base_period",
            "curr_period",
            (pl.col("pq_curr") / pl.col("p_base_q_curr")).alias("index"),
        )
    else:
        pairs = pairs.select(
            "base_period",
            "curr_period",
            (pl.col("p_curr_q_base") / pl.col("pq_base")).alias("index"),
        )

    # links[(m, k)] = B(m, k); ordered pairs with no common products are absent
    links = {
        (row["base_period"], row["curr_period"]): row["index"]
        for row in pairs.iter_rows(named=True)
    }

    unmatched_pairs = []
    skipped_pairs = []
    for i in range(M):
        for j in range(i + 1, M):
            common = product_sets[entities[i]] & product_sets[entities[j]]
            if not common:
                skipped_pairs.append((entities[i], entities[j]))
            elif len(product_sets[entities[i]] ^ product_sets[entities[j]]):
                unmatched_pairs.append(
                    (entities[i], entities[j],
                     len(product_sets[entities[i]] ^ product_sets[entities[j]]))
                )

    if panel_warnings and (unmatched_pairs or skipped_pairs):
        parts = []
        if unmatched_pairs:
            total_unmatched = sum(n for _, _, n in unmatched_pairs)
            n_pairs = M * (M - 1) // 2
            parts.append(
                f"{len(unmatched_pairs)} of {n_pairs} entity pairs have unmatched "
                f"products ({total_unmatched} unmatched product occurrences in total); "
                "each link uses the products matched between its two entities"
            )
        if skipped_pairs:
            formatted = ", ".join(f"{e} vs {f}" for e, f in skipped_pairs)
            parts.append(
                "entity pairs with no common products were skipped: " + formatted
            )
        warnings.warn(
            "Unbalanced panel: " + "; ".join(parts) + ".",
            UserWarning,
        )

    # Bridge average: index(k) = exp(mean_m [ln B(m, k) - ln B(m, b)]).
    # Diagonal links B(k, k) = B(b, b) = 1 contribute log(1) = 0; bridges
    # where either leg has no common products are skipped and the average
    # runs over the available bridges only.
    indices = []
    for entity in entities:
        log_sum = 0.0
        n_links = 0
        for bridge in entities:
            b_mk = 1.0 if bridge == entity else links.get((bridge, entity))
            b_mb = 1.0 if bridge == base else links.get((bridge, base))
            if b_mk is None or b_mb is None:
                continue
            log_sum += np.log(b_mk) - np.log(b_mb)
            n_links += 1

        if n_links == 0:
            raise ValueError(
                f"No common products between entity {entity} and any other entity; "
                "cannot compute GEKS index"
            )

        indices.append({"period": entity, value_name: float(np.exp(log_sum / n_links))})

    return pl.DataFrame(indices)


def _validate_spatial_input(df: pl.DataFrame) -> None:
    """
    Validate DataFrame for spatial index computation.

    Mirrors the checks of ``multilateral._validate_multilateral_input``
    (weighted, unbalanced mode) with entity-flavoured error messages.

    Args:
        df: Canonical DataFrame to validate.

    Raises:
        ValueError: If validation fails.
    """
    required_cols = ["product_id", "period", "aggregated_price", "aggregated_quantity"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if not df.schema["aggregated_price"].is_numeric():
        raise ValueError("aggregated_price must be numeric")

    min_price = df.select(pl.col("aggregated_price").min()).item()
    if min_price <= 0:
        raise ValueError("All aggregated_price values must be positive")

    grouped = df.group_by(["product_id", "period"]).len()
    max_count = grouped.select(pl.col("len").max()).item()
    if max_count > 1:
        raise ValueError("Each product must have exactly one observation per entity")

    n_entities = df.select("period").unique().height
    if n_entities < 2:
        raise ValueError("Spatial GEKS indices require at least 2 entities")

    n_products = df.select("product_id").unique().height
    if n_products < 2:
        raise ValueError("Spatial GEKS indices require at least 2 products")

    if not df.schema["aggregated_quantity"].is_numeric():
        raise ValueError("aggregated_quantity must be numeric")

    min_quantity = df.select(pl.col("aggregated_quantity").min()).item()
    if min_quantity <= 0:
        raise ValueError("All aggregated_quantity values must be positive")
