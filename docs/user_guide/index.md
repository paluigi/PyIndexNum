# User Guide

Welcome to the PyIndexNum user guide! This section provides comprehensive information on how to use the library effectively for economic index number calculations.

## Overview

PyIndexNum is a Python library designed for calculating price and quantity indices using modern data processing techniques. Built on top of Polars, it offers high-performance computation of bilateral and multilateral economic indices.

## Key Features

- **High Performance**: Built on Polars for efficient data processing
- **Comprehensive Index Methods**: Support for bilateral and multilateral index calculation
- **Data Preparation Tools**: Built-in utilities for data standardization and aggregation
- **Extension Methods**: Support for index splicing and rolling window calculations
- **Type Safety**: Full type annotations for better IDE support

## Installation

```bash
pip install pyindexnum
```

Or using uv:

```bash
uv add pyindexnum
```

## Quick Start

See the [Examples](../examples/workflow.md) section for a complete workflow demonstration.

## Data Requirements

Raw transaction data passed to `standardize_columns()` should contain a date
column, numeric price data, a unique product identifier, and (for weighted
indices) a quantity column — each mapped through the function's `date_col`,
`price_col`, `id_col` and `quantity_col` arguments.

Every index function consumes the aggregated panel produced by
`aggregate_time()`, with one observation per product and period:

- `product_id`: Unique product identifier
- `period`: Date identifying the period (e.g. the truncated month)
- `aggregated_price`: Numeric price for that product-period
- `aggregated_quantity`: Numeric quantity (required for weighted indices;
  GEKS-Jevons and the unweighted time-product-dummy do not need it)

## Using your own column names

If your data is already aggregated with different column names, pass them
explicitly — there is no need to rename or re-aggregate:

```python
import pyindexnum as pin

geks = pin.geks_fisher(
    df,
    period_col="month",
    price_col="avg_price",
    id_col="sku",
    quantity_col="total_qty",
)
```

All bilateral and multilateral index functions accept `period_col`,
`price_col`, `id_col` and (where quantities are used) `quantity_col`.
`remove_unbalanced()` accepts `id_col`/`period_col` and `get_summary()`
accepts `id_col`/`date_col`/`quantity_col`.

## Unbalanced panels

By default the index functions require a balanced panel — every product
present in every period — and raise a clear error otherwise. You can keep
that guarantee by balancing first (`remove_unbalanced()` or one of the
imputation functions), or skip balancing entirely with `balanced=False`,
which computes each period comparison on the products matched between the
periods involved:

```python
fisher_idx = pin.fisher(df_two_periods, balanced=False)
geks_idx = pin.geks_fisher(df_agg, balanced=False)
```

Method-specific behaviour under `balanced=False`:

- **Bilateral indices** (including each GEKS link): computed on the matched
  product set. Törnqvist expenditure shares are re-normalised over the
  matched set. Dutot compares means over the matched set — as a ratio of
  means it remains sensitive to the composition of that set.
- **GEKS**: each bilateral link uses its own matched set; links between
  periods that share no products are skipped, and each period's GEKS average
  runs over the available links. A period sharing no products with any other
  period raises an error.
- **Geary-Khamis**: the iteration runs over observed cells only. Convergence
  is well-defined for complete panels; for strongly unbalanced panels it is
  not theoretically guaranteed (`max_iter` and `tol` remain available as
  safeguards).
- **Time Product Dummy**: regression-based and natively handles unbalanced
  data — no flag needed. Products observed in a single period make their
  dummies collinear with the time dummies; consider removing such products.

Pass `panel_warnings=True` to be warned about how many products each
comparison drops.

## Workflow

1. **Data Standardization**: Use `standardize_columns()` to ensure consistent column names and types
2. **Temporal Aggregation**: Use `aggregate_time()` to aggregate data to desired frequency — its output feeds every index function directly
3. **Panel Balancing**: Optionally remove unbalanced data with `remove_unbalanced()` or impute missing values — or skip this step and use `balanced=False` in the index functions
4. **Index Calculation**: Compute bilateral or multilateral indices
5. **Extension**: Optionally apply splicing methods for chained indices

## Best Practices

- Always start with `standardize_columns()` to ensure data consistency
- Use `aggregate_time()` before index calculation to handle high-frequency data
- For unbalanced panels, choose between `remove_unbalanced()` (removes products), imputation methods (fills missing values), or `balanced=False` (matched-sample computation without imputation)
- Bilateral indices compare exactly two periods; multilateral indices can handle multiple periods

## Common Pitfalls

- Ensure data is sorted by date before aggregation
- Check for missing values that could affect index calculations
- Verify that product IDs are consistent across periods
- Use appropriate aggregation methods based on your data characteristics
