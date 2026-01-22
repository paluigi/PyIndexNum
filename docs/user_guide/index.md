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

The library expects data in a standardized format with the following columns:

- `date`: Date or datetime column
- `price`: Numeric price data
- `product_id`: Unique product identifier
- `quantity`: Numeric quantity data (optional, required for weighted indices)

## Workflow

1. **Data Standardization**: Use `standardize_columns()` to ensure consistent column names and types
2. **Temporal Aggregation**: Use `aggregate_time()` to aggregate data to desired frequency
3. **Panel Balancing**: Optionally remove unbalanced data with `remove_unbalanced()` or impute missing values
4. **Index Calculation**: Compute bilateral or multilateral indices
5. **Extension**: Optionally apply splicing methods for chained indices

## Best Practices

- Always start with `standardize_columns()` to ensure data consistency
- Use `aggregate_time()` before index calculation to handle high-frequency data
- For unbalanced panels, choose between `remove_unbalanced()` (removes products) or imputation methods (fills missing values)
- Bilateral indices compare exactly two periods; multilateral indices can handle multiple periods

## Common Pitfalls

- Ensure data is sorted by date before aggregation
- Check for missing values that could affect index calculations
- Verify that product IDs are consistent across periods
- Use appropriate aggregation methods based on your data characteristics
