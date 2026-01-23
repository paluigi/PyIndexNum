# Complete Workflow Example

This example demonstrates the typical PyIndexNum workflow for calculating economic indices from raw price data.

## Sample Data

Let's start with some sample price data for three products over several months:

```python
import polars as pl
import pyindexnum as pin

# Create sample data
data = {
    "date": [
        "2023-01-01", "2023-01-15", "2023-02-01", "2023-02-15", "2023-03-01",
        "2023-01-01", "2023-01-15", "2023-02-01", "2023-02-15", "2023-03-01",
        "2023-01-01", "2023-01-15", "2023-02-01", "2023-02-15", "2023-03-01"
    ],
    "product": ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B", "C", "C", "C", "C", "C"],
    "price": [100, 105, 110, 108, 115, 200, 195, 210, 205, 220, 50, 52, 48, 51, 49],
    "quantity": [10, 12, 11, 13, 10, 5, 6, 4, 7, 5, 20, 18, 22, 19, 21]
}

df = pl.DataFrame(data)
print(df)
```

Output:
```
shape: (15, 4)
┌────────────┬─────────┬───────┬──────────┐
│ date       ┆ product ┆ price ┆ quantity │
│ ---        ┆ ---     ┆ ---   ┆ ---      │
│ str        ┆ str     ┆ i64   ┆ i64      │
╞════════════╪═════════╪═══════╪══════════╡
│ 2023-01-01 ┆ A       ┆ 100   ┆ 10       │
│ 2023-01-15 ┆ A       ┆ 105   ┆ 12       │
│ 2023-02-01 ┆ A       ┆ 110   ┆ 11       │
│ 2023-02-15 ┆ A       ┆ 108   ┆ 13       │
│ 2023-03-01 ┆ A       ┆ 115   ┆ 10       │
│ ...        ┆ ...     ┆ ...   ┆ ...      │
│ 2023-03-01 ┆ C       ┆ 49    ┆ 21       │
└────────────┴─────────┴───────┴──────────┘
```

## Step 1: Standardize Column Names

First, standardize the column names to match PyIndexNum's expected format:

```python
df_std = pin.standardize_columns(
    df,
    date_col="date",
    price_col="price",
    id_col="product",
    quantity_col="quantity"
)
print(df_std.head())
```

Output:
```
shape: (5, 4)
┌────────────┬───────┬────────────┬──────────┐
│ date       ┆ price ┆ product_id ┆ quantity │
│ ---        ┆ ---   ┆ ---        ┆ ---      │
│ date       ┆ i64   ┆ str        ┆ i64      │
╞════════════╪═══════╪════════════╪══════════╡
│ 2023-01-01 ┆ 100   ┆ A          ┆ 10       │
│ 2023-01-15 ┆ 105   ┆ A          ┆ 12       │
│ 2023-02-01 ┆ 110   ┆ A          ┆ 11       │
│ 2023-02-15 ┆ 108   ┆ A          ┆ 13       │
│ 2023-03-01 ┆ 115   ┆ A          ┆ 10       │
└────────────┴───────┴────────────┴──────────┘
```

## Step 2: Aggregate Time Series

Aggregate the high-frequency data to monthly periods:

```python
df_agg = pin.aggregate_time(
    df_std,
    freq="1mo",
    agg_type="arithmetic"  # Arithmetic mean for prices
)
print(df_agg)
```

Output:
```
shape: (9, 4)
┌────────────┬────────────┬───────────────────┬─────────────────────┐
│ product_id ┆ period     ┆ aggregated_price  ┆ aggregated_quantity │
│ ---        ┆ ---        ┆ ---               ┆ ---                 │
│ str        ┆ date       ┆ f64               ┆ f64                 │
╞════════════╪════════════╪═══════════════════╪═════════════════════╡
│ A          ┆ 2023-01-01 ┆ 102.5             ┆ 11.0                │
│ A          ┆ 2023-02-01 ┆ 109.0             ┆ 12.0                │
│ A          ┆ 2023-03-01 ┆ 115.0             ┆ 10.0                │
│ B          ┆ 2023-01-01 ┆ 197.5             ┆ 5.5                 │
│ B          ┆ 2023-02-01 ┆ 207.5             ┆ 5.5                 │
│ B          ┆ 2023-03-01 ┆ 220.0             ┆ 5.0                 │
│ C          ┆ 2023-01-01 ┆ 51.0              ┆ 19.0                │
│ C          ┆ 2023-02-01 ┆ 49.5              ┆ 20.5                │
│ C          ┆ 2023-03-01 ┆ 49.0              ┆ 21.0                │
└────────────┴────────────┴────────────┴─────────────────────┘
```

## Step 3: Handle Unbalanced Data (Optional)

For this example, our data is already balanced. If you had missing data, you could either remove unbalanced products:

```python
df_balanced = pin.remove_unbalanced(df_agg)
```

Or impute missing values:

```python
df_imputed = pin.carry_forward_imputation(
    df_agg,
    value_cols=["aggregated_price", "aggregated_quantity"]
)
```

## Step 4: Calculate Bilateral Indices

Calculate indices comparing two specific periods. Let's compare January to February:

```python
# Filter to January and February data
df_two_periods = df_agg.filter(
    pl.col("period").is_in([
        pl.date(2023, 1, 1),
        pl.date(2023, 2, 1)
    ])
)

# Calculate various bilateral indices
laspeyres_idx = pin.laspeyres(df_two_periods)
paasche_idx = pin.paasche(df_two_periods)
fisher_idx = pin.fisher(df_two_periods)
tornqvist_idx = pin.tornqvist(df_two_periods)

print(f"Laspeyres Index: {laspeyres_idx:.4f}")
print(f"Paasche Index: {paasche_idx:.4f}")
print(f"Fisher Index: {fisher_idx:.4f}")
print(f"Törnqvist Index: {tornqvist_idx:.4f}")
```

Output:
```
Laspeyres Index: 1.0507
Paasche Index: 1.0511
Fisher Index: 1.0509
Törnqvist Index: 1.0509
```

## Step 5: Calculate Multilateral Indices (Optional)

For data spanning multiple periods, use multilateral methods:

```python
# GEKS-Fisher index for all three months
geks_fisher_idx = pin.geks_fisher(df_agg)
print(f"GEKS-Fisher Index (Jan-Mar): {geks_fisher_idx:.4f}")
```

## Step 6: Apply Extension Methods (Optional)

For chained multilateral indices:

```python
# Example of movement splicing for extending the index
extended_indices = pin.movement_splice(geks_fisher_idx1, geks_fisher_idx2)
print("Extended indices:", extended_indices)
```

## Summary

This workflow demonstrates:

1. **Data Preparation**: Standardizing columns and aggregating time series
2. **Data Quality**: Handling unbalanced panels through removal or imputation
3. **Index Calculation**: Computing bilateral indices for period-to-period comparisons
4. **Advanced Methods**: Using multilateral indices for multi-period analysis
5. **Extensions**: Applying splicing methods for chained multilateral indices

The choice of specific methods depends on your data characteristics and analytical requirements. Always consider the economic interpretation of different index formulas when selecting appropriate methods for your use case.
