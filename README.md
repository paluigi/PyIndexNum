# PyIndexNum

[![PyPI version](https://badge.fury.io/py/pyindexnum.svg)](https://pypi.org/project/pyindexnum/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/pyindexnum/badge/?version=latest)](https://pyindexnum.readthedocs.io/en/latest/?badge=latest)

A high-performance Python library for calculating economic index numbers using Polars. Designed for statisticians and economists working with price and quantity indices.

## Features

- **High Performance**: Built on Polars for efficient data processing of large datasets
- **Comprehensive Index Methods**: Support for bilateral and multilateral price/quantity indices
- **Data Preparation Tools**: Built-in utilities for data standardization and temporal aggregation
- **Flexible Input**: Every index function consumes `aggregate_time` output directly, and accepts custom column names for pre-aggregated external data
- **Unbalanced Panels**: Matched-sample computation (`balanced=False`) across bilateral and multilateral indices — no imputation or product removal required
- **Extension Methods**: Support for index splicing and rolling window calculations
- **Type Safety**: Full type annotations for better IDE support and code reliability

## What's new in 0.5.0

**Experimental spatial indices.** New `geks_paasche` and `geks_laspeyres` multilateral price indices for cross-region (spatial) comparisons, in the new `pyindexnum.experimental` module. They use the bridge-averaged regression form (Białek's 2022 GEKS-L and its Paasche mirror) because Paasche and Laspeyres bilaterals do not satisfy country reversal; the results are transitive and base invariant without requiring it, and reduce to the bilateral Fisher index when only two regions are compared.

- **Spatial-first API**: `entity_col` (default `"region"`), an explicit `base_entity`, and matched-sample support for unbalanced regional panels (`panel_warnings=True` summarises dropped products). Data from the temporal pipeline needs one rename: `df.rename({"period": "region"})`.
- Unbalanced panels are handled on matched product sets: each bilateral link uses the products matched between its two regions, entity pairs sharing no products are skipped, and a region connected to no other raises a clear error.
- Dedicated [user guide page](https://pyindexnum.readthedocs.io/en/latest/user_guide/experimental_spatial.html) and test suite; the API is experimental and may change in future releases.

## What's new in 0.4.0

**Breaking change — one unified input schema.** All index functions (bilateral and multilateral) now consume the panel schema produced by `aggregate_time`: `product_id`, `period`, `aggregated_price`, `aggregated_quantity`. The output of `aggregate_time` feeds every index function directly, with no renaming. Bilateral functions previously required `date`/`price`/`quantity` columns; existing code migrates in one line:

```python
# before 0.4.0                        # 0.4.0
laspeyres(df)                         laspeyres(df, period_col="date", price_col="price",
                                               quantity_col="quantity")
```

New features:

- **Custom column names** on every index function (`period_col`, `price_col`, `id_col`, `quantity_col`), plus on `remove_unbalanced` and `get_summary` — work directly with pre-aggregated data in any schema.
- **Unbalanced panel support** via `balanced=False`: bilateral indices and GEKS links are computed on the products matched between the periods involved (Törnqvist expenditure shares re-normalise over the matched set; GEKS averages over available links; Geary-Khamis iterates over observed cells). `panel_warnings=True` reports how many products each comparison drops. The time-product-dummy method handles unbalanced panels natively and needs no flag.
- Clear up-front errors for unbalanced input under the default `balanced=True` (previously Geary-Khamis failed with a misleading convergence error).

## Installation

### Using pip

```bash
pip install pyindexnum
```

### Using uv

```bash
uv add pyindexnum
```

### From source

```bash
git clone https://github.com/paluigi/PyIndexNum.git
cd PyIndexNum
uv sync
```

## Quick Start

Here's the typical workflow for calculating economic indices:

```python
import polars as pl
import pyindexnum as pin

# Load your price data
df = pl.read_csv("price_data.csv")

# 1. Standardize column names
df_std = pin.standardize_columns(df, date_col="date", price_col="price", id_col="product_id", quantity_col="quantity")

# 2. Aggregate to desired time frequency
df_agg = pin.aggregate_time(df_std, freq="1mo", agg_type="arithmetic")

# 3. Handle unbalanced panels (optional)
df_balanced = pin.remove_unbalanced(df_agg)
# or
df_imputed = pin.carry_forward_imputation(df_agg, ["aggregated_price", "aggregated_quantity"])

# 4. Calculate bilateral indices (two periods)
laspeyres_idx = pin.laspeyres(df_balanced)
fisher_idx = pin.fisher(df_balanced)

# 5. Calculate multilateral indices (multiple periods)
# Split data into two overlapping rolling windows
periods = sorted(df_agg["period"].unique())
window1 = df_agg.filter(pl.col("period").is_in(periods[:-1]))
window2 = df_agg.filter(pl.col("period").is_in(periods[1:]))

geks_fisher_idx1 = pin.geks_fisher(window1)
geks_fisher_idx2 = pin.geks_fisher(window2)

# 6. Apply extension methods (optional)
extended_idx = pin.movement_splice(geks_fisher_idx1, geks_fisher_idx2)
```

## Supported Index Methods

### Bilateral Indices (Two-Period Comparisons)

| Index | Formula | Use Case |
|-------|---------|----------|
| **Jevons** | Geometric mean of price relatives | Unweighted geometric average |
| **Carli** | Arithmetic mean of price relatives | Unweighted arithmetic average |
| **Dutot** | Ratio of arithmetic means of prices | Simple price average comparison |
| **Laspeyres** | Weighted by base period quantities | Fixed basket approach |
| **Paasche** | Weighted by current period quantities | Current basket approach |
| **Fisher** | Geometric mean of Laspeyres and Paasche | Ideal index (time/quantity reversal) |
| **Törnqvist** | Weighted geometric mean with average expenditure shares | Symmetric treatment |
| **Walsh** | Geometric mean of quantities as fixed basket | Alternative symmetric approach |

### Multilateral Indices (Multi-Period Comparisons)

| Index | Method | Description |
|-------|--------|-------------|
| **GEKS-Fisher** | Chained Fisher indices | Most widely used multilateral method |
| **GEKS-Törnqvist** | Chained Törnqvist indices | Alternative chaining approach |
| **GEKS-Jevons** | Unweighted geometric mean of price relatives | Transitive multilateral index (no quantity) |
| **Geary-Khamis** | System of equations | Global approach |
| **Time Product Dummy** | Regression-based | WLS with expenditure shares or unweighted OLS |

### Extension Methods

- **Movement Splice**: Chain indices using movement ratios
- **Window Splice**: Moving window chaining
- **Half Splice**: Half-year overlapping windows
- **Mean Splice**: Average of overlapping windows
- **Fixed Base Rolling Window**: Rolling window with fixed base

### Experimental: Spatial Indices

Multilateral indices for cross-region (spatial) price comparisons, built on
the bridge-averaged regression form because Paasche and Laspeyres bilaterals
do not satisfy country reversal. Their API is experimental and may change;
see the [spatial indices guide](https://pyindexnum.readthedocs.io/en/latest/user_guide/experimental_spatial.html) for details.

| Index | Method | Description |
|-------|--------|-------------|
| **GEKS-Paasche** | Bridge-averaged Paasche bilaterals | Every link into a region uses that region's own basket |
| **GEKS-Laspeyres** | Bridge-averaged Laspeyres bilaterals | Białek's (2022) GEKS-L over regions: every link uses the bridge region's basket |

```python
import pyindexnum as pin

# Spatial data: a region column replaces the period column. Data from the
# temporal pipeline needs one rename first:
df_region = df_agg.rename({"period": "region"})

paasche = pin.geks_paasche(df_region, base_entity="North")
laspeyres = pin.geks_laspeyres(df_region, base_entity="North")
```

## Data Requirements

Raw transaction data should contain a date column, a numeric price column, a unique product ID and (for weighted indices) a quantity column — `standardize_columns` maps these to the standard names via its `date_col`/`price_col`/`id_col`/`quantity_col` arguments.

Every index function then consumes the aggregated panel produced by `aggregate_time`, with one observation per product and period:

```
┌────────────┬────────────┬───────────────────┬───────────────────────┐
│ product_id ┆ period     ┆ aggregated_price  ┆ aggregated_quantity   │
│ ---        ┆ ---        ┆ ---               ┆ ---                   │
│ str        ┆ date       ┆ f64               ┆ f64                   │
╞════════════╪════════════╪═══════════════════╪═══════════════════════╡
│ A          ┆ 2023-01-01 ┆ 100.0             ┆ 10.0                  │
│ B          ┆ 2023-01-01 ┆ 200.0             ┆ 5.0                   │
│ A          ┆ 2023-02-01 ┆ 105.0             ┆ 12.0                  │
│ B          ┆ 2023-02-01 ┆ 210.0             ┆ 4.5                   │
└────────────┴────────────┴───────────────────┴───────────────────────┘
```

Already have aggregated data with different column names? Pass them explicitly — no renaming, no re-aggregation:

```python
geks_fisher_idx = pin.geks_fisher(
    df,
    period_col="month", price_col="avg_price",
    id_col="sku", quantity_col="total_qty",
    balanced=False,
)
```

## API Overview

### Data Preparation

```python
# Standardize column names and types
df_std = pin.standardize_columns(df, date_col="date", price_col="price", id_col="id")

# Aggregate time series data
df_agg = pin.aggregate_time(df_std, freq="1mo", agg_type="weighted_arithmetic")

# Handle unbalanced panels
df_balanced = pin.remove_unbalanced(df_agg)
df_imputed = pin.carry_forward_imputation(df_agg, ["aggregated_price", "aggregated_quantity"])
```

### Index Calculation

```python
# Bilateral indices (exactly two periods)
jevons = pin.jevons(df)
laspeyres = pin.laspeyres(df)
fisher = pin.fisher(df)

# Multilateral indices
geks = pin.geks_fisher(df)
gk = pin.geary_khamis(df)

# Unweighted multilateral index (no quantity required)
geks_j = pin.geks_jevons(df)

# Unbalanced panels: compute on matched products, no imputation
fisher = pin.fisher(df, balanced=False, panel_warnings=True)
geks = pin.geks_fisher(df, balanced=False)
```

### Extensions

```python
# Splicing methods
movement_spliced = pin.movement_splice(multilateral_index1, multilateral_index2)
window_spliced = pin.window_splice(multilateral_index1, multilateral_index2)
```

## Documentation

Full documentation is available at [https://pyindexnum.readthedocs.io/](https://pyindexnum.readthedocs.io/)

## Contributing

PyIndexNum is an open-source project and welcomes contributions! See our [contributing guide](https://pyindexnum.readthedocs.io/en/latest/contributing.html) for details.

### Development Setup

```bash
# Clone and setup
git clone https://github.com/paluigi/PyIndexNum.git
cd PyIndexNum
uv sync --dev

# Run tests
uv run pytest

# Build documentation
cd docs && make html
```

### Areas for Contribution

- New index methods and formulations
- Performance optimizations
- Additional data validation
- Enhanced documentation and examples
- Bug fixes and improvements

## Citation

If you use PyIndexNum in your research, please cite:

```bibtex
@software{pyindexnum,
  title = {PyIndexNum: A Python Library for Economic Index Numbers},
  author = {Palumbo, Luigi, and Yu, Mengting},
  url = {https://github.com/paluigi/PyIndexNum},
  version = {0.5.0},
}
```

## License

PyIndexNum is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Related Projects

- [Polars](https://pola.rs/): The high-performance DataFrame library that powers PyIndexNum

---

Built with ❤️ for the economic statistics community
