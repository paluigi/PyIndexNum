# PyIndexNum

[![PyPI version](https://badge.fury.io/py/pyindexnum.svg)](https://pypi.org/project/pyindexnum/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/pyindexnum/badge/?version=latest)](https://pyindexnum.readthedocs.io/en/latest/?badge=latest)

A high-performance Python library for calculating economic index numbers using Polars. Designed for statisticians and economists working with price and quantity indices.

## Features

- **High Performance**: Built on Polars for efficient data processing of large datasets
- **Comprehensive Index Methods**: Support for bilateral and multilateral price/quantity indices
- **Data Preparation Tools**: Built-in utilities for data standardization and temporal aggregation
- **Panel Data Handling**: Robust methods for dealing with unbalanced panels through removal or imputation
- **Extension Methods**: Support for index splicing and rolling window calculations
- **Type Safety**: Full type annotations for better IDE support and code reliability

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
geks_fisher_idx = pin.geks_fisher(df_agg)

# 6. Apply extension methods (optional)
extended_idx = pin.movement_splice(geks_fisher_idx1, geks_fisher_idx2)
```

## Supported Index Methods

### Bilateral Indices (Two-Period Comparisons)

| Index | Formula | Use Case |
|-------|---------|----------|
| **Jevons** | Geometric mean of price relatives | Unweighted geometric average |
| **Dudot** | Arithmetic mean of price relatives | Unweighted arithmetic average |
| **Carli** | Ratio of arithmetic means | Simple price average comparison |
| **Laspeyres** | Weighted by base period quantities | Fixed basket approach |
| **Paasche** | Weighted by current period quantities | Current basket approach |
| **Fisher** | Geometric mean of Laspeyres and Paasche | Ideal index (time/quantity reversal) |
| **Törnqvist** | Weighted geometric mean with average shares | Symmetric treatment |
| **Walsh** | Geometric mean with base quantity weights | Alternative symmetric approach |

### Multilateral Indices (Multi-Period Comparisons)

| Index | Method | Description |
|-------|--------|-------------|
| **GEKS-Fisher** | Chained Fisher indices | Most widely used multilateral method |
| **GEKS-Törnqvist** | Chained Törnqvist indices | Alternative chaining approach |
| **Geary-Khamis** | System of equations | Global approach |
| **Time Product Dummy** | Regression-based | Econometric approach |

### Extension Methods

- **Movement Splice**: Chain indices using movement ratios
- **Window Splice**: Moving window chaining
- **Half Splice**: Half-year overlapping windows
- **Mean Splice**: Average of overlapping windows
- **Fixed Base Rolling Window**: Rolling window with fixed base

## Data Requirements

Your data should contain:

- **Date column**: Date or datetime values
- **Price column**: Numeric price observations
- **Product ID column**: Unique identifier for each product/variety
- **Quantity column**: Numeric quantities (required for weighted indices)

Example data structure:
```
┌────────────┬────────────┬───────┬──────────┐
│ date       ┆ product_id ┆ price ┆ quantity │
│ ---        ┆ ---        ┆ ---   ┆ ---      │
│ date       ┆ str        ┆ f64   ┆ f64      │
╞════════════╪════════════╪═══════╪══════════╡
│ 2023-01-01 ┆ A          ┆ 100.0 ┆ 10.0     │
│ 2023-01-01 ┆ B          ┆ 200.0 ┆ 5.0      │
│ 2023-02-01 ┆ A          ┆ 105.0 ┆ 12.0     │
│ 2023-02-01 ┆ B          ┆ 210.0 ┆ 4.5      │
└────────────┴────────────┴────────────┴──────────┘
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
df_imputed = pin.carry_forward_imputation(df_agg, ["price", "quantity"])
```

### Index Calculation

```python
# Bilateral indices
jevons = pin.jevons(df)
laspeyres = pin.laspeyres(df)
fisher = pin.fisher(df)

# Multilateral indices
geks = pin.geks_fisher(df)
gk = pin.geary_khamis(df)
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
  version = {0.1.0},
}
```

## License

PyIndexNum is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Related Projects

- [Polars](https://pola.rs/): The high-performance DataFrame library that powers PyIndexNum
- [NumPy](https://numpy.org/): Fundamental package for scientific computing

---

Built with ❤️ for the economic statistics community
