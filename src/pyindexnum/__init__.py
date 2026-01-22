"""
PyIndexNum: A Python library for calculating economic index numbers.

This library provides functions for computing bilateral and multilateral
price and quantity indices using Polars for high-performance data processing.
"""

from .utils import aggregate_time, remove_unbalanced, carry_forward_imputation, carry_backward_imputation
from .bilateral import jevons, dudot, carli, laspeyres, paasche, fisher, tornqvist, walsh
from .multilateral import geks_fisher, geks_tornqvist, geary_khamis, time_product_dummy
from .extension import movement_splice, window_splice, half_splice, mean_splice, fixed_base_rolling_window

__version__ = "0.1.0"
