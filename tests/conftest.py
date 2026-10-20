"""
Shared fixtures for the PyIndexNum test suite.

All CSV-based fixtures resolve paths relative to this file so the tests can
be run from any working directory.
"""

from pathlib import Path

import polars as pl
import pytest

TESTS_DIR = Path(__file__).parent


@pytest.fixture
def perf_raw() -> pl.DataFrame:
    """Raw contents of tests/performance_price_data.csv."""
    return pl.read_csv(TESTS_DIR / "performance_price_data.csv")


@pytest.fixture
def test_data_raw() -> pl.DataFrame:
    """Raw contents of tests/test_data.csv."""
    return pl.read_csv(TESTS_DIR / "test_data.csv")
