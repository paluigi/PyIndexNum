"""
Tests for extension methods for connecting multilateral indices.
"""

import pytest
import polars as pl
import numpy as np
from datetime import date, timedelta
from pathlib import Path
from pyindexnum import movement_splice, window_splice, half_splice, mean_splice, fixed_base_rolling_window, geks_fisher


class TestExtensionMethods:
    """Test all extension methods."""

    @pytest.fixture
    def sample_indices(self):
        """Create sample multilateral indices for testing."""
        # Load and process test data
        test_data_path = Path(__file__).parent / "test_data.csv"
        df = pl.read_csv(test_data_path)

        # Rename columns to match multilateral input requirements
        df = df.rename({
            "date": "period",
            "price": "aggregated_price",
            "quantity": "aggregated_quantity"
        })

        # Cast period to Date type
        df = df.with_columns(
            pl.col("period").str.strptime(pl.Date, "%Y-%m-%d")
        )

        # Aggregate by product and period (sum quantities, average prices)
        df_agg = df.group_by(["product_id", "period"]).agg(
            pl.col("aggregated_quantity").sum().alias("aggregated_quantity"),
            pl.col("aggregated_price").mean().alias("aggregated_price")
        )

        # Get sorted unique dates
        dates = df_agg.select("period").unique().sort("period").to_series().to_list()

        # Create first window: periods 1-13
        window1_dates = dates[:13]
        df_window1 = df_agg.filter(pl.col("period").is_in(window1_dates))

        # Create second window: periods 2-14 (shifted by one period)
        window2_dates = dates[1:14]
        df_window2 = df_agg.filter(pl.col("period").is_in(window2_dates))

        # Compute multilateral indices using GEKS-Fisher
        index1 = geks_fisher(df_window1)
        index2 = geks_fisher(df_window2)

        return index1, index2

    def test_movement_splice(self, sample_indices):
        """Test movement splice method."""
        index1, index2 = sample_indices
        result = movement_splice(index1, index2)

        # Should have all periods from index1 plus the spliced period
        expected_periods = set(index1.select("period").to_series().to_list() +
                               [index2.select(pl.col("period").max()).item()])
        actual_periods = set(result.select("period").to_series().to_list())

        assert actual_periods == expected_periods
        assert result.select(pl.col("index_value").min()).item() > 0
        assert result.select(pl.col("index_value").is_finite().all()).item()

    def test_window_splice(self, sample_indices):
        """Test window splice method."""
        index1, index2 = sample_indices
        result = window_splice(index1, index2)

        # Should have all periods from index1 plus the spliced period
        expected_periods = set(index1.select("period").to_series().to_list() +
                               [index2.select(pl.col("period").max()).item()])
        actual_periods = set(result.select("period").to_series().to_list())

        assert actual_periods == expected_periods
        assert result.select(pl.col("index_value").min()).item() > 0
        assert result.select(pl.col("index_value").is_finite().all()).item()

    def test_half_splice(self, sample_indices):
        """Test half splice method."""
        index1, index2 = sample_indices
        result = half_splice(index1, index2)

        # Should have all periods from index1 plus the spliced period
        expected_periods = set(index1.select("period").to_series().to_list() +
                               [index2.select(pl.col("period").max()).item()])
        actual_periods = set(result.select("period").to_series().to_list())

        assert actual_periods == expected_periods
        assert result.select(pl.col("index_value").min()).item() > 0
        assert result.select(pl.col("index_value").is_finite().all()).item()

    def test_mean_splice(self, sample_indices):
        """Test mean splice method."""
        index1, index2 = sample_indices
        result = mean_splice(index1, index2)

        # Should have all periods from index1 plus the spliced period
        expected_periods = set(index1.select("period").to_series().to_list() +
                               [index2.select(pl.col("period").max()).item()])
        actual_periods = set(result.select("period").to_series().to_list())

        assert actual_periods == expected_periods
        assert result.select(pl.col("index_value").min()).item() > 0
        assert result.select(pl.col("index_value").is_finite().all()).item()

    def test_fixed_base_rolling_window(self, sample_indices):
        """Test fixed base rolling window method."""
        index1, index2 = sample_indices
        base_period = index1.select("period").sort("period").slice(1, 1).item()  # Second period, should be in both
        base_period_str = base_period.strftime("%Y-%m-%d")

        result = fixed_base_rolling_window(index1, index2, base_period_str)

        # Should have all periods from index1 plus the spliced period
        expected_periods = set(index1.select("period").to_series().to_list() +
                               [index2.select(pl.col("period").max()).item()])
        actual_periods = set(result.select("period").to_series().to_list())

        assert actual_periods == expected_periods
        assert result.select(pl.col("index_value").min()).item() > 0
        assert result.select(pl.col("index_value").is_finite().all()).item()




class TestInputValidation:
    """Test input validation for extension methods."""

    def test_invalid_dataframe_type(self):
        """Test validation with invalid DataFrame types."""
        with pytest.raises(ValueError, match="polars DataFrames"):
            movement_splice("not a dataframe", pl.DataFrame())

    def test_missing_columns(self):
        """Test validation with missing required columns."""
        index1 = pl.DataFrame({
            "period": [date(2023, 1, 1)],
            "value": [1.0]  # Wrong column name
        })

        index2 = pl.DataFrame({
            "period": [date(2023, 2, 1)],
            "index_value": [1.05]
        })

        with pytest.raises(ValueError, match="missing required columns"):
            movement_splice(index1, index2)

    def test_invalid_index_value_type(self):
        """Test validation with non-numeric index values."""
        index1 = pl.DataFrame({
            "period": [date(2023, 1, 1)],
            "index_value": ["invalid"]  # String instead of number
        })

        index2 = pl.DataFrame({
            "period": [date(2023, 2, 1)],
            "index_value": [1.05]
        })

        with pytest.raises(ValueError, match="must be numeric"):
            movement_splice(index1, index2)

    def test_invalid_period_type(self):
        """Test validation with non-temporal period column."""
        index1 = pl.DataFrame({
            "period": ["2023-01-01"],  # String instead of date
            "index_value": [1.0]
        })

        index2 = pl.DataFrame({
            "period": ["2023-02-01"],
            "index_value": [1.05]
        })

        with pytest.raises(ValueError, match="must be a temporal type"):
            movement_splice(index1, index2)

    def test_different_window_lengths(self):
        """Test validation with different window lengths."""
        index1 = pl.DataFrame({
            "period": [date(2023, 1, 1), date(2023, 2, 1)],
            "index_value": [1.0, 1.05]
        })

        index2 = pl.DataFrame({
            "period": [date(2023, 2, 1), date(2023, 3, 1), date(2023, 4, 1)],
            "index_value": [1.05, 1.10, 1.15]
        })

        with pytest.raises(ValueError, match="same window length"):
            movement_splice(index1, index2)

    def test_insufficient_periods(self):
        """Test validation with insufficient periods."""
        index1 = pl.DataFrame({
            "period": [date(2023, 1, 1)],
            "index_value": [1.0]
        })

        index2 = pl.DataFrame({
            "period": [date(2023, 2, 1)],
            "index_value": [1.05]
        })

        with pytest.raises(ValueError, match="at least 2 periods"):
            movement_splice(index1, index2)

    def test_duplicate_periods(self):
        """Test validation with duplicate periods."""
        index1 = pl.DataFrame({
            "period": [date(2023, 1, 1), date(2023, 1, 1)],  # Duplicate
            "index_value": [1.0, 1.05]
        })

        index2 = pl.DataFrame({
            "period": [date(2023, 2, 1), date(2023, 3, 1)],
            "index_value": [1.05, 1.10]
        })

        with pytest.raises(ValueError, match="duplicate periods"):
            movement_splice(index1, index2)

    def test_non_shifted_indices(self):
        """Test validation with indices not shifted by exactly one period."""
        index1 = pl.DataFrame({
            "period": [date(2023, 1, 1), date(2023, 2, 1)],
            "index_value": [1.0, 1.05]
        })

        index2 = pl.DataFrame({
            "period": [date(2023, 3, 1), date(2023, 4, 1)],  # No overlap
            "index_value": [1.10, 1.15]
        })

        with pytest.raises(ValueError, match="Indices must be overlapped"):
            movement_splice(index1, index2)

    def test_zero_index_values(self):
        """Test validation with zero index values."""
        index1 = pl.DataFrame({
            "period": [date(2023, 1, 1), date(2023, 2, 1)],
            "index_value": [0.0, 1.05]  # Zero value
        })

        index2 = pl.DataFrame({
            "period": [date(2023, 2, 1), date(2023, 3, 1)],
            "index_value": [1.05, 1.10]
        })

        with pytest.raises(ValueError, match="all positive index values"):
            movement_splice(index1, index2)

    def test_negative_index_values(self):
        """Test validation with negative index values."""
        index1 = pl.DataFrame({
            "period": [date(2023, 1, 1), date(2023, 2, 1)],
            "index_value": [-1.0, 1.05]  # Negative value
        })

        index2 = pl.DataFrame({
            "period": [date(2023, 2, 1), date(2023, 3, 1)],
            "index_value": [1.05, 1.10]
        })

        with pytest.raises(ValueError, match="all positive index values"):
            movement_splice(index1, index2)


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_no_overlapping_periods_mean_splice(self):
        """Test mean splice with no overlapping periods."""
        index1 = pl.DataFrame({
            "period": [date(2023, 1, 1), date(2023, 2, 1)],
            "index_value": [1.0, 1.05]
        })

        index2 = pl.DataFrame({
            "period": [date(2023, 3, 1), date(2023, 4, 1)],
            "index_value": [1.10, 1.15]
        })

        with pytest.raises(ValueError, match="Indices must be overlapped"):
            mean_splice(index1, index2)
