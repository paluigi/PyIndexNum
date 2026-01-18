"""
Tests for extension methods for connecting multilateral indices.
"""

import pytest
import polars as pl
import numpy as np
from datetime import date, timedelta
from pyindexnum import movement_splice, window_splice, half_splice, mean_splice


class TestExtensionMethods:
    """Test all extension methods."""

    @pytest.fixture
    def sample_indices(self):
        """Create sample multilateral indices for testing."""
        # Create two indices shifted by one period with 3 periods each
        index1 = pl.DataFrame({
            "period": [date(2023, 1, 1), date(2023, 2, 1), date(2023, 3, 1)],
            "index_value": [1.0, 1.05, 1.10]
        })

        index2 = pl.DataFrame({
            "period": [date(2023, 2, 1), date(2023, 3, 1), date(2023, 4, 1)],
            "index_value": [1.05, 1.10, 1.15]
        })

        return index1, index2

    @pytest.fixture
    def sample_indices_5_periods(self):
        """Create sample indices with 5 periods for half splice testing."""
        index1 = pl.DataFrame({
            "period": [
                date(2023, 1, 1), date(2023, 2, 1), date(2023, 3, 1),
                date(2023, 4, 1), date(2023, 5, 1)
            ],
            "index_value": [1.0, 1.02, 1.05, 1.08, 1.12]
        })

        index2 = pl.DataFrame({
            "period": [
                date(2023, 2, 1), date(2023, 3, 1), date(2023, 4, 1),
                date(2023, 5, 1), date(2023, 6, 1)
            ],
            "index_value": [1.02, 1.05, 1.08, 1.12, 1.16]
        })

        return index1, index2

    def test_movement_splice_basic(self, sample_indices):
        """Test basic movement splice functionality."""
        index1, index2 = sample_indices
        result = movement_splice(index1, index2)

        # Check structure
        assert isinstance(result, pl.DataFrame)
        assert result.columns == ["period", "index_value"]
        assert result.height == 1

        # Check period is correct (last period of index2)
        expected_period = date(2023, 4, 1)
        assert result.select("period").item() == expected_period

        # Check calculation: 1.10 * (1.15 / 1.10) = 1.15
        expected_value = 1.15
        assert result.select("index_value").item() == pytest.approx(expected_value)

    def test_window_splice_basic(self, sample_indices):
        """Test basic window splice functionality."""
        index1, index2 = sample_indices
        result = window_splice(index1, index2)

        # Check structure
        assert isinstance(result, pl.DataFrame)
        assert result.columns == ["period", "index_value"]
        assert result.height == 1

        # Check period is correct
        expected_period = date(2023, 4, 1)
        assert result.select("period").item() == expected_period

        # Check calculation details
        # Window rate: 1.15 / 1.05 = 1.095238
        # Base rate: 1.05 / 1.0 = 1.05
        # Result: 1.10 * 1.095238 * 1.05 = 1.20476
        expected_value = 1.10 * (1.15 / 1.05) * (1.05 / 1.0)
        assert result.select("index_value").item() == pytest.approx(expected_value)

    def test_half_splice_basic(self, sample_indices_5_periods):
        """Test basic half splice functionality with odd window length."""
        index1, index2 = sample_indices_5_periods
        result = half_splice(index1, index2)

        # Check structure
        assert isinstance(result, pl.DataFrame)
        assert result.columns == ["period", "index_value"]
        assert result.height == 1

        # Check period is correct
        expected_period = date(2023, 6, 1)
        assert result.select("period").item() == expected_period

        # Check calculation using middle period (index 2, date 2023-03-01)
        # Middle index1: 1.05, Middle index2: 1.05
        # Last index2: 1.16
        # Rate: 1.16 / 1.05 = 1.10476
        # Result: 1.05 * 1.10476 = 1.16
        expected_value = 1.05 * (1.16 / 1.05)
        assert result.select("index_value").item() == pytest.approx(expected_value)

    def test_half_splice_even_window_error(self, sample_indices):
        """Test that half splice raises error for even window length."""
        # Create indices with even window length (2 periods)
        index1 = pl.DataFrame({
            "period": [date(2023, 1, 1), date(2023, 2, 1)],
            "index_value": [1.0, 1.05]
        })

        index2 = pl.DataFrame({
            "period": [date(2023, 2, 1), date(2023, 3, 1)],
            "index_value": [1.05, 1.10]
        })

        with pytest.raises(ValueError, match="odd window length"):
            half_splice(index1, index2)

    def test_mean_splice_basic(self, sample_indices):
        """Test basic mean splice functionality."""
        index1, index2 = sample_indices
        result = mean_splice(index1, index2)

        # Check structure
        assert isinstance(result, pl.DataFrame)
        assert result.columns == ["period", "index_value"]
        assert result.height == 1

        # Check period is correct
        expected_period = date(2023, 4, 1)
        assert result.select("period").item() == expected_period

        # Check calculation for overlapping periods (2023-02-01 and 2023-03-01)
        # For 2023-02-01:
        #   rate_idx1 = 1.10 / 1.05 = 1.0476
        #   rate_idx2 = 1.15 / 1.05 = 1.0952
        #   splice_rate = 1.0952 / 1.0476 = 1.0455
        #
        # For 2023-03-01:
        #   rate_idx1 = 1.10 / 1.10 = 1.0
        #   rate_idx2 = 1.15 / 1.10 = 1.0455
        #   splice_rate = 1.0455 / 1.0 = 1.0455
        #
        # Geometric mean: (1.0455 * 1.0455)^(1/2) = 1.0455
        # Result: 1.10 * 1.0455 = 1.15

        # The calculation should result in approximately 1.15
        expected_value = 1.15
        assert result.select("index_value").item() == pytest.approx(expected_value, abs=1e-6)

    def test_all_methods_same_result_simple_case(self):
        """Test that all methods give same result for simple linear case."""
        # Create indices with constant growth rate
        index1 = pl.DataFrame({
            "period": [date(2023, 1, 1), date(2023, 2, 1), date(2023, 3, 1)],
            "index_value": [1.0, 1.1, 1.2]
        })

        index2 = pl.DataFrame({
            "period": [date(2023, 2, 1), date(2023, 3, 1), date(2023, 4, 1)],
            "index_value": [1.1, 1.2, 1.3]
        })

        # All methods should give the same result for this simple case
        movement_result = movement_splice(index1, index2)
        window_result = window_splice(index1, index2)
        mean_result = mean_splice(index1, index2)

        # Movement: 1.2 * (1.3 / 1.2) = 1.3
        assert movement_result.select("index_value").item() == pytest.approx(1.3)

        # Window: 1.2 * (1.3 / 1.1) * (1.1 / 1.0) = 1.3
        assert window_result.select("index_value").item() == pytest.approx(1.3)

        # Mean: Should also be 1.3 for this simple case
        assert mean_result.select("index_value").item() == pytest.approx(1.3, abs=1e-6)


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
            "period": [date(2023, 3, 1), date(2023, 4, 1)],  # Shifted by 2 periods
            "index_value": [1.10, 1.15]
        })

        with pytest.raises(ValueError, match="shifted by exactly one period"):
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

        with pytest.raises(ValueError, match="No overlapping periods"):
            mean_splice(index1, index2)

    def test_single_period_overlap(self):
        """Test with only one overlapping period."""
        index1 = pl.DataFrame({
            "period": [date(2023, 1, 1), date(2023, 2, 1)],
            "index_value": [1.0, 1.05]
        })

        index2 = pl.DataFrame({
            "period": [date(2023, 2, 1), date(2023, 3, 1)],
            "index_value": [1.05, 1.10]
        })

        # Should work with single overlap
        result = mean_splice(index1, index2)
        assert result.height == 1
        assert result.select("index_value").item() == pytest.approx(1.10)

    def test_large_window_length(self):
        """Test with larger window lengths."""
        periods1 = [date(2023, 1, 1) + timedelta(days=30*i) for i in range(12)]
        periods2 = [date(2023, 2, 1) + timedelta(days=30*i) for i in range(12)]

        index1 = pl.DataFrame({
            "period": periods1,
            "index_value": [1.0 + i*0.01 for i in range(12)]
        })

        index2 = pl.DataFrame({
            "period": periods2,
            "index_value": [1.01 + i*0.01 for i in range(12)]
        })

        # Should work with larger windows
        result = movement_splice(index1, index2)
        assert result.height == 1

        result = mean_splice(index1, index2)
        assert result.height == 1

    def test_different_frequencies(self):
        """Test with different time frequencies."""
        # Monthly data
        index1 = pl.DataFrame({
            "period": [date(2023, 1, 1), date(2023, 2, 1), date(2023, 3, 1)],
            "index_value": [1.0, 1.05, 1.10]
        })

        index2 = pl.DataFrame({
            "period": [date(2023, 2, 1), date(2023, 3, 1), date(2023, 4, 1)],
            "index_value": [1.05, 1.10, 1.15]
        })

        # Should work with monthly data
        result = movement_splice(index1, index2)
        assert result.height == 1

        # Weekly data
        index1_weekly = pl.DataFrame({
            "period": [date(2023, 1, 1), date(2023, 1, 8), date(2023, 1, 15)],
            "index_value": [1.0, 1.02, 1.04]
        })

        index2_weekly = pl.DataFrame({
            "period": [date(2023, 1, 8), date(2023, 1, 15), date(2023, 1, 22)],
            "index_value": [1.02, 1.04, 1.06]
        })

        # Should work with weekly data
        result = movement_splice(index1_weekly, index2_weekly)
        assert result.height == 1


class TestMathematicalProperties:
    """Test mathematical properties and relationships."""

    def test_movement_splice_monotonicity(self):
        """Test that movement splice preserves monotonicity."""
        # Increasing series
        index1 = pl.DataFrame({
            "period": [date(2023, 1, 1), date(2023, 2, 1), date(2023, 3, 1)],
            "index_value": [1.0, 1.05, 1.10]
        })

        index2 = pl.DataFrame({
            "period": [date(2023, 2, 1), date(2023, 3, 1), date(2023, 4, 1)],
            "index_value": [1.05, 1.10, 1.15]
        })

        result = movement_splice(index1, index2)
        # Should be greater than last value of index1
        assert result.select("index_value").item() > 1.10

    def test_window_splice_consistency(self):
        """Test window splice consistency with different base periods."""
        # Create indices with constant growth
        index1 = pl.DataFrame({
            "period": [date(2023, 1, 1), date(2023, 2, 1), date(2023, 3, 1)],
            "index_value": [1.0, 1.1, 1.2]
        })

        index2 = pl.DataFrame({
            "period": [date(2023, 2, 1), date(2023, 3, 1), date(2023, 4, 1)],
            "index_value": [1.1, 1.2, 1.3]
        })

        result = window_splice(index1, index2)
        # Should equal the last value of index2 for this simple case
        assert result.select("index_value").item() == pytest.approx(1.3)

    def test_mean_splice_geometric_mean_property(self):
        """Test that mean splice uses geometric mean correctly."""
        # Create indices where different overlapping periods give different rates
        index1 = pl.DataFrame({
            "period": [date(2023, 1, 1), date(2023, 2, 1), date(2023, 3, 1)],
            "index_value": [1.0, 1.05, 1.10]
        })

        index2 = pl.DataFrame({
            "period": [date(2023, 2, 1), date(2023, 3, 1), date(2023, 4, 1)],
            "index_value": [1.05, 1.12, 1.15]  # Different growth rate
        })

        result = mean_splice(index1, index2)

        # Calculate expected rates manually
        # For 2023-02-01: rate_idx1 = 1.10/1.05, rate_idx2 = 1.15/1.05
        # For 2023-03-01: rate_idx1 = 1.10/1.10, rate_idx2 = 1.15/1.12
        rate1 = (1.15 / 1.05) / (1.10 / 1.05)
        rate2 = (1.15 / 1.12) / (1.10 / 1.10)

        expected_rate = np.exp(np.mean(np.log([rate1, rate2])))
        expected_value = 1.10 * expected_rate

        assert result.select("index_value").item() == pytest.approx(expected_value, abs=1e-6)
