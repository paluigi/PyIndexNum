"""
Tests for multilateral price index functions.
"""

import pytest
import polars as pl
import numpy as np
from datetime import date
from pyindexnum import geks_fisher, geks_tornqvist, geary_khamis, time_product_dummy


class TestGEKSIndices:
    """Test GEKS multilateral indices."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return pl.DataFrame({
            "product_id": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
            "period": [
                date(2023, 1, 1), date(2023, 2, 1), date(2023, 3, 1),
                date(2023, 1, 1), date(2023, 2, 1), date(2023, 3, 1),
                date(2023, 1, 1), date(2023, 2, 1), date(2023, 3, 1)
            ],
            "aggregated_price": [100, 105, 110, 200, 210, 220, 50, 52, 54],
            "aggregated_quantity": [10, 10, 10, 20, 20, 20, 5, 5, 5]
        })

    def test_geks_fisher_basic(self, sample_data):
        """Test basic GEKS-Fisher functionality."""
        result = geks_fisher(sample_data)

        # Check structure
        assert isinstance(result, pl.DataFrame)
        assert result.columns == ["period", "index_value"]
        assert result.height == 3  # 3 periods

        # Check base period is 1.0
        assert result.filter(pl.col("period") == date(2023, 1, 1)).select("index_value").item() == pytest.approx(1.0)

        # For GEKS-Fisher with constant quantities across all periods, 
        # the multilateral index should be transitive and match bilateral Fisher.
        # Since quantities are constant (10, 20, 5), Fisher matches Laspeyres/Paasche.
        # Relative price change from period 1 to 2 for A, B, C: 1.05, 1.05, 1.04
        # Fisher(1,2) = (105*10 + 210*20 + 52*5) / (100*10 + 200*20 + 50*5)
        # = (1050 + 4200 + 260) / (1000 + 4000 + 250) = 5510 / 5250 ≈ 1.0495238
        expected_2 = 5510 / 5250
        assert result.filter(pl.col("period") == date(2023, 2, 1)).select("index_value").item() == pytest.approx(expected_2, abs=1e-6)

    def test_geks_tornqvist_basic(self, sample_data):
        """Test basic GEKS-Törnqvist functionality."""
        result = geks_tornqvist(sample_data)

        # Check structure
        assert isinstance(result, pl.DataFrame)
        assert result.columns == ["period", "index_value"]
        assert result.height == 3

        # Check base period is 1.0
        assert result.filter(pl.col("period") == date(2023, 1, 1)).select("index_value").item() == pytest.approx(1.0)

        # With constant quantities, GEKS-Tornqvist reduces to the bilateral Tornqvist.
        # GEKS(2) = [T(1,2)/T(1,1) * T(2,2)/T(2,1) * T(3,2)/T(3,1)]^(1/3)
        # = [T12 * T12 * T13/T23]^(1/3)
        # T12 = 1.0495238, T13 = 1.0990476, T23 = 1.0471869
        # GEKS(2) ≈ 1.0495238
        expected_2 = 1.0495238009
        assert result.filter(pl.col("period") == date(2023, 2, 1)).select("index_value").item() == pytest.approx(expected_2, abs=1e-6)

    def test_geks_insufficient_periods(self):
        """Test GEKS with insufficient periods."""
        df = pl.DataFrame({
            "product_id": ["A", "B"],
            "period": [date(2023, 1, 1), date(2023, 1, 1)],
            "aggregated_price": [100, 200],
            "aggregated_quantity": [10, 20]
        })

        with pytest.raises(ValueError, match="require at least 2 periods"):
            geks_fisher(df)

    def test_geks_insufficient_products(self):
        """Test GEKS with insufficient products."""
        df = pl.DataFrame({
            "product_id": ["A", "A"],
            "period": [date(2023, 1, 1), date(2023, 2, 1)],
            "aggregated_price": [100, 105],
            "aggregated_quantity": [10, 10]
        })

        with pytest.raises(ValueError, match="require at least 2 products"):
            geks_fisher(df)


class TestGearyKhamis:
    """Test Geary-Khamis multilateral index."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return pl.DataFrame({
            "product_id": ["A", "A", "A", "B", "B", "B"],
            "period": [
                date(2023, 1, 1), date(2023, 2, 1), date(2023, 3, 1),
                date(2023, 1, 1), date(2023, 2, 1), date(2023, 3, 1)
            ],
            "aggregated_price": [100, 105, 110, 200, 210, 220],
            "aggregated_quantity": [10, 10, 10, 20, 20, 20]
        })

    def test_geary_khamis_basic(self, sample_data):
        """Test basic Geary-Khamis functionality."""
        result = geary_khamis(sample_data)

        # Check structure
        assert isinstance(result, pl.DataFrame)
        assert result.columns == ["period", "index_value"]
        assert result.height == 3

        # Check base period is 1.0
        assert result.filter(pl.col("period") == date(2023, 1, 1)).select("index_value").item() == pytest.approx(1.0, abs=1e-6)

        # With constant quantities, price levels should follow the average price change
        # Period 2 vs Period 1: (105*10 + 210*20) / (100*10 + 200*20) = 5250 / 5000 = 1.05
        assert result.filter(pl.col("period") == date(2023, 2, 1)).select("index_value").item() == pytest.approx(1.05, abs=1e-6)

    def test_geary_khamis_convergence(self, sample_data):
        """Test Geary-Khamis convergence."""
        # Should converge quickly for this simple case
        result = geary_khamis(sample_data, max_iter=10)
        assert result.height == 3


class TestTimeProductDummy:
    """Test Time Product Dummy multilateral index."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return pl.DataFrame({
            "product_id": ["A", "A", "A", "B", "B", "B"],
            "period": [
                date(2023, 1, 1), date(2023, 2, 1), date(2023, 3, 1),
                date(2023, 1, 1), date(2023, 2, 1), date(2023, 3, 1)
            ],
            "aggregated_price": [100, 105, 110, 200, 210, 220],
            "aggregated_quantity": [10, 10, 10, 20, 20, 20]
        })

    @pytest.fixture
    def sample_data_no_quantity(self):
        """Create sample data without quantity column."""
        return pl.DataFrame({
            "product_id": ["A", "A", "A", "B", "B", "B"],
            "period": [
                date(2023, 1, 1), date(2023, 2, 1), date(2023, 3, 1),
                date(2023, 1, 1), date(2023, 2, 1), date(2023, 3, 1)
            ],
            "aggregated_price": [100, 105, 110, 200, 210, 220]
        })

    def test_time_product_dummy_weighted(self, sample_data):
        """Test weighted Time Product Dummy."""
        result = time_product_dummy(sample_data, weighted=True)

        assert isinstance(result, pl.DataFrame)
        assert result.columns == ["period", "index_value"]
        assert result.height == 3
        assert result.filter(pl.col("period") == date(2023, 1, 1)).select("index_value").item() == pytest.approx(1.0)
        assert result.filter(pl.col("index_value") <= 0).height == 0



    def test_time_product_dummy_no_quantity_weighted(self, sample_data_no_quantity):
        """Test that weighted TPD fails without quantity column."""
        with pytest.raises(ValueError, match="Missing required columns"):
            time_product_dummy(sample_data_no_quantity, weighted=True)

    def test_time_product_dummy_no_quantity_unweighted(self, sample_data_no_quantity):
        """Test that unweighted TPD works without quantity column."""
        result = time_product_dummy(sample_data_no_quantity, weighted=False)
        assert result.height == 3


class TestValidation:
    """Test input validation for multilateral functions."""

    def test_missing_columns(self):
        """Test validation with missing columns."""
        df = pl.DataFrame({
            "product_id": ["A", "B"],
            "period": [date(2023, 1, 1), date(2023, 1, 1)],
            "price": [100, 200]  # Wrong column name
        })

        with pytest.raises(ValueError, match="Missing required columns"):
            geks_fisher(df)

    def test_zero_price(self):
        """Test validation with zero prices."""
        df = pl.DataFrame({
            "product_id": ["A", "B"],
            "period": [date(2023, 1, 1), date(2023, 1, 1)],
            "aggregated_price": [0, 200],
            "aggregated_quantity": [10, 20]
        })

        with pytest.raises(ValueError, match="must be positive"):
            geks_fisher(df)

    def test_duplicate_observations(self):
        """Test validation with duplicate product-period observations."""
        df = pl.DataFrame({
            "product_id": ["A", "A", "B"],
            "period": [date(2023, 1, 1), date(2023, 1, 1), date(2023, 1, 1)],
            "aggregated_price": [100, 105, 200],
            "aggregated_quantity": [10, 10, 20]
        })

        with pytest.raises(ValueError, match="exactly one observation per period"):
            geks_fisher(df)
