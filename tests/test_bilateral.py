"""
Tests for bilateral price index functions.
"""

import pytest
import polars as pl
from pyindexnum.bilateral import jevons, dudot, carli, laspeyres, paasche, fisher, tornqvist, walsh
from pyindexnum.utils import standardize_columns, aggregate_time


@pytest.fixture
def sample_data():
    """Create sample standardized data for two periods."""
    return pl.DataFrame({
        "date": [
            "2023-01-01", "2023-01-01", "2023-01-01",
            "2023-02-01", "2023-02-01", "2023-02-01"
        ],
        "product_id": ["A", "B", "C", "A", "B", "C"],
        "price": [100, 200, 150, 110, 190, 160]
    })


@pytest.fixture
def aggregated_data():
    """Create aggregated data with one price per product per period."""
    df = pl.DataFrame({
        "date": [
            "2023-01-01", "2023-01-01", "2023-01-01", "2023-01-01",
            "2023-02-01", "2023-02-01", "2023-02-01", "2023-02-01"
        ],
        "product_id": ["A", "A", "B", "B", "A", "A", "B", "B"],
        "price": [100, 105, 200, 195, 110, 115, 190, 185]
    })
    return aggregate_time(df, agg_type="arithmetic")


class TestJevons:
    """Test Jevons index function."""

    def test_jevons_basic(self, sample_data):
        """Test basic Jevons calculation."""
        # Expected: geometric mean of (1.1, 0.95, 160/150)
        # (1.1 * 0.95 * 1.0667)^(1/3) ≈ 1.033
        expected = (1.1 * 0.95 * (160/150)) ** (1/3)
        result = jevons(sample_data)
        assert abs(result - expected) < 1e-6

    def test_jevons_single_product(self):
        """Test Jevons with single product."""
        df = pl.DataFrame({
            "date": ["2023-01-01", "2023-02-01"],
            "product_id": ["A", "A"],
            "price": [100, 110]
        })
        expected = 1.1
        result = jevons(df)
        assert abs(result - expected) < 1e-6

    def test_jevons_zero_price(self):
        """Test Jevons with zero price raises error."""
        df = pl.DataFrame({
            "date": ["2023-01-01", "2023-01-01", "2023-02-01", "2023-02-01"],
            "product_id": ["A", "B", "A", "B"],
            "price": [0, 200, 110, 190]
        })
        with pytest.raises(ValueError, match="All prices must be positive"):
            jevons(df)

    def test_jevons_negative_price(self):
        """Test Jevons with negative price raises error."""
        df = pl.DataFrame({
            "date": ["2023-01-01", "2023-01-01", "2023-02-01", "2023-02-01"],
            "product_id": ["A", "B", "A", "B"],
            "price": [-100, 200, 110, 190]
        })
        with pytest.raises(ValueError, match="All prices must be positive"):
            jevons(df)


class TestDudot:
    """Test Dudot index function."""

    def test_dudot_basic(self, sample_data):
        """Test basic Dudot calculation."""
        # Expected: arithmetic mean of (1.1, 0.95, 160/150)
        expected = (1.1 + 0.95 + 160/150) / 3
        result = dudot(sample_data)
        assert abs(result - expected) < 1e-6

    def test_dudot_single_product(self):
        """Test Dudot with single product."""
        df = pl.DataFrame({
            "date": ["2023-01-01", "2023-02-01"],
            "product_id": ["A", "A"],
            "price": [100, 110]
        })
        expected = 1.1
        result = dudot(df)
        assert abs(result - expected) < 1e-6

    def test_dudot_negative_price(self):
        """Test Dudot with negative price raises error."""
        df = pl.DataFrame({
            "date": ["2023-01-01", "2023-01-01", "2023-02-01", "2023-02-01"],
            "product_id": ["A", "B", "A", "B"],
            "price": [-100, 200, 110, 190]
        })
        with pytest.raises(ValueError, match="All prices must be positive"):
            dudot(df)


class TestCarli:
    """Test Carli index function."""

    def test_carli_basic(self, sample_data):
        """Test basic Carli calculation."""
        # Expected: mean_current / mean_base = (110+190+160)/3 / (100+200+150)/3
        mean_current = (110 + 190 + 160) / 3
        mean_base = (100 + 200 + 150) / 3
        expected = mean_current / mean_base
        result = carli(sample_data)
        assert abs(result - expected) < 1e-6

    def test_carli_single_product(self):
        """Test Carli with single product."""
        df = pl.DataFrame({
            "date": ["2023-01-01", "2023-02-01"],
            "product_id": ["A", "A"],
            "price": [100, 110]
        })
        expected = 110 / 100  # 1.1
        result = carli(df)
        assert abs(result - expected) < 1e-6

    def test_carli_zero_price(self):
        """Test Carli with zero price raises error."""
        df = pl.DataFrame({
            "date": ["2023-01-01", "2023-01-01", "2023-02-01", "2023-02-01"],
            "product_id": ["A", "B", "A", "B"],
            "price": [0, 200, 110, 190]
        })
        with pytest.raises(ValueError, match="All prices must be positive"):
            carli(df)

    def test_carli_negative_price(self):
        """Test Carli with negative price raises error."""
        df = pl.DataFrame({
            "date": ["2023-01-01", "2023-01-01", "2023-02-01", "2023-02-01"],
            "product_id": ["A", "B", "A", "B"],
            "price": [-100, 200, 110, 190]
        })
        with pytest.raises(ValueError, match="All prices must be positive"):
            carli(df)


class TestValidation:
    """Test input validation for bilateral functions."""

    def test_wrong_number_dates(self):
        """Test error with wrong number of dates."""
        # One date
        df = pl.DataFrame({
            "date": ["2023-01-01", "2023-01-01"],
            "product_id": ["A", "B"],
            "price": [100, 200]
        })
        with pytest.raises(ValueError, match="exactly 2 unique dates"):
            jevons(df)

        # Three dates
        df = pl.DataFrame({
            "date": ["2023-01-01", "2023-02-01", "2023-03-01"],
            "product_id": ["A", "A", "A"],
            "price": [100, 110, 120]
        })
        with pytest.raises(ValueError, match="exactly 2 unique dates"):
            jevons(df)

    def test_missing_columns(self):
        """Test error with missing required columns."""
        df = pl.DataFrame({
            "date": ["2023-01-01", "2023-02-01"],
            "price": [100, 110]
        })
        with pytest.raises(ValueError, match="Missing required columns"):
            jevons(df)

    def test_multiple_prices_per_product_period(self, aggregated_data):
        """Test error when product has multiple prices per period."""
        # Rename columns to match expected format
        df_renamed = aggregated_data.rename({"period": "date", "aggregated_price": "price"})
        # aggregated_data should be valid, but let's add a duplicate
        extra_row = df_renamed.filter(pl.col("product_id") == "A").head(1)
        df_invalid = pl.concat([df_renamed, extra_row])

        with pytest.raises(ValueError, match="exactly one price per period"):
            jevons(df_invalid)

    def test_missing_products_in_period(self, sample_data):
        """Test error when products differ between periods."""
        # Remove product C from current period
        df_invalid = sample_data.filter(
            ~((pl.col("date") == "2023-02-01") & (pl.col("product_id") == "C"))
        )
        with pytest.raises(ValueError, match="Products must be identical"):
            jevons(df_invalid)


@pytest.fixture
def sample_weighted_data():
    """Create sample standardized data with quantities for two periods."""
    return pl.DataFrame({
        "date": [
            "2023-01-01", "2023-01-01", "2023-01-01",
            "2023-02-01", "2023-02-01", "2023-02-01"
        ],
        "product_id": ["A", "B", "C", "A", "B", "C"],
        "price": [100, 200, 150, 110, 190, 160],
        "quantity": [10, 20, 15, 10, 20, 15]
    })


class TestLaspeyres:
    """Test Laspeyres index function."""

    def test_laspeyres_basic(self, sample_weighted_data):
        """Test basic Laspeyres calculation."""
        # Expected: sum(p_t * q_0) / sum(p_0 * q_0)
        # = (110*10 + 190*20 + 160*15) / (100*10 + 200*20 + 150*15)
        # = (1100 + 3800 + 2400) / (1000 + 4000 + 2250) = 7300 / 7250 ≈ 1.0076
        expected = 7300 / 7250
        result = laspeyres(sample_weighted_data)
        assert abs(result - expected) < 1e-6

    def test_laspeyres_single_product(self):
        """Test Laspeyres with single product."""
        df = pl.DataFrame({
            "date": ["2023-01-01", "2023-02-01"],
            "product_id": ["A", "A"],
            "price": [100, 110],
            "quantity": [10, 10]
        })
        expected = (110 * 10) / (100 * 10)  # 1.1
        result = laspeyres(df)
        assert abs(result - expected) < 1e-6

    def test_laspeyres_zero_quantity(self):
        """Test Laspeyres with zero quantity raises error."""
        df = pl.DataFrame({
            "date": ["2023-01-01", "2023-01-01", "2023-02-01", "2023-02-01"],
            "product_id": ["A", "B", "A", "B"],
            "price": [100, 200, 110, 190],
            "quantity": [0, 20, 10, 20]
        })
        with pytest.raises(ValueError, match="All quantities must be positive"):
            laspeyres(df)

    def test_laspeyres_missing_quantity(self):
        """Test Laspeyres with missing quantity column raises error."""
        df = pl.DataFrame({
            "date": ["2023-01-01", "2023-02-01"],
            "product_id": ["A", "A"],
            "price": [100, 110]
        })
        with pytest.raises(ValueError, match="Missing required column: quantity"):
            laspeyres(df)


class TestPaasche:
    """Test Paasche index function."""

    def test_paasche_basic(self, sample_weighted_data):
        """Test basic Paasche calculation."""
        # Expected: sum(p_t * q_t) / sum(p_0 * q_t)
        # = (110*10 + 190*20 + 160*15) / (100*10 + 200*20 + 150*15)
        # = 7300 / 7250 ≈ 1.0076
        expected = 7300 / 7250
        result = paasche(sample_weighted_data)
        assert abs(result - expected) < 1e-6

    def test_paasche_different_quantities(self):
        """Test Paasche with different quantities per period."""
        df = pl.DataFrame({
            "date": ["2023-01-01", "2023-01-01", "2023-02-01", "2023-02-01"],
            "product_id": ["A", "B", "A", "B"],
            "price": [100, 200, 110, 190],
            "quantity": [10, 20, 15, 25]
        })
        # Expected: (110*15 + 190*25) / (100*15 + 200*25) = (1650 + 4750) / (1500 + 5000) = 6400 / 6500 ≈ 0.9846
        expected = 6400 / 6500
        result = paasche(df)
        assert abs(result - expected) < 1e-6


class TestFisher:
    """Test Fisher index function."""

    def test_fisher_basic(self, sample_weighted_data):
        """Test basic Fisher calculation."""
        # Fisher = sqrt(Laspeyres * Paasche)
        laspeyres_index = laspeyres(sample_weighted_data)
        paasche_index = paasche(sample_weighted_data)
        expected = (laspeyres_index * paasche_index) ** 0.5
        result = fisher(sample_weighted_data)
        assert abs(result - expected) < 1e-6

    def test_fisher_time_reversal(self, sample_weighted_data):
        """Test Fisher satisfies time reversal test."""
        # Time reversal: Fisher(t->0) * Fisher(0->t) = 1
        # Create reversed data
        df_reversed = sample_weighted_data.with_columns(
            pl.when(pl.col("date") == "2023-01-01")
            .then(pl.lit("2023-02-01"))
            .when(pl.col("date") == "2023-02-01")
            .then(pl.lit("2023-01-01"))
            .otherwise(pl.col("date"))
            .alias("date")
        )

        fisher_forward = fisher(sample_weighted_data)
        fisher_reverse = fisher(df_reversed)

        # Should satisfy time reversal (approximately)
        assert abs((fisher_forward * fisher_reverse) - 1.0) < 1e-10


class TestTornqvist:
    """Test Törnqvist index function."""

    def test_tornqvist_basic(self):
        """Test basic Törnqvist calculation."""
        df = pl.DataFrame({
            "date": ["2023-01-01", "2023-01-01", "2023-02-01", "2023-02-01"],
            "product_id": ["A", "B", "A", "B"],
            "price": [100, 200, 110, 190],
            "quantity": [10, 20, 15, 25]
        })
        # Manual calculation:
        # Total Q = (10+15) + (20+25) = 70
        # Weight A: (10+15)/70 ≈ 0.3571, ln(110/100) ≈ 0.09531
        # Weight B: (20+25)/70 ≈ 0.6429, ln(190/200) ≈ -0.05129
        # Sum = 0.3571*0.09531 + 0.6429*(-0.05129) ≈ 0.0340 - 0.0330 ≈ 0.0010
        # exp(0.0010) ≈ 1.0010
        result = tornqvist(df)
        assert result > 1.0  # Should be slightly above 1
        assert abs(result - 1.001) < 0.001  # Approximate check


class TestWalsh:
    """Test Walsh index function."""

    def test_walsh_basic(self, sample_weighted_data):
        """Test basic Walsh calculation."""
        # Expected: sum(sqrt(p_t * p_0) * q_0) / sum(p_0 * q_0)
        # sqrt(p_t * p_0) for each:
        # A: sqrt(110*100) ≈ 104.403
        # B: sqrt(190*200) ≈ 194.935
        # C: sqrt(160*150) ≈ 154.919
        # Numerator: 104.403*10 + 194.935*20 + 154.919*15 ≈ 1044 + 3899 + 2324 ≈ 7267
        # Denominator: 100*10 + 200*20 + 150*15 = 7250
        # Result ≈ 7267/7250 ≈ 1.0023
        expected = 7267 / 7250
        result = walsh(sample_weighted_data)
        assert abs(result - expected) < 1e-3  # Allow small numerical difference


class TestWeightedValidation:
    """Test input validation for weighted bilateral functions."""

    def test_missing_quantity_column(self, sample_data):
        """Test error when quantity column is missing."""
        with pytest.raises(ValueError, match="Missing required column: quantity"):
            laspeyres(sample_data)

    def test_zero_quantity(self):
        """Test error with zero quantity."""
        df = pl.DataFrame({
            "date": ["2023-01-01", "2023-02-01"],
            "product_id": ["A", "A"],
            "price": [100, 110],
            "quantity": [0, 10]
        })
        with pytest.raises(ValueError, match="All quantities must be positive"):
            laspeyres(df)

    def test_negative_quantity(self):
        """Test error with negative quantity."""
        df = pl.DataFrame({
            "date": ["2023-01-01", "2023-02-01"],
            "product_id": ["A", "A"],
            "price": [100, 110],
            "quantity": [-10, 10]
        })
        with pytest.raises(ValueError, match="All quantities must be positive"):
            laspeyres(df)


class TestWithRealData:
    """Test with real aggregated test data."""

    @pytest.fixture
    def real_data(self):
        """Load and prepare real test data."""
        df = pl.read_csv("tests/test_data.csv")
        df_std = standardize_columns(df)
        # Filter to two dates and aggregate
        df_filtered = df_std.filter(
            pl.col("date").dt.strftime("%Y-%m-%d").is_in(["2023-01-01", "2023-01-02"])
        )
        agg_df = aggregate_time(df_filtered, agg_type="arithmetic", freq="1d")
        # Rename to expected column names
        return agg_df.rename({"period": "date", "aggregated_price": "price"})

    def test_with_real_data(self, real_data):
        """Test all indices with real aggregated data."""
        # Just check they run without error and return reasonable values
        j_index = jevons(real_data)
        d_index = dudot(real_data)
        c_index = carli(real_data)

        # All should be positive floats
        assert isinstance(j_index, float) and j_index > 0
        assert isinstance(d_index, float) and d_index > 0
        assert isinstance(c_index, float) and c_index > 0

        # Jevons and Dudot should be close for small changes
        assert abs(j_index - d_index) < 0.1  # Allow some difference

    def test_real_data_structure(self, real_data):
        """Verify real data has correct structure."""
        # Should have exactly 2 dates
        dates = real_data.select("date").unique()
        assert len(dates) == 2

        # Each product should have exactly one price per date
        grouped = real_data.group_by(["product_id", "date"]).len()
        max_count = grouped.select(pl.col("len").max()).item()
        assert max_count == 1

        # Same products in both periods
        dates_list = dates["date"].to_list()
        products_base = real_data.filter(pl.col("date") == dates_list[0]).select("product_id").sort("product_id")
        products_current = real_data.filter(pl.col("date") == dates_list[1]).select("product_id").sort("product_id")
        assert products_base.equals(products_current)
