"""
Tests for utility functions in pyindexnum.utils.
"""

import polars as pl
import pytest
from pathlib import Path
from datetime import date

from pyindexnum.utils import aggregate_time, standardize_columns, get_summary, remove_unbalanced, carry_forward_imputation, carry_backward_imputation


@pytest.fixture
def sample_data():
    """Load test data from CSV file."""
    csv_path = Path(__file__).parent / "test_data.csv"
    return pl.read_csv(csv_path)


@pytest.fixture
def small_sample_data():
    """Create a small sample dataset for testing."""
    return pl.DataFrame({
        "date": ["2023-01-01", "2023-01-15", "2023-02-01", "2023-02-15"],
        "product": ["A", "A", "A", "A"],
        "price": [100.0, 110.0, 120.0, 130.0],
        "quantity": [10, 12, 15, 18]
    })


class TestAggregateTime:
    """Test suite for aggregate_time function."""

    def test_basic_arithmetic_aggregation(self, small_sample_data):
        """Test basic arithmetic mean aggregation."""
        result = aggregate_time(
            small_sample_data,
            date_col="date",
            price_col="price",
            quantity_col="quantity",
            id_col="product",
            agg_type="arithmetic",
            freq="1mo"
        )

        # Should have 2 periods: 2023-01 and 2023-02
        assert len(result) == 2
        assert result.columns == ["product", "period", "aggregated_price", "aggregated_quantity"]

        # Check January aggregation
        jan_data = result.filter(pl.col("period") == pl.date(2023, 1, 1))
        assert jan_data["aggregated_price"][0] == 105.0  # (100 + 110) / 2
        assert jan_data["aggregated_quantity"][0] == 22  # 10 + 12

        # Check February aggregation
        feb_data = result.filter(pl.col("period") == pl.date(2023, 2, 1))
        assert feb_data["aggregated_price"][0] == 125.0  # (120 + 130) / 2
        assert feb_data["aggregated_quantity"][0] == 33  # 15 + 18

    def test_geometric_mean_aggregation(self, small_sample_data):
        """Test geometric mean aggregation."""
        result = aggregate_time(
            small_sample_data,
            date_col="date",
            price_col="price",
            id_col="product",
            agg_type="geometric",
            freq="1mo"
        )

        # Should have 2 periods
        assert len(result) == 2

        # Check January geometric mean
        jan_data = result.filter(pl.col("period") == pl.date(2023, 1, 1))
        expected_geom = (100.0 * 110.0) ** 0.5  # sqrt(100 * 110)
        actual = jan_data["aggregated_price"][0]
        assert abs(actual - expected_geom) < 1e-6

    def test_weighted_arithmetic_aggregation(self, small_sample_data):
        """Test weighted arithmetic mean aggregation."""
        result = aggregate_time(
            small_sample_data,
            date_col="date",
            price_col="price",
            quantity_col="quantity",
            id_col="product",
            agg_type="weighted_arithmetic",
            freq="1mo"
        )

        # Should have 2 periods
        assert len(result) == 2

        # Check January weighted mean: (100*10 + 110*12) / (10 + 12)
        jan_data = result.filter(pl.col("period") == pl.date(2023, 1, 1))
        expected_weighted = (100*10 + 110*12) / (10 + 12)
        assert abs(jan_data["aggregated_price"][0] - expected_weighted) < 1e-6

    def test_weighted_harmonic_aggregation(self, small_sample_data):
        """Test weighted harmonic mean aggregation."""
        result = aggregate_time(
            small_sample_data,
            date_col="date",
            price_col="price",
            quantity_col="quantity",
            id_col="product",
            agg_type="weighted_harmonic",
            freq="1mo"
        )

        # Should have 2 periods
        assert len(result) == 2

        # Check January weighted harmonic: (10 + 12) / ((10/100) + (12/110))
        jan_data = result.filter(pl.col("period") == pl.date(2023, 1, 1))
        expected_harmonic = (10 + 12) / ((10/100) + (12/110))
        assert abs(jan_data["aggregated_price"][0] - expected_harmonic) < 1e-6

    def test_weighted_without_quantity_raises_error(self, small_sample_data):
        """Test that weighted aggregation without quantity raises error."""
        with pytest.raises(ValueError, match="Weighted aggregation.*requires quantity_col"):
            aggregate_time(
                small_sample_data,
                date_col="date",
                price_col="price",
                quantity_col=None,
                id_col="product",
                agg_type="weighted_arithmetic",
                freq="1mo"
            )

    def test_different_frequencies(self, sample_data):
        """Test aggregation with different frequencies using full dataset."""
        # Test weekly aggregation
        result_weekly = aggregate_time(
            sample_data,
            date_col="date",
            price_col="price",
            quantity_col="quantity",
            id_col="product_id",
            agg_type="arithmetic",
            freq="1w"
        )

        # Should have more periods than monthly
        result_monthly = aggregate_time(
            sample_data,
            date_col="date",
            price_col="price",
            quantity_col="quantity",
            id_col="product_id",
            agg_type="arithmetic",
            freq="1mo"
        )

        assert len(result_weekly) > len(result_monthly)

    def test_multiple_products(self, sample_data):
        """Test aggregation with multiple products."""
        result = aggregate_time(
            sample_data,
            date_col="date",
            price_col="price",
            quantity_col="quantity",
            id_col="product_id",
            agg_type="arithmetic",
            freq="1mo"
        )

        # Should have multiple products per period
        unique_products = result["product_id"].unique()
        assert len(unique_products) > 1

        # Check that each product appears in multiple periods
        for product in unique_products[:2]:  # Test first 2 products
            product_data = result.filter(pl.col("product_id") == product)
            assert len(product_data) > 1

    def test_without_quantity(self, small_sample_data):
        """Test aggregation without quantity column."""
        result = aggregate_time(
            small_sample_data,
            date_col="date",
            price_col="price",
            quantity_col=None,
            id_col="product",
            agg_type="arithmetic",
            freq="1mo"
        )

        # Should not have aggregated_quantity column
        assert "aggregated_quantity" not in result.columns
        assert "aggregated_price" in result.columns

    def test_string_date_parsing(self, small_sample_data):
        """Test that string dates are properly parsed."""
        # Data already has string dates, should work
        result = aggregate_time(
            small_sample_data,
            date_col="date",
            price_col="price",
            id_col="product",
            agg_type="arithmetic",
            freq="1mo"
        )

        assert result.schema["period"] == pl.Date

    def test_invalid_inputs(self, small_sample_data):
        """Test error handling for invalid inputs."""
        # Invalid df type
        with pytest.raises(ValueError, match="df must be a polars DataFrame"):
            aggregate_time("not a dataframe", "date", "price", None, "product", "arithmetic", "1mo")

        # Missing required columns
        with pytest.raises(ValueError, match="Missing required columns"):
            aggregate_time(small_sample_data, "nonexistent", "price", None, "product", "arithmetic", "1mo")

        # Invalid date column type
        bad_data = small_sample_data.with_columns(pl.col("price").alias("bad_date"))
        with pytest.raises(ValueError, match="date_col.*must be datetime"):
            aggregate_time(bad_data, "bad_date", "price", None, "product", "arithmetic", "1mo")

        # Invalid price column type
        bad_data2 = small_sample_data.with_columns(pl.col("product").alias("bad_price"))
        with pytest.raises(ValueError, match="price_col.*must be numeric"):
            aggregate_time(bad_data2, "date", "bad_price", None, "product", "arithmetic", "1mo")

    def test_zero_and_negative_prices_geometric(self):
        """Test geometric mean handling of zero and negative prices."""
        data_with_zeros = pl.DataFrame({
            "date": ["2023-01-01", "2023-01-02", "2023-01-03"],
            "product": ["A", "A", "A"],
            "price": [100.0, 0.0, -50.0],  # Include zero and negative
            "quantity": [10, 12, 15]
        })

        result = aggregate_time(
            data_with_zeros,
            date_col="date",
            price_col="price",
            id_col="product",
            agg_type="geometric",
            freq="1mo"
        )

        # Should return null for geometric mean with invalid values
        assert result["aggregated_price"][0] is None

    def test_harmonic_mean_with_zeros(self):
        """Test harmonic mean handling of zero prices."""
        data_with_zeros = pl.DataFrame({
            "date": ["2023-01-01", "2023-01-02"],
            "product": ["A", "A"],
            "price": [100.0, 0.0],
            "quantity": [10, 12]
        })

        result = aggregate_time(
            data_with_zeros,
            date_col="date",
            price_col="price",
            id_col="product",
            agg_type="harmonic",
            freq="1mo"
        )

        # Should return null for harmonic mean with zero
        assert result["aggregated_price"][0] is None

    def test_weighted_geometric_complex(self, sample_data):
        """Test weighted geometric mean with real data."""
        result = aggregate_time(
            sample_data,
            date_col="date",
            price_col="price",
            quantity_col="quantity",
            id_col="product_id",
            agg_type="weighted_geometric",
            freq="1mo"
        )

        # Should not have null values for valid data
        valid_prices = result.filter(pl.col("aggregated_price").is_not_null())
        assert len(valid_prices) > 0

    def test_category_filtering(self, sample_data):
        """Test aggregation works with category filtering."""
        electronics_data = sample_data.filter(pl.col("category") == "electronics")

        result = aggregate_time(
            electronics_data,
            date_col="date",
            price_col="price",
            quantity_col="quantity",
            id_col="product_id",
            agg_type="arithmetic",
            freq="1mo"
        )

        # Should only have electronics products
        unique_products = result["product_id"].unique()
        electronics_products = electronics_data["product_id"].unique()
        assert all(prod in electronics_products for prod in unique_products)


class TestStandardizeColumns:
    """Test suite for standardize_columns function."""

    def test_basic_standardization_without_quantity(self):
        """Test basic column standardization without quantity."""
        df = pl.DataFrame({
            "date": ["2023-01-01", "2023-01-02"],
            "price": [100.0, 110.0],
            "product_id": ["A", "B"]
        })

        result = standardize_columns(df)

        # Check columns
        assert result.columns == ["date", "price", "product_id"]
        # Check date type
        assert result.schema["date"] == pl.Date
        # Check date values
        assert result["date"].to_list() == [date(2023, 1, 1), date(2023, 1, 2)]
        # Check other columns unchanged
        assert result["price"][0] == 100.0
        assert result["product_id"][0] == "A"

    def test_standardization_with_quantity(self):
        """Test standardization with quantity column."""
        df = pl.DataFrame({
            "date": ["2023-01-01", "2023-01-02"],
            "price": [100.0, 110.0],
            "product_id": ["A", "B"],
            "quantity": [10, 0]  # One zero quantity
        })

        result = standardize_columns(df, quantity_col="quantity")

        # Should have filtered out quantity == 0
        assert len(result) == 1
        assert result["quantity"].to_list() == [10]
        assert result["product_id"].to_list() == ["A"]

    def test_custom_column_names(self):
        """Test with custom column names."""
        df = pl.DataFrame({
            "dt": ["2023-01-01"],
            "prc": [100.0],
            "prod": ["A"],
            "qty": [5]
        })

        result = standardize_columns(
            df,
            date_col="dt",
            price_col="prc",
            id_col="prod",
            quantity_col="qty"
        )

        assert result.columns == ["date", "price", "product_id", "quantity"]
        assert result["date"].to_list() == [date(2023, 1, 1)]
        assert result["price"].to_list() == [100.0]
        assert result["product_id"].to_list() == ["A"]
        assert result["quantity"].to_list() == [5]

    def test_custom_date_format(self):
        """Test custom date format."""
        df = pl.DataFrame({
            "date": ["01/01/2023"],
            "price": [100.0],
            "product_id": ["A"]
        })

        result = standardize_columns(df, date_format="%m/%d/%Y")

        assert result["date"].to_list() == [date(2023, 1, 1)]

    def test_missing_columns_error(self):
        """Test error when required columns are missing."""
        df = pl.DataFrame({
            "date": ["2023-01-01"],
            "price": [100.0]
            # Missing product_id
        })

        with pytest.raises(ValueError, match="Missing required columns"):
            standardize_columns(df)

    def test_invalid_price_type_error(self):
        """Test error when price is not numeric."""
        df = pl.DataFrame({
            "date": ["2023-01-01"],
            "price": ["not_numeric"],
            "product_id": ["A"]
        })

        with pytest.raises(ValueError, match="Price column must be numeric"):
            standardize_columns(df)

    def test_invalid_quantity_type_error(self):
        """Test error when quantity is not numeric."""
        df = pl.DataFrame({
            "date": ["2023-01-01"],
            "price": [100.0],
            "product_id": ["A"],
            "quantity": ["not_numeric"]
        })

        with pytest.raises(ValueError, match="Quantity column must be numeric"):
            standardize_columns(df, quantity_col="quantity")

    def test_quantity_filtering_edge_cases(self):
        """Test quantity filtering with various zero values."""
        df = pl.DataFrame({
            "date": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"],
            "price": [100.0, 110.0, 120.0, 130.0],
            "product_id": ["A", "B", "C", "D"],
            "quantity": [0.0, 0.0, 1.0, -5.0]  # Various zero/negative values
        })

        result = standardize_columns(df, quantity_col="quantity")

        # Should filter out rows where quantity == 0 (including 0.0)
        # Note: negative quantities are kept as per current logic
        assert len(result) == 2
        assert result["product_id"].to_list() == ["C", "D"]
        assert result["quantity"].to_list() == [1.0, -5.0]

    def test_column_selection_only_specified(self):
        """Test that only specified columns are selected."""
        df = pl.DataFrame({
            "date": ["2023-01-01"],
            "price": [100.0],
            "product_id": ["A"],
            "extra_col": ["should_be_removed"]
        })

        result = standardize_columns(df)

        assert "extra_col" not in result.columns
        assert result.columns == ["date", "price", "product_id"]


class TestGetSummary:
    """Test suite for get_summary function."""

    def test_basic_summary_with_quantity(self, small_sample_data):
        """Test basic summary with quantity column."""
        # Standardize the data first
        df = standardize_columns(
            small_sample_data,
            date_col="date",
            price_col="price",
            id_col="product",
            quantity_col="quantity"
        )

        result = get_summary(df)

        assert result == {
            "n_products": 1,  # All same product "A"
            "start_date": date(2023, 1, 1),
            "end_date": date(2023, 2, 15),
            "quantity": True
        }

    def test_summary_without_quantity(self):
        """Test summary without quantity column."""
        df = pl.DataFrame({
            "date": [date(2023, 1, 1), date(2023, 1, 2)],
            "price": [100.0, 110.0],
            "product_id": ["A", "B"]
        })

        result = get_summary(df)

        assert result == {
            "n_products": 2,
            "start_date": date(2023, 1, 1),
            "end_date": date(2023, 1, 2),
            "quantity": False
        }

    def test_summary_with_null_quantities(self):
        """Test summary with quantity column but all null values."""
        df = pl.DataFrame({
            "date": [date(2023, 1, 1), date(2023, 1, 2)],
            "price": [100.0, 110.0],
            "product_id": ["A", "B"],
            "quantity": [None, None]
        })

        result = get_summary(df)

        assert result == {
            "n_products": 2,
            "start_date": date(2023, 1, 1),
            "end_date": date(2023, 1, 2),
            "quantity": False  # All null, so False
        }

    def test_summary_with_mixed_quantities(self):
        """Test summary with some non-null quantities."""
        df = pl.DataFrame({
            "date": [date(2023, 1, 1), date(2023, 1, 2)],
            "price": [100.0, 110.0],
            "product_id": ["A", "B"],
            "quantity": [10, None]
        })

        result = get_summary(df)

        assert result == {
            "n_products": 2,
            "start_date": date(2023, 1, 1),
            "end_date": date(2023, 1, 2),
            "quantity": True  # Has at least one non-null
        }

    def test_summary_multiple_products(self, sample_data):
        """Test summary with multiple products using sample data."""
        # Standardize the sample data
        df = standardize_columns(
            sample_data,
            date_col="date",
            price_col="price",
            id_col="product_id",
            quantity_col="quantity"
        )

        result = get_summary(df)

        # Check that we have multiple products
        assert result["n_products"] > 1
        # Check date range
        assert result["start_date"] <= result["end_date"]
        # Check quantity present
        assert result["quantity"] is True


class TestRemoveUnbalanced:
    """Test suite for remove_unbalanced function."""

    def test_remove_unbalanced_products(self):
        """Test removing products not present in all periods."""
        df = pl.DataFrame({
            "product_id": ["A", "A", "A", "B", "B", "C"],
            "period": [
                date(2023, 1, 1), date(2023, 2, 1), date(2023, 3, 1),
                date(2023, 1, 1), date(2023, 2, 1),
                date(2023, 1, 1)
            ],
            "aggregated_price": [100, 110, 120, 200, 210, 300]
        })

        result = remove_unbalanced(df)

        # Should only keep product A (present in all 3 periods)
        assert len(result) == 3
        assert result["product_id"].unique().to_list() == ["A"]
        assert result["aggregated_price"].to_list() == [100, 110, 120]

    def test_all_products_balanced(self):
        """Test when all products are present in all periods."""
        df = pl.DataFrame({
            "product_id": ["A", "A", "B", "B"],
            "period": [
                date(2023, 1, 1), date(2023, 2, 1),
                date(2023, 1, 1), date(2023, 2, 1)
            ],
            "aggregated_price": [100, 110, 200, 210]
        })

        result = remove_unbalanced(df)

        # Should keep all products
        assert len(result) == 4
        assert set(result["product_id"].unique().to_list()) == {"A", "B"}

    def test_all_products_unbalanced(self):
        """Test when all products are missing from some periods."""
        df = pl.DataFrame({
            "product_id": ["A", "B", "C"],
            "period": [
                date(2023, 1, 1),
                date(2023, 1, 1),
                date(2023, 2, 1)
            ],
            "aggregated_price": [100, 200, 300]
        })

        result = remove_unbalanced(df)

        # Should remove all products (none present in both periods)
        assert len(result) == 0

    def test_single_period_all_balanced(self):
        """Test with single period (all products balanced by definition)."""
        df = pl.DataFrame({
            "product_id": ["A", "B", "C"],
            "period": [
                date(2023, 1, 1),
                date(2023, 1, 1),
                date(2023, 1, 1)
            ],
            "aggregated_price": [100, 200, 300]
        })

        result = remove_unbalanced(df)

        # Should keep all products
        assert len(result) == 3
        assert set(result["product_id"].unique().to_list()) == {"A", "B", "C"}

    def test_preserve_columns(self):
        """Test that all columns are preserved."""
        df = pl.DataFrame({
            "product_id": ["A", "A", "B"],
            "period": [
                date(2023, 1, 1), date(2023, 2, 1),
                date(2023, 1, 1)
            ],
            "aggregated_price": [100, 110, 200],
            "aggregated_quantity": [10, 12, 15],
            "extra_col": ["x", "y", "z"]
        })

        result = remove_unbalanced(df)

        # Should preserve all columns
        assert result.columns == df.columns
        # Should only keep product A
        assert len(result) == 2
        assert result["product_id"].unique().to_list() == ["A"]

    def test_missing_product_id_column_error(self):
        """Test error when product_id column is missing."""
        df = pl.DataFrame({
            "period": [date(2023, 1, 1)],
            "aggregated_price": [100]
        })

        with pytest.raises(ValueError, match="Missing required columns"):
            remove_unbalanced(df)

    def test_missing_period_column_error(self):
        """Test error when period column is missing."""
        df = pl.DataFrame({
            "product_id": ["A"],
            "aggregated_price": [100]
        })

        with pytest.raises(ValueError, match="Missing required columns"):
            remove_unbalanced(df)


@pytest.fixture
def unbalanced_aggregated_data():
    """Create unbalanced aggregated data for imputation testing."""
    return pl.DataFrame({
        "product_id": ["A", "A", "B", "C", "C"],
        "period": [
            date(2023, 1, 1), date(2023, 2, 1),  # A has both periods
            date(2023, 1, 1),  # B missing period 2
            date(2023, 1, 1), date(2023, 2, 1)   # C has both periods
        ],
        "aggregated_price": [100.0, 110.0, 200.0, 300.0, 320.0],
        "aggregated_quantity": [10, 12, 15, 20, 22]
    })


class TestCarryForwardImputation:
    """Test suite for carry_forward_imputation function."""

    def test_basic_forward_fill_creates_balanced_panel(self, unbalanced_aggregated_data):
        """Test that forward imputation creates balanced panel and fills missing values."""
        result = carry_forward_imputation(
            unbalanced_aggregated_data,
            value_cols=["aggregated_price", "aggregated_quantity"]
        )

        # Should have 6 rows: 3 products × 2 periods
        assert len(result) == 6
        assert result.columns == ["product_id", "period", "aggregated_price", "aggregated_quantity"]

        # Check product A (already complete)
        a_data = result.filter(pl.col("product_id") == "A").sort("period")
        assert a_data["aggregated_price"].to_list() == [100.0, 110.0]
        assert a_data["aggregated_quantity"].to_list() == [10, 12]

        # Check product B (missing period 2, should be forward filled)
        b_data = result.filter(pl.col("product_id") == "B").sort("period")
        assert b_data["aggregated_price"].to_list() == [200.0, 200.0]  # Forward filled
        assert b_data["aggregated_quantity"].to_list() == [15, 15]  # Forward filled

        # Check product C (already complete)
        c_data = result.filter(pl.col("product_id") == "C").sort("period")
        assert c_data["aggregated_price"].to_list() == [300.0, 320.0]
        assert c_data["aggregated_quantity"].to_list() == [20, 22]

    def test_forward_fill_only_price(self, unbalanced_aggregated_data):
        """Test forward imputation on price only."""
        result = carry_forward_imputation(
            unbalanced_aggregated_data,
            value_cols=["aggregated_price"]
        )

        # Quantity should remain as is (with nulls for missing periods)
        b_data = result.filter(pl.col("product_id") == "B").sort("period")
        assert b_data["aggregated_price"].to_list() == [200.0, 200.0]
        assert b_data["aggregated_quantity"].to_list() == [15, None]  # Not filled

    def test_forward_fill_no_missing_periods(self):
        """Test forward fill when data is already balanced."""
        balanced_data = pl.DataFrame({
            "product_id": ["A", "A", "B", "B"],
            "period": [
                date(2023, 1, 1), date(2023, 2, 1),
                date(2023, 1, 1), date(2023, 2, 1)
            ],
            "aggregated_price": [100.0, 110.0, 200.0, 210.0]
        })

        result = carry_forward_imputation(balanced_data, ["aggregated_price"])

        # Should remain unchanged (sort to ensure consistent order)
        result_sorted = result.sort(["product_id", "period"])
        assert len(result_sorted) == 4
        assert result_sorted["aggregated_price"].to_list() == [100.0, 110.0, 200.0, 210.0]

    def test_forward_fill_custom_columns(self, unbalanced_aggregated_data):
        """Test forward imputation with custom column names."""
        custom_data = unbalanced_aggregated_data.rename({
            "product_id": "prod",
            "period": "time",
            "aggregated_price": "price"
        })

        result = carry_forward_imputation(
            custom_data,
            value_cols=["price"],
            id_col="prod",
            time_col="time"
        )

        # Should have created balanced panel
        assert len(result) == 6
        b_data = result.filter(pl.col("prod") == "B").sort("time")
        assert b_data["price"].to_list() == [200.0, 200.0]

    def test_forward_fill_missing_required_columns(self):
        """Test error when required columns are missing."""
        df = pl.DataFrame({
            "product_id": ["A"],
            "aggregated_price": [100.0]
            # Missing period
        })

        with pytest.raises(ValueError, match="Missing required columns"):
            carry_forward_imputation(df, ["aggregated_price"])

    def test_forward_fill_leading_nulls_remain(self):
        """Test that leading nulls (no previous value) remain null."""
        data_with_leading_nulls = pl.DataFrame({
            "product_id": ["A", "A"],
            "period": [date(2023, 1, 1), date(2023, 2, 1)],
            "aggregated_price": [None, 110.0]  # Leading null
        })

        result = carry_forward_imputation(data_with_leading_nulls, ["aggregated_price"])

        # Leading null should remain null, second value filled
        a_data = result.filter(pl.col("product_id") == "A").sort("period")
        assert a_data["aggregated_price"].to_list() == [None, 110.0]


class TestCarryBackwardImputation:
    """Test suite for carry_backward_imputation function."""

    def test_basic_backward_fill_creates_balanced_panel(self, unbalanced_aggregated_data):
        """Test that backward imputation creates balanced panel and fills missing values."""
        result = carry_backward_imputation(
            unbalanced_aggregated_data,
            value_cols=["aggregated_price", "aggregated_quantity"]
        )

        # Should have 6 rows: 3 products × 2 periods
        assert len(result) == 6

        # Check product A (already complete)
        a_data = result.filter(pl.col("product_id") == "A").sort("period")
        assert a_data["aggregated_price"].to_list() == [100.0, 110.0]

        # Check product B (missing period 2, no future data so remains null)
        b_data = result.filter(pl.col("product_id") == "B").sort("period")
        assert b_data["aggregated_price"].to_list() == [200.0, None]  # Not filled (no future value)
        assert b_data["aggregated_quantity"].to_list() == [15, None]  # Not filled (no future value)

        # Check product C (already complete)
        c_data = result.filter(pl.col("product_id") == "C").sort("period")
        assert c_data["aggregated_price"].to_list() == [300.0, 320.0]

    def test_backward_fill_only_quantity(self, unbalanced_aggregated_data):
        """Test backward imputation on quantity only."""
        result = carry_backward_imputation(
            unbalanced_aggregated_data,
            value_cols=["aggregated_quantity"]
        )

        b_data = result.filter(pl.col("product_id") == "B").sort("period")
        assert b_data["aggregated_quantity"].to_list() == [15, None]  # Not filled (no future value)
        assert b_data["aggregated_price"].to_list() == [200.0, None]  # Not filled

    def test_backward_fill_no_missing_periods(self):
        """Test backward fill when data is already balanced."""
        balanced_data = pl.DataFrame({
            "product_id": ["A", "A", "B", "B"],
            "period": [
                date(2023, 1, 1), date(2023, 2, 1),
                date(2023, 1, 1), date(2023, 2, 1)
            ],
            "aggregated_price": [100.0, 110.0, 200.0, 210.0]
        })

        result = carry_backward_imputation(balanced_data, ["aggregated_price"])

        # Should remain unchanged (sort to ensure consistent order)
        result_sorted = result.sort(["product_id", "period"])
        assert len(result_sorted) == 4
        assert result_sorted["aggregated_price"].to_list() == [100.0, 110.0, 200.0, 210.0]

    def test_backward_fill_custom_columns(self, unbalanced_aggregated_data):
        """Test backward imputation with custom column names."""
        custom_data = unbalanced_aggregated_data.rename({
            "product_id": "prod",
            "period": "time",
            "aggregated_price": "price"
        })

        result = carry_backward_imputation(
            custom_data,
            value_cols=["price"],
            id_col="prod",
            time_col="time"
        )

        # Should have created balanced panel
        assert len(result) == 6
        b_data = result.filter(pl.col("prod") == "B").sort("time")
        assert b_data["price"].to_list() == [200.0, None]  # Not filled (no future value)

    def test_backward_fill_missing_required_columns(self):
        """Test error when required columns are missing."""
        df = pl.DataFrame({
            "product_id": ["A"],
            "aggregated_price": [100.0]
            # Missing period
        })

        with pytest.raises(ValueError, match="Missing required columns"):
            carry_backward_imputation(df, ["aggregated_price"])

    def test_backward_fill_trailing_nulls_remain(self):
        """Test that trailing nulls (no future value) remain null."""
        data_with_trailing_nulls = pl.DataFrame({
            "product_id": ["A", "A"],
            "period": [date(2023, 1, 1), date(2023, 2, 1)],
            "aggregated_price": [100.0, None]  # Trailing null
        })

        result = carry_backward_imputation(data_with_trailing_nulls, ["aggregated_price"])

        # Trailing null should remain null
        a_data = result.filter(pl.col("product_id") == "A").sort("period")
        assert a_data["aggregated_price"].to_list() == [100.0, None]
