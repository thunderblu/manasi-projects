import pytest
import pandas as pd
import os
import sys
from pathlib import Path

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_acquisition import StockDataAcquisition

class TestDataAcquisition:
    @pytest.fixture
    def data_acquirer(self):
        """Fixture to create data acquirer instance"""
        return StockDataAcquisition(output_dir='data/raw')

    def test_fetch_stock_data_valid_symbol(self, data_acquirer):
        """Test fetching data for a valid stock symbol"""
        df = data_acquirer.fetch_stock_data('AAPL', '2023-01-01', '2024-01-01')
        
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert all(col in df.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume'])

    def test_fetch_stock_data_invalid_symbol(self, data_acquirer):
        """Test fetching data for an invalid stock symbol"""
        with pytest.raises(Exception):
            data_acquirer.fetch_stock_data('INVALID_SYMBOL')

    def test_fetch_stock_data_invalid_dates(self, data_acquirer):
        """Test fetching data with invalid dates"""
        with pytest.raises(Exception):
            data_acquirer.fetch_stock_data('AAPL', '2025-01-01', '2024-01-01')

    def test_output_file_creation(self, data_acquirer):
        """Test if output file is created"""
        symbol = 'AAPL'
        start_date = '2023-01-01'
        end_date = '2024-01-01'
        
        data_acquirer.fetch_stock_data(symbol, start_date, end_date)
        expected_file = f"{symbol}.csv"
        assert os.path.exists(os.path.join('data/raw', expected_file)) 