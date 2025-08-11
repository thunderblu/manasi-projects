import pytest
import pandas as pd
import numpy as np
from src.data_preprocessing import StockDataPreprocessor

class TestDataPreprocessing:
    @pytest.fixture
    def preprocessor(self):
        """Fixture to create preprocessor instance"""
        return StockDataPreprocessor()

    @pytest.fixture
    def sample_data(self):
        """Fixture to create sample stock data"""
        dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
        data = {
            'Open': np.random.uniform(100, 200, len(dates)),
            'High': np.random.uniform(150, 250, len(dates)),
            'Low': np.random.uniform(50, 150, len(dates)),
            'Close': np.random.uniform(100, 200, len(dates)),
            'Volume': np.random.uniform(1000000, 5000000, len(dates))
        }
        df = pd.DataFrame(data, index=dates)
        return df

    def test_clean_data(self, preprocessor, sample_data):
        """Test data cleaning"""
        # Insert some NaN values
        sample_data.iloc[5:10, 0] = np.nan
        
        cleaned_data = preprocessor.clean_data(sample_data)
        
        assert cleaned_data.isnull().sum().sum() == 0
        assert isinstance(cleaned_data, pd.DataFrame)
        assert cleaned_data.shape == sample_data.shape

    def test_engineer_features(self, preprocessor, sample_data):
        """Test feature engineering"""
        featured_data = preprocessor.engineer_features(sample_data)
        
        expected_features = ['MA5', 'MA20', 'RSI', 'Daily_Return', 'Volatility']
        
        assert all(feature in featured_data.columns for feature in expected_features)
        assert featured_data.shape[0] > 0
        assert featured_data.shape[1] > sample_data.shape[1]

    def test_scale_features(self, preprocessor, sample_data):
        """Test feature scaling"""
        columns_to_scale = ['Open', 'High', 'Low', 'Close', 'Volume']
        scaled_data, scaler = preprocessor.scale_features(sample_data, columns_to_scale)
        
        # Check if values are scaled between 0 and 1 with tolerance
        tolerance = 1e-10  # Define a small tolerance for floating-point comparison
        for col in columns_to_scale:
            assert scaled_data[col].min() >= -tolerance  # Allow small negative values
            assert scaled_data[col].max() <= 1 + tolerance  # Allow slightly over 1

    def test_prepare_sequences(self, preprocessor, sample_data):
        """Test sequence preparation"""
        sequence_length = 10
        X, y = preprocessor.prepare_sequences(sample_data, sequence_length)
        
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.shape[1] == sequence_length
        assert X.shape[2] == sample_data.shape[1]
        assert len(y) == len(X)