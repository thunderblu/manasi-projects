import pytest
import pandas as pd
import os
import numpy as np

class TestDataValidation:
    @pytest.fixture
    def raw_data_dir(self):
        return 'data/raw'

    @pytest.fixture
    def processed_data_dir(self):
        return 'data/processed'

    def test_raw_data_files_exist(self, raw_data_dir):
        """Test if raw data files exist"""
        assert os.path.exists(raw_data_dir)
        csv_files = [f for f in os.listdir(raw_data_dir) if f.endswith('.csv')]
        assert len(csv_files) > 0

    def test_raw_data_structure(self, raw_data_dir):
        """Test structure of raw data files"""
        for file in os.listdir(raw_data_dir):
            if file.endswith('.csv'):
                df = pd.read_csv(os.path.join(raw_data_dir, file))
                required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                assert all(col in df.columns for col in required_columns)
                assert not df.empty

    def test_processed_data_files_exist(self, processed_data_dir):
        """Test if processed data files exist"""
        assert os.path.exists(processed_data_dir)
        csv_files = [f for f in os.listdir(processed_data_dir) if f.endswith('.csv')]
        assert len(csv_files) > 0

    def test_processed_data_structure(self, processed_data_dir):
        """Test structure of processed data files"""
        for file in os.listdir(processed_data_dir):
            if file.endswith('.csv'):
                df = pd.read_csv(os.path.join(processed_data_dir, file))
                required_features = ['MA5', 'MA20', 'RSI']
                assert all(col in df.columns for col in required_features)
                assert not df.empty

    def test_data_consistency(self, raw_data_dir, processed_data_dir):
        """Test consistency between raw and processed data"""
        raw_files = [f for f in os.listdir(raw_data_dir) if f.endswith('.csv')]
        processed_files = [f for f in os.listdir(processed_data_dir) if f.endswith('.csv')]
        
        for raw_file in raw_files:
            processed_file = f"{raw_file}"
            assert processed_file in processed_files

    def test_data_quality(self, processed_data_dir):
        """Test quality of processed data"""
        TOLERANCE = 1e-10  # Small tolerance for floating-point comparison
        
        # Define which columns should be scaled to [-1, 1]
        SCALED_FEATURES = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'MA5', 'MA20', 'RSI', 'Daily_Return', 
            'Volatility'
        ]
        
        # Define features that can have wider ranges
        WIDER_RANGE_FEATURES = [
            'MACD', 'Signal_Line', 'Momentum'  # These can have larger ranges
        ]
        
        for file in os.listdir(processed_data_dir):
            if file.endswith('.csv'):
                df = pd.read_csv(os.path.join(processed_data_dir, file))
                
                # Check for missing values
                assert df.isnull().sum().sum() == 0, f"Found missing values in {file}"
                
                # Check for infinite values
                assert not np.isinf(df.select_dtypes(include=np.number).values).any(), \
                    f"Found infinite values in {file}"
                
                # Check scaled features are between -1 and 1
                scaled_columns = [col for col in SCALED_FEATURES if col in df.columns]
                for col in scaled_columns:
                    min_val = df[col].min()
                    max_val = df[col].max()
                    assert min_val >= -1 - TOLERANCE, \
                        f"Column {col} has values below -1: {min_val}"
                    assert max_val <= 1 + TOLERANCE, \
                        f"Column {col} has values above 1: {max_val}"
                
                # Check wider range features are finite
                wider_range_columns = [col for col in WIDER_RANGE_FEATURES if col in df.columns]
                for col in wider_range_columns:
                    assert df[col].notna().all(), f"Found NaN values in column {col}"
                    assert np.isfinite(df[col]).all(), f"Found infinite values in column {col}"
                
                # For remaining non-scaled features, just check for finite values
                other_columns = [col for col in df.select_dtypes(include=[np.number]).columns 
                               if col not in scaled_columns and col not in wider_range_columns]
                for col in other_columns:
                    assert df[col].notna().all(), f"Found NaN values in column {col}"
                    assert np.isfinite(df[col]).all(), f"Found infinite values in column {col}"