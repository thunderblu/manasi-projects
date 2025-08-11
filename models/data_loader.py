import os
import pandas as pd
import numpy as np
from typing import Tuple, Optional
from models.utils.logger import setup_logger
from sklearn.impute import SimpleImputer

class StockDataLoader:
    def __init__(self, data_dir: str = "data/mitigated"):
        """
        Initialize the data loader
        Args:
            data_dir: Directory containing the mitigated data
        """
        self.data_dir = data_dir
        self.logger = setup_logger("StockDataLoader")
        self.imputer = SimpleImputer(strategy='mean')

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the data by handling missing values and outliers
        """
        try:
            # Handle missing values
            self.logger.info("Preprocessing data...")
            
            # Check for missing values
            missing_values = df.isnull().sum()
            if missing_values.any():
                self.logger.info(f"Found missing values:\n{missing_values[missing_values > 0]}")
                
                # Fill missing values using SimpleImputer
                df_filled = pd.DataFrame(
                    self.imputer.fit_transform(df),
                    columns=df.columns,
                    index=df.index
                )
                
                self.logger.info("Missing values have been imputed")
                return df_filled
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error preprocessing data: {str(e)}")
            raise

    def load_data(self, symbols: list = ["AAPL", "GOOGL", "MSFT"]) -> dict:
        """
        Load and preprocess data for specified symbols
        Args:
            symbols: List of stock symbols to load
        Returns:
            Dictionary containing DataFrames for each symbol
        """
        data_dict = {}
        try:
            for symbol in symbols:
                file_path = os.path.join(self.data_dir, f"{symbol}_mitigated.csv")
                
                # Load data using pandas
                df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
                
                # Convert timezone-aware dates to timezone-naive
                df.index = df.index.tz_localize(None)
                
                # Preprocess the data
                df = self.preprocess_data(df)
                
                data_dict[symbol] = df
                self.logger.info(f"Successfully loaded and preprocessed data for {symbol}")
                
            return data_dict
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise

    def split_data(self, 
                   df: pd.DataFrame, 
                   train_ratio: float = 0.7,
                   val_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets
        Args:
            df: Input DataFrame
            train_ratio: Ratio of training data
            val_ratio: Ratio of validation data
        Returns:
            Tuple of train, validation, and test DataFrames
        """
        try:
            # Calculate split points
            n = len(df)
            train_size = int(n * train_ratio)
            val_size = int(n * val_ratio)
            
            # Split the data
            train_data = df[:train_size]
            val_data = df[train_size:train_size + val_size]
            test_data = df[train_size + val_size:]
            
            self.logger.info(f"Data split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
            
            return train_data, val_data, test_data
            
        except Exception as e:
            self.logger.error(f"Error splitting data: {str(e)}")
            raise

    def get_features_and_target(self, 
                              df: pd.DataFrame,
                              target_col: str = 'Close',
                              feature_cols: Optional[list] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Separate features and target variables
        Args:
            df: Input DataFrame
            target_col: Name of target column
            feature_cols: List of feature columns to use
        Returns:
            Tuple of features DataFrame and target Series
        """
        try:
            if feature_cols is None:
                # Use all columns except target as features
                feature_cols = [col for col in df.columns if col != target_col]
            
            X = df[feature_cols]
            y = df[target_col]
            
            self.logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"Error separating features and target: {str(e)}")
            raise 