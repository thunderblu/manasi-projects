import pandas as pd
from typing import Dict
import logging

class DataSlicer:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def create_slices(self, X: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Create meaningful slices of the dataset"""
        slices = {}
        
        try:
            # Get available columns
            columns = X.columns.tolist()
            
            # Price-based slices (using target variable)
            if 'target' in columns:
                price_quartiles = pd.qcut(X['target'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
                for quartile in ['Q1', 'Q2', 'Q3', 'Q4']:
                    slices[f'price_{quartile}'] = X[price_quartiles == quartile]
            
            # Volatility slices
            if 'volatility' in columns:
                vol_quartiles = pd.qcut(X['volatility'], q=4, labels=['Low', 'Medium-Low', 'Medium-High', 'High'])
                for quartile in vol_quartiles.unique():
                    slices[f'volatility_{quartile}'] = X[vol_quartiles == quartile]
            
            # Volume slices
            volume_cols = [col for col in columns if 'volume' in col.lower()]
            if volume_cols:
                volume_col = volume_cols[0]  # Use the first volume-related column
                volume_quartiles = pd.qcut(X[volume_col], q=4, labels=['Low', 'Medium-Low', 'Medium-High', 'High'])
                for quartile in volume_quartiles.unique():
                    slices[f'volume_{quartile}'] = X[volume_quartiles == quartile]
            
            # Market trend slices (if returns are available)
            returns_cols = [col for col in columns if 'return' in col.lower()]
            if returns_cols:
                returns_col = returns_cols[0]
                slices['bull_market'] = X[X[returns_col] > 0]
                slices['bear_market'] = X[X[returns_col] < 0]
            
            if not slices:
                # If no specific slices could be created, create generic quartile slices
                # using the first numerical column
                num_cols = X.select_dtypes(include=['float64', 'int64']).columns
                if len(num_cols) > 0:
                    generic_quartiles = pd.qcut(X[num_cols[0]], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
                    for quartile in ['Q1', 'Q2', 'Q3', 'Q4']:
                        slices[f'generic_{quartile}'] = X[generic_quartiles == quartile]
            
            self.logger.info(f"Created {len(slices)} data slices successfully")
            return slices
            
        except Exception as e:
            self.logger.error(f"Error in creating slices: {str(e)}")
            raise