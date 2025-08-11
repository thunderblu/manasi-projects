import pandas as pd
import numpy as np
from scipy import stats
from typing import List, Dict, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class AnomalyDetector:
    def __init__(self, zscore_threshold: float = 3.0, missing_threshold: float = 0.1):
        """
        Initialize anomaly detector
        Args:
            zscore_threshold: Z-score threshold for outlier detection
            missing_threshold: Maximum acceptable percentage of missing values
        """
        self.zscore_threshold = zscore_threshold
        self.missing_threshold = missing_threshold
        
    def detect_outliers(self, df: pd.DataFrame) -> Dict[str, List[Dict]]:
        """Detect outliers using Z-score method"""
        outliers = {}
        
        # Add specific columns to check for outliers
        columns_to_check = [
            'Returns', 'Open_Close_Ret', 'High_Low_Range',
            'MA_Ratio', 'RSI', 'Market_Mood',
            'GSPC_Return', 'GDAXI_Return', 'GC_Return', 'NSEI_Return'
        ]
        
        for column in columns_to_check:
            if column in df.columns:
                # Handle cases where column might be entirely NaN
                if df[column].dropna().empty:
                    continue
                
                try:
                    z_scores = np.abs(stats.zscore(df[column].dropna()))
                    outlier_indices = np.where(z_scores > self.zscore_threshold)[0]
                    
                    if len(outlier_indices) > 0:
                        outliers[column] = [{
                            'index': idx,
                            'value': float(df[column].iloc[idx]),  # Convert to float for JSON serialization
                            'zscore': float(z_scores.iloc[idx]),
                            'timestamp': df.index[idx].strftime('%Y-%m-%d %H:%M:%S') if isinstance(df.index, pd.DatetimeIndex) else str(idx)
                        } for idx in outlier_indices]
                except Exception as e:
                    logger.warning(f"Could not process column {column} for outliers: {str(e)}")
                    continue
                
        return outliers
    
    def detect_pattern_anomalies(self, df: pd.DataFrame) -> List[Dict]:
        """Detect anomalies in price patterns"""
        anomalies = []
        
        try:
            # Detect sudden price changes
            daily_returns = df['Close'].pct_change()
            volatility = daily_returns.rolling(window=20).std()
            
            # Detect price jumps > 3 standard deviations
            large_moves = np.abs(daily_returns) > (3 * volatility)
            large_moves = large_moves[large_moves].index  # Get only True indices
            
            if len(large_moves) > 0:
                anomalies.extend([{
                    'type': 'price_jump',
                    'timestamp': idx.strftime('%Y-%m-%d %H:%M:%S'),
                    'value': float(daily_returns[idx]),
                    'threshold': float(3 * volatility[idx])
                } for idx in large_moves])
            
            # Detect volume spikes
            volume_mean = df['Volume'].rolling(window=20).mean()
            volume_std = df['Volume'].rolling(window=20).std()
            volume_spikes = df['Volume'] > (volume_mean + 3 * volume_std)
            volume_spikes = volume_spikes[volume_spikes].index
            
            if len(volume_spikes) > 0:
                anomalies.extend([{
                    'type': 'volume_spike',
                    'timestamp': idx.strftime('%Y-%m-%d %H:%M:%S'),
                    'value': float(df['Volume'][idx]),
                    'threshold': float(volume_mean[idx] + 3 * volume_std[idx])
                } for idx in volume_spikes])
                
        except Exception as e:
            logger.warning(f"Error in pattern anomaly detection: {str(e)}")
            
        return anomalies
    
    def check_data_quality(self, df: pd.DataFrame) -> List[str]:
        """Check for data quality issues"""
        issues = []
        
        try:
            # Check for missing values
            missing_pct = df.isnull().sum() / len(df)
            for column, pct in missing_pct.items():
                if pct > self.missing_threshold:
                    issues.append(f"High missing values in {column}: {pct:.2%}")
            
            # Check for zero or negative prices
            price_columns = ['Open', 'High', 'Low', 'Close']
            for col in price_columns:
                if col in df.columns:
                    if (df[col] <= 0).any():
                        issues.append(f"Invalid prices detected in {col}")
            
            # Check for price inconsistencies
            if all(col in df.columns for col in ['High', 'Low', 'Close']):
                invalid_prices = (df['Low'] > df['High']) | (df['Close'] > df['High']) | (df['Close'] < df['Low'])
                if invalid_prices.any():
                    issues.append(f"Price inconsistencies detected on {len(invalid_prices[invalid_prices])} rows")
                    
        except Exception as e:
            logger.warning(f"Error in data quality check: {str(e)}")
            issues.append(f"Data quality check error: {str(e)}")
        
        return issues 
    
    def handle_outliers(self, df: pd.DataFrame, method: str = 'winsorize') -> Tuple[pd.DataFrame, Dict]:
        """
        Handle outliers using specified method
        Args:
            df: Input DataFrame
            method: 'winsorize', 'clip', 'remove', or 'impute'
        Returns:
            Cleaned DataFrame and handling report
        """
        df_cleaned = df.copy()
        handling_report = {}

        try:
            for column in df.select_dtypes(include=[np.number]).columns:
                # Detect outliers
                z_scores = np.abs(stats.zscore(df[column].dropna()))
                outlier_mask = z_scores > self.zscore_threshold
                
                if outlier_mask.any():
                    original_values = df[column][outlier_mask]
                    
                    if method == 'winsorize':
                        # Winsorize: Cap at percentiles
                        lower, upper = np.percentile(df[column], [1, 99])
                        df_cleaned[column] = df_cleaned[column].clip(lower, upper)
                        
                    elif method == 'clip':
                        # Clip at z-score threshold
                        mean, std = df[column].mean(), df[column].std()
                        lower = mean - (self.zscore_threshold * std)
                        upper = mean + (self.zscore_threshold * std)
                        df_cleaned[column] = df_cleaned[column].clip(lower, upper)
                        
                    elif method == 'remove':
                        # Remove rows with outliers
                        df_cleaned = df_cleaned[~outlier_mask]
                        
                    elif method == 'impute':
                        # Impute with rolling median
                        window = 5
                        rolling_median = df[column].rolling(window=window, center=True).median()
                        df_cleaned.loc[outlier_mask, column] = rolling_median[outlier_mask]
                    
                    # Convert timestamps to strings in the original_values dictionary
                    original_values_dict = {
                        str(idx): float(val) 
                        for idx, val in original_values.items()
                    }
                    
                    new_values_dict = {
                        str(idx): float(val) 
                        for idx, val in df_cleaned[column][outlier_mask].items()
                    }
                    
                    # Record handling in report
                    handling_report[column] = {
                        'outliers_found': int(sum(outlier_mask)),
                        'handling_method': method,
                        'original_values': original_values_dict,
                        'new_values': new_values_dict
                    }
            
        except Exception as e:
            logger.error(f"Error handling outliers: {str(e)}")
            
        return df_cleaned, handling_report

    def handle_pattern_anomalies(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict]]:
        """Handle pattern anomalies in price and volume data"""
        df_cleaned = df.copy()
        handling_report = []
        
        try:
            # Handle price jumps
            daily_returns = df['Close'].pct_change()
            volatility = daily_returns.rolling(window=20).std()
            large_moves = np.abs(daily_returns) > (3 * volatility)
            
            if large_moves.any():
                # Smooth out large price movements
                window = 3
                df_cleaned.loc[large_moves, 'Close'] = df['Close'].rolling(
                    window=window, center=True, min_periods=1
                ).mean()[large_moves]
                
                handling_report.append({
                    'type': 'price_jump',
                    'count': int(large_moves.sum()),  # Convert np.int64 to int
                    'handling': f'Smoothed using {window}-day moving average'
                })
            
            # Handle volume spikes
            volume_mean = df['Volume'].rolling(window=20).mean()
            volume_std = df['Volume'].rolling(window=20).std()
            volume_spikes = df['Volume'] > (volume_mean + 3 * volume_std)
            
            if volume_spikes.any():
                # Replace spike with rolling median
                df_cleaned.loc[volume_spikes, 'Volume'] = df['Volume'].rolling(
                    window=5, center=True, min_periods=1
                ).median()[volume_spikes]
                
                handling_report.append({
                    'type': 'volume_spike',
                    'count': int(volume_spikes.sum()),  # Convert np.int64 to int
                    'handling': 'Replaced with 5-day median volume'
                })
                
            # Convert any remaining numpy types in the report
            handling_report = [{
                'type': str(item['type']),
                'count': int(item['count']),
                'handling': str(item['handling'])
            } for item in handling_report]
                
        except Exception as e:
            logger.error(f"Error handling pattern anomalies: {str(e)}")
            
        return df_cleaned, handling_report

    def handle_missing_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Handle missing data with appropriate methods"""
        df_cleaned = df.copy()
        handling_report = {}
        
        try:
            for column in df.columns:
                missing_count = df[column].isnull().sum()
                if missing_count > 0:
                    if column in ['Open', 'High', 'Low', 'Close']:
                        # Fill price data using forward fill then backward fill
                        df_cleaned[column] = df_cleaned[column].ffill().bfill()
                        method = 'forward/backward fill'
                    
                    elif column == 'Volume':
                        # Fill volume with rolling median
                        window = 5
                        rolling_median = df[column].rolling(window=window, center=True).median()
                        df_cleaned[column] = df_cleaned[column].fillna(rolling_median)
                        method = f'{window}-day rolling median'
                    
                    else:
                        # For other columns, use interpolation
                        df_cleaned[column] = df_cleaned[column].interpolate(method='linear')
                        method = 'linear interpolation'
                    
                    handling_report[column] = {
                        'missing_count': missing_count,
                        'handling_method': method
                    }
            
        except Exception as e:
            logger.error(f"Error handling missing data: {str(e)}")
            
        return df_cleaned, handling_report

    def clean_data(self, df: pd.DataFrame, outlier_method: str = 'winsorize') -> Tuple[pd.DataFrame, Dict]:
        """Complete data cleaning pipeline"""
        cleaning_report = {}
        
        try:
            # Handle missing data
            df_cleaned, missing_report = self.handle_missing_data(df)
            cleaning_report['missing_data'] = missing_report
            
            # Handle outliers
            df_cleaned, outlier_report = self.handle_outliers(df_cleaned, method=outlier_method)
            cleaning_report['outliers'] = outlier_report
            
            # Handle pattern anomalies
            df_cleaned, pattern_report = self.handle_pattern_anomalies(df_cleaned)
            cleaning_report['pattern_anomalies'] = pattern_report
            
            # Log summary
            logger.info("Data cleaning completed:")
            logger.info(f"- Missing values handled in {len(missing_report)} columns")
            logger.info(f"- Outliers handled in {len(outlier_report)} columns")
            logger.info(f"- Pattern anomalies handled: {len(pattern_report)} types")
            
        except Exception as e:
            logger.error(f"Error in data cleaning pipeline: {str(e)}")
            
        return df_cleaned, cleaning_report