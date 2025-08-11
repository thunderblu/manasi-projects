import great_expectations as ge
import pandas as pd
import logging
from typing import Dict, List, Optional
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class DataValidator:
    def __init__(self, stats_dir: str = '/opt/airflow/data/stats'):
        """Initialize validator with directory for storing statistics"""
        self.stats_dir = stats_dir
        os.makedirs(stats_dir, exist_ok=True)
        
    def validate_data(self, df: pd.DataFrame, symbol: str, stage: str) -> List[str]:
        """Validate data using Great Expectations and save results as CSV"""
        try:
            # First, validate the DataFrame structure
            expected_columns = {
                'Open', 'High', 'Low', 'Close', 'Volume',
                'Returns', 'Open_Close_Ret', 'High_Low_Range',
                'MA5', 'MA20', 'MA_Ratio', 'RSI', 'Market_Mood',
                'GSPC_Return', 'GDAXI_Return', 'GC_Return', 'NSEI_Return'  # New global market returns
            }
            missing_columns = expected_columns - set(df.columns)
            if missing_columns:
                return [f"Missing required columns: {', '.join(missing_columns)}"]

            ge_df = ge.from_pandas(df)
            validation_results = []
            
            # Only validate columns that exist in the DataFrame
            for column in expected_columns.intersection(set(df.columns)):
                validation_results.extend([
                    ge_df.expect_column_values_to_not_be_null(column),
                    ge_df.expect_column_values_to_be_between(
                        column, 
                        min_value=float(0), 
                        mostly=0.99
                    ) if column != 'Volume' else ge_df.expect_column_values_to_be_between(
                        column,
                        min_value=0,
                        mostly=0.99
                    )
                ])

            # Price relationship validations only if all price columns exist
            price_columns = {'High', 'Low', 'Open', 'Close'}
            if price_columns.issubset(set(df.columns)):
                high_values = df['High'].astype(float)
                low_values = df['Low'].astype(float)
                
                validation_results.extend([
                    ge_df.expect_column_values_to_be_between(
                        'Low', 
                        min_value=float(low_values.min()), 
                        max_value=float(high_values.max()), 
                        mostly=0.99
                    ),
                    ge_df.expect_column_values_to_be_between(
                        'Close', 
                        min_value=float(low_values.min()), 
                        max_value=float(high_values.max()), 
                        mostly=0.99
                    ),
                    ge_df.expect_column_values_to_be_between(
                        'Open', 
                        min_value=float(low_values.min()), 
                        max_value=float(high_values.max()), 
                        mostly=0.99
                    )
                ])

            # Convert validation results to DataFrame
            validation_data = []
            for result in validation_results:
                validation_data.append({
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'expectation_type': result.expectation_config.expectation_type,
                    'column': result.expectation_config.kwargs.get('column'),
                    'success': result.success,
                    'unexpected_count': result.result.get('unexpected_count', 0),
                    'unexpected_percent': result.result.get('unexpected_percent', 0)
                })
            
            # Save validation results as CSV
            validation_df = pd.DataFrame(validation_data)
            validation_path = os.path.join(self.stats_dir, f'{symbol}_{stage}_validation.csv')
            validation_df.to_csv(validation_path, index=False)
            
            # Return issues list
            issues = [
                f"{row['expectation_type']} ({row['column']}): {row['unexpected_count']} failures"
                for _, row in validation_df[~validation_df['success']].iterrows()
            ]
            
            return issues
            
        except Exception as e:
            logger.error(f"Validation error: {str(e)}")
            return [f"Validation error: {str(e)}"]

    def generate_statistics(self, df: pd.DataFrame, symbol: str, stage: str) -> Dict:
        """Generate comprehensive statistics and save as CSV"""
        try:
            # Basic statistics
            basic_stats = pd.DataFrame({
                'metric': ['row_count', 'column_count'],
                'value': [len(df), len(df.columns)]
            })
            
            # Missing values
            missing_stats = pd.DataFrame({
                'column': df.columns,
                'missing_count': df.isnull().sum().values
            })
            
            # Numerical statistics
            numeric_stats_list = []
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            for col in numeric_cols:
                numeric_stats_list.append({
                    'column': col,
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'median': df[col].median(),
                    'skew': df[col].skew(),
                    'kurtosis': df[col].kurtosis()
                })
            numeric_stats = pd.DataFrame(numeric_stats_list)
            
            # Temporal statistics if date index exists
            if isinstance(df.index, pd.DatetimeIndex):
                temporal_stats = pd.DataFrame({
                    'metric': ['start_date', 'end_date', 'date_range_days', 'missing_dates'],
                    'value': [
                        df.index.min().strftime('%Y-%m-%d'),
                        df.index.max().strftime('%Y-%m-%d'),
                        (df.index.max() - df.index.min()).days,
                        len(pd.date_range(start=df.index.min(), end=df.index.max()).difference(df.index))
                    ]
                })
                temporal_stats_path = os.path.join(self.stats_dir, f'{symbol}_{stage}_temporal_stats.csv')
                temporal_stats.to_csv(temporal_stats_path, index=False)
            
            # Save all statistics as CSV files
            basic_stats_path = os.path.join(self.stats_dir, f'{symbol}_{stage}_basic_stats.csv')
            missing_stats_path = os.path.join(self.stats_dir, f'{symbol}_{stage}_missing_stats.csv')
            numeric_stats_path = os.path.join(self.stats_dir, f'{symbol}_{stage}_numeric_stats.csv')
            
            basic_stats.to_csv(basic_stats_path, index=False)
            missing_stats.to_csv(missing_stats_path, index=False)
            numeric_stats.to_csv(numeric_stats_path, index=False)
            
            return {
                'basic_stats_path': basic_stats_path,
                'missing_stats_path': missing_stats_path,
                'numeric_stats_path': numeric_stats_path
            }
            
        except Exception as e:
            logger.error(f"Statistics generation error: {str(e)}")
            return {'error': str(e)}

    def check_data_drift(self, current_df: pd.DataFrame, reference_df: pd.DataFrame,
                        threshold: float = 0.1, symbol: str = None) -> List[str]:
        """Check for data drift and save results as CSV"""
        try:
            drift_data = []
            numeric_cols = current_df.select_dtypes(include=['int64', 'float64']).columns
            
            for col in numeric_cols:
                curr_mean = current_df[col].mean()
                ref_mean = reference_df[col].mean()
                curr_std = current_df[col].std()
                ref_std = reference_df[col].std()
                
                mean_drift = abs(curr_mean - ref_mean) / ref_mean
                std_drift = abs(curr_std - ref_std) / ref_std
                
                drift_data.append({
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'column': col,
                    'current_mean': curr_mean,
                    'reference_mean': ref_mean,
                    'mean_drift': mean_drift,
                    'current_std': curr_std,
                    'reference_std': ref_std,
                    'std_drift': std_drift,
                    'threshold_exceeded': mean_drift > threshold or std_drift > threshold
                })
            
            # Save drift results as CSV
            drift_df = pd.DataFrame(drift_data)
            if symbol:
                drift_path = os.path.join(self.stats_dir, f'{symbol}_drift_analysis.csv')
                drift_df.to_csv(drift_path, index=False)
            
            # Generate drift issues list
            drift_issues = [
                f"Mean drift detected in {row['column']}: {row['mean_drift']:.2%} change"
                for _, row in drift_df[drift_df['mean_drift'] > threshold].iterrows()
            ]
            drift_issues.extend([
                f"Std drift detected in {row['column']}: {row['std_drift']:.2%} change"
                for _, row in drift_df[drift_df['std_drift'] > threshold].iterrows()
            ])
            
            return drift_issues
            
        except Exception as e:
            logger.error(f"Data drift check error: {str(e)}")
            return [f"Drift check error: {str(e)}"]
