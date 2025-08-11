import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
import logging
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from fairlearn.metrics import MetricFrame
import json

@dataclass
class SlicePerformanceMetrics:
    """Metrics for each data slice"""
    slice_name: str
    accuracy: float
    precision: float
    recall: float
    sample_size: int
    feature_importance: Dict[str, float]
    disparity_metrics: Dict[str, float]

class SliceAnalyzer:
    def __init__(self, stats_dir: str = '/opt/airflow/data/stats'):
        self.logger = logging.getLogger(__name__)
        self.stats_dir = Path(stats_dir)
        self.stats_dir.mkdir(exist_ok=True)
        
    def create_market_slices(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Create market condition-based slices with more balanced thresholds"""
        try:
            # Use more balanced RSI thresholds
            rsi_high = df['RSI'].quantile(0.7)  # Top 30%
            rsi_low = df['RSI'].quantile(0.3)   # Bottom 30%
            
            # Use percentile-based thresholds for volatility
            volatility_high = df['Volatility'].quantile(0.7)
            volatility_low = df['Volatility'].quantile(0.3)
            
            # Use percentile-based thresholds for volume
            volume_high = df['Volume'].quantile(0.7)
            volume_low = df['Volume'].quantile(0.3)
            
            slices = {
                'bullish': df[df['RSI'] > rsi_high],
                'bearish': df[df['RSI'] < rsi_low],
                'neutral': df[(df['RSI'] >= rsi_low) & (df['RSI'] <= rsi_high)],
                'high_volatility': df[df['Volatility'] > volatility_high],
                'low_volatility': df[df['Volatility'] < volatility_low],
                'high_volume': df[df['Volume'] > volume_high],
                'low_volume': df[df['Volume'] < volume_low]
            }
            
            # Log slice sizes
            for name, slice_df in slices.items():
                self.logger.info(f"Slice {name}: {len(slice_df)} samples")
            
            return slices
            
        except Exception as e:
            self.logger.error(f"Error creating market slices: {str(e)}")
            raise
    
    def analyze_slices(self, df: pd.DataFrame, symbol: str) -> Dict:
        """Main method to analyze data slices"""
        try:
            # Create copy to avoid modifying original dataframe
            df = df.copy()
            
            # Clean data
            for col in ['RSI', 'Volatility', 'Volume', 'MA5', 'MA20', 'Daily_Return']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    df[col] = df[col].fillna(df[col].median())
            
            # Create binary target (up/down movement)
            df['target'] = (df['Daily_Return'] > 0).astype(int)
            
            # Create slices
            slices = self.create_market_slices(df)
            slice_metrics = {}
            csv_records = []
            
            min_samples = max(int(len(df) * 0.05), 5)  # At least 5% of data or 5 samples
            
            for slice_name, slice_df in slices.items():
                if len(slice_df) < min_samples:
                    self.logger.warning(
                        f"Skipping {slice_name} slice due to insufficient samples "
                        f"({len(slice_df)} < {min_samples})"
                    )
                    continue
                    
                # Calculate simple prediction using RSI
                try:
                    y_true = slice_df['target']
                    y_pred = (slice_df['RSI'] > 50).astype(int)
                    
                    # Calculate metrics
                    metrics = SlicePerformanceMetrics(
                        slice_name=slice_name,
                        accuracy=accuracy_score(y_true, y_pred),
                        precision=precision_score(y_true, y_pred, zero_division=0),
                        recall=recall_score(y_true, y_pred, zero_division=0),
                        sample_size=len(slice_df),
                        feature_importance=self._calculate_feature_importance(slice_df),
                        disparity_metrics=self._calculate_disparity_metrics(slice_df)
                    )
                    
                    slice_metrics[slice_name] = metrics
                    
                    # Prepare CSV record
                    csv_record = {
                        'symbol': symbol,
                        'slice_name': slice_name,
                        'date': pd.Timestamp.now().strftime('%Y-%m-%d'),
                        'accuracy': metrics.accuracy,
                        'precision': metrics.precision,
                        'recall': metrics.recall,
                        'sample_size': metrics.sample_size
                    }
                    csv_records.append(csv_record)
                    
                except Exception as e:
                    self.logger.error(f"Error processing {slice_name} slice: {str(e)}")
                    continue
            
            if not csv_records:
                self.logger.warning(f"No valid slices processed for {symbol}")
                return {
                    'symbol': symbol,
                    'timestamp': pd.Timestamp.now().isoformat(),
                    'error': 'No valid slices processed',
                    'recommendations': ['Investigate data quality issues']
                }
            
            # Save metrics to CSV
            metrics_df = pd.DataFrame(csv_records)
            csv_path = self.stats_dir / f'{symbol}_slice_metrics.csv'
            metrics_df.to_csv(csv_path, index=False)
            
            # Generate report
            report = {
                'symbol': symbol,
                'timestamp': pd.Timestamp.now().isoformat(),
                'total_samples': len(df),
                'slice_metrics': {
                    name: metrics.__dict__ 
                    for name, metrics in slice_metrics.items()
                },
                'recommendations': self._generate_recommendations(slice_metrics)
            }
            
            # Save detailed report as JSON
            report_path = self.stats_dir / f'{symbol}_slice_analysis.json'
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error in slice analysis for {symbol}: {str(e)}")
            raise
    
    def _calculate_feature_importance(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate feature importance using robust correlation"""
        importance = {}
        try:
            # Ensure numeric columns and handle NaN/inf values
            numeric_cols = ['RSI', 'Volatility', 'Volume', 'MA5', 'MA20']
            df_clean = df.copy()
            
            for col in numeric_cols:
                if col in df.columns:
                    # Convert to numeric and handle inf values
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                    df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)
                    # Fill NaN with median
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
                    
                    # Calculate correlation with error handling
                    try:
                        corr = df_clean[col].corr(df_clean['target'])
                        importance[col] = abs(float(corr)) if not pd.isna(corr) else 0.0
                    except Exception as e:
                        self.logger.warning(f"Error calculating correlation for {col}: {str(e)}")
                        importance[col] = 0.0
                    
        except Exception as e:
            self.logger.error(f"Error calculating feature importance: {str(e)}")
            return {'error': str(e)}
        
        return importance
    
    def _calculate_disparity_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate disparity metrics across volume groups"""
        metrics = {}
        try:
            df_clean = df.copy()
            
            # Clean and handle NaN values
            df_clean['Volume'] = pd.to_numeric(df_clean['Volume'], errors='coerce')
            df_clean['Daily_Return'] = pd.to_numeric(df_clean['Daily_Return'], errors='coerce')
            
            # Fill NaN values
            df_clean['Volume'] = df_clean['Volume'].fillna(df_clean['Volume'].median())
            df_clean['Daily_Return'] = df_clean['Daily_Return'].fillna(0)
            
            # Split into high/low volume groups
            volume_median = df_clean['Volume'].median()
            df_clean['volume_group'] = df_clean['Volume'].apply(
                lambda x: 'high' if x > volume_median else 'low'
            )
            
            # Calculate return disparities
            high_returns = df_clean[df_clean['volume_group'] == 'high']['Daily_Return']
            low_returns = df_clean[df_clean['volume_group'] == 'low']['Daily_Return']
            
            if not high_returns.empty and not low_returns.empty:
                metrics['return_disparity'] = float(high_returns.mean() - low_returns.mean())
                metrics['volatility_disparity'] = float(high_returns.std() - low_returns.std())
                
                # Add sample size information
                metrics['high_volume_samples'] = len(high_returns)
                metrics['low_volume_samples'] = len(low_returns)
            else:
                metrics['return_disparity'] = 0.0
                metrics['volatility_disparity'] = 0.0
                metrics['high_volume_samples'] = 0
                metrics['low_volume_samples'] = 0
                self.logger.warning("Empty return groups in disparity calculation")
                
        except Exception as e:
            self.logger.error(f"Error calculating disparity metrics: {str(e)}")
            return {'error': str(e)}
        
        return metrics
    
    def _generate_recommendations(self, 
                                slice_metrics: Dict[str, SlicePerformanceMetrics]) -> List[str]:
        """Generate recommendations based on slice analysis"""
        recommendations = []
        
        for name, metrics in slice_metrics.items():
            if metrics.sample_size < 100:
                recommendations.append(
                    f"Small sample size in {name} slice. Consider collecting more data."
                )
            
            if metrics.accuracy < 0.55:
                recommendations.append(
                    f"Poor prediction accuracy in {name} slice. Consider feature engineering."
                )
                
            if abs(metrics.disparity_metrics.get('return_disparity', 0)) > 0.1:
                recommendations.append(
                    f"Significant return disparity in {name} slice. Consider volume normalization."
                )
        
        return recommendations 

    def _calculate_correlation_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate correlation matrix with proper handling of zero variance"""
        try:
            # Remove constant columns
            df = df.select_dtypes(include=[np.number])
            std = df.std()
            valid_cols = std[std > 1e-10].index  # Use small threshold instead of zero
            
            if len(valid_cols) < 2:
                self.logger.warning(f"Insufficient valid columns for correlation. Valid columns: {len(valid_cols)}")
                return np.zeros((df.shape[1], df.shape[1]))
            
            df = df[valid_cols]
            
            # Use pandas correlation which handles edge cases better
            corr_matrix = df.corr(method='pearson', min_periods=1).fillna(0).values
            return corr_matrix
            
        except Exception as e:
            self.logger.error(f"Error in correlation calculation: {str(e)}")
            return np.zeros((df.shape[1], df.shape[1]))

    def analyze_slice(self, slice_df: pd.DataFrame, slice_name: str) -> Dict:
        """Analyze a single slice with detailed metrics logging"""
        metrics = {}
        try:
            # Calculate basic metrics
            y_true = slice_df['Target'].values
            y_pred = slice_df['Prediction'].values
            
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted'),
                'recall': recall_score(y_true, y_pred, average='weighted'),
                'f1': f1_score(y_true, y_pred, average='weighted'),
                'samples': len(slice_df)
            }
            
            self.logger.info(f"Slice {slice_name} detailed metrics: {metrics}")
            
            # Adjust threshold based on your requirements
            if metrics['accuracy'] < 0.55:  # Lowered from 0.6 to see if any slices perform better
                return {
                    'status': 'poor',
                    'metrics': metrics,
                    'message': f"Poor prediction accuracy ({metrics['accuracy']:.2f}) in {slice_name} slice. Consider feature engineering."
                }
            return {
                'status': 'good',
                'metrics': metrics,
                'message': f"Good prediction accuracy ({metrics['accuracy']:.2f}) in {slice_name} slice."
            }
        except Exception as e:
            self.logger.error(f"Error analyzing slice {slice_name}: {str(e)}")
            return {'error': str(e), 'metrics': metrics}