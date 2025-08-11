import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
import logging
from pathlib import Path
import json

@dataclass
class BiasMetrics:
    """Metrics for each data slice"""
    mean_return: float
    volatility: float
    sample_size: int
    distribution_stats: Dict[str, float]
    feature_correlations: Dict[str, float]

class MarketBiasDetector:
    def __init__(self, stats_dir: str = '/opt/airflow/data/stats'):
        self.logger = logging.getLogger(__name__)
        self.stats_dir = Path(stats_dir)
        self.stats_dir.mkdir(exist_ok=True)
        
    def create_market_condition_slices(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Create slices based on market conditions"""
        slices = {}
        
        # Trend-based slices using RSI
        slices['overbought'] = df[df['RSI'] > 70]
        slices['oversold'] = df[df['RSI'] < 30]
        slices['neutral'] = df[(df['RSI'] >= 30) & (df['RSI'] <= 70)]
        
        # Volatility-based slices
        vol_median = df['Volatility'].median()
        slices['high_volatility'] = df[df['Volatility'] > vol_median]
        slices['low_volatility'] = df[df['Volatility'] <= vol_median]
        
        # Return-based slices
        slices['positive_returns'] = df[df['Daily_Return'] > 0]
        slices['negative_returns'] = df[df['Daily_Return'] < 0]
        
        # Volume-based slices
        vol_75th = df['Volume'].quantile(0.75)
        slices['high_volume'] = df[df['Volume'] > vol_75th]
        slices['normal_volume'] = df[df['Volume'] <= vol_75th]
        
        return slices
    
    def calculate_slice_metrics(self, slice_df: pd.DataFrame) -> BiasMetrics:
        """Calculate metrics for a given slice"""
        return BiasMetrics(
            mean_return=float(slice_df['Daily_Return'].mean()),
            volatility=float(slice_df['Volatility'].mean()),
            sample_size=len(slice_df),
            distribution_stats={
                'return_skew': float(slice_df['Daily_Return'].skew()),
                'return_kurtosis': float(slice_df['Daily_Return'].kurtosis()),
                'volume_skew': float(slice_df['Volume'].skew())
            },
            feature_correlations={
                'volume_return_corr': float(slice_df['Volume'].corr(slice_df['Daily_Return'])),
                'volatility_return_corr': float(slice_df['Volatility'].corr(slice_df['Daily_Return'])),
                'rsi_return_corr': float(slice_df['RSI'].corr(slice_df['Daily_Return'])),
                'sp500_return_corr': float(slice_df['GSPC_Return'].corr(slice_df['Daily_Return'])),
                'dax_return_corr': float(slice_df['GDAXI_Return'].corr(slice_df['Daily_Return'])),
                'gold_return_corr': float(slice_df['GC_Return'].corr(slice_df['Daily_Return'])),
                'nifty_return_corr': float(slice_df['NSEI_Return'].corr(slice_df['Daily_Return']))
            }
        )
    
    def detect_bias(self, df: pd.DataFrame, symbol: str) -> Dict:
        """Main method to detect and analyze bias in the data"""
        try:
            slices = self.create_market_condition_slices(df)
            slice_metrics = {}
            bias_indicators = []
            
            # For CSV export
            csv_records = []
            
            # Calculate metrics for each slice
            for slice_name, slice_df in slices.items():
                metrics = self.calculate_slice_metrics(slice_df)
                slice_metrics[slice_name] = metrics
                
                # Create CSV record
                csv_record = {
                    'symbol': symbol,
                    'slice_name': slice_name,
                    'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d'),
                    'sample_size': metrics.sample_size,
                    'mean_return': metrics.mean_return,
                    'volatility': metrics.volatility,
                    'return_skew': metrics.distribution_stats['return_skew'],
                    'return_kurtosis': metrics.distribution_stats['return_kurtosis'],
                    'volume_skew': metrics.distribution_stats['volume_skew'],
                    'volume_return_correlation': metrics.feature_correlations['volume_return_corr'],
                    'volatility_return_correlation': metrics.feature_correlations['volatility_return_corr'],
                    'rsi_return_correlation': metrics.feature_correlations['rsi_return_corr'],
                    'sp500_return_correlation': metrics.feature_correlations['sp500_return_corr'],
                    'dax_return_correlation': metrics.feature_correlations['dax_return_corr'],
                    'gold_return_correlation': metrics.feature_correlations['gold_return_corr'],
                    'nifty_return_correlation': metrics.feature_correlations['nifty_return_corr']
                }
                csv_records.append(csv_record)
                
                # Check for potential biases
                if metrics.sample_size < len(df) * 0.05:
                    bias_indicators.append(f"Underrepresented slice: {slice_name}")
                    csv_record['bias_type'] = 'underrepresented'
                    
                if abs(metrics.distribution_stats['return_skew']) > 1.0:
                    bias_indicators.append(f"Significant return skew in {slice_name}")
                    csv_record['bias_type'] = 'skewed_returns'
                    
                if abs(metrics.feature_correlations['volume_return_corr']) > 0.7:
                    bias_indicators.append(f"Strong volume-return correlation in {slice_name}")
                    csv_record['bias_type'] = 'high_correlation'
            
            # Save as CSV for model development
            csv_df = pd.DataFrame(csv_records)
            csv_path = self.stats_dir / f'{symbol}_bias_metrics.csv'
            csv_df.to_csv(csv_path, index=False)
            
            # Generate comprehensive JSON report
            report = {
                'symbol': symbol,
                'timestamp': pd.Timestamp.now().isoformat(),
                'total_samples': len(df),
                'slice_metrics': {
                    name: {
                        'metrics': metrics.__dict__
                    } for name, metrics in slice_metrics.items()
                },
                'bias_indicators': bias_indicators,
                'recommendations': self._generate_recommendations(slice_metrics)
            }
            
            # Save detailed report as JSON
            report_path = self.stats_dir / f'{symbol}_bias_report.json'
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error in bias detection: {str(e)}")
            raise
    
    def _generate_recommendations(self, slice_metrics: Dict[str, BiasMetrics]) -> List[str]:
        """Generate recommendations based on bias analysis"""
        recommendations = []
        
        # Sample size balance
        sizes = [m.sample_size for m in slice_metrics.values()]
        size_cv = np.std(sizes) / np.mean(sizes)
        if size_cv > 0.5:
            recommendations.append("Consider resampling to balance slice sizes")
            
        # Return distribution
        skews = [m.distribution_stats['return_skew'] for m in slice_metrics.values()]
        if any(abs(s) > 1.0 for s in skews):
            recommendations.append("Consider return distribution normalization")
            
        # Correlation patterns
        vol_corrs = [m.feature_correlations['volume_return_corr'] for m in slice_metrics.values()]
        if any(abs(c) > 0.7 for c in vol_corrs):
            recommendations.append("Consider volume normalization")
            
        return recommendations 