import pandas as pd
import numpy as np
from fairlearn.metrics import MetricFrame
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime
from models.utils.logger import setup_logger
import seaborn as sns
import mlflow
from typing import Dict

class BiasChecker:
    def __init__(self, config):
        self.config = config
        self.logger = setup_logger("BiasChecker")
        self.bias_metrics = {}
        
    def custom_mse(self, y_true, y_pred):
        """Custom MSE function for MetricFrame"""
        return mean_squared_error(y_true, y_pred)
        
    def check_bias(self, model, X: pd.DataFrame, y: pd.Series, volatility_values: pd.Series) -> Dict:
        """
        Check for bias in model predictions across different volatility groups
        
        Args:
            model: Trained model
            X: Feature data
            y: Target values
            volatility_values: Series of volatility values
        
        Returns:
            Dict containing bias metrics
        """
        try:
            # Get predictions
            y_pred = model.predict(X)
            
            # Calculate overall bias
            overall_mse = mean_squared_error(y, y_pred)
            
            # Calculate group-wise bias
            quartiles = pd.qcut(volatility_values, q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
            group_mse = {}
            
            for group in ['Q1', 'Q2', 'Q3', 'Q4']:
                mask = quartiles == group
                if any(mask):
                    group_mse[f'group_{group}'] = mean_squared_error(
                        y[mask], y_pred[mask]
                    )
            
            # Store metrics in instance variable with 'by_group' structure
            self.bias_metrics = {
                'overall_bias': float(overall_mse),
                'by_group': {k: float(v) for k, v in group_mse.items()}
            }
            
            self.logger.info(f"Overall bias metric: {overall_mse:.4f}")
            self.logger.info(f"Group-wise bias metrics: {group_mse}")
            
            return self.bias_metrics
            
        except Exception as e:
            self.logger.error(f"Error in bias checking: {str(e)}")
            raise
            
    def visualize_bias(self, save_dir: Path):
        """Create and save bias visualization"""
        try:
            if not hasattr(self, 'bias_metrics') or 'by_group' not in self.bias_metrics:
                self.logger.warning("No bias metrics available for visualization")
                return
                
            # Use the by_group metrics directly
            group_metrics = self.bias_metrics['by_group']
            
            # Create DataFrame with proper column names
            data = []
            for group, value in group_metrics.items():
                data.append({
                    'group': group.replace('group_', ''),  # Remove 'group_' prefix
                    'value': value
                })
            df = pd.DataFrame(data)
            
            plt.figure(figsize=(10, 6))
            sns.barplot(data=df, x='group', y='value')
            plt.title('Bias Across Groups')
            plt.xlabel('Volatility Quartiles')
            plt.ylabel('Mean Squared Error')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save plot
            plot_path = save_dir / 'bias_visualization.png'
            plt.savefig(plot_path)
            plt.close()
            
            # Log to MLflow
            mlflow.log_artifact(str(plot_path))
            
            self.logger.info(f"Bias visualization saved to {plot_path}")
            
        except Exception as e:
            self.logger.error(f"Error in bias visualization: {str(e)}")
            raise
            
    def save_bias_report(self, save_dir: Path):
        """
        Save bias metrics to a JSON file
        """
        try:
            if not self.bias_metrics:
                self.logger.warning("No bias metrics available to save")
                return
                
            bias_report_path = save_dir / 'bias_report.json'
            with open(bias_report_path, 'w') as f:
                json.dump(self.bias_metrics, f, indent=4)
                
            self.logger.info(f"Bias report saved to {bias_report_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving bias report: {str(e)}")
            raise 