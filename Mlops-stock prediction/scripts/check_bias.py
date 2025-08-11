import mlflow
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, Tuple, List
from sklearn.metrics import mean_squared_error

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BiasDetector:
    def __init__(self, config_path: str = 'config/pipeline_config.yml'):
        self.config = self._load_config(config_path)
        self.bias_thresholds = self.config['bias_thresholds']
        
    def _load_config(self, config_path: str) -> dict:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def create_slices(self, X: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Create data slices for bias detection"""
        slices = {}
        
        try:
            # Volatility-based slices
            if 'volatility' in X.columns:
                vol_quartiles = pd.qcut(X['volatility'], q=4, 
                                      labels=['low', 'medium_low', 'medium_high', 'high'])
                for label in vol_quartiles.unique():
                    slices[f'volatility_{label}'] = X[vol_quartiles == label]
            
            # Volume-based slices
            if 'volume' in X.columns:
                vol_quartiles = pd.qcut(X['volume'], q=4, 
                                      labels=['low', 'medium_low', 'medium_high', 'high'])
                for label in vol_quartiles.unique():
                    slices[f'volume_{label}'] = X[vol_quartiles == label]
            
            logger.info(f"Created {len(slices)} data slices")
            return slices
            
        except Exception as e:
            logger.error(f"Error creating slices: {str(e)}")
            return {}
    
    def evaluate_slice_metrics(self, model, X: pd.DataFrame, y: np.ndarray, 
                             slices: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Calculate metrics for each slice"""
        metrics = {}
        baseline_mse = mean_squared_error(y, model.predict(X))
        
        for slice_name, slice_data in slices.items():
            slice_indices = slice_data.index
            y_slice = y[slice_indices]
            y_pred_slice = model.predict(slice_data)
            
            slice_mse = mean_squared_error(y_slice, y_pred_slice)
            relative_diff = abs(slice_mse - baseline_mse) / baseline_mse
            
            metrics[slice_name] = {
                'mse': slice_mse,
                'relative_difference': relative_diff,
                'size': len(slice_data)
            }
            
        return metrics
    
    def detect_bias(self, metrics: Dict[str, Dict]) -> Tuple[bool, List[str]]:
        """Detect bias based on slice metrics"""
        biased_slices = []
        
        for slice_name, slice_metrics in metrics.items():
            # Check if slice performance is significantly worse
            if slice_metrics['relative_difference'] > self.bias_thresholds['max_relative_difference']:
                biased_slices.append(slice_name)
                logger.warning(f"Detected bias in slice {slice_name}: "
                             f"relative difference = {slice_metrics['relative_difference']:.4f}")
        
        has_bias = len(biased_slices) > 0
        return has_bias, biased_slices
    
    def check_model_bias(self) -> Tuple[bool, Dict]:
        """Main method to check model bias"""
        try:
            # Get latest MLflow run
            client = mlflow.tracking.MlflowClient()
            latest_run = client.search_runs(
                experiment_ids=[self.config['experiment_id']],
                order_by=["attributes.start_time DESC"],
                max_results=1
            )[0]
            
            # Load model and validation data
            model = mlflow.sklearn.load_model(f"runs:/{latest_run.info.run_id}/model")
            validation_data = mlflow.artifacts.load_dict(f"runs:/{latest_run.info.run_id}/validation_data.json")
            
            X_val = pd.DataFrame(validation_data['X_val'])
            y_val = np.array(validation_data['y_val'])
            
            # Create data slices
            slices = self.create_slices(X_val)
            
            # Calculate metrics for each slice
            slice_metrics = self.evaluate_slice_metrics(model, X_val, y_val, slices)
            
            # Detect bias
            has_bias, biased_slices = self.detect_bias(slice_metrics)
            
            # Log results to MLflow
            with mlflow.start_run(run_id=latest_run.info.run_id, nested=True):
                mlflow.log_dict(slice_metrics, "bias_metrics.json")
                mlflow.log_dict({
                    'has_bias': has_bias,
                    'biased_slices': biased_slices
                }, "bias_detection_results.json")
            
            return not has_bias, {
                'slice_metrics': slice_metrics,
                'biased_slices': biased_slices
            }
            
        except Exception as e:
            logger.error(f"Bias detection failed: {str(e)}")
            return False, {}

def main():
    detector = BiasDetector()
    success, results = detector.check_model_bias()
    
    if success:
        logger.info("No significant bias detected!")
    else:
        logger.error(f"Detected bias in slices: {results.get('biased_slices', [])}")
        
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 