import mlflow
import yaml
import numpy as np
from pathlib import Path
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelValidator:
    def __init__(self, config_path: str = 'config/pipeline_config.yml'):
        self.config = self._load_config(config_path)
        self.thresholds = self.config['validation_thresholds']
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate comprehensive model metrics"""
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }
    
    def check_thresholds(self, metrics: Dict) -> Tuple[bool, Dict]:
        """Check if metrics meet defined thresholds"""
        results = {}
        for metric, value in metrics.items():
            if metric in self.thresholds:
                passed = value <= self.thresholds[metric]
                results[metric] = {
                    'value': value,
                    'threshold': self.thresholds[metric],
                    'passed': passed
                }
                logger.info(f"{metric}: {value:.6f} (threshold: {self.thresholds[metric]}) - {'PASSED' if passed else 'FAILED'}")
        
        all_passed = all(result['passed'] for result in results.values())
        return all_passed, results

    def validate_latest_model(self) -> Tuple[bool, Dict]:
        """Validate the latest trained model"""
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
            
            # Calculate metrics
            y_val = validation_data['y_val']
            y_pred = model.predict(validation_data['X_val'])
            metrics = self.calculate_metrics(y_val, y_pred)
            
            # Check against thresholds
            passed, results = self.check_thresholds(metrics)
            
            # Log validation results to MLflow
            with mlflow.start_run(run_id=latest_run.info.run_id, nested=True):
                mlflow.log_metrics({f"val_{k}": v for k, v in metrics.items()})
                mlflow.log_dict(results, "validation_results.json")
            
            return passed, results
            
        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            return False, {}

def main():
    validator = ModelValidator()
    success, results = validator.validate_latest_model()
    
    if success:
        logger.info("Model validation passed all thresholds!")
    else:
        logger.error("Model validation failed!")
        
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
