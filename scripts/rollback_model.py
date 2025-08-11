import mlflow
import yaml
import logging
from google.cloud import aiplatform
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelRollback:
    def __init__(self, config_path: str = 'config/pipeline_config.yml'):
        self.config = self._load_config(config_path)
        self.rollback_config = self.config['rollback']
        self.registry_config = self.config['model_registry']
        
    def _load_config(self, config_path: str) -> dict:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _get_current_model(self) -> Optional[aiplatform.Model]:
        """Get currently deployed model"""
        try:
            aiplatform.init(
                project=self.registry_config['gcp_project'],
                location=self.registry_config['region']
            )
            
            models = aiplatform.Model.list(
                filter=f'display_name="{self.registry_config["model_name"]}"',
                order_by="create_time desc"
            )
            
            return models[0] if models else None
            
        except Exception as e:
            logger.error(f"Error getting current model: {str(e)}")
            return None
    
    def _get_previous_stable_model(self) -> Optional[aiplatform.Model]:
        """Get the last known stable model"""
        try:
            models = aiplatform.Model.list(
                filter=f'display_name="{self.registry_config["model_name"]}" AND labels.status="stable"',
                order_by="create_time desc"
            )
            
            return models[1] if len(models) > 1 else None  # Get second-to-last stable model
            
        except Exception as e:
            logger.error(f"Error getting previous stable model: {str(e)}")
            return None
    
    def _compare_model_performance(self, current_model: aiplatform.Model, 
                                 previous_model: aiplatform.Model) -> Tuple[bool, Dict]:
        """Compare performance metrics between models"""
        try:
            current_metrics = current_model.get_model_evaluation()
            previous_metrics = previous_model.get_model_evaluation()
            
            # Compare key metrics
            performance_diff = {
                'mse': current_metrics.metrics['mse'] - previous_metrics.metrics['mse'],
                'mae': current_metrics.metrics['mae'] - previous_metrics.metrics['mae'],
                'r2': current_metrics.metrics['r2'] - previous_metrics.metrics['r2']
            }
            
            # Check if current model is significantly worse
            needs_rollback = (
                performance_diff['mse'] > self.rollback_config['mse_threshold'] or
                performance_diff['mae'] > self.rollback_config['mae_threshold'] or
                performance_diff['r2'] < -self.rollback_config['r2_threshold']
            )
            
            return needs_rollback, performance_diff
            
        except Exception as e:
            logger.error(f"Error comparing model performance: {str(e)}")
            return True, {}  # Conservative approach: rollback on error
    
    def perform_rollback(self) -> bool:
        """Execute model rollback if necessary"""
        try:
            # Get current and previous models
            current_model = self._get_current_model()
            previous_model = self._get_previous_stable_model()
            
            if not current_model or not previous_model:
                logger.error("Could not find current or previous model")
                return False
            
            # Compare performance
            needs_rollback, performance_diff = self._compare_model_performance(
                current_model, previous_model
            )
            
            if needs_rollback:
                logger.warning("Performance degradation detected. Initiating rollback...")
                logger.info(f"Performance differences: {performance_diff}")
                
                # Update model versions
                current_model.update(labels={'status': 'rolled_back'})
                previous_model.update(labels={'status': 'active'})
                
                # Deploy previous model
                endpoint = aiplatform.Endpoint(current_model.serving_endpoint)
                endpoint.deploy(previous_model)
                
                # Log rollback to MLflow
                with mlflow.start_run(run_name="model_rollback"):
                    mlflow.log_dict({
                        'rollback_reason': 'performance_degradation',
                        'performance_diff': performance_diff,
                        'timestamp': datetime.now().isoformat(),
                        'from_model': current_model.resource_name,
                        'to_model': previous_model.resource_name
                    }, "rollback_info.json")
                
                logger.info("Rollback completed successfully")
                return True
            else:
                logger.info("No rollback needed - current model performing well")
                return True
                
        except Exception as e:
            logger.error(f"Rollback failed: {str(e)}")
            return False

def main():
    rollback = ModelRollback()
    success = rollback.perform_rollback()
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 