from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict
import logging

class MetricsEvaluator:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def evaluate_slice_metrics(self, model, slices: Dict, y) -> Dict[str, Dict[str, float]]:
        """Evaluate model performance on different slices"""
        metrics = {}
        
        try:
            for slice_name, slice_data in slices.items():
                if len(slice_data) > 0:
                    slice_y = y[slice_data.index]
                    predictions = model.predict(slice_data)
                    
                    metrics[slice_name] = {
                        'mse': mean_squared_error(slice_y, predictions),
                        'mae': mean_absolute_error(slice_y, predictions),
                        'r2': r2_score(slice_y, predictions),
                        'size': len(slice_data)
                    }
            
            self.logger.info(f"Evaluated metrics for {len(metrics)} slices")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error in evaluating metrics: {str(e)}")
            raise 