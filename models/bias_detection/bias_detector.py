import numpy as np
from typing import Dict, List
import logging

class BiasDetector:
    def __init__(self, threshold: float = 0.1):
        self.threshold = threshold
        self.logger = logging.getLogger(self.__class__.__name__)

    def detect_bias(self, metrics: Dict[str, Dict[str, float]]) -> List[Dict]:
        """Detect significant performance disparities across slices"""
        try:
            biased_slices = []
            baseline_mse = np.mean([m['mse'] for m in metrics.values()])
            
            for slice_name, slice_metrics in metrics.items():
                relative_diff = abs(slice_metrics['mse'] - baseline_mse) / baseline_mse
                if relative_diff > self.threshold:
                    biased_slices.append({
                        'slice': slice_name,
                        'relative_difference': relative_diff,
                        'metrics': slice_metrics
                    })
            
            self.logger.info(f"Detected {len(biased_slices)} biased slices")
            return biased_slices
            
        except Exception as e:
            self.logger.error(f"Error in bias detection: {str(e)}")
            raise 