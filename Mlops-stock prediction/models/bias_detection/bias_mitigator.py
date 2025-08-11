import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

class BiasMitigator:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def mitigate_bias(self, X: pd.DataFrame, y: pd.Series, 
                     biased_slices: List[Dict], 
                     slices: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.Series]:
        """Apply bias mitigation techniques"""
        try:
            if not biased_slices:
                return X, y
                
            # Calculate weights for each sample
            weights = pd.Series(1.0, index=X.index)
            
            for bias_info in biased_slices:
                slice_name = bias_info['slice']
                relative_diff = bias_info['relative_difference']
                
                if slice_name in slices:
                    slice_indices = slices[slice_name].index
                    weights[slice_indices] *= (1 + relative_diff)
            
            # Normalize weights
            weights = weights / weights.mean()
            
            # Resample data using weights
            resampled_indices = np.random.choice(
                X.index,
                size=len(X),
                p=weights/weights.sum(),
                replace=True
            )
            
            self.logger.info("Bias mitigation completed successfully")
            return X.loc[resampled_indices], y.loc[resampled_indices]
            
        except Exception as e:
            self.logger.error(f"Error in bias mitigation: {str(e)}")
            raise 