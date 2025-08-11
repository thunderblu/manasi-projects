import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict
from models.utils.logger import setup_logger

class ModelValidator:
    def __init__(self):
        self.logger = setup_logger("ModelValidator")

    def validate(self, model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Validate the model on the test dataset and compute metrics.
        
        Args:
            model: Trained model to validate
            X_test: Test features
            y_test: Test target
        
        Returns:
            Dictionary of validation metrics
        """
        try:
            test_preds = model.predict(X_test)
            
            test_metrics = {
                'test_mse': mean_squared_error(y_test, test_preds),
                'test_mae': mean_absolute_error(y_test, test_preds),
                'test_r2': r2_score(y_test, test_preds)
            }
            
            self.logger.info(f"Validation on test set")
            self.logger.info(f"Test MSE: {test_metrics['test_mse']:.4f}")
            self.logger.info(f"Test MAE: {test_metrics['test_mae']:.4f}")
            self.logger.info(f"Test R²: {test_metrics['test_r2']:.4f}")
            
            return test_metrics
            
        except Exception as e:
            self.logger.error(f"Error during validation: {str(e)}")
            raise 