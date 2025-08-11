from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from typing import Dict, Any
from models.utils.logger import setup_logger

class HyperparameterTuner:
    def __init__(self, config):
        self.config = config
        self.logger = setup_logger("HyperparameterTuner")
        self.best_params = {}
        self.cv_results = {}
        
    def tune_model(self, model, X_train, y_train, model_type: str) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using RandomizedSearchCV
        """
        try:
            # Define scoring metrics
            scoring = {
                'neg_mse': make_scorer(mean_squared_error, greater_is_better=False),
                'neg_mae': make_scorer(mean_absolute_error, greater_is_better=False),
                'r2': make_scorer(r2_score)
            }
            
            # Select parameter grid based on model type
            param_grid = (self.config.rf_param_grid if model_type == "random_forest" 
                         else self.config.xgb_param_grid)
            
            # Initialize RandomizedSearchCV
            random_search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_grid,
                n_iter=self.config.n_trials,
                cv=self.config.n_cv_folds,
                scoring=scoring,
                refit='neg_mse',  # Optimize for MSE
                n_jobs=-1,
                random_state=self.config.random_state,
                verbose=1
            )
            
            # Perform hyperparameter search
            self.logger.info(f"Starting hyperparameter tuning for {model_type}...")
            random_search.fit(X_train, y_train)
            
            # Store results
            self.best_params[model_type] = random_search.best_params_
            self.cv_results[model_type] = random_search.cv_results_
            
            # Log results
            self.logger.info(f"Best parameters for {model_type}: {random_search.best_params_}")
            self.logger.info(f"Best cross-validation MSE: {-random_search.best_score_:.4f}")
            
            return random_search.best_estimator_
            
        except Exception as e:
            self.logger.error(f"Error during hyperparameter tuning: {str(e)}")
            raise
            
    def save_tuning_results(self, save_dir):
        """Save hyperparameter tuning results"""
        try:
            results = {
                'best_parameters': self.best_params,
                'cv_results_summary': {
                    model_type: {
                        'mean_test_neg_mse': float(np.mean(results['test_neg_mse'])),
                        'std_test_neg_mse': float(np.std(results['test_neg_mse'])),
                        'mean_test_r2': float(np.mean(results['test_r2'])),
                        'std_test_r2': float(np.std(results['test_r2']))
                    }
                    for model_type, results in self.cv_results.items()
                }
            }
            
            with open(save_dir / 'hyperparameter_tuning_results.json', 'w') as f:
                json.dump(results, f, indent=4)
                
        except Exception as e:
            self.logger.error(f"Error saving tuning results: {str(e)}")
            raise 