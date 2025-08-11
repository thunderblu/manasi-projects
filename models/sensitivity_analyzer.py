import shap
import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List
from models.utils.logger import setup_logger
from sklearn.metrics import mean_squared_error

class SensitivityAnalyzer:
    def __init__(self, config):
        self.config = config
        self.logger = setup_logger("SensitivityAnalyzer")
        
    def analyze_feature_importance(self, model, X_train, X_test, feature_names: List[str]):
        """
        Analyze feature importance using SHAP values
        """
        try:
            # Calculate SHAP values
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            
            # Create SHAP summary plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
            plt.tight_layout()
            
            # Save plot
            plt.savefig('feature_importance_shap.png')
            plt.close()
            
            # Calculate mean absolute SHAP values for each feature
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': np.abs(shap_values).mean(axis=0)
            })
            feature_importance = feature_importance.sort_values('importance', ascending=False)
            
            return feature_importance
            
        except Exception as e:
            self.logger.error(f"Error in SHAP analysis: {str(e)}")
            raise

    def analyze_lime(self, model, X_train, X_test, feature_names: List[str]):
        """
        Analyze local feature importance using LIME
        """
        try:
            self.logger.info("Starting LIME analysis...")
            
            # Convert DataFrames to numpy arrays if they aren't already
            X_train_array = X_train.values if hasattr(X_train, 'values') else X_train
            X_test_array = X_test.values if hasattr(X_test, 'values') else X_test
            
            # Create the LIME explainer
            explainer = lime.lime_tabular.LimeTabularExplainer(
                X_train_array,
                feature_names=feature_names,
                class_names=['price'],
                mode='regression'
            )
            
            # Get explanation for first test instance
            exp = explainer.explain_instance(
                X_test_array[0],
                model.predict,
                num_features=len(feature_names)
            )
            
            # Save explanation
            exp.save_to_file('lime_explanation.html')
            self.logger.info("LIME analysis completed successfully")
            
            return exp.as_list()
            
        except Exception as e:
            self.logger.error(f"Error in LIME analysis: {str(e)}")
            raise

    def analyze_hyperparameter_sensitivity(self, model_trainer, X_train, y_train, model_type: str, param_ranges: Dict[str, List[Any]]):
        """
        Analyze model sensitivity to hyperparameter changes
        """
        try:
            results = []
            
            # Get best parameters from hyperparameter tuner for the specific model type
            best_params = model_trainer.hyperparameter_tuner.best_params.get(model_type, {})
            if not best_params:
                self.logger.warning(f"No best parameters found for {model_type}, using default parameters")
                best_params = {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'min_samples_split': 2 if model_type == 'random_forest' else None,
                    'learning_rate': 0.1 if model_type == 'xgboost' else None
                }
                # Remove None values
                best_params = {k: v for k, v in best_params.items() if v is not None}
            
            # Test each parameter independently
            for param_name, param_values in param_ranges.items():
                param_scores = []
                
                for value in param_values:
                    # Create parameter grid with single parameter varying
                    params = best_params.copy()
                    params[param_name] = value
                    
                    # Create and train a new model with these parameters
                    model = model_trainer.create_model(model_type, params)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_train)
                    score = mean_squared_error(y_train, y_pred)
                    param_scores.append(score)
                
                results.append({
                    'parameter': param_name,
                    'values': param_values,
                    'scores': param_scores
                })
            
            # Create sensitivity plots
            self._plot_parameter_sensitivity(results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in hyperparameter sensitivity analysis: {str(e)}")
            raise

    def _plot_parameter_sensitivity(self, results: List[Dict]):
        """
        Create plots for hyperparameter sensitivity analysis
        """
        try:
            n_params = len(results)
            fig, axes = plt.subplots(n_params, 1, figsize=(10, 5*n_params))
            
            if n_params == 1:
                axes = [axes]
            
            for ax, result in zip(axes, results):
                ax.plot(result['values'], result['scores'], 'o-')
                ax.set_xlabel(result['parameter'])
                ax.set_ylabel('Score (MSE)')
                ax.set_title(f"Sensitivity to {result['parameter']}")
                ax.grid(True)
            
            plt.tight_layout()
            plt.savefig('hyperparameter_sensitivity.png')
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error in plotting sensitivity: {str(e)}")
            raise