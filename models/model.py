import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Any, Tuple
import joblib
from pathlib import Path
import json
from datetime import datetime

from models.utils.logger import setup_logger
from models.config import ModelConfig
from models.hyperparameter_tuner import HyperparameterTuner
from models.experiment_tracker import ExperimentTracker
import mlflow
import os

class ModelTrainer:
    def __init__(self, config: ModelConfig):
        """
        Initialize the model trainer
        Args:
            config: ModelConfig instance containing model parameters
        """
        self.config = config
        self.logger = setup_logger("ModelTrainer")
        self.models = {}
        self.best_model = None
        self.best_score = float('inf')
        self.metrics = {}
        self.hyperparameter_tuner = HyperparameterTuner(config)
        self.experiment_tracker = ExperimentTracker(config)
        self.best_params = {}

    def create_model(self, model_type: str, params: Dict = None) -> Any:
        """
        Create a model instance based on the specified type
        """
        if params is None:
            params = {}
            
        if model_type == "random_forest":
            return RandomForestRegressor(
                n_estimators=params.get('n_estimators', self.config.n_estimators),
                max_depth=params.get('max_depth', self.config.max_depth),
                random_state=self.config.random_state,
                **{k: v for k, v in params.items() if k not in ['n_estimators', 'max_depth']}
            )
        elif model_type == "xgboost":
            return XGBRegressor(
                n_estimators=params.get('n_estimators', self.config.n_estimators),
                max_depth=params.get('max_depth', self.config.max_depth),
                learning_rate=params.get('learning_rate', self.config.learning_rate),
                random_state=self.config.random_state,
                **{k: v for k, v in params.items() if k not in ['n_estimators', 'max_depth', 'learning_rate']}
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def train_and_evaluate(self, 
                          X_train: pd.DataFrame, 
                          y_train: pd.Series,
                          X_val: pd.DataFrame, 
                          y_val: pd.Series,
                          model_type: str) -> Dict[str, float]:
        """
        Train and evaluate a single model
        """
        try:
            # Start nested run for model training
            self.experiment_tracker.start_run(model_type, "STOCK", nested=True)
            
            # Create base model
            base_model = self.create_model(model_type)
            
            # Perform hyperparameter tuning if enabled
            if self.config.hyperparameter_tuning:
                self.logger.info(f"Starting hyperparameter tuning for {model_type}")
                model = self.hyperparameter_tuner.tune_model(
                    base_model, X_train, y_train, model_type
                )
            else:
                model = base_model
            
            # Log parameters
            self.experiment_tracker.log_parameters({
                "model_type": model_type,
                **model.get_params()
            })
            
            # Make predictions
            train_preds = model.predict(X_train)
            val_preds = model.predict(X_val)
            
            # Calculate metrics
            metrics = {
                'train_mse': mean_squared_error(y_train, train_preds),
                'train_mae': mean_absolute_error(y_train, train_preds),
                'train_r2': r2_score(y_train, train_preds),
                'val_mse': mean_squared_error(y_val, val_preds),
                'val_mae': mean_absolute_error(y_val, val_preds),
                'val_r2': r2_score(y_val, val_preds)
            }
            
            # Log metrics
            self.experiment_tracker.log_metrics(metrics)
            
            # Log model
            self.experiment_tracker.log_model(model, model_type)
            
            # Create and log visualizations
            save_dir = Path(self.config.model_save_path) / datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir.mkdir(parents=True, exist_ok=True)
            
            self.experiment_tracker.log_feature_importance(
                model, X_train.columns, model_type, save_dir
            )
            
            # End MLflow run
            self.experiment_tracker.end_run()
            
            # Store model and metrics
            self.models[model_type] = model
            self.metrics[model_type] = metrics
            
            # Update best model if this one performs better
            if metrics['val_mse'] < self.best_score:
                self.best_score = metrics['val_mse']
                self.best_model = model_type
            
            self.logger.info(f"Completed training for {model_type}")
            self.logger.info(f"Validation MSE: {metrics['val_mse']:.4f}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error training {model_type}: {str(e)}")
            raise
        finally:
            if mlflow.active_run():
                mlflow.end_run()

    def train_all_models(self, 
                        X_train: pd.DataFrame, 
                        y_train: pd.Series,
                        X_val: pd.DataFrame, 
                        y_val: pd.Series) -> Dict[str, Dict[str, float]]:
        """
        Train and evaluate all specified models
        """
        model_types = ["random_forest", "xgboost"]
        all_metrics = {}
        
        for model_type in model_types:
            self.logger.info(f"Training {model_type}...")
            metrics = self.train_and_evaluate(X_train, y_train, X_val, y_val, model_type)
            all_metrics[model_type] = metrics
        
        self.logger.info(f"Best performing model: {self.best_model}")
        return all_metrics

    def save_model(self, symbol: str, best_model: str) -> None:
        """
        Save the best model and its metrics
        """
        if best_model is None:
            self.logger.warning("No model to save!")
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = Path(self.config.model_save_path) / symbol / timestamp
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = save_dir / f"{best_model}_model.joblib"
        joblib.dump(self.models[best_model], model_path)
        
        # Save metrics
        metrics_path = save_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics[best_model], f, indent=4)
        
        # Save configuration
        config_path = save_dir / "config.json"
        with open(config_path, 'w') as f:
            config_dict = {k: str(v) if isinstance(v, Path) else v 
                          for k, v in self.config.__dict__.items()}
            json.dump(config_dict, f, indent=4)
            
        self.logger.info(f"Model and metrics saved to {save_dir}")

    def load_model(self, model_path: str) -> Any:
        """
        Load a saved model
        """
        try:
            model = joblib.load(model_path)
            self.logger.info(f"Model loaded from {model_path}")
            return model
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
