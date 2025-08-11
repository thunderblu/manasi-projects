import mlflow
import mlflow.sklearn
import mlflow.xgboost
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from models.utils.logger import setup_logger
from datetime import datetime

class ExperimentTracker:
    def __init__(self, config):
        self.config = config
        self.logger = setup_logger("ExperimentTracker")
        
        # Set up MLflow with absolute path
        mlflow_dir = Path(config.project_root) / "mlruns"
        mlflow.set_tracking_uri(f"file://{mlflow_dir.absolute()}")
        self.logger.info(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
        
        # Set experiment to production
        self.experiment_name = "stock_prediction_production"
        mlflow.set_experiment(self.experiment_name)
        self.logger.info(f"MLflow experiment: {self.experiment_name}")
        
    def start_run(self, model_type: str, symbol: str, nested: bool = True) -> None:
        """Start a new MLflow run"""
        run_name = f"{model_type}_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Only end the run if it's not nested
        if mlflow.active_run() and not nested:
            self.logger.info("Ending existing run before starting new one")
            mlflow.end_run()
            
        self.active_run = mlflow.start_run(run_name=run_name, nested=nested)
        self.logger.info(f"Started MLflow run: {run_name}")
    
    def end_run(self) -> None:
        """End the current MLflow run"""
        if mlflow.active_run():
            mlflow.end_run()
            self.logger.info("Ended MLflow run")
    
    def log_parameters(self, params: Dict[str, Any]) -> None:
        """Log model parameters"""
        if mlflow.active_run():
            mlflow.log_params(params)
    
    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log model metrics"""
        if mlflow.active_run():
            mlflow.log_metrics(metrics)
    
    def log_model(self, model: Any, model_type: str) -> None:
        """Log the model"""
        if not mlflow.active_run():
            return
            
        if model_type == "random_forest":
            mlflow.sklearn.log_model(model, "model")
        elif model_type == "xgboost":
            mlflow.xgboost.log_model(model, "model")
            
    def create_performance_plots(self, 
                               all_metrics: Dict[str, Dict[str, float]], 
                               save_dir: Path) -> None:
        """Create and save performance comparison plots"""
        try:
            # Prepare data for plotting
            plot_data = []
            for model_type, metrics in all_metrics.items():
                for metric_name, value in metrics.items():
                    plot_data.append({
                        'Model': model_type,
                        'Metric': metric_name,
                        'Value': value
                    })
            df_plot = pd.DataFrame(plot_data)
            
            # Create performance comparison plot
            plt.figure(figsize=(12, 6))
            sns.barplot(data=df_plot[df_plot['Metric'].isin(['val_mse', 'val_r2'])],
                       x='Metric', y='Value', hue='Model')
            plt.title('Model Performance Comparison')
            plt.tight_layout()
            
            # Save plot
            plot_path = save_dir / 'performance_comparison.png'
            plt.savefig(plot_path)
            mlflow.log_artifact(str(plot_path))
            plt.close()
            
            # Create detailed metrics heatmap
            plt.figure(figsize=(10, 6))
            metrics_pivot = df_plot.pivot(index='Model', columns='Metric', values='Value')
            sns.heatmap(metrics_pivot, annot=True, fmt='.4f', cmap='YlOrRd')
            plt.title('Detailed Metrics Comparison')
            plt.tight_layout()
            
            # Save heatmap
            heatmap_path = save_dir / 'metrics_heatmap.png'
            plt.savefig(heatmap_path)
            mlflow.log_artifact(str(heatmap_path))
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error creating performance plots: {str(e)}")
            raise
            
    def log_feature_importance(self, 
                             model: Any, 
                             feature_names: list, 
                             model_type: str,
                             save_dir: Path) -> None:
        """Create and log feature importance plot"""
        try:
            plt.figure(figsize=(10, 6))
            
            if model_type == "random_forest":
                importances = model.feature_importances_
            elif model_type == "xgboost":
                importances = model.feature_importances_
                
            indices = np.argsort(importances)[::-1]
            
            plt.title(f"Feature Importances ({model_type})")
            plt.bar(range(len(importances)), importances[indices])
            plt.xticks(range(len(importances)), 
                      [feature_names[i] for i in indices], 
                      rotation=45, ha='right')
            plt.tight_layout()
            
            # Save plot
            importance_path = save_dir / f'{model_type}_feature_importance.png'
            plt.savefig(importance_path)
            mlflow.log_artifact(str(importance_path))
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error creating feature importance plot: {str(e)}")
            raise
            
    def end_run(self) -> None:
        """End the current MLflow run"""
        mlflow.end_run() 