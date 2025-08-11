from models.data_loader import StockDataLoader
from models.config import ModelConfig
from models.model import ModelTrainer
from models.model_validator import ModelValidator
from models.bias_checker import BiasChecker
from models.bias_detection import BiasAnalyzer
from models.model_selector import ModelSelector
from models.utils.logger import setup_logger
from pathlib import Path
from datetime import datetime
from models.model_registry import ModelRegistry as GCPModelRegistry
import os
from models.experiment_tracker import ExperimentTracker
import mlflow
from models.sensitivity_analyzer import SensitivityAnalyzer

def verify_credentials():
    """Verify GCP credentials exist and are accessible"""
    # Get the project root directory (where your main project folder is)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Construct the path to credentials relative to project root
    credentials_path = os.path.join(project_root, "credentials", "latest.json")
    
    if not os.path.exists(credentials_path):
        raise FileNotFoundError(
            f"GCP credentials not found at {credentials_path}. "
            "Please ensure the credentials file is in the correct location."
        )
    return credentials_path

def main():
    logger = setup_logger("TrainingPipeline")
    config = ModelConfig()
    
    try:
        # Verify credentials
        verify_credentials()
        
        # Set GCP credentials
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = config.gcp_credentials_path
        logger.info("GCP credentials loaded successfully")
        
        # Initialize model registry
        model_registry = GCPModelRegistry(config)
        
        # Set up MLflow experiment
        mlflow_dir = Path(config.project_root) / "mlruns"
        mlflow.set_tracking_uri(f"file://{mlflow_dir.absolute()}")
        experiment_name = "stock_prediction_production"
        mlflow.set_experiment(experiment_name)
        logger.info(f"MLflow experiment: {experiment_name}")
        
        data_loader = StockDataLoader(config.data_dir)
        data_dict = data_loader.load_data(config.symbols)
        
        for symbol, df in data_dict.items():
            logger.info(f"Processing {symbol}")
            
            with mlflow.start_run(run_name=f"full_pipeline_{symbol}") as parent_run:
                # Create save directory at the beginning
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_dir = Path(config.model_save_path) / symbol / timestamp
                save_dir.mkdir(parents=True, exist_ok=True)
                
                # Log general parameters
                mlflow.log_params({
                    "symbol": symbol,
                    "train_ratio": config.train_ratio,
                    "val_ratio": config.val_ratio,
                    "random_state": config.random_state
                })
                
                # Data splitting and feature extraction
                train_data, val_data, test_data = data_loader.split_data(
                    df,
                    train_ratio=config.train_ratio,
                    val_ratio=config.val_ratio
                )
                
                X_train, y_train = data_loader.get_features_and_target(train_data)
                X_val, y_val = data_loader.get_features_and_target(val_data)
                X_test, y_test = data_loader.get_features_and_target(test_data)
                
                # Train models
                model_trainer = ModelTrainer(config)
                validation_metrics = {}
                bias_metrics = {}
                
                # Initialize both bias checkers
                bias_checker = BiasChecker(config)
                bias_analyzer = BiasAnalyzer(config)
                
                for model_type in ["random_forest", "xgboost"]:
                    logger.info(f"Training {model_type}")
                    metrics = model_trainer.train_and_evaluate(
                        X_train, y_train, X_val, y_val, model_type
                    )
                    validation_metrics[model_type] = metrics
                    
                    # Use both bias checking methods
                    # 1. Original bias checking
                    bias_metrics[model_type] = bias_checker.check_bias(
                        model_trainer.models[model_type],
                        X_val,
                        y_val,
                        X_val['volatility']
                    )
                    
                    # 2. New detailed bias analysis
                    try:
                        biased_slices, slice_metrics = bias_analyzer.analyze_bias(
                            model_trainer.models[model_type],
                            X_val,  # Features
                            y_val,  # Target
                            save_dir / model_type
                        )
                        
                        if biased_slices:
                            logger.warning(f"Detected bias in {len(biased_slices)} slices for {model_type}")
                            X_train_reweighted, y_train_reweighted = bias_analyzer.mitigate_bias(
                                X_train,
                                y_train,
                                biased_slices
                            )
                            # Retrain if necessary
                            model_trainer.train(X_train_reweighted, y_train_reweighted, model_type)
                    except Exception as e:
                        logger.warning(f"Advanced bias analysis failed: {str(e)}. Continuing with basic bias checking.")
                    
                    # Continue with existing model selection code ...
                
                # Initialize sensitivity analyzer
                sensitivity_analyzer = SensitivityAnalyzer(config)
                
                # Perform sensitivity analysis for the trained models
                for model_type in ["random_forest", "xgboost"]:
                    with mlflow.start_run(run_name=f"sensitivity_analysis_{model_type}", nested=True):
                        logger.info(f"Performing sensitivity analysis for {model_type}")
                        
                        # 1. Feature importance using SHAP
                        feature_importance = sensitivity_analyzer.analyze_feature_importance(
                            model_trainer.models[model_type],
                            X_train,
                            X_test,
                            feature_names=X_train.columns.tolist()
                        )
                        
                        # 2. LIME explanation
                        sensitivity_analyzer.analyze_lime(
                            model_trainer.models[model_type],
                            X_train,
                            X_test,
                            feature_names=X_train.columns.tolist()
                        )
                        
                        # 3. Hyperparameter sensitivity (your existing code)
                        if model_type == "random_forest":
                            param_ranges = {
                                'n_estimators': [100, 200, 300, 400, 500],
                                'max_depth': [3, 4, 5, 6, 7, 8],
                                'min_samples_split': [2, 5, 10]
                            }
                        else:  # xgboost
                            param_ranges = {
                                'n_estimators': [100, 200, 300, 400, 500],
                                'max_depth': [3, 4, 5, 6, 7, 8],
                                'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2]
                            }
                        
                        sensitivity_results = sensitivity_analyzer.analyze_hyperparameter_sensitivity(
                            model_trainer,
                            X_train,
                            y_train,
                            model_type,
                            param_ranges
                        )
                        
                        # Log all sensitivity analysis artifacts
                        mlflow.log_artifact('feature_importance_shap.png')
                        mlflow.log_artifact('lime_explanation.html')
                        
                        # Log existing sensitivity results
                        for result in sensitivity_results:
                            for value, score in zip(result['values'], result['scores']):
                                mlflow.log_metric(
                                    f"{model_type}_{result['parameter']}_sensitivity_{value}", 
                                    score
                                )
                        
                        logger.info(f"Completed sensitivity analysis for {model_type}")
                
                # Select best model
                model_selector = ModelSelector(config)
                try:
                    best_model, combined_scores = model_selector.select_model(
                        validation_metrics, bias_metrics
                    )
                    
                    # Log selection results
                    with mlflow.start_run(run_name=f"model_selection", nested=True):
                        mlflow.log_param("best_model", best_model)
                        for model_name, score in combined_scores.items():
                            mlflow.log_metric(f"combined_score_{model_name}", score)
                    
                    # Save model selection report
                    model_selector.save_selection_report(save_dir, best_model, combined_scores)
                    
                    # Create and save visualizations
                    bias_checker.visualize_bias(save_dir)
                    bias_checker.save_bias_report(save_dir)
                    
                    # Log artifacts
                    mlflow.log_artifacts(str(save_dir), "results")
                    
                    # Save model and continue with existing GCP logic
                    model_trainer.save_model(symbol, best_model)
                    
                    # Push to GCP
                    version = model_registry.push_model(
                        model_path=str(save_dir),
                        model_name=f"{symbol.lower()}_stock_predictor",
                        version=timestamp
                    )
                    
                    logger.info(f"Model pushed to registry with version: {version}")
                    logger.info(f"Completed processing for {symbol}")
                    
                except Exception as e:
                    logger.error(f"Error in model selection: {str(e)}")
                    raise
                
                # Create and log performance plots at the end of parent run
                experiment_tracker = ExperimentTracker(config)
                experiment_tracker.create_performance_plots(
                    validation_metrics,
                    save_dir
                )
                
    except Exception as e:
        logger.error(f"Error in training pipeline: {str(e)}")
        raise
    finally:
        while mlflow.active_run():
            mlflow.end_run()

if __name__ == "__main__":
    main()
