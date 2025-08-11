import mlflow
import os
from pathlib import Path
from models.config import ModelConfig
from models.data_loader import StockDataLoader
from models.model import ModelTrainer
import logging

def test_mlflow_logging():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("TestMLflow")
    
    try:
        # Initialize configuration
        config = ModelConfig()
        
        # Set MLflow tracking URI explicitly
        mlflow_dir = Path("mlruns").absolute()
        mlflow.set_tracking_uri(f"file://{mlflow_dir}")
        logger.info(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
        
        # Create experiment
        experiment_name = "stock_prediction_test"
        mlflow.set_experiment(experiment_name)
        logger.info(f"MLflow experiment: {experiment_name}")
        
        # Load data
        data_loader = StockDataLoader(config.data_dir)
        data_dict = data_loader.load_data(symbols=["AAPL"])
        df = data_dict["AAPL"]
        
        # Split data
        train_data, val_data, test_data = data_loader.split_data(
            df,
            train_ratio=config.train_ratio,
            val_ratio=config.val_ratio
        )
        
        X_train, y_train = data_loader.get_features_and_target(train_data)
        X_val, y_val = data_loader.get_features_and_target(val_data)
        
        # Initialize model trainer
        model_trainer = ModelTrainer(config)
        
        # Train Random Forest
        logger.info("Training Random Forest...")
        rf_metrics = model_trainer.train_and_evaluate(
            X_train, y_train, X_val, y_val, "random_forest"
        )
        
        # Create and log plots
        save_dir = Path("model_results")
        save_dir.mkdir(exist_ok=True)
        
        model_trainer.experiment_tracker.create_performance_plots(
            {"random_forest": rf_metrics},
            save_dir
        )
        
        logger.info("MLflow test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error in MLflow test: {str(e)}")
        return False
    finally:
        # Ensure any active run is ended
        if mlflow.active_run():
            mlflow.end_run()

if __name__ == "__main__":
    success = test_mlflow_logging()
    print(f"MLflow test {'successful' if success else 'failed'}")
    
    if success:
        print("\nTo view results:")
        print("1. Run 'mlflow ui' in terminal")
        print("2. Open http://localhost:5000 in your browser")
        print(f"3. Look for experiment: stock_prediction_test") 