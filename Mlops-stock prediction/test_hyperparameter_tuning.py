from models.config import ModelConfig
from models.data_loader import StockDataLoader
from models.model import ModelTrainer
import pandas as pd
import logging

def test_hyperparameter_tuning():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("TestHyperparameterTuning")
    
    try:
        # Initialize configuration
        config = ModelConfig()
        config.hyperparameter_tuning = True  # Enable tuning
        config.n_trials = 5  # Reduce trials for testing
        config.n_cv_folds = 3  # Reduce folds for testing
        
        # Load data
        logger.info("Loading data...")
        data_loader = StockDataLoader(config.data_dir)
        data_dict = data_loader.load_data(symbols=["AAPL"])  # Test with single stock
        
        # Process single stock
        df = data_dict["AAPL"]
        
        # Split data
        train_data, val_data, test_data = data_loader.split_data(
            df,
            train_ratio=config.train_ratio,
            val_ratio=config.val_ratio
        )
        
        # Get features and target
        X_train, y_train = data_loader.get_features_and_target(train_data)
        X_val, y_val = data_loader.get_features_and_target(val_data)
        
        # Initialize model trainer
        logger.info("Initializing model trainer...")
        model_trainer = ModelTrainer(config)
        
        # Test Random Forest tuning
        logger.info("Testing Random Forest hyperparameter tuning...")
        rf_metrics = model_trainer.train_and_evaluate(
            X_train, y_train, X_val, y_val, "random_forest"
        )
        logger.info(f"Random Forest metrics: {rf_metrics}")
        
        # Test XGBoost tuning
        logger.info("Testing XGBoost hyperparameter tuning...")
        xgb_metrics = model_trainer.train_and_evaluate(
            X_train, y_train, X_val, y_val, "xgboost"
        )
        logger.info(f"XGBoost metrics: {xgb_metrics}")
        
        # Print best parameters
        logger.info("Best parameters:")
        logger.info(f"Random Forest: {model_trainer.hyperparameter_tuner.best_params.get('random_forest')}")
        logger.info(f"XGBoost: {model_trainer.hyperparameter_tuner.best_params.get('xgboost')}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in hyperparameter tuning test: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_hyperparameter_tuning()
    print(f"Test {'successful' if success else 'failed'}") 