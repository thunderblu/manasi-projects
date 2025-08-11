from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path
import os

@dataclass
class ModelConfig:
    # Data parameters
    data_dir: str = "data/mitigated"
    symbols: List[str] = ("AAPL", "GOOGL", "MSFT")
    target_col: str = "Close"
    feature_cols: Optional[List[str]] = None
    
    # Training parameters
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_state: int = 42
    
    # Model parameters
    model_type: str = "random_forest"  # Options: "random_forest", "xgboost"
    n_estimators: int = 100  # Number of trees in the forest
    max_depth: Optional[int] = None  # Maximum depth of the tree
    learning_rate: float = 0.1  # Only used for xgboost
    
    # Paths
    model_save_path: str = "models/saved_models"
    log_dir: str = "logs"
    
    # Logging parameters
    log_level: int = 20  # INFO level
    save_logs: bool = True
    
    # Get the absolute path to the project root directory
    project_root: Path = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # GCP settings
    gcp_project_id: str = "team02-441605"
    gcp_location: str = "us-central1"
    gcp_repository: str = "stock-models"
    
    # Use relative path for credentials
    @property
    def gcp_credentials_path(self) -> str:
        return str(self.project_root / "credentials" / "latest.json")
    
    # Hyperparameter tuning settings
    hyperparameter_tuning: bool = True
    n_cv_folds: int = 5
    n_trials: int = 50  # Number of trials for random search
    
    # Hyperparameter search spaces
    rf_param_grid = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }
    
    xgb_param_grid = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [3, 4, 5, 6, 7, 8],
        'learning_rate': [0.01, 0.05, 0.1, 0.15],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5]
    }
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        assert self.train_ratio + self.val_ratio + self.test_ratio == 1.0, \
            "Train, validation, and test ratios must sum to 1"
        
        # Create necessary directories using project_root
        Path(self.project_root / self.model_save_path).mkdir(parents=True, exist_ok=True)
        Path(self.project_root / self.log_dir).mkdir(parents=True, exist_ok=True)
