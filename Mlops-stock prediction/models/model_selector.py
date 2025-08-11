from typing import Dict, Tuple, Any
from models.utils.logger import setup_logger

class ModelSelector:
    def __init__(self, config):
        self.config = config
        self.logger = setup_logger("ModelSelector")
        
    def select_model(
        self,
        validation_metrics: Dict[str, Dict[str, float]],
        bias_metrics: Dict[str, Dict[str, float]]
    ) -> Tuple[str, Dict[str, float]]:
        """
        Select the best model based on validation metrics and bias scores.
        
        Args:
            validation_metrics: Dictionary of model validation metrics
            bias_metrics: Dictionary of model bias metrics
            
        Returns:
            Tuple of (best_model_name, combined_scores)
        """
        try:
            combined_scores = {}
            
            for model_name in validation_metrics.keys():
                # Get validation score (using MSE)
                val_score = 1 - validation_metrics[model_name]['val_mse']
                
                # Get bias score (using overall bias)
                bias_score = 1 - bias_metrics[model_name]['overall_bias']  # Changed from 'overall' to 'overall_bias'
                
                # Combine scores (you can adjust weights as needed)
                combined_scores[model_name] = 0.7 * val_score + 0.3 * bias_score
            
            # Select best model
            best_model = max(combined_scores.items(), key=lambda x: x[1])[0]
            
            self.logger.info(f"Selected model: {best_model}")
            self.logger.info(f"Model scores: {combined_scores}")
            
            return best_model, combined_scores
            
        except Exception as e:
            self.logger.error(f"Error in model selection: {str(e)}")
            raise
            
    def save_selection_report(self, save_dir, best_model: str, scores: Dict[str, float]) -> None:
        """Save model selection report"""
        try:
            report_path = save_dir / "model_selection_report.txt"
            
            with open(report_path, 'w') as f:
                f.write("Model Selection Report\n")
                f.write("====================\n\n")
                f.write(f"Selected Model: {best_model}\n\n")
                f.write("Model Scores:\n")
                for model, score in scores.items():
                    f.write(f"{model}: {score:.4f}\n")
                    
            self.logger.info(f"Selection report saved to {report_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving selection report: {str(e)}")
            raise 