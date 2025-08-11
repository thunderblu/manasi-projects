import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict
import logging

class BiasVisualizer:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def create_visualizations(self, metrics: Dict[str, Dict[str, float]], 
                            save_dir: Path):
        """Create visualizations of bias analysis"""
        try:
            plt.figure(figsize=(12, 6))
            
            # Plot MSE across slices
            slice_names = list(metrics.keys())
            mse_values = [m['mse'] for m in metrics.values()]
            
            plt.subplot(1, 2, 1)
            sns.barplot(x=mse_values, y=slice_names)
            plt.title('MSE Across Different Slices')
            plt.xlabel('Mean Squared Error')
            
            # Plot relative sample sizes
            sizes = [m['size'] for m in metrics.values()]
            
            plt.subplot(1, 2, 2)
            sns.barplot(x=[s/sum(sizes) for s in sizes], y=slice_names)
            plt.title('Relative Size of Each Slice')
            plt.xlabel('Proportion of Dataset')
            
            plt.tight_layout()
            plt.savefig(save_dir / 'bias_analysis.png')
            plt.close()
            
            self.logger.info(f"Saved bias visualizations to {save_dir}")
            
        except Exception as e:
            self.logger.error(f"Error in creating visualizations: {str(e)}")
            raise 