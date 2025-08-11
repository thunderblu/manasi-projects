import json
import numpy as np
from pathlib import Path
from typing import Dict, List
import logging

class BiasReportGenerator:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def generate_report(self, metrics: Dict[str, Dict[str, float]], 
                       biased_slices: List[Dict]) -> Dict:
        """Generate a comprehensive bias analysis report"""
        try:
            report = {
                'overall_metrics': {
                    'mean_mse': np.mean([m['mse'] for m in metrics.values()]),
                    'std_mse': np.std([m['mse'] for m in metrics.values()]),
                    'total_samples': sum(m['size'] for m in metrics.values())
                },
                'slice_metrics': metrics,
                'detected_bias': {
                    'num_biased_slices': len(biased_slices),
                    'biased_slices': biased_slices
                },
                'recommendations': []
            }
            
            # Add recommendations based on findings
            if biased_slices:
                report['recommendations'].extend([
                    "Consider collecting more data for underperforming slices",
                    "Implement bias mitigation techniques such as reweighting or resampling",
                    "Monitor these slices closely in production"
                ])
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error in generating report: {str(e)}")
            raise

    def save_report(self, report: Dict, save_dir: Path):
        """Save the bias analysis report"""
        try:
            with open(save_dir / 'bias_report.json', 'w') as f:
                json.dump(report, f, indent=4)
            self.logger.info(f"Saved bias report to {save_dir}")
            
        except Exception as e:
            self.logger.error(f"Error in saving report: {str(e)}")
            raise 