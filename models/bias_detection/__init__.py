from .slicer import DataSlicer
from .metrics_evaluator import MetricsEvaluator
from .bias_detector import BiasDetector
from .bias_mitigator import BiasMitigator
from .visualizer import BiasVisualizer
from .report_generator import BiasReportGenerator

class BiasAnalyzer:
    def __init__(self, config):
        self.config = config
        self.slicer = DataSlicer()
        self.metrics_evaluator = MetricsEvaluator()
        self.bias_detector = BiasDetector()
        self.bias_mitigator = BiasMitigator()
        self.visualizer = BiasVisualizer()
        self.report_generator = BiasReportGenerator()

    def analyze_bias(self, model, X, y, save_dir):
        """Perform complete bias analysis"""
        # Create data slices
        slices = self.slicer.create_slices(X)
        
        # Evaluate metrics for each slice
        metrics = self.metrics_evaluator.evaluate_slice_metrics(model, slices, y)
        
        # Detect bias
        biased_slices = self.bias_detector.detect_bias(metrics)
        
        # Generate visualizations
        self.visualizer.create_visualizations(metrics, save_dir)
        
        # Generate and save report
        report = self.report_generator.generate_report(metrics, biased_slices)
        self.report_generator.save_report(report, save_dir)
        
        return biased_slices, metrics

    def mitigate_bias(self, X, y, biased_slices):
        """Apply bias mitigation if needed"""
        slices = self.slicer.create_slices(X)
        return self.bias_mitigator.mitigate_bias(X, y, biased_slices, slices) 