import unittest
import os
import sys
from pathlib import Path

class TestMLPipeline(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment"""
        self.test_data = {
            'X_train': None,  # Add your test data
            'y_train': None,
            'X_val': None,
            'y_val': None
        }
    
    def test_model_validation(self):
        """Test model validation script"""
        sys.path.append('scripts')
        from validate_model import ModelValidator
        
        validator = ModelValidator()
        success, results = validator.validate_latest_model()
        self.assertIsNotNone(results)
    
    def test_bias_detection(self):
        """Test bias detection script"""
        sys.path.append('scripts')
        from check_bias import BiasDetector
        
        detector = BiasDetector()
        success, results = detector.check_model_bias()
        self.assertIsNotNone(results)
    
    def test_model_deployment(self):
        """Test model deployment script"""
        sys.path.append('scripts')
        from deploy_model import ModelDeployer
        
        deployer = ModelDeployer()
        success = deployer.push_to_registry()
        self.assertTrue(success)
    
    def test_notifications(self):
        """Test notification system"""
        sys.path.append('scripts')
        from send_notifications import PipelineNotifier
        
        notifier = PipelineNotifier()
        status = {
            'training_passed': True,
            'validation_passed': True,
            'bias_check_passed': True,
            'deployment_passed': True
        }
        success = notifier.send_email_notification(status)
        self.assertTrue(success)
    
    def test_rollback(self):
        """Test rollback mechanism"""
        sys.path.append('scripts')
        from rollback_model import ModelRollback
        
        rollback = ModelRollback()
        success = rollback.perform_rollback()
        self.assertIsNotNone(success)

if __name__ == '__main__':
    unittest.main() 