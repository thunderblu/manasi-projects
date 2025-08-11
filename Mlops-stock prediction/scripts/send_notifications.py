import smtplib
import yaml
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PipelineNotifier:
    def __init__(self, config_path: str = 'config/pipeline_config.yml'):
        self.config = self._load_config(config_path)
        self.notification_config = self.config['notifications']
        self.email_config = self.notification_config['email']
        
    def _load_config(self, config_path: str) -> dict:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _create_email_content(self, status: Dict) -> str:
        """Create email content based on pipeline status"""
        content = []
        content.append("ML Pipeline Status Update\n")
        content.append("=" * 30 + "\n")
        
        # Training Status
        content.append(f"Model Training: {'✅ Passed' if status.get('training_passed') else '❌ Failed'}")
        if 'training_metrics' in status:
            content.append("Training Metrics:")
            for metric, value in status['training_metrics'].items():
                content.append(f"- {metric}: {value:.4f}")
        
        # Validation Status
        content.append(f"\nModel Validation: {'✅ Passed' if status.get('validation_passed') else '❌ Failed'}")
        if 'validation_metrics' in status:
            content.append("Validation Metrics:")
            for metric, value in status['validation_metrics'].items():
                content.append(f"- {metric}: {value:.4f}")
        
        # Bias Check Status
        content.append(f"\nBias Check: {'✅ Passed' if status.get('bias_check_passed') else '❌ Failed'}")
        if 'biased_slices' in status:
            content.append("Biased Slices:")
            for slice_name in status['biased_slices']:
                content.append(f"- {slice_name}")
        
        # Deployment Status
        content.append(f"\nModel Deployment: {'✅ Passed' if status.get('deployment_passed') else '❌ Failed'}")
        if 'model_registry_info' in status:
            content.append("Registry Information:")
            content.append(f"- Model ID: {status['model_registry_info'].get('model_id', 'N/A')}")
            content.append(f"- Version: {status['model_registry_info'].get('version', 'N/A')}")
        
        return "\n".join(content)
    
    def send_email_notification(self, status: Dict) -> bool:
        """Send email notification about pipeline status"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_config['sender']
            msg['To'] = 'ananthareddy6060@gmail.com'  # Your email address
            
            # Set subject based on overall status
            overall_success = all([
                status.get('training_passed', False),
                status.get('validation_passed', False),
                status.get('bias_check_passed', False),
                status.get('deployment_passed', False)
            ])
            
            msg['Subject'] = f"ML Pipeline {'Success ✅' if overall_success else 'Failure ❌'}"
            
            # Create email content
            body = self._create_email_content(status)
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email using SMTP
            with smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port']) as server:
                server.starttls()
                server.login(self.email_config['username'], self.email_config['password'])
                server.send_message(msg)
            
            logger.info(f"Successfully sent notification email to ananthareddy6060@gmail.com")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email notification: {str(e)}")
            return False

def main():
    # Get pipeline status from environment or files
    status = {
        'training_passed': True,  # Replace with actual status
        'validation_passed': True,
        'bias_check_passed': True,
        'deployment_passed': True,
        'training_metrics': {
            'mse': 0.001,
            'mae': 0.02
        },
        'validation_metrics': {
            'mse': 0.0012,
            'mae': 0.022
        },
        'biased_slices': [],
        'model_registry_info': {
            'model_id': 'model-123',
            'version': 'v1.0'
        }
    }
    
    notifier = PipelineNotifier()
    success = notifier.send_email_notification(status)
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 