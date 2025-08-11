import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

class StockMonitor:
    def __init__(self, email_config: Dict[str, str], logger: logging.Logger):
        self.email_config = email_config
        self.logger = logger
        
    def check_data_quality(self, df: pd.DataFrame) -> List[str]:
        """Check for data quality issues"""
        issues = []
        
        # Check for missing values
        missing = df.isnull().sum()
        if missing.any():
            issues.append(f"Missing values detected: {missing[missing > 0].to_dict()}")
            
        # Check for duplicate indices
        duplicates = df.index.duplicated()
        if duplicates.any():
            issues.append(f"Duplicate timestamps detected: {df.index[duplicates].tolist()}")
            
        # Check for outliers using IQR
        for column in df.select_dtypes(include=[np.number]).columns:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[column] < Q1 - 1.5 * IQR) | (df[column] > Q3 + 1.5 * IQR)]
            if not outliers.empty:
                issues.append(f"Outliers detected in {column}: {len(outliers)} points")
        
        return issues
    
    def send_alert(self, subject: str, body: str):
        """Send email alert"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_config['sender']
            msg['To'] = self.email_config['recipient']
            msg['Subject'] = subject
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
            server.starttls()
            server.login(self.email_config['username'], self.email_config['password'])
            server.send_message(msg)
            server.quit()
            
            self.logger.info(f"Alert sent: {subject}")
            
        except Exception as e:
            self.logger.error(f"Failed to send alert: {str(e)}") 