from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable
from src.data_acquisition import StockDataAcquisition
from src.data_preprocessing import StockDataPreprocessor
from src.utils.logging_config import setup_logger
from src.utils.monitoring import StockMonitor
from src.data_validation import DataValidator
from src.anomaly_detection import AnomalyDetector
from src.bias_detection import MarketBiasDetector
from src.slice_analysis import SliceAnalyzer
from src.bias_mitigation import BiasMitigation
import os
import pandas as pd
import json
import logging

# Set up logging
logger = setup_logger('stock_prediction_pipeline')

# Email configuration using environment variables with default values
email_config = {
    'smtp_server': os.getenv('AIRFLOW_VAR_smtp_server', 'smtp.gmail.com'),
    'smtp_port': os.getenv('AIRFLOW_VAR_smtp_port', '587'),
    'sender': os.getenv('AIRFLOW_VAR_alert_sender_email', 'ananthareddy12321@gmail.com'),
    'recipient': os.getenv('AIRFLOW_VAR_alert_recipient_email', 'dheerajkumar.1379@gmail.com'),
    'username': os.getenv('AIRFLOW_VAR_smtp_username', 'ananthareddy12321@gmail.com'),
    'password': os.getenv('AIRFLOW_VAR_smtp_password', 'jaei dczj aokn fcaf')
}

# Initialize monitor
monitor = StockMonitor(email_config, logger)

# Initialize validator
validator = DataValidator()

# Initialize anomaly detector
anomaly_detector = AnomalyDetector()

# Initialize market bias detector
bias_detector = MarketBiasDetector()

# Initialize slice analyzer
slice_analyzer = SliceAnalyzer()

def acquire_stock_data(**context):
    """Task to acquire stock data"""
    logger.info("Starting data acquisition")
    try:
        symbols = ['AAPL', 'GOOGL', 'MSFT']
        data_acquirer = StockDataAcquisition(output_dir='/opt/airflow/data/raw')
        
        for symbol in symbols:
            logger.info(f"Fetching data for {symbol}")
            df = data_acquirer.fetch_stock_data(symbol)
            
            # Check data quality
            issues = monitor.check_data_quality(df)
            if issues:
                alert_msg = f"Data quality issues for {symbol}:\n" + "\n".join(issues)
                monitor.send_alert(
                    f"Data Quality Alert - {symbol}",
                    alert_msg
                )
                logger.warning(alert_msg)
            
            logger.info(f"Successfully acquired data for {symbol}")
            
    except Exception as e:
        error_msg = f"Failed to acquire data: {str(e)}"
        logger.error(error_msg)
        monitor.send_alert("Data Acquisition Failed", error_msg)
        raise

def validate_stock_data(**context):
    """Task to validate stock data at different stages"""
    logger.info("Starting data validation")
    try:
        preprocessor = StockDataPreprocessor(
            input_dir='/opt/airflow/data/raw',
            output_dir='/opt/airflow/data/processed'
        )
        
        for symbol in ['AAPL', 'GOOGL', 'MSFT']:
            logger.info(f"Validating {symbol} data")
            
            # Validate raw data
            raw_path = os.path.join(preprocessor.input_dir, f"{symbol}.csv")
            if os.path.exists(raw_path):
                try:
                    # Add error handling for CSV reading
                    raw_df = pd.read_csv(raw_path, on_bad_lines='skip')
                    issues = validator.validate_data(raw_df, symbol, 'raw')
                    if issues:
                        monitor.send_alert(f"Data Quality Alert - {symbol} (Raw)", "\n".join(issues))
                except Exception as e:
                    error_msg = f"Error reading raw data for {symbol}: {str(e)}"
                    logger.error(error_msg)
                    monitor.send_alert(f"Data Reading Error - {symbol} (Raw)", error_msg)
                    continue
            
            # Validate processed data
            processed_path = os.path.join(preprocessor.output_dir, f"{symbol}.csv")
            if os.path.exists(processed_path):
                try:
                    processed_df = pd.read_csv(processed_path, on_bad_lines='skip')
                    issues = validator.validate_data(processed_df, symbol, 'processed')
                    if issues:
                        monitor.send_alert(f"Data Quality Alert - {symbol} (Processed)", "\n".join(issues))
                except Exception as e:
                    error_msg = f"Error reading processed data for {symbol}: {str(e)}"
                    logger.error(error_msg)
                    monitor.send_alert(f"Data Reading Error - {symbol} (Processed)", error_msg)
                    continue
            
            logger.info(f"Completed validation for {symbol}")
            
    except Exception as e:
        error_msg = f"Validation failed: {str(e)}"
        logger.error(error_msg)
        monitor.send_alert("Validation Failed", error_msg)
        raise

def preprocess_stock_data(**context):
    """Task to preprocess stock data"""
    logger.info("Starting data preprocessing with enhanced features")
    try:
        preprocessor = StockDataPreprocessor(
            input_dir='/opt/airflow/data/raw',
            output_dir='/opt/airflow/data/processed'
        )
        
        for symbol in ['AAPL', 'GOOGL', 'MSFT']:
            logger.info(f"Processing {symbol} data with global market features")
            
            # Load and process data
            df = preprocessor.load_data(f"{symbol}.csv")
            df_cleaned = preprocessor.clean_data(df)
            df_featured = preprocessor.engineer_features(df_cleaned)
            
            # Log the new features
            logger.info(f"Added features for {symbol}: {list(df_featured.columns)}")
            
            # Continue with scaling and saving
            df_processed, _ = preprocessor.scale_features(df_featured)
            output_path = os.path.join(preprocessor.output_dir, f"{symbol}.csv")
            df_processed.to_csv(output_path)
            
            logger.info(f"Successfully processed {symbol} data with enhanced features")
            
    except Exception as e:
        error_msg = f"Failed to process data: {str(e)}"
        logger.error(error_msg)
        monitor.send_alert("Data Processing Failed", error_msg)
        raise

def detect_anomalies(**context):
    """Task to detect anomalies and clean stock data"""
    logger.info("Starting anomaly detection and cleaning")
    try:
        preprocessor = StockDataPreprocessor(
            input_dir='/opt/airflow/data/raw',
            output_dir='/opt/airflow/data/processed'
        )
        
        for symbol in ['AAPL', 'GOOGL', 'MSFT']:
            logger.info(f"Processing anomalies for {symbol}")
            
            # Load processed data
            processed_path = os.path.join(preprocessor.output_dir, f"{symbol}.csv")
            if os.path.exists(processed_path):
                df = pd.read_csv(processed_path)
                df['Date'] = pd.to_datetime(df['Date'], utc=True)
                df.set_index('Date', inplace=True)
                
                # 1. Detect and handle outliers
                outliers = anomaly_detector.detect_outliers(df)
                if outliers:
                    alert_msg = "Outliers detected:\n"
                    for col, values in outliers.items():
                        alert_msg += f"\n{col}:\n"
                        for v in values[:5]:
                            alert_msg += f"- {v['timestamp']}: {v['value']:.2f} (z-score: {v['zscore']:.2f})\n"
                    monitor.send_alert(f"Outlier Alert - {symbol}", alert_msg)
                    
                    # Handle outliers
                    df_cleaned, outlier_report = anomaly_detector.handle_outliers(df, method='winsorize')
                    logger.info(f"Outlier handling report for {symbol}:\n{json.dumps(outlier_report, indent=2)}")
                
                # 2. Detect and handle pattern anomalies
                pattern_anomalies = anomaly_detector.detect_pattern_anomalies(df_cleaned)
                if pattern_anomalies:
                    alert_msg = "Pattern anomalies detected:\n"
                    for anomaly in pattern_anomalies[:5]:
                        alert_msg += f"- {anomaly['type']} at {anomaly['timestamp']}: {anomaly['value']:.2f}\n"
                    monitor.send_alert(f"Pattern Anomaly Alert - {symbol}", alert_msg)
                    
                    # Handle pattern anomalies
                    df_cleaned, pattern_report = anomaly_detector.handle_pattern_anomalies(df_cleaned)
                    logger.info(f"Pattern anomaly handling report for {symbol}:\n{json.dumps(pattern_report, indent=2)}")
                
                # 3. Check and handle data quality issues
                quality_issues = anomaly_detector.check_data_quality(df_cleaned)
                if quality_issues:
                    alert_msg = "Data quality issues detected:\n" + "\n".join(quality_issues)
                    monitor.send_alert(f"Data Quality Alert - {symbol}", alert_msg)
                    
                    # Handle missing data
                    df_cleaned, missing_report = anomaly_detector.handle_missing_data(df_cleaned)
                    logger.info(f"Missing data handling report for {symbol}:\n{json.dumps(missing_report, indent=2)}")
                
                # 4. Run complete cleaning pipeline
                final_df, cleaning_report = anomaly_detector.clean_data(df_cleaned, outlier_method='winsorize')
                
                # Save cleaned data
                cleaned_path = os.path.join(preprocessor.output_dir, f"{symbol}_cleaned.csv")
                final_df.to_csv(cleaned_path)
                
                # Log cleaning summary
                logger.info(f"Cleaning summary for {symbol}:")
                logger.info(f"- Original shape: {df.shape}")
                logger.info(f"- Cleaned shape: {final_df.shape}")
                logger.info(f"- Cleaning report:\n{json.dumps(cleaning_report, indent=2)}")
            
            logger.info(f"Completed anomaly detection and cleaning for {symbol}")
            
    except Exception as e:
        error_msg = f"Anomaly detection and cleaning failed: {str(e)}"
        logger.error(error_msg)
        monitor.send_alert("Anomaly Detection Failed", error_msg)
        raise

def detect_market_bias(**context):
    """Task to detect market bias in stock data"""
    logger.info("Starting market bias detection")
    try:
        bias_detector = MarketBiasDetector()
        
        for symbol in ['AAPL', 'GOOGL', 'MSFT']:
            logger.info(f"Detecting bias for {symbol}")
            
            # Read cleaned data
            cleaned_path = f'/opt/airflow/data/processed/{symbol}_cleaned.csv'
            df = pd.read_csv(cleaned_path, parse_dates=['Date'], index_col='Date')
            
            # Detect bias
            bias_report = bias_detector.detect_bias(df, symbol)
            
            # Log findings
            if bias_report['bias_indicators']:
                logger.warning(f"Bias detected for {symbol}:")
                for indicator in bias_report['bias_indicators']:
                    logger.warning(f"- {indicator}")
                    
            if bias_report['recommendations']:
                logger.info(f"Recommendations for {symbol}:")
                for rec in bias_report['recommendations']:
                    logger.info(f"- {rec}")
            
            logger.info(f"Completed bias detection for {symbol}")
                    
    except Exception as e:
        error_msg = f"Market bias detection failed: {str(e)}"
        logger.error(error_msg)
        raise

def analyze_data_slices(**context):
    """Task to analyze data slices"""
    logger.info("Starting slice analysis")
    try:
        for symbol in ['AAPL', 'GOOGL', 'MSFT']:
            logger.info(f"Analyzing slices for {symbol}")
            
            # Read cleaned data
            cleaned_path = f'/opt/airflow/data/processed/{symbol}_cleaned.csv'
            df = pd.read_csv(cleaned_path, parse_dates=['Date'], index_col='Date')
            
            # Analyze slices
            slice_report = slice_analyzer.analyze_slices(df, symbol)
            
            # Log findings
            if slice_report['recommendations']:
                logger.info(f"Recommendations for {symbol}:")
                for rec in slice_report['recommendations']:
                    logger.info(f"- {rec}")
            
            logger.info(f"Completed slice analysis for {symbol}")
                    
    except Exception as e:
        error_msg = f"Slice analysis failed: {str(e)}"
        logger.error(error_msg)
        raise

def mitigate_bias(**context):
    """Apply bias mitigation techniques"""
    try:
        logger = logging.getLogger(__name__)
        mitigator = BiasMitigation(logger=logger)
        
        # Create mitigated directory if it doesn't exist
        mitigated_dir = '/opt/airflow/data/mitigated'
        os.makedirs(mitigated_dir, exist_ok=True)
        logger.info(f"Created directory: {mitigated_dir}")
        
        mitigated_data = {}
        for stock_symbol in ['AAPL', 'GOOGL', 'MSFT']:
            logger.info(f"Applying bias mitigation for {stock_symbol}")
            
            # Read the cleaned data directly
            input_path = f'/opt/airflow/data/processed/{stock_symbol}_cleaned.csv'
            df = pd.read_csv(input_path, parse_dates=['Date'], index_col='Date')
            
            # Apply mitigation
            mitigated_df = mitigator.mitigate_bias(df, {})
            
            # Save mitigated data
            output_path = f'{mitigated_dir}/{stock_symbol}_mitigated.csv'
            mitigated_df.to_csv(output_path)
            logger.info(f"Saved mitigated data to {output_path}")
            
            mitigated_data[stock_symbol] = output_path
            
        # Push paths to mitigated data files
        ti = context['task_instance']
        ti.xcom_push(key='mitigated_data_paths', value=mitigated_data)
        logger.info("Bias mitigation completed for all stocks")
        
    except Exception as e:
        logger.error(f"Error in bias mitigation: {str(e)}")
        raise

# Define DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'stock_prediction_pipeline',
    default_args=default_args,
    description='Stock prediction pipeline with monitoring',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2024, 1, 1),
    catchup=False
)

# Define tasks
acquire_data_task = PythonOperator(
    task_id='acquire_stock_data',
    python_callable=acquire_stock_data,
    provide_context=True,
    dag=dag,
)

preprocess_data_task = PythonOperator(
    task_id='preprocess_stock_data',
    python_callable=preprocess_stock_data,
    provide_context=True,
    dag=dag,
)

validate_data_task = PythonOperator(
    task_id='validate_stock_data',
    python_callable=validate_stock_data,
    provide_context=True,
    dag=dag,
)

detect_anomalies_task = PythonOperator(
    task_id='detect_anomalies',
    python_callable=detect_anomalies,
    provide_context=True,
    dag=dag,
)

detect_market_bias_task = PythonOperator(
    task_id='detect_market_bias',
    python_callable=detect_market_bias,
    provide_context=True,
    dag=dag,
)

analyze_slices_task = PythonOperator(
    task_id='analyze_data_slices',
    python_callable=analyze_data_slices,
    provide_context=True,
    dag=dag,
)

mitigate_bias_task = PythonOperator(
    task_id='mitigate_bias',
    python_callable=mitigate_bias,
    provide_context=True,
    dag=dag
)

# Set task dependencies
acquire_data_task >> preprocess_data_task >> validate_data_task >> \
    detect_anomalies_task >> detect_market_bias_task >> analyze_slices_task >> \
    mitigate_bias_task