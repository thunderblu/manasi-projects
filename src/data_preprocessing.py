import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import logging
from typing import List, Optional, Tuple
import os
import yfinance as yf

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StockDataPreprocessor:
    def __init__(self, input_dir: str = 'data/raw', output_dir: str = 'data/processed'):
        """
        Initialize the preprocessor with input and output directories
        
        Args:
            input_dir (str): Directory containing raw data
            output_dir (str): Directory to save processed data
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.scaler = MinMaxScaler()
        os.makedirs(output_dir, exist_ok=True)
        
    def load_data(self, filename: str) -> pd.DataFrame:
        """Load data from CSV file"""
        try:
            # Read CSV with more flexible parsing
            df = pd.read_csv(
                os.path.join(self.input_dir, filename),
                parse_dates=['Date'],
                index_col='Date'
            )
            
            # Ensure index is datetime type and convert to UTC
            df.index = pd.to_datetime(df.index, utc=True).tz_convert(None)
            
            # Ensure we have the required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Select only the columns we need
            df = df[required_columns]
            
            logger.info(f"Successfully loaded data with shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the dataframe by handling missing values and outliers
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        try:
            df_cleaned = df.copy()
            
            # Remove any duplicate indices
            df_cleaned = df_cleaned[~df_cleaned.index.duplicated(keep='first')]
            
            # Sort by date
            df_cleaned.sort_index(inplace=True)
            
            # Handle missing values using newer methods
            df_cleaned = df_cleaned.ffill().bfill()
            
            # Remove outliers using IQR method
            for column in df_cleaned.columns:
                Q1 = df_cleaned[column].quantile(0.25)
                Q3 = df_cleaned[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df_cleaned[column] = df_cleaned[column].clip(lower=lower_bound, upper=upper_bound)
            
            logger.info(f"Successfully cleaned data with shape: {df_cleaned.shape}")
            return df_cleaned
            
        except Exception as e:
            logger.error(f"Error cleaning data: {str(e)}")
            raise

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create technical indicators and additional features
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with additional features
        """
        try:
            df_featured = df.copy()
            
            # Ensure index is datetime
            if not isinstance(df_featured.index, pd.DatetimeIndex):
                df_featured.index = pd.to_datetime(df_featured.index)
            
            # Price-based features
            df_featured['Returns'] = df_featured['Close'].pct_change()
            df_featured['Open_Close_Ret'] = (df_featured['Close'] - df_featured['Open']) / df_featured['Open']
            df_featured['High_Low_Range'] = (df_featured['High'] - df_featured['Low']) / df_featured['Low']
            
            # Moving averages
            df_featured['MA5'] = df_featured['Close'].rolling(window=5, min_periods=1).mean()
            df_featured['MA20'] = df_featured['Close'].rolling(window=20, min_periods=1).mean()
            df_featured['MA_Ratio'] = df_featured['MA5'] / df_featured['MA20']
            
            # RSI
            delta = df_featured['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            rs = gain / loss
            df_featured['RSI'] = 100 - (100 / (1 + rs))
            
            # Market Mood (MACD-based)
            short_ema = df_featured['Close'].ewm(span=12, adjust=False).mean()
            long_ema = df_featured['Close'].ewm(span=26, adjust=False).mean()
            macd = short_ema - long_ema
            signal = macd.ewm(span=9, adjust=False).mean()
            df_featured['Market_Mood'] = macd - signal
            
            # Volatility features
            df_featured['Daily_Return'] = df_featured['Close'].pct_change()
            df_featured['Volatility'] = df_featured['Daily_Return'].rolling(window=30, min_periods=1).std() * np.sqrt(252)
            
            # Calendar features
            df_featured['Day_of_Week'] = df_featured.index.dayofweek
            df_featured['Month'] = df_featured.index.month
            df_featured['Quarter'] = df_featured.index.quarter
            df_featured['Days_to_Month_End'] = df_featured.index.days_in_month - df_featured.index.day
            
            # Global market features (if available)
            try:
                global_data = self._fetch_global_data(
                    df_featured.index[0].strftime('%Y-%m-%d'),
                    df_featured.index[-1].strftime('%Y-%m-%d')
                )
                
                if not global_data.empty:
                    global_data = global_data.reindex(df_featured.index, method='ffill')
                    for col in global_data.columns:
                        name = col.replace('^', '').replace('=F', '')
                        df_featured[f'{name}_Return'] = global_data[col].pct_change(fill_method=None)
                    logger.info("Successfully added global market features")
            except Exception as e:
                logger.warning(f"Could not add global market features: {str(e)}")
            
            # Fill any remaining NaN values
            df_featured = df_featured.bfill().ffill()
            
            logger.info(f"Successfully engineered features with shape: {df_featured.shape}")
            return df_featured
            
        except Exception as e:
            logger.error(f"Error engineering features: {str(e)}")
            raise

    def _fetch_global_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Helper method to fetch global market indicators"""
        try:
            tickers = ['^GSPC', '^GDAXI', 'GC=F', '^NSEI']
            all_data = pd.DataFrame()
            
            for ticker in tickers:
                try:
                    df = yf.download(
                        ticker,
                        start=start_date,
                        end=end_date,
                        progress=False
                    )
                    
                    if not df.empty:
                        df.index = df.index.tz_localize(None)
                        all_data[ticker] = df['Close']
                        logger.info(f"Successfully fetched {ticker} data")
                        
                except Exception as e:
                    logger.warning(f"Failed to fetch {ticker}: {str(e)}")
                    continue
                    
            return all_data
            
        except Exception as e:
            logger.error(f"Error fetching global data: {str(e)}")
            return pd.DataFrame()

    def scale_features(self, df: pd.DataFrame, columns_to_scale: Optional[List[str]] = None) -> Tuple[pd.DataFrame, MinMaxScaler]:
        """
        Scale features using MinMaxScaler
        
        Args:
            df (pd.DataFrame): Input dataframe
            columns_to_scale (List[str]): List of columns to scale
            
        Returns:
            Tuple[pd.DataFrame, MinMaxScaler]: Scaled dataframe and scaler object
        """
        try:
            df_scaled = df.copy()
            
            # If no columns specified, scale all numeric columns
            if columns_to_scale is None:
                columns_to_scale = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Ensure all specified columns exist
            missing_columns = [col for col in columns_to_scale if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing columns for scaling: {missing_columns}")
            
            # Scale the features
            scaled_data = self.scaler.fit_transform(df_scaled[columns_to_scale])
            df_scaled[columns_to_scale] = scaled_data
            
            logger.info(f"Successfully scaled features: {columns_to_scale}")
            return df_scaled, self.scaler
            
        except Exception as e:
            logger.error(f"Error scaling features: {str(e)}")
            raise

    def prepare_sequences(self, df: pd.DataFrame, sequence_length: int, target_column: str = 'Close') -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for time series prediction
        
        Args:
            df (pd.DataFrame): Input dataframe
            sequence_length (int): Length of input sequences
            target_column (str): Column to predict
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: X and y arrays for training
        """
        try:
            X, y = [], []
            data = df.values
            
            for i in range(len(data) - sequence_length):
                X.append(data[i:(i + sequence_length)])
                y.append(data[i + sequence_length][df.columns.get_loc(target_column)])
            
            return np.array(X), np.array(y)
            
        except Exception as e:
            logger.error(f"Error preparing sequences: {str(e)}")
            raise

def main():
    """Example usage of the preprocessor"""
    try:
        # Initialize preprocessor
        preprocessor = StockDataPreprocessor()
        
        # Get list of CSV files in the raw directory
        csv_files = [f for f in os.listdir(preprocessor.input_dir) if f.endswith('.csv')]
        print(f"Found {len(csv_files)} CSV files: {csv_files}")
        
        for file in csv_files:
            print(f"\nProcessing {file}...")
            
            # Load data
            df = preprocessor.load_data(file)
            print(f"Loaded data shape: {df.shape}")
            
            # Clean data
            df_cleaned = preprocessor.clean_data(df)
            print(f"Cleaned data shape: {df_cleaned.shape}")
            
            # Engineer features
            df_featured = preprocessor.engineer_features(df_cleaned)
            print(f"Featured data shape: {df_featured.shape}")
            print("Added features:", 
                  [col for col in df_featured.columns if col not in df_cleaned.columns])
            
            # Scale features
            columns_to_scale = ['Open', 'High', 'Low', 'Close', 'Volume', 
                              'MA5', 'MA20', 'RSI', 'MACD']
            df_scaled, scaler = preprocessor.scale_features(df_featured, columns_to_scale)
            
            # Prepare sequences
            X, y = preprocessor.prepare_sequences(df_scaled, sequence_length=10)
            
            # Save processed data
            output_filename = f'processed_{file}'
            output_path = os.path.join(preprocessor.output_dir, output_filename)
            df_scaled.to_csv(output_path)
            
            print(f"\nResults for {file}:")
            print(f"- Processed data saved to: {output_path}")
            print(f"- Processed data shape: {df_scaled.shape}")
            print(f"- Sequence data shape: X={X.shape}, y={y.shape}")
            
        print("\nAll files processed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    main()