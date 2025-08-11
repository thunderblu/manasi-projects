import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StockDataAcquisition:
    def __init__(self, output_dir='data/raw'):
        """
        Initialize the data acquisition class
        
        Args:
            output_dir (str): Directory to save the downloaded data
        """
        self.output_dir = output_dir
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories if they don't exist"""
        os.makedirs(self.output_dir, exist_ok=True)
    
    def fetch_stock_data(self, symbol, start_date=None, end_date=None, interval='1d'):
        """
        Fetch stock data from Yahoo Finance
        
        Args:
            symbol (str): Stock symbol (e.g., 'AAPL')
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
            interval (str): Data interval ('1d', '1wk', '1mo')
            
        Returns:
            pandas.DataFrame: Downloaded stock data
        """
        try:
            # If no dates provided, use last 5 years
            if not start_date:
                start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            logger.info(f"Fetching data for {symbol} from {start_date} to {end_date}")
            
            # Create ticker object
            ticker = yf.Ticker(symbol)
            
            # Download data
            df = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval
            )
            
            # Basic data validation
            if df.empty:
                raise ValueError(f"No data retrieved for {symbol}")
            
            # Save to CSV
            output_file = os.path.join(
                self.output_dir, 
                f"{symbol}.csv"
            )
            df.to_csv(output_file)
            logger.info(f"Data saved to {output_file}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            raise

def main():
    """Main function to demonstrate usage"""
    # Example usage
    symbols = ['AAPL', 'GOOGL', 'MSFT']  # Example stock symbols
    data_acquirer = StockDataAcquisition()
    
    for symbol in symbols:
        try:
            df = data_acquirer.fetch_stock_data(
                symbol=symbol,
                start_date='2020-01-01',
                end_date='2024-01-01'
            )
            print(f"\nPreview of {symbol} data:")
            print(df.head())
            
        except Exception as e:
            print(f"Failed to process {symbol}: {str(e)}")

if __name__ == "__main__":
    main() 