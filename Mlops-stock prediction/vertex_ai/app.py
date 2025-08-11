import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta
from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
import pandas as pd
import plotly.graph_objects as go
import json
import os

def load_tickers():
    """Load ticker symbols from JSON files"""
    tickers_data = []
    ticker_files = ['ticker1.json', 'ticker2.json', 'ticker3.json']
    
    try:
        for file in ticker_files:
            file_path = os.path.join('tickers', file)
            with open(file_path, 'r') as f:
                data = json.load(f)
                tickers_data.extend(data)
        
        # Create a list of tuples (symbol, name, sector, industry) for the dropdown
        ticker_options = [(ticker['symbol'], 
                          f"{ticker['symbol']} - {ticker['name']}", 
                          ticker.get('sector', ''), 
                          ticker.get('industry', '')) 
                         for ticker in tickers_data]
        return sorted(ticker_options, key=lambda x: x[0])
    except Exception as e:
        st.error(f"Error loading ticker data: {str(e)}")
        return []

def filter_tickers(ticker_options, sector=None, industry=None, search_text=''):
    """Filter tickers based on sector, industry, and search text"""
    filtered_tickers = ticker_options
    
    if sector and sector != 'All Sectors':
        filtered_tickers = [t for t in filtered_tickers if t[2] == sector]
    
    if industry and industry != 'All Industries':
        filtered_tickers = [t for t in filtered_tickers if t[3] == industry]
    
    if search_text:
        search_text = search_text.lower()
        filtered_tickers = [t for t in filtered_tickers if search_text in t[1].lower()]
    
    return filtered_tickers

def get_stock_data(symbol):
    """Fetch stock data using yfinance"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(start=start_date, end=end_date)
        return df
    except Exception as e:
        st.error(f"Error fetching stock data: {str(e)}")
        return None

def make_prediction(instance_dict):
    """Make prediction using Vertex AI endpoint"""
    project = "322603165747"
    endpoint_id = "4602032306335514624"
    location = "us-east1"
    api_endpoint = "us-east1-aiplatform.googleapis.com"

    try:
        client_options = {"api_endpoint": api_endpoint}
        client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)

        instance = json_format.ParseDict(instance_dict, Value())
        instances = [instance]
        
        parameters_dict = {}
        parameters = json_format.ParseDict(parameters_dict, Value())

        endpoint = client.endpoint_path(
            project=project, location=location, endpoint=endpoint_id
        )

        response = client.predict(
            endpoint=endpoint, instances=instances, parameters=parameters
        )

        prediction = dict(response.predictions[0])
        return prediction
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None

def main():
    st.title("Stock Price Predictor")
    
    # Load ticker options
    ticker_options = load_tickers()
    
    if ticker_options:
        # Get unique sectors and industries
        sectors = sorted(list(set(t[2] for t in ticker_options if t[2])))
        sectors.insert(0, 'All Sectors')
        
        industries = sorted(list(set(t[3] for t in ticker_options if t[3])))
        industries.insert(0, 'All Industries')
        
        # Create filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_sector = st.selectbox("Filter by Sector", sectors)
        
        with col2:
            selected_industry = st.selectbox("Filter by Industry", industries)
        
        with col3:
            search_text = st.text_input("Search Stocks", "")
        
        # Filter tickers based on selection
        filtered_tickers = filter_tickers(ticker_options, selected_sector, selected_industry, search_text)
        
        # Display filtered tickers in a selectbox
        if filtered_tickers:
            selected_option = st.selectbox(
                "Choose a stock",
                options=[option[1] for option in filtered_tickers],
                index=None,
                placeholder="Select a stock..."
            )
            if selected_option:
                symbol = selected_option.split(' - ')[0]
            else:
                symbol = None
                
            # Show number of results
            st.caption(f"Showing {len(filtered_tickers)} stocks")
        else:
            st.warning("No stocks match your filters")
            symbol = None
        
        # Add a date range selector
        st.subheader("Select Date Range")
        col4, col5 = st.columns(2)
        with col4:
            days = st.slider("Number of days of historical data", 
                            min_value=5, 
                            max_value=30, 
                            value=7)
        
        if symbol and st.button("Predict", type="primary"):
            # Show loading spinner
            with st.spinner(f'Fetching data for {symbol}...'):
                df = get_stock_data(symbol)
            
            if df is not None and not df.empty:
                # Create tabs for different views
                tab1, tab2 = st.tabs(["📈 Prediction", "📊 Historical Data"])
                
                with tab1:
                    # Prepare features for prediction
                    latest_data = df.iloc[-1]
                    instance_dict = {
                        "Date": latest_data.name.strftime("%m/%d/%y"),
                        "Open": str(latest_data['Open']),
                        "High": str(latest_data['High']),
                        "Low": str(latest_data['Low']),
                        "Volume": str(latest_data['Volume'])
                    }
                    
                    # Make prediction
                    with st.spinner('Making prediction...'):
                        prediction = make_prediction(instance_dict)
                    
                    if prediction:
                        st.subheader("Prediction Results")
                        
                        # Create metrics display
                        metrics_cols = st.columns(3)
                        with metrics_cols[0]:
                            st.metric(
                                "Predicted Price", 
                                f"${prediction['value']:.2f}",
                                delta=f"{((prediction['value'] - latest_data['Close'])/latest_data['Close']*100):.2f}%"
                            )
                        with metrics_cols[1]:
                            st.metric("Lower Bound", f"${prediction['lower_bound']:.2f}")
                        with metrics_cols[2]:
                            st.metric("Upper Bound", f"${prediction['upper_bound']:.2f}")
                        
                        # Create visualization
                        fig = go.Figure()
                        
                        # Add historical prices
                        fig.add_trace(go.Scatter(
                            x=df.index,
                            y=df['Close'],
                            name='Historical Close Price',
                            line=dict(color='blue')
                        ))
                        
                        # Add prediction point
                        fig.add_trace(go.Scatter(
                            x=[df.index[-1] + timedelta(days=1)],
                            y=[prediction['value']],
                            name='Prediction',
                            mode='markers',
                            marker=dict(size=10, color='red'),
                            error_y=dict(
                                type='data',
                                symmetric=False,
                                array=[prediction['upper_bound'] - prediction['value']],
                                arrayminus=[prediction['value'] - prediction['lower_bound']],
                            )
                        ))
                        
                        fig.update_layout(
                            title=f'{symbol} Stock Price Prediction',
                            xaxis_title='Date',
                            yaxis_title='Price ($)',
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    st.subheader("Historical Data")
                    # Add summary statistics
                    summary_cols = st.columns(4)
                    with summary_cols[0]:
                        st.metric("Current Price", f"${df['Close'][-1]:.2f}")
                    with summary_cols[1]:
                        st.metric("Day High", f"${df['High'][-1]:.2f}")
                    with summary_cols[2]:
                        st.metric("Day Low", f"${df['Low'][-1]:.2f}")
                    with summary_cols[3]:
                        st.metric("Volume", f"{df['Volume'][-1]:,.0f}")
                    
                    # Show the dataframe with formatted numbers
                    st.dataframe(
                        df.style.format({
                            'Open': '${:.2f}',
                            'High': '${:.2f}',
                            'Low': '${:.2f}',
                            'Close': '${:.2f}',
                            'Volume': '{:,.0f}'
                        })
                    )
    else:
        st.error("Failed to load ticker data. Please check the JSON files.")

if __name__ == "__main__":
    main()
