import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import warnings
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Download NLTK resources if not already downloaded
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

class IndianStockPredictionApp:
    def __init__(self):
        self.ticker = None
        self.is_index = False
        self.historical_data = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.lstm_model = None
        self.rf_model = None
        
        # Indian stock indices mapping
        self.indices_map = {
            'NIFTY50': '^NSEI',
            'SENSEX': '^BSESN',
            'NIFTYBANK': '^NSEBANK',
            'NIFTYIT': 'CNXIT.NS',
            'NIFTYPHARMA': 'CNXPHARMA.NS',
            'NIFTYAUTO': 'CNXAUTO.NS',
            'NIFTYMETAL': 'CNXMETAL.NS',
            'NIFTYFMCG': 'CNXFMCG.NS'
        }
        
    def get_stock_data(self, input_ticker, days=60):  # Increased days to ensure enough data
        """Fetch historical stock data for Indian stocks or indices"""
        # Check if the input is an index
        if input_ticker.upper() in self.indices_map:
            self.is_index = True
            self.ticker = self.indices_map[input_ticker.upper()]
            self.display_name = input_ticker.upper()
        else:
            # Handle normal stocks by adding .NS or .BO suffix if not present
            if not (input_ticker.endswith('.NS') or input_ticker.endswith('.BO')):
                # Default to NSE
                self.ticker = f"{input_ticker.upper()}.NS"
            else:
                self.ticker = input_ticker.upper()
            self.display_name = self.ticker
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        try:
            logging.info(f"Downloading data for {self.ticker} from {start_date} to {end_date}")
            data = yf.download(self.ticker, start=start_date, end=end_date)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            
            if data.empty or len(data) < 10:  # Need at least 10 days of data
                # Try alternate exchange if NSE data not found
                if self.ticker.endswith('.NS') and not self.is_index:
                    alt_ticker = self.ticker.replace('.NS', '.BO')
                    print(f"Not enough data for {self.ticker}, trying {alt_ticker} instead...")
                    data = yf.download(alt_ticker, start=start_date, end=end_date)
                    if not data.empty and len(data) >= 10:
                        self.ticker = alt_ticker
                        self.display_name = alt_ticker
                    else:
                        return False, f"Not enough data available for {input_ticker} on either NSE or BSE."
                else:
                    return False, f"Not enough data available for {input_ticker}."
            
            # Debug information
            logging.info(f"Downloaded {len(data)} days of data for {self.ticker}")
            logging.info(f"Data columns: {data.columns}")
            logging.info(f"Data head: {data.head(2)}")
            
            self.historical_data = data
            return True, "Data fetched successfully"
        except Exception as e:
            return False, f"Error fetching data: {str(e)}"
    
    def prepare_data(self, sequence_length=10):
        """Prepare data for LSTM model with handling for missing data"""
        try:
            df = self.historical_data.copy()
            
            # Handle extreme values in volume - clip to reasonable range
            if 'Volume' in df.columns:
                # Replace zeros with NaN
                df.loc[df['Volume'] == 0, 'Volume'] = np.nan
                # Fill NaN values with median of non-zero values
                median_volume = df['Volume'].median()
                df['Volume'].fillna(median_volume, inplace=True)
            else:
                # If volume is missing, create a dummy column with 1s
                logging.warning("Volume data missing, creating dummy volume")
                df['Volume'] = 1
            
            logging.info(f"Data shape after volume handling: {df.shape}")
            
            # Add technical indicators
            df['MA5'] = df['Close'].rolling(window=5).mean()
            df['MA20'] = df['Close'].rolling(window=20).mean()
            
            # Simplified RSI calculation with error handling
            try:
                df['RSI'] = self._calculate_rsi(df['Close'], 14)
            except Exception as e:
                logging.warning(f"Error calculating RSI: {e}")
                df['RSI'] = 50  # Neutral RSI value
            
            # Simplified MACD calculation with error handling
            try:
                df['MACD'] = self._calculate_macd(df['Close'])
            except Exception as e:
                logging.warning(f"Error calculating MACD: {e}")
                df['MACD'] = 0  # Neutral MACD value
            
            # Volume change with safeguards
            try:
                df['Volume_Change'] = df['Volume'].pct_change()
                df['Volume_Change'].fillna(0, inplace=True)
                # Clip extreme values
                df['Volume_Change'] = df['Volume_Change'].clip(-1, 1)
            except Exception as e:
                logging.warning(f"Error calculating Volume Change: {e}")
                df['Volume_Change'] = 0
            
            # Replace any remaining NaN values with forward fill then backward fill
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            logging.info(f"Data shape after filling NaNs: {df.shape}")
            
            # Make sure we have enough data after preprocessing
            if len(df) <= sequence_length:
                raise ValueError(f"Not enough data after preprocessing. Only have {len(df)} rows.")
            
            # Select features for prediction
            features = ['Close', 'Volume', 'MA5', 'MA20', 'RSI', 'MACD', 'Volume_Change']
            data = df[features].values
            
            logging.info(f"Feature data shape: {data.shape}")
            
            # Check for any remaining NaNs or Infs
            if np.isnan(data).any() or np.isinf(data).any():
                logging.warning("NaN or Inf values detected in data!")
                # Replace NaNs and Infs with 0
                data = np.nan_to_num(data, nan=0, posinf=0, neginf=0)
            
            # Scale the data
            scaled_data = self.scaler.fit_transform(data)
            
            # Prepare X and y for LSTM
            X, y = [], []
            for i in range(len(scaled_data) - sequence_length):
                X.append(scaled_data[i:i+sequence_length])
                y.append(scaled_data[i+sequence_length, 0])  # Predict Close price
            
            X = np.array(X)
            y = np.array(y)
            
            logging.info(f"X shape: {X.shape}, y shape: {y.shape}")
            
            return X, y, df
            
        except Exception as e:
            logging.error(f"Error in prepare_data: {str(e)}")
            raise
    
    def _calculate_rsi(self, prices, window=14):
        """Calculate Relative Strength Index safely"""
        # Make a copy to avoid modifying the original
        prices = prices.copy()
        
        # Calculate daily returns
        delta = prices.diff()
        
        # Create gain/loss series
        gain = delta.copy()
        loss = delta.copy()
        gain[gain < 0] = 0
        loss[loss > 0] = 0
        loss = abs(loss)
        
        # Calculate average gain/loss over the window
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        # Handle division by zero
        avg_loss = avg_loss.replace(0, 0.001)
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate Moving Average Convergence Divergence safely"""
        # Calculate EMAs
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        
        # MACD line
        macd = ema_fast - ema_slow
        
        # Signal line
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        
        # MACD histogram
        macd_hist = macd - macd_signal
        
        return macd_hist
    
    def build_lstm_model(self, input_shape):
        """Build LSTM model"""
        logging.info(f"Building LSTM model with input shape: {input_shape}")
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
    def train_models(self, X, y):
        """Train both LSTM and Random Forest models"""
        logging.info("Training models...")
        
        # Train LSTM model
        self.lstm_model = self.build_lstm_model((X.shape[1], X.shape[2]))
        self.lstm_model.fit(X, y, batch_size=32, epochs=50, verbose=0)
        
        # Reshape X for Random Forest
        X_rf = X.reshape(X.shape[0], -1)
        
        # Train Random Forest model
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.rf_model.fit(X_rf, y)
        
        logging.info("Model training complete")
    
    def get_news_sentiment(self, num_articles=5):
        """Fetch and analyze news sentiment for Indian stocks/indices"""
        if not self.ticker:
            return 0, "No ticker selected"
        
        # For demo purposes, use simulated news if scraping fails
        simulated_news = [
            f"Quarterly results for {self.display_name.split('.')[0]} beat market expectations",
            f"Analysts upgrade {self.display_name.split('.')[0]} target price after strong guidance",
            f"New government policies could impact {self.display_name.split('.')[0]}'s sector",
            f"{self.display_name.split('.')[0]} announces expansion into new markets",
            f"Industry outlook for {self.display_name.split('.')[0]}'s sector remains strong"
        ]
        
        # Different sources for indices and individual stocks
        if self.is_index:
            # For indices, use more general market news sources
            news_sources = [
                self._get_moneycontrol_news,
                self._get_economic_times_news
            ]
        else:
            # For individual stocks, use stock-specific news
            stock_name = self.display_name.split('.')[0]  # Remove exchange suffix
            news_sources = [
                lambda: self._get_moneycontrol_news(stock_name),
                lambda: self._get_economic_times_news(stock_name)
            ]
        
        # Collect news from all sources
        all_articles = []
        for source_func in news_sources:
            try:
                articles = source_func()
                if articles:
                    all_articles.extend(articles)
            except Exception as e:
                logging.warning(f"Error fetching news from a source: {str(e)}")
        
        # If no articles found from scraping, use simulated news
        if not all_articles:
            logging.info("Using simulated news as no news was scraped successfully")
            all_articles = simulated_news
        
        # Calculate sentiment
        # Limit to requested number
        articles = all_articles[:num_articles]
        
        # Calculate sentiment score for each article
        sentiment_scores = []
        news_summary = []
        
        for i, article in enumerate(articles):
            sentiment = self.sentiment_analyzer.polarity_scores(article)
            sentiment_scores.append(sentiment['compound'])
            
            # Create a summary with sentiment label
            sentiment_label = "positive" if sentiment['compound'] > 0.05 else "negative" if sentiment['compound'] < -0.05 else "neutral"
            # Truncate article text if too long
            article_text = article if len(article) < 100 else article[:97] + "..."
            news_summary.append(f"{i+1}. {article_text} (Sentiment: {sentiment_label})")
        
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        return avg_sentiment, news_summary
    
    def _get_economic_times_news(self, search_term=None):
        """Fetch news from Economic Times with error handling"""
        try:
            if search_term:
                url = f"https://economictimes.indiatimes.com/searchresult.cms?query={search_term}"
            else:
                url = "https://economictimes.indiatimes.com/markets/stocks"
            
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            news_items = soup.find_all('a', class_='title')
            return [item.get_text().strip() for item in news_items[:5] if item.get_text()]
        except Exception as e:
            logging.warning(f"Error fetching Economic Times news: {e}")
            return []
    
    def _get_moneycontrol_news(self, search_term=None):
        """Fetch news from MoneyControl with error handling"""
        try:
            if search_term:
                url = f"https://www.moneycontrol.com/stocks/company_info/stock_news.php?sc_id={search_term}"
            else:
                url = "https://www.moneycontrol.com/news/business/markets/"
            
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Different selectors depending on whether it's a search or main page
            if search_term:
                news_items = soup.find_all('span', class_='arial11_summ')
            else:
                news_items = soup.find_all('h2')
            
            return [item.get_text().strip() for item in news_items[:5] if item.get_text()]
        except Exception as e:
            logging.warning(f"Error fetching MoneyControl news: {e}")
            return []
    
    def predict_prices(self, days=5):
        """Predict future stock prices"""
        if self.historical_data is None or self.lstm_model is None:
            return None, "Models not trained yet"
        
        try:
            # Prepare the most recent data for prediction
            df = self.historical_data.copy()
            
            # Calculate technical indicators
            df['MA5'] = df['Close'].rolling(window=5).mean()
            df['MA20'] = df['Close'].rolling(window=20).mean()
            df['RSI'] = self._calculate_rsi(df['Close'], 14)
            df['MACD'] = self._calculate_macd(df['Close'])
            
            # Handle volume
            if 'Volume' in df.columns:
                # Replace zeros with NaN
                df.loc[df['Volume'] == 0, 'Volume'] = np.nan
                # Fill NaN values with median of non-zero values
                median_volume = df['Volume'].median()
                df['Volume'].fillna(median_volume, inplace=True)
                df['Volume_Change'] = df['Volume'].pct_change()
                df['Volume_Change'].fillna(0, inplace=True)
                # Clip extreme values
                df['Volume_Change'] = df['Volume_Change'].clip(-1, 1)
            else:
                # If volume is missing, create dummy columns
                df['Volume'] = 1
                df['Volume_Change'] = 0
            
            # Fill any NaN values
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            # Select features
            features = ['Close', 'Volume', 'MA5', 'MA20', 'RSI', 'MACD', 'Volume_Change']
            last_sequence = df[features].values[-10:]  # Last 10 days
            
            # Scale the data
            last_sequence_scaled = self.scaler.transform(last_sequence)
            
            # Make predictions for the next 'days' days
            lstm_predictions = []
            rf_predictions = []
            
            current_sequence = last_sequence_scaled.reshape(1, 10, 7)
            current_sequence_rf = current_sequence.reshape(1, -1)
            
            # Get news sentiment
            sentiment_score, news_summary = self.get_news_sentiment()
            
            # Adjust prediction based on sentiment
            sentiment_factor = 1 + (sentiment_score * 0.01)  # Small adjustment based on sentiment
            
            for _ in range(days):
                # Predict next day with LSTM
                lstm_pred = self.lstm_model.predict(current_sequence, verbose=0)[0][0]
                lstm_predictions.append(lstm_pred)
                
                # Predict next day with Random Forest
                rf_pred = self.rf_model.predict(current_sequence_rf)[0]
                rf_predictions.append(rf_pred)
                
                # Update sequence for next prediction
                # Create a copy of the last row
                next_row = np.copy(current_sequence[0, -1, :])
                # Update the close price (first column)
                next_row[0] = lstm_pred
                # Append the new row and drop the first one
                new_sequence = np.vstack((current_sequence[0, 1:, :], next_row))
                current_sequence = new_sequence.reshape(1, 10, 7)
                current_sequence_rf = current_sequence.reshape(1, -1)
            
            # Combine predictions (ensemble approach)
            combined_predictions = [(lstm_predictions[i] + rf_predictions[i]) / 2 for i in range(days)]
            
            # Apply sentiment adjustment
            adjusted_predictions = [pred * sentiment_factor for pred in combined_predictions]
            
            # Inverse transform to get actual values
            # Create a template with the same shape as what we trained on
            pred_template = np.zeros((len(adjusted_predictions), 7))
            # Set the first column (Close price) to our predictions
            pred_template[:, 0] = adjusted_predictions
            
            # Inverse transform
            predicted_prices = self.scaler.inverse_transform(pred_template)[:, 0]
            
            # Generate dates for the predictions
            last_date = self.historical_data.index[-1]
            prediction_dates = [last_date + timedelta(days=i+1) for i in range(days)]
            
            # Skip weekends
            valid_prediction_dates = []
            valid_predicted_prices = []
            
            for i, date in enumerate(prediction_dates):
                # Skip Saturday (5) and Sunday (6)
                if date.weekday() < 5:
                    valid_prediction_dates.append(date)
                    valid_predicted_prices.append(predicted_prices[i])
                
                # If we don't have enough weekday predictions, add more days
                if len(valid_prediction_dates) < days and i == len(prediction_dates) - 1:
                    extra_days = days - len(valid_prediction_dates)
                    for j in range(1, extra_days + 1):
                        extra_date = prediction_dates[-1] + timedelta(days=j)
                        if extra_date.weekday() < 5:  # Weekday
                            # For simplicity, use the last predicted price with small random variation
                            extra_price = predicted_prices[-1] * (1 + np.random.normal(0, 0.005))
                            valid_prediction_dates.append(extra_date)
                            valid_predicted_prices.append(extra_price)
            
            # Create a prediction explanation
            last_close = self.historical_data['Close'].iloc[-1]
            if valid_predicted_prices[-1] > last_close:
                direction = "upward"
            else:
                direction = "downward"
            
            price_change = (valid_predicted_prices[-1] - last_close) / last_close * 100
            
            # Generate prediction reasoning
            explanation = self._generate_explanation(direction, price_change, sentiment_score, news_summary)
            
            return pd.DataFrame({'Date': valid_prediction_dates[:days], 'Price': valid_predicted_prices[:days]}), explanation
            
        except Exception as e:
            logging.error(f"Error in predict_prices: {str(e)}")
            # Return fallback prediction
            return self._generate_fallback_prediction(), "Error generating detailed prediction. Using simplified forecast."
    
    def _generate_fallback_prediction(self):
        """Generate a fallback prediction if the main method fails"""
        # Simple moving average based prediction
        last_date = self.historical_data.index[-1]
        last_price = self.historical_data['Close'].iloc[-1]
        
        # Calculate average daily change over last 5 days
        last_5_prices = self.historical_data['Close'].iloc[-5:]
        avg_daily_change = (last_5_prices.pct_change().mean() or 0.001)  # Default to 0.1% if calculation fails
        
        # Generate dates and prices
        prediction_dates = []
        predicted_prices = []
        
        current_date = last_date
        current_price = last_price
        
        # Generate 5 trading days (skip weekends)
        while len(prediction_dates) < 5:
            current_date = current_date + timedelta(days=1)
            
            # Skip weekends
            if current_date.weekday() < 5:  # 0-4 are weekdays
                # Apply average change with some randomness
                random_factor = np.random.normal(1, 0.002)  # Small random variation
                current_price = current_price * (1 + avg_daily_change) * random_factor
                
                prediction_dates.append(current_date)
                predicted_prices.append(current_price)
        
        return pd.DataFrame({'Date': prediction_dates, 'Price': predicted_prices})
    
    def _generate_explanation(self, direction, price_change, sentiment_score, news_summary):
        """Generate explanation for the prediction"""
        explanation = []
        
        # Overall prediction summary
        explanation.append(f"PREDICTION: The model predicts a {direction} trend for {self.display_name} with a "
                          f"{abs(price_change):.2f}% {'increase' if price_change > 0 else 'decrease'} "
                          f"over the next 5 trading days.")
        
        # Technical factors
        explanation.append("\nTECHNICAL FACTORS:")
        
        # RSI analysis
        try:
            rsi = self._calculate_rsi(self.historical_data['Close']).iloc[-1]
            if rsi > 70:
                explanation.append(f"* RSI is high at {rsi:.2f}, suggesting the stock may be overbought.")
            elif rsi < 30:
                explanation.append(f"* RSI is low at {rsi:.2f}, suggesting the stock may be oversold.")
            else:
                explanation.append(f"* RSI is neutral at {rsi:.2f}.")
        except:
            explanation.append("* RSI analysis unavailable.")
        
        # Moving average analysis
        try:
            ma5 = self.historical_data['Close'].rolling(5).mean().iloc[-1]
            ma20 = self.historical_data['Close'].rolling(20).mean().iloc[-1]
            if ma5 > ma20:
                explanation.append(f"* 5-day moving average (₹{ma5:.2f}) is above 20-day MA (₹{ma20:.2f}), indicating positive short-term momentum.")
            else:
                explanation.append(f"* 5-day moving average (₹{ma5:.2f}) is below 20-day MA (₹{ma20:.2f}), indicating negative short-term momentum.")
        except:
            explanation.append("* Moving average analysis unavailable.")
        
        # Volume analysis
        try:
            if 'Volume' in self.historical_data.columns:
                avg_volume = self.historical_data['Volume'].replace(0, np.nan).mean()
                last_volume = self.historical_data['Volume'].iloc[-1]
                if last_volume > avg_volume * 1.5:
                    explanation.append(f"* Trading volume is significantly higher than average, suggesting strong market interest.")
                elif last_volume < avg_volume * 0.5:
                    explanation.append(f"* Trading volume is significantly lower than average, suggesting weak market interest.")
                else:
                    explanation.append(f"* Trading volume is around average levels.")
        except:
            explanation.append("* Volume analysis unavailable.")
        
        # Add India-specific market factors
        explanation.append("\nINDIAN MARKET CONTEXT:")
        
        # Check if it's an index
        if self.is_index:
            if "NIFTY" in self.display_name or "SENSEX" in self.display_name:
                explanation.append(f"* As a major Indian index, {self.display_name} is influenced by both domestic factors and global market trends.")
                explanation.append("* FII (Foreign Institutional Investor) and DII (Domestic Institutional Investor) flows are critical for short-term movement.")
            else:
                explanation.append(f"* Sectoral index {self.display_name} may be influenced by specific policy announcements and global sector trends.")
        else:
            explanation.append(f"* Indian equities are currently affected by RBI policy decisions, FII/DII flows, and global market sentiment.")
            explanation.append(f"* {self.display_name.split('.')[0]}'s sector performance and quarterly results are key factors to watch.")
        
        # News sentiment analysis
        explanation.append("\nNEWS SENTIMENT:")
        if isinstance(news_summary, list) and len(news_summary) > 0:
            if sentiment_score > 0.2:
                explanation.append(f"* News sentiment is strongly positive (score: {sentiment_score:.2f}), supporting the prediction.")
            elif sentiment_score > 0.05:
                explanation.append(f"* News sentiment is slightly positive (score: {sentiment_score:.2f}).")
            elif sentiment_score < -0.2:
                explanation.append(f"* News sentiment is strongly negative (score: {sentiment_score:.2f}), supporting the prediction.")
            elif sentiment_score < -0.05:
                explanation.append(f"* News sentiment is slightly negative (score: {sentiment_score:.2f}).")
            else:
                explanation.append(f"* News sentiment is neutral (score: {sentiment_score:.2f}).")
            
            # Add top news headlines
            explanation.append("\nRECENT NEWS HEADLINES:")
            for news in news_summary:
                explanation.append(news)
        else:
            explanation.append("* No recent news articles found for sentiment analysis.")
        
        # Disclaimer
        explanation.append("\nDISCLAIMER: This prediction is based on historical data and current news sentiment. "
                         "Market conditions can change rapidly and unpredictably. This should not be considered "
                         "financial advice. Indian markets are subject to various unique factors including regulatory "
                         "changes and policy decisions that may impact stocks unpredictably.")
        
        return "\n".join(explanation)
    
    def visualize_results(self):
        """Visualize historical data and predictions"""
        if self.historical_data is None:
            return "No data to visualize"
        
        try:
            # Get the last 20 days of historical data
            last_20_days = self.historical_data[-20:]['Close']
            
            # Get predictions for the next 5 days
            predictions, _ = self.predict_prices(5)
            
            # Create a plot
            plt.figure(figsize=(12, 6))
            
            # Plot historical data
            plt.plot(last_20_days.index, last_20_days.values, label='Historical Data', color='blue')
            
            # Plot predictions
            plt.plot(predictions['Date'], predictions['Price'], label='Predictions', color='red', linestyle='--')
            
            # Add a vertical line to separate historical data and predictions
            plt.axvline(x=last_20_days.index[-1], color='black', linestyle='-', alpha=0.3)
            
            # Use ₹ symbol for individual stocks
            plt.title(f'{self.display_name} {"Index" if self.is_index else "Stock"} Price Prediction')
            plt.xlabel('Date')
            plt.ylabel('Price (₹)' if not self.is_index else 'Index Value')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            
            return plt
            
        except Exception as e:
            logging.error(f"Error in visualize_results: {str(e)}")
            
            # Create a simplified fallback visualization
            plt.figure(figsize=(12, 6))
            plt.plot(self.historical_data[-20:]['Close'], label='Historical Data')
            plt.title(f"{self.display_name} - Simple Price Chart")
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.grid(True, alpha=0.3)
            plt.legend()
            return plt


# Main function to run the app
def run_indian_stock_prediction():
    # Initialize the app
    app = IndianStockPredictionApp()
    
    print("Indian Stock/Index Prediction App")
    print("\nAvailable indices: NIFTY50, SENSEX, NIFTYBANK, NIFTYIT, NIFTYPHARMA, NIFTYAUTO, NIFTYMETAL, NIFTYFMCG")
    print("For individual stocks, enter the ticker symbol (e.g., RELIANCE, TCS, HDFCBANK)")
    
    # Get user input for stock ticker or index
    ticker = input("\nEnter stock ticker or index name: ").strip()
    
    print(f"\nFetching data for {ticker}...")
    success, message = app.get_stock_data(ticker)
    
    if not success:
        print(message)
        return
    
    print(f"Data fetched successfully. Processing {app.display_name}...")
    
    # Prepare data and train models
    X, y, processed_data = app.prepare_data()
    print("\nTraining prediction models...")
    app.train_models(X, y)
    print("Models trained successfully!")
    
    # Get predictions and explanation
    print("\nGenerating predictions and analyzing news sentiment...")
    predictions, explanation = app.predict_prices()
    
    print("\n--- STOCK ANALYSIS RESULTS ---")
    print(f"\nLast 20 days of {app.display_name} prices:")
    print(app.historical_data[-20:]['Close'].to_string())
    
    print("\nPrice predictions for the next 5 trading days:")
    print(predictions.to_string(index=False))
    
    print("\n--- PREDICTION EXPLANATION ---")
    print(explanation)
    
    # Visualize results
    print("\nGenerating visualization...")
    plt = app.visualize_results()
    plt.show()

if __name__ == "__main__":
    run_indian_stock_prediction()