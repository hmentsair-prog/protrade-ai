from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import requests
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def run_ai_prediction(coin_symbol):
    # 🧠 स्मार्ट सप्लायर राउटिंग
    if "USDT" in coin_symbol:
        url = f"https://api.binance.com/api/v3/klines?symbol={coin_symbol}&interval=1d&limit=100"
        response = requests.get(url)
        data = response.json()
        df = pd.DataFrame(data, columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'])
    else:
        raw_df = yf.Ticker(coin_symbol).history(period="100d", interval="1d")
        df = raw_df.reset_index()

    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['Price_Change'] = df['Close'] - df['Open']
    df = df.dropna()
    
    X = df[['Open', 'High', 'Low', 'Volume', 'SMA_10', 'Price_Change']]
    y = df['Close'].shift(-1)
    
    X_today = X.iloc[-1:]
    X_train = X.iloc[:-1]
    y_train = y.dropna()
    
    ai_model = RandomForestRegressor(n_estimators=100, random_state=42)
    ai_model.fit(X_train, y_train)
    
    prediction = ai_model.predict(X_today)[0]
    current_price = df['Close'].iloc[-1]
    
    trend = "BULLISH 🚀" if prediction > current_price else "BEARISH 📉"
    
    return {
        "coin": coin_symbol,
        "current_price": round(current_price, 4),
        "predicted_price": round(prediction, 4),
        "trend": trend
    }

@app.get("/")
def read_root():
    return {"message": "ProTrade AI Server Active!"}

@app.get("/api/predict/{coin}")
def get_prediction(coin: str):
    return run_ai_prediction(coin)
