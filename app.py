import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# 1. Konfigurasi Halaman Utama
st.set_page_config(page_title="AI Trading Predictor", layout="wide")
st.title("📈 Aplikasi Grafik Trading AI")
st.subheader("Prediksi Arah Harga Saham & Kripto Menggunakan Machine Learning")

# 2. Sidebar Input Pengguna
st.sidebar.header("Pengaturan AI & Pasar")
ticker = st.sidebar.text_input("Simbol Emiten (Contoh: AAPL, BTC-USD, BBCA.JK)", value="BTC-USD")
start_date = st.sidebar.date_input("Tanggal Mulai Data", pd.to_datetime("2024-01-01"))
end_date = st.sidebar.date_input("Tanggal Akhir Data", pd.to_datetime("today"))

# 3. Mengunduh Data Pasar (yFinance)
@st.cache_data
def load_data(symbol, start, end):
    data = yf.download(symbol, start=start, end=end)
    return data

data = load_data(ticker, start_date, end_date)

if data.empty:
    st.error("Data tidak ditemukan. Silakan periksa kembali simbol emiten Anda.")
else:
    # 4. Rekayasa Fitur AI (Technical Indicators)
    df = data.copy()
    df['Sma_10'] = df['Close'].rolling(window=10).mean()
    df['Sma_30'] = df['Close'].rolling(window=30).mean()
    df['Return'] = df['Close'].pct_change()
    
    # Target: Jika harga besok lebih tinggi dari hari ini = 1 (Naik), jika tidak = 0 (Turun)
    df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
    df.dropna(inplace=True)
    
    # Fitur untuk AI belajar
    features = ['Close', 'Sma_10', 'Sma_30', 'Return']
    X = df[features]
    y = df['Target']
    
    # 5. Pelatihan Model AI (Random Forest)
    train_size = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Prediksi untuk hari esok
    latest_data = X.iloc[[-1]]
    latest_data_scaled = scaler.transform(latest_data)
    prediction = model.predict(latest_data_scaled)
    prediction_proba = model.predict_proba(latest_data_scaled)
    
    # 6. Menampilkan Metrik AI ke Pengguna
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label=f"Harga Penutupan Terakhir ({ticker})", value=f"${df['Close'].iloc[-1]:,.2f}")
    with col2:
        hasil_prediksi = "📈 NAIK" if prediction == 1 else "📉 TURUN"
        probabilitas = prediction_proba[0][1] if prediction == 1 else prediction_proba[0][0]
        st.metric(label="Prediksi Arah Harga Esok Hari", value=hasil_prediksi, delta=f"Keyakinan: {probabilitas*100:.1f}%")

    # 7. Pembuatan Grafik Interaktif (Candlestick)
    st.markdown("---")
    st.subheader("Grafik Analisis Teknikal")
    
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Harga"
    ))
    fig.add_trace(go.Scatter(x=df.index, y=df['Sma_10'], mode='lines', name='SMA 10', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=df.index, y=df['Sma_30'], mode='lines', name='SMA 30', line=dict(color='blue')))
    
    fig.update_layout(xaxis_rangeslider_visible=False, template="plotly_dark", height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    if st.checkbox("Lihat Data Mentah"):
        st.write(df.tail(10))
