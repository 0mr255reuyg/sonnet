import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import time
from datetime import datetime

# ─── SAYFA AYARI ────────────────────────────────────────────────
st.set_page_config(page_title="BIST Radar", layout="wide")

# ─── HİSSE LİSTESİ (Örnek 20 Hisse - Hızlı Test İçin) ───────────
# 350 tane çok uzun sürüyor, önce bunu dene çalışıyor mu diye.
# Çalışırsa listeyi sonradan uzatırız.
BIST_TICKERS = [
    "AKBNK","GARAN","ISCTR","YKBNK","HALKB",
    "THYAO","PGSUS","TAVHL",
    "TUPRS","PETKM","AYGAZ",
    "ASELS","OTKAR",
    "TOASO","FROTO","ARCLK",
    "BIMAS","MGROS","SOKM",
    "EREGL","KRDMD"
]
YF_TICKERS = [f"{t}.IS" for t in BIST_TICKERS]

# ─── ANALİZ FONKSİYONU ──────────────────────────────────────────
def analyze_stock(ticker):
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period="6mo", timeout=10)
        info = tk.info
        
        if hist.empty: return None
        
        close = hist["Close"]
        rsi = ta.rsi(close, length=14).iloc[-1]
        
        # Basit Skorlama
        score = 50
        if rsi < 30: score += 30
        elif rsi > 70: score -= 20
        
        price = float(close.iloc[-1])
        change = float((price - close.iloc[-2]) / close.iloc[-2] * 100)
        
        return {
            "Hisse": ticker.replace(".IS", ""),
            "Fiyat": round(price, 2),
            "Değişim": round(change, 2),
            "RSI": round(rsi, 1),
            "Skor": min(100, max(0, score)),
            "Sektör": info.get("sector", "Diğer")
        }
    except:
        return None

# ─── ARAYÜZ ─────────────────────────────────────────────────────
st.title("📈 BIST Radar (Streamlit)")

if "data" not in st.session_state:
    st.session_state.data = []

# Sidebar
st.sidebar.header("Kontroller")
if st.sidebar.button("🔄 Taramayı Başlat", type="primary"):
    with st.spinner("Hisseler taranıyor..."):
        results = []
        bar = st.progress(0)
        for i, t in enumerate(YF_TICKERS):
            res = analyze_stock(t)
            if res: results.append(res)
            bar.progress((i+1)/len(YF_TICKERS))
            time.sleep(0.2) # Hata almamak için yavaşlat
        
        results.sort(key=lambda x: x["Skor"], reverse=True)
        st.session_state.data = results
        st.session_state.time = datetime.now().strftime("%H:%M")
        bar.empty()
        st.success("Tarama Bitti!")

# Sonuçları Göster
if st.session_state.data:
    st.caption(f"Son güncelleme: {st.session_state.time}")
    df = pd.DataFrame(st.session_state.data)
    
    # Filtre
    min_score = st.slider("Minimum Skor", 0, 100, 40)
    filtered = df[df["Skor"] >= min_score]
    
    st.dataframe(filtered, use_container_width=True)
    
    # Detay
    st.subheader("Detay İnceleme")
    cols = st.columns(3)
    for i, row in filtered.head(3).iterrows():
        with cols[i % 3]:
            st.metric(row["Hisse"], f"₺{row['Fiyat']}", f"{row['Değişim']}%")
            st.caption(f"RSI: {row['RSI']} | Skor: {row['Skor']}")
else:
    st.info("👈 Sol taraftaki butona basarak taramayı başlat.")
