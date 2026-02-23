"""
BIST RADAR - Streamlit Versiyonu
Gerçek veri: yfinance (.IS uzantısı)
Teknik analiz: pandas-ta
Deploy: Streamlit Cloud (GitHub)
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import logging
import time
import plotly.graph_objects as go
from datetime import datetime

# ─── SAYFA AYARLARI ────────────────────────────────────────────────
st.set_page_config(page_title="BIST Radar", page_icon="📈", layout="wide")

# ─── LOGGING ──────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

# ─── 350+ BIST HİSSE LİSTESİ ──────────────────────────────────────
BIST_TICKERS = [
    # BANKALAR
    "AKBNK","GARAN","ISCTR","YKBNK","HALKB","VAKBN","QNBFB","ALBRK",
    "SKBNK","TSKB","DENIZ",
    # HOLDİNGLER
    "KCHOL","SAHOL","DOHOL","TKFEN","ISMEN","ISGSY",
    # HAVAYOLLARI & ULAŞIM
    "THYAO","PGSUS","TAVHL","CLEBI",
    # ENERJİ & PETROL
    "TUPRS","PETKM","AYGAZ","AKSEN","ZOREN","ENJSA","AYDEM",
    # SAVUNMA & MAKİNE
    "ASELS","OTKAR","KATMR",
    # OTOMOTİV
    "TOASO","FROTO","ARCLK","KARSN","TTRAK","BFREN","DITAS","JANTS",
    # PERAKENDE
    "BIMAS","MGROS","SOKM","CRFSA","MAVI","BIZIM","TKNSA","VAKKO",
    # TEKNOLOJİ & TELEKOMÜNİKASYON
    "TCELL","TTKOM","LOGO","INDES","ARENA","LINK","DGATE","FONET",
    "PENTA","PKART","MTRKS","NETAS",
    # CAM & SERAMİK
    "SISE","TRKCM","ANACM","KUTPO","EGSER","USAK",
    # İNŞAAT & GYO
    "ENKAI","EKGYO","ISGYO","TRGYO","SNGYO","RYGYO","VKGYO",
    "NUGYO","OZKGY","HLGYO","MSGYO","PGYO","AKMGY",
    # GIDA & İÇECEK
    "ULKER","AEFES","CCOLA","TATGD","DARDL","BANVT","MERKO",
    "KNFRT","SELVA","KENT","TUKAS","ERSU","KERVT",
    # MADENCİLİK
    "KOZAL","KOZAA",
    # KİMYA & İLAÇ
    "ECILC","DEVA","ALKIM","GUBRF","HEKTS","POLHO","BAGFS",
    "GOODY","SEKUR","KLKIM","EPLAS",
    # SİGORTA & FİNANS
    "ANHYT","ANSGR","AKGRT","RAYSG","TURSG","LIDER","UNLU","INFO",
    # TEKSTİL
    "KORDS","SKTAS","SNPAM","YUNSA","SUWEN","BRKO","BOSSA",
    "KRTEK","ARSAN","LUKSK","MNDRS",
    # LOJİSTİK
    "RYSAS","BNTAS","KONTR",
    # ÇİMENTO
    "AKCNS","CIMSA","BOLUC","ADANA","AFYON","GOLTS","KONYA",
    "MRDIN","NUHCM","UNYEC","BUCIM","BASCM","OSMEN",
    # KAĞIT & AMBALAJ
    "KARTN","OLMIP","KAPLM","BAKAB","TEZOL",
    # ELEKTRİK
    "AKENR","GESAN","EMKEL","PAMEL",
    # METAL & DEMİR ÇELİK
    "EREGL","KRDMD","KRDMA","KRDMB","ISDMR","BRSAN","CELHA",
    "SARKY","BORLS","DMSAS","GEDIK","BURCE","OZBAL","IMASM",
    "CEMTS","CEMAS","OYLUM","TUCLK","SIMAS","KRSAN","SAMAT",
    # SAĞLIK
    "MPARK","LKMNH","MEDTR",
    # SPOR
    "BJKAS","FENER","GSRAY",
    # DİĞER
    "SODSN","AEFES","GUBRF","POLHO","KORDS","GENTS",
    "TTRAK","EGPRO","BRMEN","YATAS","BMELK","PRDGS",
    "HUBVC","KAREL","ESCOM","VBTYZ","OBASE","EDATA",
    "MACKO","ARENA","DGATE","LINK","FONET","MTRKS",
]

# Tekrar edenleri temizle ve .IS uzantısı ekle
BIST_TICKERS = list(dict.fromkeys(BIST_TICKERS))
YF_TICKERS = [f"{t}.IS" for t in BIST_TICKERS]

# ─── STREAMLİT CACHE ──────────────────────────────────────────────
@st.cache_data(ttl=300)  # 5 dakika cache
def get_cached_data():
    """Cache'den veri çek (Streamlit standardı)."""
    return None

# ─── TEKNİK ANALİZ FONKSİYONLARI (AYNEN KORUNDU) ─────────────────
def calc_indicators(df: pd.DataFrame) -> dict:
    """Fiyat geçmişinden tüm teknik göstergeleri hesapla."""
    if df is None or len(df) < 30:
        return {}

    close = df["Close"].squeeze()
    high  = df["High"].squeeze()
    low   = df["Low"].squeeze()
    vol   = df["Volume"].squeeze()

    ind = {}

    try:
        # RSI
        rsi = ta.rsi(close, length=14)
        ind["rsi"] = round(float(rsi.iloc[-1]), 1) if rsi is not None and not rsi.empty else 50.0

        # MACD
        macd_df = ta.macd(close, fast=12, slow=26, signal=9)
        if macd_df is not None and not macd_df.empty:
            ind["macd"]        = round(float(macd_df.iloc[-1, 0]), 3)
            ind["macd_signal"] = round(float(macd_df.iloc[-1, 2]), 3)
            ind["macd_hist"]   = round(float(macd_df.iloc[-1, 1]), 3)
        else:
            ind["macd"] = ind["macd_signal"] = ind["macd_hist"] = 0.0

        # Bollinger Bands
        bb = ta.bbands(close, length=20)
        if bb is not None and not bb.empty:
            upper = float(bb.iloc[-1, 0])
            lower = float(bb.iloc[-1, 2])
            c     = float(close.iloc[-1])
            ind["bb_upper"]   = round(upper, 2)
            ind["bb_lower"]   = round(lower, 2)
            ind["bb_percent"] = round((c - lower) / (upper - lower), 3) if (upper - lower) > 0 else 0.5
        else:
            ind["bb_percent"] = 0.5

        # Stochastic RSI
        stoch = ta.stochrsi(close, length=14)
        if stoch is not None and not stoch.empty:
            ind["stoch_rsi"] = round(float(stoch.iloc[-1, 0]) * 100, 1)
        else:
            ind["stoch_rsi"] = 50.0

        # ADX
        adx_df = ta.adx(high, low, close, length=14)
        if adx_df is not None and not adx_df.empty:
            ind["adx"] = round(float(adx_df.iloc[-1, 0]), 1)
        else:
            ind["adx"] = 20.0

        # EMA 50 / 200
        ema50  = ta.ema(close, length=50)
        ema200 = ta.ema(close, length=200)
        if ema50 is not None and ema200 is not None and len(ema50) > 0 and len(ema200) > 0:
            e50 = float(ema50.iloc[-1])
            e200 = float(ema200.iloc[-1])
            ind["ema50"]        = round(e50, 2)
            ind["ema200"]       = round(e200, 2)
            ind["ema_golden"]   = e50 > e200
        else:
            ind["ema50"] = ind["ema200"] = 0.0
            ind["ema_golden"] = False

        # CCI
        cci = ta.cci(high, low, close, length=20)
        ind["cci"] = round(float(cci.iloc[-1]), 1) if cci is not None and not cci.empty else 0.0

        # Williams %R
        willr = ta.willr(high, low, close, length=14)
        ind["williams_r"] = round(float(willr.iloc[-1]), 1) if willr is not None and not willr.empty else -50.0

        # OBV
        obv = ta.obv(close, vol)
        if obv is not None and len(obv) >= 10:
            obv_recent = obv.iloc[-10:]
            ind["obv_trend"] = round(float((obv_recent.iloc[-1] - obv_recent.iloc[0]) / (abs(obv_recent.iloc[0]) + 1e-9)), 3)
        else:
            ind["obv_trend"] = 0.0

        # Momentum
        mom = ta.mom(close, length=10)
        ind["momentum"] = round(float(mom.iloc[-1]), 2) if mom is not None and not mom.empty else 0.0

        # Hacim anomalisi
        if len(vol) >= 21:
            avg_vol = float(vol.iloc[-21:-1].mean())
            last_vol = float(vol.iloc[-1])
            ind["vol_multiplier"] = round(last_vol / avg_vol, 2) if avg_vol > 0 else 1.0
        else:
            ind["vol_multiplier"] = 1.0

        # 52 haftalık high/low
        ind["w52_high"] = round(float(high.iloc[-252:].max()), 2)
        ind["w52_low"]  = round(float(low.iloc[-252:].min()), 2)

        # Fiyat geçmişi
        hist_close = close.iloc[-90:].round(2).tolist()
        hist_vol   = vol.iloc[-90:].astype(int).tolist()
        ind["price_history"]  = hist_close
        ind["volume_history"] = hist_vol

    except Exception as e:
        log.warning(f"Gösterge hesaplama hatası: {e}")

    return ind


def score_stock(info: dict, ind: dict) -> dict:
    """Bileşik fırsat skoru hesapla (AYNEN KORUNDU)."""
    tech_score = 0
    fund_score = 0
    vol_score  = 0
    signals    = []
    flags      = []

    rsi        = ind.get("rsi", 50)
    macd_hist  = ind.get("macd_hist", 0)
    bb_pct     = ind.get("bb_percent", 0.5)
    stoch      = ind.get("stoch_rsi", 50)
    adx        = ind.get("adx", 20)
    ema_golden = ind.get("ema_golden", False)
    cci        = ind.get("cci", 0)
    wr         = ind.get("williams_r", -50)
    obv        = ind.get("obv_trend", 0)
    mom        = ind.get("momentum", 0)
    vol_mult   = ind.get("vol_multiplier", 1)

    # ── Teknik ──
    if rsi < 30:
        tech_score += 22; signals.append("RSI Aşırı Satım")
    elif rsi < 45:
        tech_score += 12
    elif rsi > 72:
        tech_score -= 12

    if macd_hist > 0:
        tech_score += 16; signals.append("MACD Pozitif")
    else:
        tech_score -= 6

    if bb_pct < 0.18:
        tech_score += 20; signals.append("BB Alt Bant Sıkışma")
    elif bb_pct > 0.88:
        tech_score -= 10

    if stoch < 20:
        tech_score += 14; signals.append("Stoch Aşırı Sat")
    elif stoch > 82:
        tech_score -= 8

    if adx > 35:
        tech_score += 10
    if ema_golden:
        tech_score += 14; signals.append("Altın Kesişim EMA")
    if wr < -80:
        tech_score += 10; signals.append("Williams Aşırı Sat")
    if abs(cci) > 150:
        tech_score += 6
    if mom > 5:
        tech_score += 8
    if obv > 0.3:
        tech_score += 8
    tech_score = max(0, min(100, tech_score))

    # ── Temel ──
    pe = info.get("trailingPE") or info.get("forwardPE") or 0
    pb = info.get("priceToBook") or 0
    rev_growth = (info.get("revenueGrowth") or 0) * 100
    ebitda_margin = (info.get("ebitdaMargins") or 0) * 100
    profit_margin = (info.get("profitMargins") or 0) * 100
    de_ratio = info.get("debtToEquity") or 0
    roe = (info.get("returnOnEquity") or 0) * 100

    if 0 < pe < 8:
        fund_score += 25; signals.append("Çok Ucuz F/K")
    elif 0 < pe < 14:
        fund_score += 14; signals.append("Ucuz F/K")
    elif pe > 25:
        fund_score -= 8

    if 0 < pb < 0.8:
        fund_score += 22; signals.append("Defter Altı PD/DD")
    elif 0 < pb < 1.5:
        fund_score += 10

    if rev_growth > 30:
        fund_score += 20; signals.append("Güçlü Büyüme")
    elif rev_growth > 10:
        fund_score += 10
    elif rev_growth < -5:
        fund_score -= 8

    if ebitda_margin > 25:
        fund_score += 8
    if profit_margin > 15:
        fund_score += 8
    if 0 < de_ratio < 50:
        fund_score += 6
    if roe > 20:
        fund_score += 8
    fund_score = max(0, min(100, fund_score))

    # ── Hacim ──
    if vol_mult > 4:
        vol_score += 40
        signals.append("GÜÇLÜ HACİM ANOMALİSİ")
        flags.append({"icon": "🔴", "detail": f"{vol_mult:.1f}x hacim patlaması"})
    elif vol_mult > 2.5:
        vol_score += 25
        signals.append("Yüksek Hacim")
        flags.append({"icon": "🔊", "detail": f"{vol_mult:.1f}x yüksek hacim"})
    elif vol_mult > 1.5:
        vol_score += 12

    if obv > 0.4:
        vol_score += 15
    vol_score = max(0, min(100, vol_score))

    overall = round(tech_score * 0.4 + fund_score * 0.3 + vol_score * 0.3)

    sig_type = "izle"
    if overall >= 68:
        sig_type = "guclu"
    elif overall >= 50:
        sig_type = "al"
    elif overall < 28:
        sig_type = "dikkat"

    return {
        "tech_score": tech_score,
        "fund_score": fund_score,
        "vol_score": vol_score,
        "score": overall,
        "type": sig_type,
        "signals": signals[:6],
        "flags": flags,
        "has_anomaly": vol_mult > 2.5,
    }


def fetch_stock(ticker_is: str) -> dict | None:
    """Tek bir hisseyi yfinance'dan çek ve analiz et."""
    ticker_bist = ticker_is.replace(".IS", "")
    try:
        yf_ticker = yf.Ticker(ticker_is)
        hist = yf_ticker.history(period="1y", timeout=15)
        if hist.empty or len(hist) < 20:
            return None

        info = {}
        try:
            info = yf_ticker.info or {}
        except Exception:
            pass

        last_close   = float(hist["Close"].iloc[-1])
        prev_close   = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else last_close
        day_change   = round((last_close - prev_close) / prev_close * 100, 2)
        mcap_raw     = info.get("marketCap") or 0
        mcap_billion = round(mcap_raw / 1e9, 1)

        ind = calc_indicators(hist)
        if not ind:
            return None

        scoring = score_stock(info, ind)

        sector   = info.get("sector") or info.get("industry") or "Diğer"
        sector_tr = {
            "Financial Services": "Finans",
            "Industrials": "Sanayi",
            "Consumer Defensive": "Savunma Tük.",
            "Consumer Cyclical": "Döngüsel Tük.",
            "Energy": "Enerji",
            "Basic Materials": "Hammadde",
            "Technology": "Teknoloji",
            "Communication Services": "Telekomünikasyon",
            "Healthcare": "Sağlık",
            "Real Estate": "GYO",
            "Utilities": "Kamu Hizmetleri",
        }.get(sector, sector)

        return {
            "ticker":      ticker_bist,
            "name":        info.get("longName") or info.get("shortName") or ticker_bist,
            "sector":      sector_tr,
            "index":       "BIST",
            "price":       round(last_close, 2),
            "day_change":  day_change,
            "mcap":        mcap_billion,
            "rsi":         ind.get("rsi", 50),
            "macd":        ind.get("macd", 0),
            "macd_hist":   ind.get("macd_hist", 0),
            "bb_percent":  ind.get("bb_percent", 0.5),
            "stoch_rsi":   ind.get("stoch_rsi", 50),
            "adx":         ind.get("adx", 20),
            "ema_golden":  ind.get("ema_golden", False),
            "ema50":       ind.get("ema50", 0),
            "ema200":      ind.get("ema200", 0),
            "cci":         ind.get("cci", 0),
            "williams_r":  ind.get("williams_r", -50),
            "obv_trend":   ind.get("obv_trend", 0),
            "momentum":    ind.get("momentum", 0),
            "vol_mult":    ind.get("vol_multiplier", 1),
            "w52_high":    ind.get("w52_high", 0),
            "w52_low":     ind.get("w52_low", 0),
            "pe":          round(info.get("trailingPE") or info.get("forwardPE") or 0, 1),
            "pb":          round(info.get("priceToBook") or 0, 2),
            "rev_growth":  round((info.get("revenueGrowth") or 0) * 100, 1),
            "ebitda_margin": round((info.get("ebitdaMargins") or 0) * 100, 1),
            "net_margin":  round((info.get("profitMargins") or 0) * 100, 1),
            "de_ratio":    round(info.get("debtToEquity") or 0, 2),
            "roe":         round((info.get("returnOnEquity") or 0) * 100, 1),
            "foreign_pct": round(info.get("heldPercentInstitutions", 0) * 100, 1),
            "price_history":  ind.get("price_history", []),
            "volume_history": ind.get("volume_history", []),
            **scoring,
        }

    except Exception as e:
        log.warning(f"{ticker_is} hatası: {e}")
        return None


def run_full_scan(progress_bar=None, status_text=None):
    """Tüm hisseleri tara ve sonuçları döndür."""
    results = []
    start = time.time()

    for i, ticker in enumerate(YF_TICKERS):
        result = fetch_stock(ticker)
        if result:
            results.append(result)
        
        # İlerleme çubuğunu güncelle
        if progress_bar:
            progress_bar.progress((i + 1) / len(YF_TICKERS))
        if status_text and i % 20 == 0:
            status_text.text(f"İşleniyor: {i+1}/{len(YF_TICKERS)} — {ticker}")
        
        time.sleep(0.15)  # Streamlit Cloud rate limit için daha agresif bekleme

    results.sort(key=lambda x: x["score"], reverse=True)
    elapsed = round(time.time() - start, 1)
    log.info(f"Tarama tamamlandı — {len(results)} hisse — {elapsed}s")
    
    return results, datetime.now().isoformat()


# ─── STREAMLİT ARAYÜZÜ ────────────────────────────────────────────
def main():
    st.title("📈 BIST Radar Pro")
    st.markdown("*Teknik analiz + Temel analiz + Hacim anomalisi ile fırsat tarama*")

    # Sidebar filtreleri
    st.sidebar.header("🔍 Filtreler")
    
    min_score = st.sidebar.slider("Minimum Skor", 0, 100, 50)
    sector_filter = st.sidebar.multiselect("Sektör Seç", options=["Tümü"] + list(set(["Finans", "Sanayi", "Enerji", "Teknoloji", "GYO", "Hammadde"])), default=["Tümü"])
    show_only_anomaly = st.sidebar.checkbox("Sadece Hacim Anomalisi", value=False)
    
    # Ana buton
    if st.sidebar.button("🔄 Taramayı Başlat", type="primary"):
        with st.spinner("BIST hisseleri taranıyor... Bu işlem 2-4 dakika sürebilir."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results, last_update = run_full_scan(progress_bar, status_text)
            
            # Session state'e kaydet
            st.session_state["results"] = results
            st.session_state["last_update"] = last_update
            st.session_state["has_data"] = True
            
            progress_bar.empty()
            status_text.empty()
            st.success(f"✅ Tarama tamamlandı! {len(results)} hisse analiz edildi.")

    # Eğer veri varsa göster
    if st.session_state.get("has_data"):
        results = st.session_state["results"]
        last_update = st.session_state.get("last_update", "Bilinmiyor")
        
        st.caption(f"🕐 Son güncelleme: {last_update}")
        
        # Filtreleme
        filtered = [r for r in results if r["score"] >= min_score]
        if show_only_anomaly:
            filtered = [r for r in filtered if r.get("has_anomaly", False)]
        
        # Özet kartları
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Toplam Hisse", len(results))
        with col2:
            st.metric("Filtrelenen", len(filtered))
        with col3:
            guclu_count = len([r for r in filtered if r["type"] == "guclu"])
            st.metric("🔥 Güçlü Sinyal", guclu_count)
        
        # Tablo görünümü
        st.subheader("📊 Tarama Sonuçları")
        
        # Tablo için sütunları hazırla
        table_data = []
        for r in filtered:
            table_data.append({
                "Hisse": r["ticker"],
                "Fiyat": f"₺{r['price']}",
                "Günlük": f"{r['day_change']:+.2f}%",
                "Skor": r["score"],
                "Sinyal": r["type"].upper(),
                "RSI": r["rsi"],
                "Hacim": f"{r['vol_mult']}x",
                "Sektör": r["sector"]
            })
        
        df = pd.DataFrame(table_data)
        
        # Skor ve sinyal bazlı renklendirme
        def color_score(val):
            if val >= 68:
                return "background-color: #22c55e; color: white"
            elif val >= 50:
                return "background-color: #eab308; color: black"
            elif val < 28:
                return "background-color: #ef4444; color: white"
            return ""
        
        st.dataframe(
            df.style.map(color_score, subset=["Skor"]),
            use_container_width=True,
            hide_index=True
        )
        
        # Detaylı inceleme (expander)
        st.subheader("🔍 Hisse Detay İnceleme")
        selected_ticker = st.selectbox("Hisse Seç", [r["ticker"] for r in filtered])
        
        if selected_ticker:
            selected_data = next((r for r in results if r["ticker"] == selected_ticker), None)
            if selected_data:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Fiyat grafiği
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=list(range(len(selected_data["price_history"]))),
                        y=selected_data["price_history"],
                        mode="lines",
                        name="Fiyat",
                        line=dict(color="#3b82f6", width=2)
                    ))
                    fig.update_layout(
                        title=f"{selected_ticker} - 90 Günlük Fiyat",
                        xaxis_title="Gün",
                        yaxis_title="Fiyat (₺)",
                        template="plotly_white",
                        height=300,
                        margin=dict(l=20, r=20, t=40, b=20)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Özet kartları
                    st.metric("Fiyat", f"₺{selected_data['price']}", f"{selected_data['day_change']:+.2f}%")
                    st.metric("Skor", selected_data["score"], 
                             delta="GÜÇLÜ" if selected_data["score"]>=68 else "AL" if selected_data["score"]>=50 else "İZLE")
                    st.metric("RSI", selected_data["rsi"])
                    st.metric("MACD", f"{selected_data['macd_hist']:+.3f}")
                    st.metric("Hacim Çarpanı", f"{selected_data['vol_mult']}x")
                    
                    if selected_data["signals"]:
                        st.markdown("📌 **Sinyaller:**")
                        for sig in selected_data["signals"]:
                            st.caption(f"• {sig}")
                
                # Teknik detaylar
                with st.expander("📋 Tüm Teknik Veriler"):
                    st.json({
                        "EMA50": selected_data["ema50"],
                        "EMA200": selected_data["ema200"],
                        "Altın Kesişim": "✅" if selected_data["ema_golden"] else "❌",
                        "BB %": selected_data["bb_percent"],
                        "Stoch RSI": selected_data["stoch_rsi"],
                        "ADX": selected_data["adx"],
                        "CCI": selected_data["cci"],
                        "Williams %R": selected_data["williams_r"]
                    })
                
                # Temel veriler
                with st.expander("🏢 Temel Analiz Verileri"):
                    st.json({
                        "F/K": selected_data["pe"],
                        "PD/DD": selected_data["pb"],
                        "Büyüme %": selected_data["rev_growth"],
                        "FAVÖK Marj %": selected_data["ebitda_margin"],
                        "Net Kar Marj %": selected_data["net_margin"],
                        "Borç/Özkaynak": selected_data["de_ratio"],
                        "Özkaynak Kârlılığı %": selected_data["roe"]
                    })
    
    else:
        # İlk açılış ekranı
        st.info("👈 Sol menüden **'Taramayı Başlat'** butonuna basarak analizi başlatabilirsin.")
        st.markdown("""
        ### 🎯 Nasıl Kullanılır?
        1. **Taramayı Başlat** butonuna tıkla (2-4 dk sürer)
        2. Filtrelerle hisseleri daralt (Skor, Sektör, Hacim)
        3. Tablodan ilgilendiğin hisseyi seç
        4. Grafik ve detay verilerini incele
        
        > ⚠️ **Not:** Streamlit Cloud ücretsiz planda çalıştığı için ilk tarama biraz yavaş olabilir. Sabırlı ol kanka! 🚀
        """)


if __name__ == "__main__":
    # Session state başlat
    if "has_data" not in st.session_state:
        st.session_state["has_data"] = False
    if "results" not in st.session_state:
        st.session_state["results"] = []
    
    main()
