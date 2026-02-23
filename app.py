"""
BIST RADAR - Flask Backend
Gerçek veri: yfinance (.IS uzantısı)
Teknik analiz: pandas-ta
Deploy: Render.com (ücretsiz)
"""

from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
from apscheduler.schedulers.background import BackgroundScheduler
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import logging
import os
import json
import time
from datetime import datetime, timedelta

# ─── SETUP ────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static')
CORS(app)

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
YF_TICKERS   = [f"{t}.IS" for t in BIST_TICKERS]

# ─── CACHE ────────────────────────────────────────────────────────
_cache = {
    "data": [],
    "market": {},
    "last_update": None,
    "is_loading": False,
}

# ─── TEKNİK ANALİZ ────────────────────────────────────────────────
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

        # Hacim anomalisi (son gün / 20 günlük ortalama)
        if len(vol) >= 21:
            avg_vol = float(vol.iloc[-21:-1].mean())
            last_vol = float(vol.iloc[-1])
            ind["vol_multiplier"] = round(last_vol / avg_vol, 2) if avg_vol > 0 else 1.0
        else:
            ind["vol_multiplier"] = 1.0

        # 52 haftalık high/low
        ind["w52_high"] = round(float(high.iloc[-252:].max()), 2)
        ind["w52_low"]  = round(float(low.iloc[-252:].min()), 2)

        # Fiyat geçmişi (grafik için son 90 kapanış)
        hist_close = close.iloc[-90:].round(2).tolist()
        hist_vol   = vol.iloc[-90:].astype(int).tolist()
        ind["price_history"]  = hist_close
        ind["volume_history"] = hist_vol

    except Exception as e:
        log.warning(f"Gösterge hesaplama hatası: {e}")

    return ind


def score_stock(info: dict, ind: dict) -> dict:
    """Bileşik fırsat skoru hesapla."""
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


# ─── VERİ ÇEKME ───────────────────────────────────────────────────
def fetch_stock(ticker_is: str) -> dict | None:
    """Tek bir hisseyi yfinance'dan çek ve analiz et."""
    ticker_bist = ticker_is.replace(".IS", "")
    try:
        yf_ticker = yf.Ticker(ticker_is)

        # Geçmiş veri (1 yıl günlük)
        hist = yf_ticker.history(period="1y", timeout=15)
        if hist.empty or len(hist) < 20:
            return None

        info = {}
        try:
            info = yf_ticker.info or {}
        except Exception:
            pass

        # Temel fiyat bilgileri
        last_close   = float(hist["Close"].iloc[-1])
        prev_close   = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else last_close
        day_change   = round((last_close - prev_close) / prev_close * 100, 2)
        mcap_raw     = info.get("marketCap") or 0
        mcap_billion = round(mcap_raw / 1e9, 1)  # Milyar TL

        # Göstergeler
        ind = calc_indicators(hist)
        if not ind:
            return None

        # Skorlama
        scoring = score_stock(info, ind)

        # Sektör bilgisi
        sector   = info.get("sector") or info.get("industry") or "Diğer"
        # Türkçe sektör çevirisi
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
            # Fiyat
            "price":       round(last_close, 2),
            "day_change":  day_change,
            "mcap":        mcap_billion,
            # Teknik göstergeler
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
            # Temel
            "pe":          round(info.get("trailingPE") or info.get("forwardPE") or 0, 1),
            "pb":          round(info.get("priceToBook") or 0, 2),
            "rev_growth":  round((info.get("revenueGrowth") or 0) * 100, 1),
            "ebitda_margin": round((info.get("ebitdaMargins") or 0) * 100, 1),
            "net_margin":  round((info.get("profitMargins") or 0) * 100, 1),
            "de_ratio":    round(info.get("debtToEquity") or 0, 2),
            "roe":         round((info.get("returnOnEquity") or 0) * 100, 1),
            "foreign_pct": round(info.get("heldPercentInstitutions", 0) * 100, 1),
            # Grafik verisi
            "price_history":  ind.get("price_history", []),
            "volume_history": ind.get("volume_history", []),
            # Skorlar
            **scoring,
        }

    except Exception as e:
        log.warning(f"{ticker_is} hatası: {e}")
        return None


def run_full_scan():
    """Tüm hisseleri tara ve cache'e kaydet."""
    if _cache["is_loading"]:
        log.info("Zaten taranıyor, atlanıyor.")
        return

    _cache["is_loading"] = True
    log.info(f"Tarama başladı — {len(YF_TICKERS)} hisse")
    results = []
    start = time.time()

    for i, ticker in enumerate(YF_TICKERS):
        result = fetch_stock(ticker)
        if result:
            results.append(result)
        if i % 20 == 0:
            log.info(f"  {i+1}/{len(YF_TICKERS)} — {ticker} — {len(results)} başarılı")
        time.sleep(0.4)  # Rate limit için bekleme

    results.sort(key=lambda x: x["score"], reverse=True)
    _cache["data"] = results
    _cache["last_update"] = datetime.now().isoformat()
    _cache["is_loading"] = False

    elapsed = round(time.time() - start, 1)
    log.info(f"Tarama tamamlandı — {len(results)} hisse — {elapsed}s")


# ─── FLASK ROUTES ─────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/scan")
def api_scan():
    """Tam tarama sonuçlarını döndür."""
    if not _cache["data"]:
        return jsonify({
            "status": "loading",
            "message": "İlk tarama devam ediyor, lütfen bekleyin (2-4 dakika)...",
            "data": [],
        })

    return jsonify({
        "status": "ok",
        "count":  len(_cache["data"]),
        "last_update": _cache["last_update"],
        "is_loading":  _cache["is_loading"],
        "data":  _cache["data"],
    })


@app.route("/api/stock/<ticker>")
def api_stock(ticker):
    """Tek hisse anlık veri."""
    ticker_is = f"{ticker.upper()}.IS"
    result = fetch_stock(ticker_is)
    if not result:
        return jsonify({"error": "Hisse bulunamadı"}), 404
    return jsonify(result)


@app.route("/api/status")
def api_status():
    return jsonify({
        "status": "ok",
        "cached_stocks": len(_cache["data"]),
        "last_update": _cache["last_update"],
        "is_loading": _cache["is_loading"],
    })


@app.route("/api/refresh", methods=["POST"])
def api_refresh():
    """Manuel yenileme tetikle."""
    if _cache["is_loading"]:
        return jsonify({"message": "Zaten taranıyor..."}), 409
    import threading
    threading.Thread(target=run_full_scan, daemon=True).start()
    return jsonify({"message": "Tarama başlatıldı"})


# ─── BAŞLANGIÇ ────────────────────────────────────────────────────
if __name__ == "__main__":
    # İlk taramayı arka planda başlat
    import threading
    log.info("BIST Radar başlatılıyor...")
    threading.Thread(target=run_full_scan, daemon=True).start()

    # Her 4 saatte bir otomatik yenile
    scheduler = BackgroundScheduler()
    scheduler.add_job(run_full_scan, "interval", hours=4, id="auto_scan")
    scheduler.start()

    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
