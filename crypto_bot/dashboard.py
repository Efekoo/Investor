import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import json
from pathlib import Path
from sqlalchemy import create_engine, text
import time

st.set_page_config(page_title="Crypto Bot Dashboard", layout="wide", page_icon="📈")

# Veritabanı URL'sini al ve temizle
DB_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@db:5432/investor_db")
STATE_FILE = os.getenv("STATE_FILE", "/app/runtime/state.json")

@st.cache_resource
def get_engine():
    # PostgreSQL için ekstra parametreleri temizliyoruz
    return create_engine(DB_URL)

def load_data(query: str) -> pd.DataFrame:
    try:
        engine = get_engine()
        with engine.connect() as conn:
            return pd.read_sql(text(query), conn)
    except Exception as e:
        st.error(f"DB Bağlantı Hatası: {e}")
        # Hata detayını yan menüde göster (Geliştirici için)
        st.sidebar.error(f"Detay: {str(e)}")
        return pd.DataFrame()

def load_state() -> dict:
    try:
        p = Path(STATE_FILE)
        if p.exists():
            return json.loads(p.read_text())
    except Exception as e:
        st.sidebar.warning(f"State dosyası okunamadı: {e}")
    return {}

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Kontrol Paneli")
refresh_rate = st.sidebar.slider("Otomatik yenileme (sn)", 5, 60, 10)
if st.sidebar.button("🔄 Manuel Yenile"):
    st.cache_resource.clear()
    st.rerun()

st.sidebar.info(f"Bağlanılan DB: {DB_URL.split('@')[-1] if '@' in DB_URL else 'Local'}")

state = load_state()

# ── Başlık ───────────────────────────────────────────────────────────────────
st.title("🚀 Crypto Trading Bot Dashboard")

# ── 1. Anlık Durum (state.json'dan) ──────────────────────────────────────────
st.subheader("💼 Anlık Durum")

paper_cash = float(state.get("paper_cash", 0.0))
last_prices = state.get("last_prices", {})
portfolio = state.get("portfolio", {})
positions_raw = portfolio.get("positions", {})

# Pozisyonlar liste veya sözlük olabilir, her iki durumu da yönetelim
if isinstance(positions_raw, list):
    positions = {p["symbol"]: p for p in positions_raw if isinstance(p, dict) and "symbol" in p}
else:
    positions = positions_raw

total_equity = paper_cash
if isinstance(positions, dict):
    for sym, pos in positions.items():
        qty = float(pos.get("qty", 0))
        mark = float(last_prices.get(sym, pos.get("entry_price", 0)))
        total_equity += qty * mark

col1, col2, col3, col4 = st.columns(4)
col1.metric("Nakit Bakiye", f"${paper_cash:,.2f}")
col2.metric("Toplam Varlık", f"${total_equity:,.2f}")
col3.metric("Açık Pozisyon", len(positions))
col4.metric("Mod", "PAPER 🟡")

# Açık pozisyonlar tablosu
if positions:
    rows = []
    for sym, pos in positions.items():
        entry = float(pos.get("entry_price", 0))
        qty = float(pos.get("qty", 0))
        mark = float(last_prices.get(sym, entry))
        pnl = (mark - entry) * qty
        pnl_pct = (mark - entry) / entry * 100 if entry > 0 else 0
        rows.append({
            "Sembol": sym,
            "Giriş": f"${entry:,.2f}",
            "Güncel": f"${mark:,.2f}",
            "Miktar": f"{qty:.6f}",
            "PnL ($)": f"${pnl:+,.2f}",
            "PnL (%)": f"{pnl_pct:+.2f}%",
            "Strateji": pos.get("strategy_name", "-"),
        })
    st.table(pd.DataFrame(rows))
else:
    st.info("Şu an açık pozisyon yok.")

st.divider()

# ── 2. Equity Eğrisi ─────────────────────────────────────────────────────────
st.subheader("📊 Equity Grafiği")
balance_df = load_data("SELECT * FROM balance_history ORDER BY timestamp ASC")

if not balance_df.empty:
    fig = px.line(balance_df, x='timestamp', y='total_balance', title='Toplam Varlık Gelişimi')
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Veritabanında henüz bakiye geçmişi bulunamadı.")

st.divider()

# ── 3. İşlem Geçmişi ─────────────────────────────────────────────────────────
st.subheader("📜 Son İşlemler")
trades_df = load_data("SELECT * FROM trades ORDER BY opened_at DESC LIMIT 20")

if not trades_df.empty:
    st.dataframe(trades_df, use_container_width=True)
else:
    st.info("Henüz tamamlanmış işlem yok.")

# ── Otomatik Yenileme ─────────────────────────────────────────────────────────
time.sleep(refresh_rate)
st.rerun()
