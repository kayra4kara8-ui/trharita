import json, textwrap, os, pandas as pd, numpy as np
from pathlib import Path

app_code = r'''
# -*- coding: utf-8 -*-
"""
📊 TİCARİ PORTFÖY ANALİZ PLATFORMU (Production-Ready / Single File)
Territory × Zaman × Coğrafya × Rekabet × Tahminleme

- streamlit, pandas, numpy, plotly, sklearn, openpyxl
- Türkiye GeoJSON (il bazlı) ile tam eşleşme + toleranslı normalizasyon
- KPI Dashboard, Zaman Serisi & Tahmin, Türkiye Haritası, BCG & Segmentasyon, Manager Scorecard, Raporlama (Excel + opsiyonel PDF)

Çalıştırma:
    streamlit run ticari_portfoy_app.py
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, r2_score

warnings.filterwarnings("ignore")

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Ticari Portföy Analiz Platformu",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# UI - CSS (Corporate)
# =============================================================================
st.markdown(
    """
<style>
/* Header */
.main-header{
    font-size:2.6rem;font-weight:800;text-align:center;padding:1.2rem 0;margin:0.2rem 0 1.2rem 0;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
}

/* Cards */
.kpi-wrap{display:flex;gap:0.8rem;flex-wrap:wrap;margin-top:0.2rem;margin-bottom:1rem}
.kpi-card{
    flex:1 1 220px;background:white;border-radius:14px;padding:1rem 1.1rem;
    box-shadow:0 6px 18px rgba(0,0,0,0.08);
    border-left:5px solid #3B82F6;
}
.kpi-title{font-size:0.85rem;color:#6B7280;font-weight:600;margin-bottom:0.4rem}
.kpi-value{font-size:1.8rem;color:#111827;font-weight:800;line-height:1.2}
.kpi-sub{font-size:0.85rem;color:#374151;font-weight:600;margin-top:0.25rem}
.badge{
    display:inline-block;padding:0.18rem 0.55rem;border-radius:999px;font-weight:700;font-size:0.78rem;
    background:#EEF2FF;color:#3730A3;margin-left:0.5rem
}

/* Tabs */
.stTabs [data-baseweb="tab-list"]{gap:0.8rem;background:#F9FAFB;padding:0.6rem;border-radius:12px}
.stTabs [data-baseweb="tab"]{height:3.1rem;padding:0 1.2rem;background:#fff;border-radius:10px;font-weight:700}
.stTabs [aria-selected="true"]{
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white !important;
}

/* Small helper */
.small-note{font-size:0.86rem;color:#6B7280}
</style>
""",
    unsafe_allow_html=True,
)

# =============================================================================
# CONSTANTS
# =============================================================================

# Zorunlu kolonlar
REQUIRED_COLS = ["DATE", "MANAGER", "TERRITORIES", "REGION", "CITY"]

# Ürün kolonları (PF + Rakip)
PRODUCT_MAP: Dict[str, Dict[str, str]] = {
    "TROCMETAM": {"pf": "TROCMETAM", "rakip": "DIGER TROCMETAM"},
    "CORTIPOL": {"pf": "CORTIPOL", "rakip": "DIGER CORTIPOL"},
    "DEKSAMETAZON": {"pf": "DEKSAMETAZON", "rakip": "DIGER DEKSAMETAZON"},
    "PF IZOTONIK": {"pf": "PF IZOTONIK", "rakip": "DIGER IZOTONIK"},
}

# Kullanıcı spesindeki il listesi (kanonik)
CANONICAL_CITIES: List[str] = """
Adana Adıyaman Afyonkarahisar Ağrı Aksaray Ankara Amasya Antalya Artvin İstanbul Aydın Balıkesir Bilecik Bolu İzmir Burdur Bursa
Çanakkale Çankırı Çorum Denizli Diyarbakır Edirne Elâzığ Erzincan Erzurum Eskişehir Gaziantep Kocaeli Giresun Gümüşhane Van
Hatay Isparta Kahramanmaraş Karabük Kars Kastamonu Kayseri Kırıkkale Kırklareli Kırşehir Konya Karaman Kütahya Malatya Manisa
Mardin Mersin Muğla Muş Nevşehir Niğde Ordu Rize Sakarya Samsun Sinop Sivas Tekirdağ Tokat Trabzon Şanlıurfa Uşak Yalova
Yozgat Zonguldak
""".split()

# Bazı excel varyasyonları -> kanonik
CITY_ALIAS = {
    "K MARAS": "Kahramanmaraş",
    "K. MARAS": "Kahramanmaraş",
    "KAHRAMANMARAS": "Kahramanmaraş",
    "SANLIURFA": "Şanlıurfa",
    "SIRNAK": "Şanlıurfa" if False else "Şırnak",  # dikkat: typo guard
    "AFYON": "Afyonkarahisar",
    "TEKIRDAG": "Tekirdağ",
    "CANAKKALE": "Çanakkale",
    "CANKIRI": "Çankırı",
    "CORUM": "Çorum",
    "DUZCE": "Düzce",
    "ESKISEHIR": "Eskişehir",
    "GUMUSHANE": "Gümüşhane",
    "IGDIR": "Iğdır",
    "ISTANBUL": "İstanbul",
    "IZMIR": "İzmir",
    "ELAZIG": "Elâzığ",
    "BINGOL": "Bingöl",
    "AGRI": "Ağrı",
    "MUGLA": "Muğla",
    "MUS": "Muş",
    "NEVSEHIR": "Nevşehir",
    "NIGDE": "Niğde",
    "USAK": "Uşak",
    "HAKKARI": "Hakkari",
}

# =============================================================================
# UTILS
# =============================================================================

def safe_divide(a, b):
    a = np.asarray(a, dtype="float64")
    b = np.asarray(b, dtype="float64")
    return np.where(b != 0, a / b, 0.0)

def fmt_num(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "0"
    x = float(x)
    if abs(x) >= 1_000_000:
        return f"{x/1_000_000:.1f}M"
    if abs(x) >= 1_000:
        return f"{x/1_000:.1f}K"
    return f"{x:,.0f}"

def fix_mojibake(s: str) -> str:
    """
    GeoJSON gibi kaynaklarda görülebilen Ã¼ / ÄŸ vb. bozulmaları düzeltmeye çalışır.
    """
    try:
        if any(ch in s for ch in ["Ã", "Ä", "Å", "Ð", "Þ", "Ý"]):
            return s.encode("latin1").decode("utf-8")
    except Exception:
        pass
    return s

def norm_key(s: str) -> str:
    """
    Eşleşme anahtarı: Türkçe karakterleri ASCII'ye yaklaştırıp, boşluk/nokta kaldırır, uppercase yapar.
    """
    if s is None:
        return ""
    s = str(s).strip()
    s = fix_mojibake(s)
    tr = {
        "İ": "I", "I": "I", "ı": "I",
        "Ş": "S", "ş": "S",
        "Ğ": "G", "ğ": "G",
        "Ü": "U", "ü": "U",
        "Ö": "O", "ö": "O",
        "Ç": "C", "ç": "C",
        "Â": "A", "â": "A",
        "Ê": "E", "ê": "E",
        "Î": "I", "î": "I",
        "Û": "U", "û": "U",
        "Á": "A", "À": "A",
        "É": "E", "È": "E",
    }
    for k, v in tr.items():
        s = s.replace(k, v)
    s = s.upper()
    s = s.replace(".", " ").replace("-", " ")
    s = " ".join(s.split())
    s = s.replace(" ", "")
    return s

def canonical_city(city: str, canonical_map: Dict[str, str]) -> Optional[str]:
    """
    Excel CITY -> kanonik şehir adı.
    Tolerans: alias map + kanonik map.
    """
    if pd.isna(city):
        return None
    raw = str(city).strip()
    if not raw:
        return None
    key = norm_key(raw)
    # alias
    alias_key = norm_key(raw.replace(".", " "))
    if alias_key in {norm_key(k): v for k, v in CITY_ALIAS.items()}:
        # rebuild alias map once is expensive; but acceptable small. We'll do faster below in global cache.
        pass
    return canonical_map.get(key) or canonical_map.get(norm_key(CITY_ALIAS.get(raw.upper().strip(), ""))) or raw.strip()

@st.cache_data(show_spinner=False)
def build_canonical_map() -> Dict[str, str]:
    """
    Kanonik şehir listesi -> {norm_key: canonical}
    """
    m = {}
    for c in CANONICAL_CITIES:
        m[norm_key(c)] = c
    # aliasları da ekle
    for k, v in CITY_ALIAS.items():
        m[norm_key(k)] = v
    return m

def get_product_cols(product: str) -> Dict[str, str]:
    return PRODUCT_MAP.get(product, {"pf": product, "rakip": f"DIGER {product}"})

# =============================================================================
# DATA LOADING & VALIDATION
# =============================================================================

@st.cache_data(show_spinner=False)
def load_excel(file) -> Tuple[Optional[pd.DataFrame], List[str]]:
    """
    Excel oku + doğrula + normalize et.
    Return: (df, warnings)
    """
    notes: List[str] = []
    try:
        df = pd.read_excel(file)
    except Exception as e:
        return None, [f"Excel okunamadı: {e}"]

    # Kolon doğrulama
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        return None, [f"Eksik zorunlu kolon(lar): {', '.join(missing)}"]

    # DATE parse
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["DATE"]).copy()
    dropped = before - len(df)
    if dropped > 0:
        notes.append(f"DATE parse edilemeyen {dropped} satır atlandı.")

    # Standardize strings
    for col in ["MANAGER", "TERRITORIES", "REGION"]:
        df[col] = df[col].astype(str).str.strip().str.upper()

    # City normalize (canonical)
    canon_map = build_canonical_map()
    df["CITY"] = df["CITY"].astype(str).str.strip()
    df["CITY_CANON"] = df["CITY"].apply(lambda x: canon_map.get(norm_key(x)) if pd.notna(x) else None)

    missing_city = df["CITY_CANON"].isna().sum()
    if missing_city:
        notes.append(f"Şehir eşleşemeyen {missing_city} satır var (CITY_CANON boş). Haritada 'Bilinmiyor' altında toplanır.")

    df["CITY_CANON"] = df["CITY_CANON"].fillna("Bilinmiyor")

    # Time fields
    df["YIL_AY"] = df["DATE"].dt.to_period("M").astype(str)
    df["YIL"] = df["DATE"].dt.year
    df["AY"] = df["DATE"].dt.month
    df["QUARTER"] = df["DATE"].dt.quarter

    # Numeric columns: zorunlu + hesap kolonları hariç kalanlar
    reserved = set(REQUIRED_COLS + ["CITY_CANON", "YIL_AY", "YIL", "AY", "QUARTER"])
    num_cols = [c for c in df.columns if c not in reserved]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    return df, notes

@st.cache_data(show_spinner=False)
def load_geojson(path: str) -> Tuple[dict, Dict[str, str], List[str]]:
    """
    GeoJSON yükle, properties.name alanını kanonik şehir isimlerine normalize edecek şekilde mapping çıkar.
    Returns: (geojson, geo_key_to_canonical, warnings)
      - geo_key_to_canonical: norm_key(geo_feature_name) -> canonical_city
    """
    warn: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        geo = json.load(f)

    canon_map = build_canonical_map()

    # geojson name list
    features = geo.get("features", [])
    geo_map: Dict[str, str] = {}
    updated = 0
    unmatched = 0

    for ft in features:
        props = ft.get("properties", {})
        nm = props.get("name", "")
        nm_fixed = fix_mojibake(str(nm))
        k = norm_key(nm_fixed)

        canon = canon_map.get(k)
        if canon is None:
            # bazı geojson'lar İngilizce/ascii olabilir; yine de title case yapıp dene
            canon = canon_map.get(norm_key(nm_fixed.title()))
        if canon is None:
            unmatched += 1
            canon = nm_fixed  # fallback
        else:
            updated += 1

        # Choropleth için feature name'i kanonik yap (eşleşme kalitesi artar)
        ft.setdefault("properties", {})["name"] = canon
        geo_map[k] = canon

    if unmatched:
        warn.append(f"GeoJSON içinde kanonik listeyle eşleşemeyen {unmatched} il ismi var (yine de harita çalışır).")
    if updated:
        warn.append(f"GeoJSON il isimleri normalize edildi (feature.properties.name güncellendi).")

    return geo, geo_map, warn

# =============================================================================
# CORE ANALYTICS
# =============================================================================

def filter_df(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp,
              territory: str, region: str, manager: str) -> pd.DataFrame:
    d = df[(df["DATE"] >= start) & (df["DATE"] <= end)].copy()
    if territory != "TÜMÜ":
        d = d[d["TERRITORIES"] == territory]
    if region != "TÜMÜ":
        d = d[d["REGION"] == region]
    if manager != "TÜMÜ":
        d = d[d["MANAGER"] == manager]
    return d

def territory_perf(df: pd.DataFrame, product: str) -> pd.DataFrame:
    cols = get_product_cols(product)
    for c in [cols["pf"], cols["rakip"]]:
        if c not in df.columns:
            return pd.DataFrame()

    group_cols = ["TERRITORIES"]
    for c in ["REGION", "CITY_CANON", "MANAGER"]:
        if c in df.columns:
            group_cols.append(c)

    out = df.groupby(group_cols, as_index=False).agg({cols["pf"]: "sum", cols["rakip"]: "sum"})
    out = out.rename(columns={cols["pf"]: "PF_Satis", cols["rakip"]: "Rakip_Satis"})
    out["Toplam_Pazar"] = out["PF_Satis"] + out["Rakip_Satis"]
    out["Pazar_Payi_%"] = safe_divide(out["PF_Satis"], out["Toplam_Pazar"]) * 100
    out["Goreceli_Pazar_Payi"] = safe_divide(out["PF_Satis"], out["Rakip_Satis"])
    out["Buyume_Potansiyeli"] = out["Toplam_Pazar"] - out["PF_Satis"]
    total_pf = out["PF_Satis"].sum()
    out["Agirlik_%"] = safe_divide(out["PF_Satis"], total_pf) * 100
    return out.sort_values("PF_Satis", ascending=False)

def time_series_monthly(df: pd.DataFrame, product: str, territory: str = "TÜMÜ") -> pd.DataFrame:
    cols = get_product_cols(product)
    if cols["pf"] not in df.columns or cols["rakip"] not in df.columns:
        return pd.DataFrame()

    d = df.copy()
    if territory != "TÜMÜ":
        d = d[d["TERRITORIES"] == territory]

    ts = d.groupby("YIL_AY", as_index=False).agg({cols["pf"]: "sum", cols["rakip"]: "sum"}).sort_values("YIL_AY")
    ts = ts.rename(columns={"YIL_AY": "Period", cols["pf"]: "PF_Satis", cols["rakip"]: "Rakip_Satis"})
    ts["Toplam_Pazar"] = ts["PF_Satis"] + ts["Rakip_Satis"]
    ts["Pazar_Payi_%"] = safe_divide(ts["PF_Satis"], ts["Toplam_Pazar"]) * 100

    ts["PF_Buyume_%"] = ts["PF_Satis"].pct_change() * 100
    ts["Rakip_Buyume_%"] = ts["Rakip_Satis"].pct_change() * 100

    ts["MA_3"] = ts["PF_Satis"].rolling(window=3, min_periods=1).mean()
    ts["MA_6"] = ts["PF_Satis"].rolling(window=6, min_periods=1).mean()

    # Trend (LR)
    if len(ts) >= 3:
        x = np.arange(len(ts)).reshape(-1, 1)
        y = ts["PF_Satis"].values
        model = LinearRegression().fit(x, y)
        ts["Trend"] = model.predict(x)
    else:
        ts["Trend"] = ts["PF_Satis"]

    return ts

def forecast_lr(ts: pd.DataFrame, months_ahead: int = 6) -> Tuple[Optional[pd.DataFrame], Dict[str, float]]:
    if ts is None or len(ts) < 3:
        return None, {}

    x = np.arange(len(ts)).reshape(-1, 1)
    y = ts["PF_Satis"].values.astype(float)

    model = LinearRegression().fit(x, y)
    y_pred = model.predict(x)

    # Future
    future_x = np.arange(len(ts), len(ts) + months_ahead).reshape(-1, 1)
    forecast = model.predict(future_x)

    # Future periods
    last_period = ts["Period"].iloc[-1]
    future_periods = pd.period_range(start=last_period, periods=months_ahead + 1, freq="M")[1:].astype(str)

    fdf = pd.DataFrame({"Period": future_periods, "PF_Satis_Tahmin": forecast})

    metrics = {
        "MAPE_%": float(mean_absolute_percentage_error(y, y_pred) * 100),
        "R2": float(r2_score(y, y_pred)),
        "Ortalama_MAE": float(np.mean(np.abs(y - y_pred))),
    }
    return fdf, metrics

def bcg_matrix(df: pd.DataFrame, product: str) -> pd.DataFrame:
    terr = territory_perf(df, product)
    if terr.empty:
        return terr

    cols = get_product_cols(product)
    d = df.sort_values("DATE")
    mid = len(d) // 2
    first = d.iloc[:mid].groupby("TERRITORIES")[cols["pf"]].sum()
    second = d.iloc[mid:].groupby("TERRITORIES")[cols["pf"]].sum()

    growth = {}
    for t in terr["TERRITORIES"].unique():
        a = float(first.get(t, 0))
        b = float(second.get(t, 0))
        growth[t] = ((b - a) / a * 100) if a > 0 else (0.0 if b == 0 else 100.0)

    terr["Pazar_Buyume_%"] = terr["TERRITORIES"].map(growth).fillna(0.0)

    med_share = terr["Goreceli_Pazar_Payi"].median()
    med_growth = terr["Pazar_Buyume_%"].median()

    def label(row):
        hi_share = row["Goreceli_Pazar_Payi"] >= med_share
        hi_growth = row["Pazar_Buyume_%"] >= med_growth
        if hi_share and hi_growth:
            return "⭐ Star"
        if hi_share and (not hi_growth):
            return "🐄 Cash Cow"
        if (not hi_share) and hi_growth:
            return "❓ Question Mark"
        return "🐶 Dog"

    terr["BCG_Kategori"] = terr.apply(label, axis=1)
    return terr

def segment_territories(bcg_df: pd.DataFrame) -> pd.DataFrame:
    if bcg_df.empty:
        return bcg_df
    out = bcg_df.copy()
    val_med = out["PF_Satis"].median()
    growth_med = out["Pazar_Buyume_%"].median()

    def seg(row):
        hv = row["PF_Satis"] >= val_med
        hg = row["Pazar_Buyume_%"] >= growth_med
        if hv and hg:
            return "🔥 High Value – High Growth"
        if hv and (not hg):
            return "💎 High Value – Low Growth"
        if (not hv) and hg:
            return "🚀 Low Value – High Growth"
        return "⚠️ Low Value – Low Growth"

    out["Segment"] = out.apply(seg, axis=1)
    return out

def manager_scorecard(df: pd.DataFrame, product: str) -> pd.DataFrame:
    cols = get_product_cols(product)
    if cols["pf"] not in df.columns or cols["rakip"] not in df.columns:
        return pd.DataFrame()

    m = df.groupby("MANAGER", as_index=False).agg(
        PF_Satis=(cols["pf"], "sum"),
        Rakip_Satis=(cols["rakip"], "sum"),
        Territory_Coverage=("TERRITORIES", "nunique"),
    )
    m["Toplam_Pazar"] = m["PF_Satis"] + m["Rakip_Satis"]
    m["Pazar_Payi_%"] = safe_divide(m["PF_Satis"], m["Toplam_Pazar"]) * 100

    # büyüme: manager bazında son 3 ay vs önceki 3 ay (aylık veri varsayımı)
    if "YIL_AY" in df.columns:
        last_period = df["YIL_AY"].max()
        periods = sorted(df["YIL_AY"].unique())
        if len(periods) >= 6:
            last3 = set(periods[-3:])
            prev3 = set(periods[-6:-3])
            last_df = df[df["YIL_AY"].isin(last3)]
            prev_df = df[df["YIL_AY"].isin(prev3)]
            last_sum = last_df.groupby("MANAGER")[cols["pf"]].sum()
            prev_sum = prev_df.groupby("MANAGER")[cols["pf"]].sum()
            growth = ((last_sum - prev_sum) / prev_sum.replace(0, np.nan) * 100).replace([np.inf, -np.inf], np.nan).fillna(0)
            m["Buyume_%"] = m["MANAGER"].map(growth).fillna(0.0)
        else:
            m["Buyume_%"] = 0.0
    else:
        m["Buyume_%"] = 0.0

    m["Rank"] = m["PF_Satis"].rank(ascending=False, method="dense").astype(int)
    m = m.sort_values(["PF_Satis", "Pazar_Payi_%"], ascending=False)
    return m

# =============================================================================
# VISUALS
# =============================================================================

def kpi_card(title: str, value: str, sub: str = "") -> str:
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    return f"""
<div class="kpi-card">
  <div class="kpi-title">{title}</div>
  <div class="kpi-value">{value}</div>
  {sub_html}
</div>
"""

def chart_territory_bar(terr: pd.DataFrame, top_n: int = 20) -> go.Figure:
    d = terr.head(top_n)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=d["TERRITORIES"], y=d["PF_Satis"], name="PF", marker_color="#3B82F6"))
    fig.add_trace(go.Bar(x=d["TERRITORIES"], y=d["Rakip_Satis"], name="Rakip", marker_color="#EF4444"))
    fig.update_layout(
        title=f"Top {top_n} Territory – PF vs Rakip",
        barmode="group",
        height=520,
        xaxis=dict(tickangle=-45),
        hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig

def chart_time_series(ts: pd.DataFrame, fdf: Optional[pd.DataFrame]) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=ts["Period"], y=ts["PF_Satis"], mode="lines+markers", name="PF (Gerçek)",
                             line=dict(color="#3B82F6", width=3), marker=dict(size=8)))
    fig.add_trace(go.Scatter(x=ts["Period"], y=ts["MA_3"], mode="lines", name="MA-3",
                             line=dict(color="#10B981", width=2, dash="dot")))
    fig.add_trace(go.Scatter(x=ts["Period"], y=ts["MA_6"], mode="lines", name="MA-6",
                             line=dict(color="#8B5CF6", width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=ts["Period"], y=ts["Trend"], mode="lines", name="Trend",
                             line=dict(color="#F59E0B", width=2, dash="longdash")))

    if fdf is not None and len(fdf) > 0:
        fig.add_trace(go.Scatter(x=fdf["Period"], y=fdf["PF_Satis_Tahmin"], mode="lines+markers", name="Tahmin (LR)",
                                 line=dict(color="#EC4899", width=3, dash="dash"), marker=dict(size=10, symbol="diamond")))

    fig.update_layout(
        title="Zaman Serisi – Trend, Hareketli Ortalamalar & Tahmin",
        height=520,
        hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"),
    )
    return fig

def chart_bcg(bcg: pd.DataFrame) -> go.Figure:
    color_map = {"⭐ Star":"#FFD700","🐄 Cash Cow":"#10B981","❓ Question Mark":"#3B82F6","🐶 Dog":"#9CA3AF"}
    fig = px.scatter(
        bcg,
        x="Goreceli_Pazar_Payi",
        y="Pazar_Buyume_%",
        size="PF_Satis",
        color="BCG_Kategori",
        color_discrete_map=color_map,
        hover_name="TERRITORIES",
        hover_data={"PF_Satis":":,.0f","Pazar_Payi_%":":.1f","Toplam_Pazar":":,.0f"},
        labels={"Goreceli_Pazar_Payi":"Göreceli Pazar Payı (PF/Rakip)","Pazar_Buyume_%":"Pazar Büyüme Hızı (%)"},
        size_max=60,
    )
    med_x = bcg["Goreceli_Pazar_Payi"].median()
    med_y = bcg["Pazar_Buyume_%"].median()
    fig.add_vline(x=med_x, line_dash="dash", line_color="rgba(255,255,255,0.45)")
    fig.add_hline(y=med_y, line_dash="dash", line_color="rgba(255,255,255,0.45)")
    fig.update_layout(
        title="BCG Matrix",
        height=650,
        plot_bgcolor="#0b1220",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e5e7eb"),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.10)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.10)")
    return fig

def chart_turkey_choropleth(
    df: pd.DataFrame,
    geo: dict,
    product: str,
    metric: str = "PF_Satis",
) -> go.Figure:
    cols = get_product_cols(product)
    city = df.groupby("CITY_CANON", as_index=False).agg(
        PF_Satis=(cols["pf"], "sum"),
        Rakip_Satis=(cols["rakip"], "sum"),
    )
    city["Toplam_Pazar"] = city["PF_Satis"] + city["Rakip_Satis"]
    city["Pazar_Payi_%"] = safe_divide(city["PF_Satis"], city["Toplam_Pazar"]) * 100

    # 81 il kapsama garantisi: kanonik illeri ekle, olmayanlara 0 ver
    canon = pd.DataFrame({"CITY_CANON": CANONICAL_CITIES})
    city = canon.merge(city, on="CITY_CANON", how="left").fillna(0)

    z_col = metric if metric in city.columns else "PF_Satis"

    fig = px.choropleth(
        city,
        geojson=geo,
        featureidkey="properties.name",
        locations="CITY_CANON",
        color=z_col,
        hover_name="CITY_CANON",
        hover_data={"PF_Satis":":,.0f","Rakip_Satis":":,.0f","Toplam_Pazar":":,.0f","Pazar_Payi_%":":.1f"},
        title="Türkiye – İl Bazlı Performans",
    )

    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(height=720, margin=dict(l=0,r=0,t=50,b=0), paper_bgcolor="rgba(0,0,0,0)")
    return fig

# =============================================================================
# REPORTING
# =============================================================================

def export_excel(sheets: Dict[str, pd.DataFrame]) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for name, df in sheets.items():
            dd = df.copy()
            dd.to_excel(writer, sheet_name=name[:31], index=False)
    return output.getvalue()

def export_pdf_optional(summary_text: str) -> Optional[bytes]:
    """
    Opsiyonel: reportlab kuruluysa basit executive summary PDF üret.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas

        buf = BytesIO()
        c = canvas.Canvas(buf, pagesize=A4)
        width, height = A4
        y = height - 60
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, y, "Executive Summary")
        y -= 30
        c.setFont("Helvetica", 11)
        for line in summary_text.split("\n"):
            if y < 60:
                c.showPage()
                y = height - 60
                c.setFont("Helvetica", 11)
            c.drawString(50, y, line[:110])
            y -= 16
        c.showPage()
        c.save()
        return buf.getvalue()
    except Exception:
        return None

# =============================================================================
# APP
# =============================================================================

def run_app():
    st.markdown('<div class="main-header">💊 Ticari Portföy Analiz Platformu</div>', unsafe_allow_html=True)
    st.markdown("**Kurumsal Seviye | Territory × Zaman × Coğrafya × Rekabet × Tahminleme**")

    # Sidebar - data
    st.sidebar.header("📂 Veri Yönetimi")
    up = st.sidebar.file_uploader("Excel yükle (.xlsx/.xls)", type=["xlsx", "xls"])

    # GeoJSON (sabit path; kullanıcı yükledi)
    geo_path = "/mnt/data/turkey.geojson"

    if up is None:
        st.info("Sol menüden Excel dosyasını yükleyin.")
        st.stop()

    df, notes = load_excel(up)
    if df is None or df.empty:
        st.error("Veri okunamadı veya boş.")
        if notes:
            st.warning(" | ".join(notes))
        st.stop()

    # GeoJSON load
    geo, geo_map, geo_notes = load_geojson(geo_path)

    # Sidebar summary
    st.sidebar.success(f"✅ {len(df):,} satır yüklendi")
    with st.sidebar.expander("📌 Veri Özeti", expanded=False):
        st.write(f"📅 Tarih: {df['DATE'].min().date()} → {df['DATE'].max().date()}")
        st.write(f"🏢 Territory: {df['TERRITORIES'].nunique()}")
        st.write(f"🗺️ Region: {df['REGION'].nunique()}")
        st.write(f"🏙️ City: {df['CITY_CANON'].nunique()}")
        st.write(f"👤 Manager: {df['MANAGER'].nunique()}")

    if notes:
        st.warning(" | ".join(notes))
    if geo_notes:
        st.info(" | ".join(geo_notes))

    # Sidebar - controls
    st.sidebar.header("🎯 Analiz Parametreleri")

    product = st.sidebar.selectbox("Ürün", list(PRODUCT_MAP.keys()), index=0)
    cols = get_product_cols(product)
    # Ürün kolon kontrol
    missing_prod = [c for c in [cols["pf"], cols["rakip"]] if c not in df.columns]
    if missing_prod:
        st.error(f"Bu ürün için gerekli kolonlar yok: {', '.join(missing_prod)}")
        st.stop()

    # Date range
    min_d, max_d = df["DATE"].min(), df["DATE"].max()
    date_mode = st.sidebar.radio("Tarih Aralığı", ["Tüm Dönem", "Özel"], horizontal=True)
    if date_mode == "Özel":
        start = st.sidebar.date_input("Başlangıç", value=min_d.date(), min_value=min_d.date(), max_value=max_d.date())
        end = st.sidebar.date_input("Bitiş", value=max_d.date(), min_value=min_d.date(), max_value=max_d.date())
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    else:
        start_ts, end_ts = min_d, max_d

    # Filters
    territory = st.sidebar.selectbox("Territory", ["TÜMÜ"] + sorted(df["TERRITORIES"].unique().tolist()))
    region = st.sidebar.selectbox("Region", ["TÜMÜ"] + sorted(df["REGION"].unique().tolist()))
    manager = st.sidebar.selectbox("Manager", ["TÜMÜ"] + sorted(df["MANAGER"].unique().tolist()))

    dff = filter_df(df, start_ts, end_ts, territory, region, manager)
    if dff.empty:
        st.warning("Seçilen filtrelerde veri yok. Filtreleri genişletin.")
        st.stop()

    # Tabs (spec’e uygun)
    t1, t2, t3, t4, t5 = st.tabs([
        "1) Executive Dashboard",
        "2) Zaman Serisi & Tahmin",
        "3) Coğrafi Analiz (Türkiye Haritası)",
        "4) Stratejik Analizler",
        "5) Raporlama",
    ])

    # =========================
    # TAB 1 - Executive
    # =========================
    with t1:
        terr = territory_perf(dff, product)
        total_pf = float(dff[cols["pf"]].sum())
        total_rk = float(dff[cols["rakip"]].sum())
        total_mkt = total_pf + total_rk
        share = float((total_pf / total_mkt * 100) if total_mkt > 0 else 0)

        kpi_html = '<div class="kpi-wrap">'
        kpi_html += kpi_card("PF Satış", fmt_num(total_pf), f"{territory} | {region} | {manager}")
        kpi_html += kpi_card("Toplam Pazar", fmt_num(total_mkt), f"Rakip: {fmt_num(total_rk)}")
        kpi_html += kpi_card("Pazar Payı", f"%{share:.1f}", f"Göreceli: {(total_pf/total_rk) if total_rk>0 else 0:.2f}")
        kpi_html += kpi_card("Aktif Territory", f"{dff['TERRITORIES'].nunique():,}", f"Şehir: {dff['CITY_CANON'].nunique():,}")
        kpi_html += '</div>'
        st.markdown(kpi_html, unsafe_allow_html=True)

        st.markdown("---")
        colA, colB = st.columns([1.2, 1])
        with colA:
            st.subheader("🏢 Territory Performans (Top)")
            if terr.empty:
                st.info("Territory performansı hesaplanamadı (kolon kontrol edin).")
            else:
                st.plotly_chart(chart_territory_bar(terr, top_n=min(20, len(terr))), use_container_width=True)
        with colB:
            st.subheader("🎯 Satış Dağılımı")
            if terr.empty:
                st.info("Veri yok.")
            else:
                pie = px.pie(
                    terr.head(10),
                    values="PF_Satis",
                    names="TERRITORIES",
                    title="Top 10 Territory – PF Satış Payı",
                )
                pie.update_layout(height=520)
                st.plotly_chart(pie, use_container_width=True)

        st.markdown("---")
        st.subheader("📋 Territory Detay Tablosu")
        if terr.empty:
            st.info("Veri yok.")
        else:
            show = terr.copy()
            show = show.rename(columns={"CITY_CANON":"CITY"})
            cols_show = ["TERRITORIES","REGION","CITY","MANAGER","PF_Satis","Rakip_Satis","Toplam_Pazar","Pazar_Payi_%","Buyume_Potansiyeli"]
            cols_show = [c for c in cols_show if c in show.columns]
            show = show[cols_show]
            st.dataframe(
                show.style.format({
                    "PF_Satis":"{:,.0f}",
                    "Rakip_Satis":"{:,.0f}",
                    "Toplam_Pazar":"{:,.0f}",
                    "Pazar_Payi_%":"{:.1f}%",
                    "Buyume_Potansiyeli":"{:,.0f}",
                }).background_gradient(subset=["Pazar_Payi_%"], cmap="RdYlGn"),
                use_container_width=True,
                height=420,
            )

    # =========================
    # TAB 2 - Time Series & Forecast
    # =========================
    with t2:
        st.subheader("📈 Aylık Zaman Serisi (YIL-AY) + Tahmin")
        ts_terr = st.selectbox("Zaman serisi için Territory", ["TÜMÜ"] + sorted(dff["TERRITORIES"].unique().tolist()), index=0, key="tsTerr")
        ts = time_series_monthly(dff, product, territory=ts_terr)

        if ts.empty:
            st.warning("Zaman serisi üretilemedi.")
        else:
            months_ahead = st.slider("Tahmin Ufku (ay)", min_value=6, max_value=18, value=6, step=1)
            fdf, metrics = forecast_lr(ts, months_ahead=months_ahead)

            col1, col2, col3 = st.columns(3)
            col1.metric("Ortalama PF", f"{ts['PF_Satis'].mean():,.0f}")
            col2.metric("Ortalama Büyüme", f"%{(np.nanmean(ts['PF_Buyume_%'].values) if len(ts)>1 else 0):.1f}")
            col3.metric("Ortalama Pay", f"%{ts['Pazar_Payi_%'].mean():.1f}")

            st.plotly_chart(chart_time_series(ts, fdf), use_container_width=True)

            with st.expander("📌 Tahmin Modeli Kalitesi (LR Baseline)", expanded=False):
                if metrics:
                    st.write({k: (round(v, 3) if isinstance(v, float) else v) for k, v in metrics.items()})
                else:
                    st.write("Yetersiz veri (min 3 dönem gerekli).")

            st.subheader("📋 Zaman Serisi Tablosu")
            st.dataframe(
                ts.style.format({
                    "PF_Satis":"{:,.0f}",
                    "Rakip_Satis":"{:,.0f}",
                    "Toplam_Pazar":"{:,.0f}",
                    "Pazar_Payi_%":"{:.1f}%",
                    "PF_Buyume_%":"{:.1f}%",
                    "Rakip_Buyume_%":"{:.1f}%",
                    "MA_3":"{:,.0f}",
                    "MA_6":"{:,.0f}",
                    "Trend":"{:,.0f}",
                }),
                use_container_width=True,
                height=420,
            )

    # =========================
    # TAB 3 - Geo
    # =========================
    with t3:
        st.subheader("🗺️ Türkiye Haritası (GeoJSON)")

        metric = st.selectbox(
            "Harita metriği",
            ["PF_Satis", "Rakip_Satis", "Toplam_Pazar", "Pazar_Payi_%"],
            index=0,
        )

        fig = chart_turkey_choropleth(dff, geo, product, metric=metric)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="small-note">Not: Haritada boş il kalmaması için 81 il kanonik listeden tamamlanır; veri yoksa değer 0 gösterilir.</div>', unsafe_allow_html=True)

    # =========================
    # TAB 4 - Strategic
    # =========================
    with t4:
        st.subheader("⭐ Stratejik Analizler (BCG + Segmentasyon + Manager)")
        bcg = bcg_matrix(dff, product)
        if bcg.empty:
            st.warning("BCG hesaplanamadı.")
        else:
            bcg2 = segment_territories(bcg)

            col1, col2 = st.columns([1.05, 0.95])
            with col1:
                st.plotly_chart(chart_bcg(bcg2), use_container_width=True)
            with col2:
                st.markdown("#### 📌 BCG Dağılımı")
                vc = bcg2["BCG_Kategori"].value_counts()
                for k in ["⭐ Star","🐄 Cash Cow","❓ Question Mark","🐶 Dog"]:
                    st.metric(k, int(vc.get(k, 0)))
                st.markdown("---")
                st.markdown("#### 🧩 Segment Dağılımı")
                sv = bcg2["Segment"].value_counts()
                for k, v in sv.items():
                    st.write(f"- **{k}**: {int(v)}")

            st.markdown("---")
            st.subheader("📋 Territory – BCG & Segment Detayı")
            show = bcg2.rename(columns={"CITY_CANON":"CITY"}).copy()
            keep = ["TERRITORIES","REGION","CITY","MANAGER","PF_Satis","Rakip_Satis","Toplam_Pazar","Pazar_Payi_%","Goreceli_Pazar_Payi","Pazar_Buyume_%","BCG_Kategori","Segment","Buyume_Potansiyeli"]
            keep = [c for c in keep if c in show.columns]
            show = show[keep].sort_values(["BCG_Kategori","PF_Satis"], ascending=[True, False])
            st.dataframe(
                show.style.format({
                    "PF_Satis":"{:,.0f}",
                    "Rakip_Satis":"{:,.0f}",
                    "Toplam_Pazar":"{:,.0f}",
                    "Pazar_Payi_%":"{:.1f}%",
                    "Goreceli_Pazar_Payi":"{:.2f}",
                    "Pazar_Buyume_%":"{:.1f}%",
                    "Buyume_Potansiyeli":"{:,.0f}",
                }),
                use_container_width=True,
                height=520,
            )

        st.markdown("---")
        st.subheader("👥 Manager Scorecard")
        ms = manager_scorecard(dff, product)
        if ms.empty:
            st.info("Manager scorecard üretilemedi.")
        else:
            st.dataframe(
                ms.style.format({
                    "PF_Satis":"{:,.0f}",
                    "Rakip_Satis":"{:,.0f}",
                    "Toplam_Pazar":"{:,.0f}",
                    "Pazar_Payi_%":"{:.1f}%",
                    "Buyume_%":"{:.1f}%",
                }).background_gradient(subset=["Pazar_Payi_%"], cmap="RdYlGn"),
                use_container_width=True,
                height=420,
            )

    # =========================
    # TAB 5 - Reporting
    # =========================
    with t5:
        st.subheader("📥 Raporlama (Excel + Opsiyonel PDF)")

        terr = territory_perf(dff, product)
        ts = time_series_monthly(dff, product, territory="TÜMÜ")
        bcg = segment_territories(bcg_matrix(dff, product))
        ms = manager_scorecard(dff, product)

        sheets = {
            "Territory_Performans": terr.rename(columns={"CITY_CANON":"CITY"}) if not terr.empty else pd.DataFrame(),
            "Zaman_Serisi": ts if not ts.empty else pd.DataFrame(),
            "BCG_Matrix": bcg.rename(columns={"CITY_CANON":"CITY"}) if not bcg.empty else pd.DataFrame(),
            "Manager_Scorecard": ms if not ms.empty else pd.DataFrame(),
        }

        xbytes = export_excel(sheets)
        st.download_button(
            "⬇️ Excel Raporu İndir",
            data=xbytes,
            file_name=f"ticari_portfoy_rapor_{product}_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        st.markdown("---")
        st.markdown("#### 🧾 Executive Summary (PDF opsiyonel)")
        summary = f"""Ürün: {product}
Tarih: {start_ts.date()} → {end_ts.date()}
Filtreler: Territory={territory}, Region={region}, Manager={manager}
PF Satış: {total_pf:,.0f}
Toplam Pazar: {total_mkt:,.0f}
Pazar Payı: %{share:.1f}
Aktif Territory: {dff['TERRITORIES'].nunique()}
"""
        pdf = export_pdf_optional(summary)
        if pdf:
            st.download_button(
                "⬇️ PDF Executive Summary İndir",
                data=pdf,
                file_name=f"executive_summary_{product}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
            )
        else:
            st.info("PDF üretimi bu ortamda devre dışı olabilir (reportlab gerekli).")

if __name__ == "__main__":
    run_app()
'''

out_path = Path("/mnt/data/ticari_portfoy_app.py")
out_path.write_text(app_code, encoding="utf-8")
str(out_path), out_path.stat().st_size

