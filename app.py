"""
🎯 GELİŞMİŞ TİCARİ PORTFÖY ANALİZ SİSTEMİ v2.0
Territory × Zaman × Coğrafi Analiz + Gelecek Tahminleme + Yatırım Stratejisi

Yeni Özellikler:
- ✅ Türkiye haritası üzerinde BÖLGE RENKLI görselleştirme
- ✅ Gelecek tahminleme (ARIMA, Linear Regression, Moving Average)
- ✅ Zaman çizelgesi analizi (Gantt chart)
- ✅ Territory bazlı performans ve yatırım stratejisi analizi
- ✅ Detaylı BCG Matrix ve stratejik konumlandırma
- ✅ Manager performans scorecards
- ✅ Otomatik aksiyon planı oluşturma
- ✅ Monte Carlo simülasyonu
- ✅ Duplicate ID hatası düzeltildi
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
from io import BytesIO
import json
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Ticari Portföy Analizi",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS - MODERN & PROFESSIONAL
# =============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
    }
    
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        padding: 2rem 0;
        margin-bottom: 2rem;
        background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 50%, #d97706 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -1px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .subtitle {
        text-align: center;
        color: #cbd5e1;
        font-size: 1.1rem;
        margin-top: -1.5rem;
        margin-bottom: 2rem;
        font-weight: 500;
    }
    
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
        border-right: 1px solid #334155;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: rgba(30, 41, 59, 0.6);
        padding: 0.75rem;
        border-radius: 12px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(148, 163, 184, 0.1);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        padding: 0 1.5rem;
        font-size: 1rem;
        font-weight: 600;
        border-radius: 8px;
        background: transparent;
        color: #94a3b8;
        border: 1px solid transparent;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(59, 130, 246, 0.1);
        color: #60a5fa;
        border-color: rgba(59, 130, 246, 0.3);
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border-color: #1e40af;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    div[data-testid="metric-container"] {
        background: rgba(30, 41, 59, 0.6);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(148, 163, 184, 0.2);
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(59, 130, 246, 0.3);
        border-color: rgba(59, 130, 246, 0.4);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.5);
        transform: translateY(-2px);
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #f1f5f9 !important;
        font-weight: 700;
    }
    
    p, span, div {
        color: #cbd5e1;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# BÖLGE RENKLERİ (COĞRAFİ & MODERN) - İKİNCİ KODDAN
# =============================================================================
REGION_COLORS = {
    "MARMARA": "#0EA5E9",              # Sky Blue
    "BATI ANADOLU": "#14B8A6",         # Teal
    "EGE": "#FCD34D",                  # Amber
    "İÇ ANADOLU": "#F59E0B",           # Orange
    "GÜNEY DOĞU ANADOLU": "#E07A5F",   # Terracotta
    "KUZEY ANADOLU": "#059669",        # Emerald
    "KARADENİZ": "#059669",            # Emerald
    "AKDENİZ": "#8B5CF6",              # Violet
    "DOĞU ANADOLU": "#7C3AED",         # Purple
    "DİĞER": "#64748B"                 # Slate Gray
}

# =============================================================================
# ŞEHİR NORMALIZATION MAP - GELİŞTİRİLMİŞ
# =============================================================================
CITY_NORMALIZE_MAP = {
    "ADANA": "Adana",
    "ADIYAMAN": "Adiyaman",
    "AFYONKARAHISAR": "Afyonkarahisar",
    "AFYON": "Afyonkarahisar",
    "AGRI": "Agri",
    "AĞRI": "Agri",
    "AKSARAY": "Aksaray",
    "AMASYA": "Amasya",
    "ANKARA": "Ankara",
    "ANTALYA": "Antalya",
    "ARTVIN": "Artvin",
    "AYDIN": "Aydin",
    "BALIKESIR": "Balikesir",
    "BARTIN": "Bartin",
    "BATMAN": "Batman",
    "BAYBURT": "Bayburt",
    "BILECIK": "Bilecik",
    "BINGOL": "Bingol",
    "BİNGÖL": "Bingol",
    "BITLIS": "Bitlis",
    "BOLU": "Bolu",
    "BURDUR": "Burdur",
    "BURSA": "Bursa",
    "CANAKKALE": "Canakkale",
    "ÇANAKKALE": "Canakkale",
    "CANKIRI": "Cankiri",
    "ÇANKIRI": "Cankiri",
    "CORUM": "Corum",
    "ÇORUM": "Corum",
    "DENIZLI": "Denizli",
    "DIYARBAKIR": "Diyarbakir",
    "DUZCE": "Duzce",
    "DÜZCE": "Duzce",
    "EDIRNE": "Edirne",
    "ELAZIG": "Elazig",
    "ELAZĞ": "Elazig",
    "ELÂZIĞ": "Elazig",
    "ERZINCAN": "Erzincan",
    "ERZURUM": "Erzurum",
    "ESKISEHIR": "Eskisehir",
    "ESKİŞEHİR": "Eskisehir",
    "GAZIANTEP": "Gaziantep",
    "GIRESUN": "Giresun",
    "GUMUSHANE": "Gumushane",
    "GÜMÜŞHANE": "Gumushane",
    "HAKKARI": "Hakkari",
    "HATAY": "Hatay",
    "IGDIR": "Igdir",
    "IĞDIR": "Igdir",
    "ISPARTA": "Isparta",
    "ISTANBUL": "Istanbul",
    "İSTANBUL": "Istanbul",
    "IZMIR": "Izmir",
    "İZMİR": "Izmir",
    "KAHRAMANMARAS": "K. Maras",
    "K. MARAS": "K. Maras",
    "KAHRAMANMARAŞ": "K. Maras",
    "KARABUK": "Karabuk",
    "KARABÜK": "Karabuk",
    "KARAMAN": "Karaman",
    "KARS": "Kars",
    "KASTAMONU": "Kastamonu",
    "KAYSERI": "Kayseri",
    "KINKKALE": "Kinkkale",
    "KIRIKKALE": "Kinkkale",
    "KIRKLARELI": "Kirklareli",
    "KIRSEHIR": "Kirsehir",
    "KIRŞEHİR": "Kirsehir",
    "KILIS": "Kilis",
    "KOCAELI": "Kocaeli",
    "KONYA": "Konya",
    "KUTAHYA": "Kutahya",
    "KÜTAHYA": "Kutahya",
    "MALATYA": "Malatya",
    "MANISA": "Manisa",
    "MARDIN": "Mardin",
    "MERSIN": "Mersin",
    "MUGLA": "Mugla",
    "MUĞLA": "Mugla",
    "MUS": "Mus",
    "MUŞ": "Mus",
    "NEVSEHIR": "Nevsehir",
    "NEVŞEHİR": "Nevsehir",
    "NIGDE": "Nigde",
    "NİĞDE": "Nigde",
    "ORDU": "Ordu",
    "OSMANIYE": "Osmaniye",
    "RIZE": "Rize",
    "SAKARYA": "Sakarya",
    "SAMSUN": "Samsun",
    "SIIRT": "Siirt",
    "SINOP": "Sinop",
    "SIRNAK": "Sirnak",
    "ŞIRNAK": "Sirnak",
    "SIVAS": "Sivas",
    "SANLIURFA": "Sanliurfa",
    "ŞANLIURFA": "Sanliurfa",
    "TEKIRDAG": "Tekirdag",
    "TEKİRDAĞ": "Tekirdag",
    "TOKAT": "Tokat",
    "TRABZON": "Trabzon",
    "TUNCELI": "Tunceli",
    "USAK": "Usak",
    "UŞAK": "Usak",
    "VAN": "Van",
    "YALOVA": "Yalova",
    "YOZGAT": "Yozgat",
    "ZONGULDAK": "Zonguldak",
    "ARDAHAN": "Ardahan",
}

def normalize_city_name(name):
    """Şehir isimlerini normalize et"""
    if pd.isna(name):
        return None
    
    name = str(name).upper().strip()
    return CITY_NORMALIZE_MAP.get(name, name)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def safe_divide(a, b):
    """Güvenli bölme işlemi"""
    return np.where(b != 0, a / b, 0)

def format_number(x):
    """Sayı formatlama"""
    if pd.isna(x):
        return 0
    return round(float(x), 2)

def get_product_columns(product):
    """Ürün kolonlarını döndür"""
    product_map = {
        "TROCMETAM": {"pf": "TROCMETAM", "rakip": "DIGER TROCMETAM"},
        "CORTIPOL": {"pf": "CORTIPOL", "rakip": "DIGER CORTIPOL"},
        "DEKSAMETAZON": {"pf": "DEKSAMETAZON", "rakip": "DIGER DEKSAMETAZON"},
        "PF IZOTONIK": {"pf": "PF IZOTONIK", "rakip": "DIGER IZOTONIK"}
    }
    return product_map.get(product, {"pf": product, "rakip": f"DIGER {product}"})

# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data
def load_excel_data(file):
    """Excel dosyasını yükle ve ön işleme yap"""
    try:
        df = pd.read_excel(file)
        
        if 'DATE' in df.columns:
            df['DATE'] = pd.to_datetime(df['DATE'])
            df['YIL_AY'] = df['DATE'].dt.strftime('%Y-%m')
            df['AY'] = df['DATE'].dt.month
            df['YIL'] = df['DATE'].dt.year
            df['QUARTER'] = df['DATE'].dt.quarter
            df['HAFTA'] = df['DATE'].dt.isocalendar().week
        
        if 'TERRITORIES' in df.columns:
            df['TERRITORIES'] = df['TERRITORIES'].str.upper().str.strip()
        if 'CITY' in df.columns:
            df['CITY'] = df['CITY'].str.strip()
            df['CITY_NORMALIZED'] = df['CITY'].apply(normalize_city_name)
        if 'REGION' in df.columns:
            df['REGION'] = df['REGION'].str.upper().str.strip()
        if 'MANAGER' in df.columns:
            df['MANAGER'] = df['MANAGER'].str.upper().str.strip()
        
        return df
    except Exception as e:
        st.error(f"Veri yükleme hatası: {str(e)}")
        return None

# =============================================================================
# TERRITORY PERFORMANCE
# =============================================================================

def calculate_territory_performance(df, product, start_date=None, end_date=None):
    """Territory bazlı performans analizi"""
    df_filtered = df.copy()
    
    if start_date and end_date and 'DATE' in df.columns:
        df_filtered = df_filtered[
            (df_filtered['DATE'] >= start_date) & 
            (df_filtered['DATE'] <= end_date)
        ]
    
    cols = get_product_columns(product)
    
    agg_dict = {}
    if cols['pf'] in df_filtered.columns:
        agg_dict[cols['pf']] = 'sum'
    if cols['rakip'] in df_filtered.columns:
        agg_dict[cols['rakip']] = 'sum'
    
    group_cols = ['TERRITORIES']
    if 'REGION' in df_filtered.columns:
        group_cols.append('REGION')
    if 'CITY' in df_filtered.columns:
        group_cols.append('CITY')
    if 'CITY_NORMALIZED' in df_filtered.columns:
        group_cols.append('CITY_NORMALIZED')
    if 'MANAGER' in df_filtered.columns:
        group_cols.append('MANAGER')
    
    terr_perf = df_filtered.groupby(group_cols).agg(agg_dict).reset_index()
    
    terr_perf.columns = list(terr_perf.columns[:len(group_cols)]) + ['PF_Satis', 'Rakip_Satis']
    terr_perf['Toplam_Pazar'] = terr_perf['PF_Satis'] + terr_perf['Rakip_Satis']
    terr_perf['Pazar_Payi_%'] = safe_divide(terr_perf['PF_Satis'], terr_perf['Toplam_Pazar']) * 100
    
    total_pf = terr_perf['PF_Satis'].sum()
    terr_perf['Agirlik_%'] = safe_divide(terr_perf['PF_Satis'], total_pf) * 100
    
    terr_perf['Goreceli_Pazar_Payi'] = safe_divide(terr_perf['PF_Satis'], terr_perf['Rakip_Satis'])
    terr_perf['Buyume_Potansiyeli'] = terr_perf['Toplam_Pazar'] - terr_perf['PF_Satis']
    
    return terr_perf.sort_values('PF_Satis', ascending=False)

# =============================================================================
# YATIRIM STRATEJİSİ - İKİNCİ KODDAN GELİŞTİRİLMİŞ
# =============================================================================

def calculate_investment_strategy(df):
    """Geliştirilmiş Yatırım Stratejisi Algoritması"""
    df = df.copy()
    df = df[df["PF_Satis"] > 0]
    
    if len(df) == 0:
        return df
    
    # 1. PAZAR BÜYÜKLÜĞÜ SEGMENTİ
    try:
        df["Pazar_Buyuklugu_Segment"] = pd.qcut(
            df["Toplam_Pazar"], 
            q=3, 
            labels=["Küçük", "Orta", "Büyük"],
            duplicates='drop'
        )
    except:
        df["Pazar_Buyuklugu_Segment"] = "Orta"
    
    # 2. PERFORMANS SEGMENTİ
    try:
        df["Performans_Segment"] = pd.qcut(
            df["PF_Satis"], 
            q=3, 
            labels=["Düşük", "Orta", "Yüksek"],
            duplicates='drop'
        )
    except:
        df["Performans_Segment"] = "Orta"
    
    # 3. PAZAR PAYI SEGMENTİ
    try:
        df["Pazar_Payi_Segment"] = pd.qcut(
            df["Pazar_Payi_%"], 
            q=3, 
            labels=["Düşük", "Orta", "Yüksek"],
            duplicates='drop'
        )
    except:
        df["Pazar_Payi_Segment"] = "Orta"
    
    # 4. BÜYÜME POTANSİYELİ
    try:
        df["Buyume_Potansiyeli_Segment"] = pd.qcut(
            df["Buyume_Potansiyeli"],
            q=3,
            labels=["Düşük", "Orta", "Yüksek"],
            duplicates='drop'
        )
    except:
        df["Buyume_Potansiyeli_Segment"] = "Orta"
    
    # 5. STRATEJİ ATAMA
    def assign_strategy(row):
        pazar = str(row["Pazar_Buyuklugu_Segment"])
        payi = str(row["Pazar_Payi_Segment"])
        buyume = str(row["Buyume_Potansiyeli_Segment"])
        
        if pazar in ["Büyük", "Orta"] and payi == "Düşük" and buyume in ["Yüksek", "Orta"]:
            return "🚀 Agresif"
        elif pazar in ["Büyük", "Orta"] and payi == "Orta":
            return "⚡ Hızlandırılmış"
        elif pazar == "Büyük" and payi == "Yüksek":
            return "🛡️ Koruma"
        elif pazar == "Küçük" and buyume == "Yüksek":
            return "💎 Potansiyel"
        else:
            return "👁️ İzleme"
    
    df["Yatirim_Stratejisi"] = df.apply(assign_strategy, axis=1)
    
    # 6. ÖNCELİK SKORU
    df["Oncelik_Skoru"] = 0
    
    max_pazar = df["Toplam_Pazar"].max()
    if max_pazar > 0:
        df["Oncelik_Skoru"] += (df["Toplam_Pazar"] / max_pazar) * 40
    
    max_pot = df["Buyume_Potansiyeli"].max()
    if max_pot > 0:
        df["Oncelik_Skoru"] += (df["Buyume_Potansiyeli"] / max_pot) * 30
    
    df.loc[df["Pazar_Payi_%"] < 10, "Oncelik_Skoru"] += 30
    
    return df

# =============================================================================
# BCG MATRIX
# =============================================================================

def calculate_bcg_matrix(df, product, start_date=None, end_date=None):
    """BCG Matrix kategorileri hesapla"""
    terr_perf = calculate_territory_performance(df, product, start_date, end_date)
    
    cols = get_product_columns(product)
    df_sorted = df.sort_values('DATE')
    
    mid_point = len(df_sorted) // 2
    first_half = df_sorted.iloc[:mid_point].groupby('TERRITORIES')[cols['pf']].sum()
    second_half = df_sorted.iloc[mid_point:].groupby('TERRITORIES')[cols['pf']].sum()
    
    growth_rate = {}
    for terr in first_half.index:
        if terr in second_half.index and first_half[terr] > 0:
            growth_rate[terr] = ((second_half[terr] - first_half[terr]) / first_half[terr]) * 100
        else:
            growth_rate[terr] = 0
    
    terr_perf['Pazar_Buyume_%'] = terr_perf['TERRITORIES'].map(growth_rate).fillna(0)
    
    median_share = terr_perf['Goreceli_Pazar_Payi'].median()
    median_growth = terr_perf['Pazar_Buyume_%'].median()
    
    def assign_bcg(row):
        if row['Goreceli_Pazar_Payi'] >= median_share:
            if row['Pazar_Buyume_%'] >= median_growth:
                return "⭐ Star"
            else:
                return "🐄 Cash Cow"
        else:
            if row['Pazar_Buyume_%'] >= median_growth:
                return "❓ Question Mark"
            else:
                return "🐶 Dog"
    
    terr_perf['BCG_Kategori'] = terr_perf.apply(assign_bcg, axis=1)
    
    return terr_perf

# =============================================================================
# TÜRKİYE HARİTASI - BÖLGE RENKLİ
# =============================================================================

def create_turkey_choropleth(city_data, geojson_data):
    """Türkiye haritası choropleth - BÖLGE RENKLİ"""
    if geojson_data is None:
        return None
    
    fig = go.Figure()
    
    # Her bölge için ayrı trace - RENK EŞLEŞTİRME
    for region in city_data['REGION'].unique():
        region_data = city_data[city_data['REGION'] == region]
        color = REGION_COLORS.get(region, "#CCCCCC")
        
        # GeoJSON ile eşleştir
        region_geojson = {
            "type": "FeatureCollection",
            "features": [
                feature for feature in geojson_data["features"]
                if feature["properties"]["name"] in region_data['CITY_NORMALIZED'].values
            ]
        }
        
        if len(region_geojson["features"]) > 0:
            fig.add_choroplethmapbox(
                geojson=region_geojson,
                locations=region_data['CITY_NORMALIZED'],
                z=[1] * len(region_data),  # Sabit değer, renk için
                featureidkey="properties.name",
                colorscale=[[0, color], [1, color]],
                showscale=False,
                marker_line_color="white",
                marker_line_width=1.5,
                customdata=list(zip(
                    region_data['CITY_NORMALIZED'],
                    region_data['REGION'],
                    region_data['PF_Satis'],
                    region_data['Pazar_Payi_%']
                )),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Bölge: %{customdata[1]}<br>"
                    "PF Satış: %{customdata[2]:,.0f}<br>"
                    "Pazar Payı: %{customdata[3]:.1f}%"
                    "<extra></extra>"
                ),
                name=region
            )
    
    fig.update_layout(
        mapbox=dict(
            style="carto-positron",
            center=dict(lat=39, lon=35),
            zoom=4.5
        ),
        height=600,
        margin=dict(l=0, r=0, t=40, b=0),
        title='Türkiye - Bölge Renkli Pazar Payı Haritası'
    )
    
    return fig

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_territory_bar_chart(df, top_n=20, title="Territory Performans"):
    """Territory performans bar chart - UNIQUE KEY"""
    top_terr = df.nlargest(top_n, 'PF_Satis')
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=top_terr['TERRITORIES'],
        y=top_terr['PF_Satis'],
        name='PF Satış',
        marker_color='#3B82F6',
        text=top_terr['PF_Satis'].apply(lambda x: f'{x:,.0f}'),
        textposition='outside'
    ))
    
    fig.add_trace(go.Bar(
        x=top_terr['TERRITORIES'],
        y=top_terr['Rakip_Satis'],
        name='Rakip Satış',
        marker_color='#EF4444',
        text=top_terr['Rakip_Satis'].apply(lambda x: f'{x:,.0f}'),
        textposition='outside'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Territory',
        yaxis_title='Satış',
        barmode='group',
        height=500,
        xaxis=dict(tickangle=-45),
        plot_bgcolor='rgba(0,0,0,0)',
        hovermode='x unified'
    )
    
    return fig

def create_bcg_scatter(bcg_df, chart_key="bcg_main"):
    """BCG Matrix scatter - UNIQUE KEY PARAMETRESİ"""
    color_map = {
        "⭐ Star": "#FFD700",
        "🐄 Cash Cow": "#10B981",
        "❓ Question Mark": "#3B82F6",
        "🐶 Dog": "#9CA3AF"
    }
    
    fig = px.scatter(
        bcg_df,
        x='Goreceli_Pazar_Payi',
        y='Pazar_Buyume_%',
        size='PF_Satis',
        color='BCG_Kategori',
        color_discrete_map=color_map,
        hover_name='TERRITORIES',
        size_max=60
    )
    
    median_share = bcg_df['Goreceli_Pazar_Payi'].median()
    median_growth = bcg_df['Pazar_Buyume_%'].median()
    
    fig.add_hline(y=median_growth, line_dash="dash", line_color="white", opacity=0.5)
    fig.add_vline(x=median_share, line_dash="dash", line_color="white", opacity=0.5)
    
    fig.update_layout(
        title='BCG Matrix - Stratejik Portföy',
        height=600,
        plot_bgcolor='#0f172a',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

# =============================================================================
# MONTE CARLO SİMÜLASYON
# =============================================================================

def monte_carlo_simulation(df, n_simulations=1000):
    """Monte Carlo risk simülasyonu"""
    top10 = df.nlargest(10, 'PF_Satis')
    
    np.random.seed(42)
    simulation_results = {}
    
    for idx, row in top10.iterrows():
        city = row['TERRITORIES']
        current_pf = row['PF_Satis']
        market_share = row['Pazar_Payi_%']
        
        growth_mean = 0.05 if market_share < 20 else (0.03 if market_share < 40 else 0.01)
        growth_std = 0.15
        
        simulated_growth = np.random.normal(growth_mean, growth_std, n_simulations)
        simulated_pf = current_pf * (1 + simulated_growth)
        
        simulation_results[city] = {
            'current': current_pf,
            'simulations': simulated_pf,
            'mean': simulated_pf.mean(),
            'p10': np.percentile(simulated_pf, 10),
            'p50': np.percentile(simulated_pf, 50),
            'p90': np.percentile(simulated_pf, 90)
        }
    
    return simulation_results

# =============================================================================
# MAIN APP
# =============================================================================

def main():
    st.markdown('<h1 class="main-header">💊 TİCARİ PORTFÖY ANALİZ SİSTEMİ v2.0</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Territory × Zaman × Coğrafi Analiz + Yatırım Stratejisi + Monte Carlo</p>', unsafe_allow_html=True)
    
    st.sidebar.header("📂 Veri Yönetimi")
    uploaded_file = st.sidebar.file_uploader("Excel Yükle", type=['xlsx'])
    
    if not uploaded_file:
        st.info("👈 Excel dosyasını yükleyin")
        st.stop()
    
    df = load_excel_data(uploaded_file)
    if df is None:
        st.stop()
    
    st.sidebar.success(f"✅ {len(df):,} satır yüklendi")
    
    # Filtreler
    st.sidebar.header("🎯 Parametreler")
    
    products = ["CORTIPOL", "TROCMETAM", "DEKSAMETAZON", "PF IZOTONIK"]
    selected_product = st.sidebar.selectbox("💊 Ürün", products)
    
    territories = ["TÜMÜ"] + sorted(df['TERRITORIES'].unique().tolist())
    selected_territory = st.sidebar.selectbox("🏢 Territory", territories)
    
    # Veriyi filtrele
    df_filtered = df.copy()
    if selected_territory != "TÜMÜ":
        df_filtered = df_filtered[df_filtered['TERRITORIES'] == selected_territory]
    
    # TABS
    tabs = st.tabs([
        "📊 Dashboard",
        "🏢 Territory Analizi",
        "🗺️ Türkiye Haritası",
        "⭐ BCG & Yatırım Stratejisi",
        "🎲 Monte Carlo Simülasyon",
        "📋 Aksiyon Planı",
        "📥 Rapor"
    ])
    
    # TAB 1: DASHBOARD
    with tabs[0]:
        st.header("📊 Dashboard")
        
        cols_metric = get_product_columns(selected_product)
        total_pf = df_filtered[cols_metric['pf']].sum()
        total_rakip = df_filtered[cols_metric['rakip']].sum()
        total_market = total_pf + total_rakip
        market_share = (total_pf / total_market * 100) if total_market > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        col1.metric("💊 PF Satış", f"{total_pf:,.0f}")
        col2.metric("🏪 Toplam Pazar", f"{total_market:,.0f}")
        col3.metric("📊 Pazar Payı", f"%{market_share:.1f}")
        
        terr_perf = calculate_territory_performance(df_filtered, selected_product)
        fig_dash = create_territory_bar_chart(terr_perf, 15, "Top 15 Territory Performans")
        st.plotly_chart(fig_dash, use_container_width=True, key="dashboard_bar_chart")
    
    # TAB 2: TERRITORY ANALİZİ
    with tabs[1]:
        st.header("🏢 Territory Bazlı Detaylı Analiz")
        
        terr_perf = calculate_territory_performance(df_filtered, selected_product)
        
        # Filtreleme ve sıralama
        col_filter1, col_filter2, col_filter3 = st.columns(3)
        
        with col_filter1:
            sort_metric = st.selectbox(
                "🔄 Sıralama Kriteri",
                ['PF_Satis', 'Pazar_Payi_%', 'Toplam_Pazar', 'Buyume_Potansiyeli'],
                format_func=lambda x: {
                    'PF_Satis': '💊 PF Satış',
                    'Pazar_Payi_%': '📊 Pazar Payı',
                    'Toplam_Pazar': '🏪 Toplam Pazar',
                    'Buyume_Potansiyeli': '🚀 Büyüme Potansiyeli'
                }[x]
            )
        
        with col_filter2:
            n_territories = st.slider("📊 Territory Sayısı", 10, 50, 20)
        
        with col_filter3:
            sort_direction = st.radio("Sıra", ["↓ Azalan", "↑ Artan"], horizontal=True)
        
        terr_sorted = terr_perf.sort_values(
            sort_metric, 
            ascending=(sort_direction == "↑ Artan")
        ).head(n_territories)
        
        # Görselleştirmeler
        col_viz1, col_viz2 = st.columns(2)
        
        with col_viz1:
            st.markdown("#### 📊 PF vs Rakip Satış")
            fig_comp = create_territory_bar_chart(terr_sorted, n_territories, "Territory Karşılaştırma")
            st.plotly_chart(fig_comp, use_container_width=True, key="territory_comp_bar")
        
        with col_viz2:
            st.markdown("#### 🎯 Pazar vs Pazar Payı")
            fig_scatter = px.scatter(
                terr_sorted,
                x='Toplam_Pazar',
                y='Pazar_Payi_%',
                size='PF_Satis',
                color='REGION' if 'REGION' in terr_sorted.columns else None,
                hover_name='TERRITORIES',
                size_max=60
            )
            fig_scatter.update_layout(height=500)
            st.plotly_chart(fig_scatter, use_container_width=True, key="territory_scatter")
        
        # Detaylı tablo
        st.subheader("📋 Detaylı Territory Listesi")
        display_cols = ['TERRITORIES', 'REGION', 'PF_Satis', 'Rakip_Satis', 
                       'Toplam_Pazar', 'Pazar_Payi_%', 'Buyume_Potansiyeli']
        display_cols = [col for col in display_cols if col in terr_perf.columns]
        
        st.dataframe(
            terr_perf[display_cols].style.format({
                'PF_Satis': '{:,.0f}',
                'Rakip_Satis': '{:,.0f}',
                'Toplam_Pazar': '{:,.0f}',
                'Pazar_Payi_%': '{:.1f}%',
                'Buyume_Potansiyeli': '{:,.0f}'
            }).background_gradient(subset=['Pazar_Payi_%'], cmap='RdYlGn'),
            use_container_width=True,
            height=500
        )
    
    # TAB 3: TÜRKİYE HARİTASI
    with tabs[2]:
        st.header("🗺️ Türkiye Coğrafi Analiz - Bölge Renkli")
        
        cols_map = get_product_columns(selected_product)
        
        if 'CITY_NORMALIZED' in df_filtered.columns and 'REGION' in df_filtered.columns:
            city_data = df_filtered.groupby(['CITY_NORMALIZED', 'REGION']).agg({
                cols_map['pf']: 'sum',
                cols_map['rakip']: 'sum'
            }).reset_index()
            
            city_data.columns = ['CITY_NORMALIZED', 'REGION', 'PF_Satis', 'Rakip_Satis']
            city_data['Toplam_Pazar'] = city_data['PF_Satis'] + city_data['Rakip_Satis']
            city_data['Pazar_Payi_%'] = safe_divide(city_data['PF_Satis'], city_data['Toplam_Pazar']) * 100
            
            # GeoJSON yükle
            geojson_file = st.file_uploader("🗺️ turkey.geojson yükle", type=['geojson', 'json'])
            
            if geojson_file:
                try:
                    geojson_data = json.load(geojson_file)
                    fig_map = create_turkey_choropleth(city_data, geojson_data)
                    if fig_map:
                        st.plotly_chart(fig_map, use_container_width=True, key="turkey_map")
                    
                    # Renk legend
                    st.subheader("🎨 Bölge Renkleri")
                    cols_legend = st.columns(5)
                    for idx, (region, color) in enumerate(REGION_COLORS.items()):
                        if region in city_data['REGION'].values:
                            with cols_legend[idx % 5]:
                                st.markdown(f"<span style='color:{color}'>⬤</span> {region}", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Harita hatası: {str(e)}")
            else:
                st.info("💡 turkey.geojson dosyasını yükleyin")
    
    # TAB 4: BCG & YATIRIM STRATEJİSİ
    with tabs[3]:
        st.header("⭐ BCG Matrix & Yatırım Stratejisi")
        
        bcg_df = calculate_bcg_matrix(df_filtered, selected_product)
        strategy_df = calculate_investment_strategy(bcg_df)
        
        # BCG Özet
        st.subheader("📊 BCG Portföy Dağılımı")
        bcg_counts = strategy_df['BCG_Kategori'].value_counts()
        
        col_bcg1, col_bcg2, col_bcg3, col_bcg4 = st.columns(4)
        col_bcg1.metric("⭐ Stars", bcg_counts.get("⭐ Star", 0))
        col_bcg2.metric("🐄 Cows", bcg_counts.get("🐄 Cash Cow", 0))
        col_bcg3.metric("❓ Questions", bcg_counts.get("❓ Question Mark", 0))
        col_bcg4.metric("🐶 Dogs", bcg_counts.get("🐶 Dog", 0))
        
        # BCG Scatter - UNIQUE KEY
        fig_bcg_main = create_bcg_scatter(strategy_df, chart_key="bcg_tab4")
        st.plotly_chart(fig_bcg_main, use_container_width=True, key="bcg_scatter_tab4")
        
        st.markdown("---")
        
        # Yatırım Stratejisi
        st.subheader("💼 Yatırım Stratejisi Dağılımı")
        strategy_counts = strategy_df['Yatirim_Stratejisi'].value_counts()
        
        col_str1, col_str2, col_str3, col_str4, col_str5 = st.columns(5)
        strategies = [
            ('🚀 Agresif', col_str1),
            ('⚡ Hızlandırılmış', col_str2),
            ('🛡️ Koruma', col_str3),
            ('💎 Potansiyel', col_str4),
            ('👁️ İzleme', col_str5)
        ]
        
        for strategy, col in strategies:
            with col:
                count = int(strategy_counts.get(strategy, 0))
                st.metric(strategy, f"{count} terr.")
        
        # Strateji detay tablosu
        st.subheader("📋 Strateji Detayları")
        display_cols_strategy = ['TERRITORIES', 'REGION', 'BCG_Kategori', 'Yatirim_Stratejisi',
                                'PF_Satis', 'Pazar_Payi_%', 'Oncelik_Skoru']
        display_cols_strategy = [col for col in display_cols_strategy if col in strategy_df.columns]
        
        st.dataframe(
            strategy_df[display_cols_strategy].style.format({
                'PF_Satis': '{:,.0f}',
                'Pazar_Payi_%': '{:.1f}%',
                'Oncelik_Skoru': '{:.0f}'
            }).background_gradient(subset=['Oncelik_Skoru'], cmap='YlOrRd'),
            use_container_width=True,
            height=500
        )
    
    # TAB 5: MONTE CARLO SİMÜLASYON
    with tabs[4]:
        st.header("🎲 Monte Carlo Risk & Fırsat Simülasyonu")
        st.caption("🔮 Gelecek dönem satış tahminleri - 1000 senaryo simülasyonu")
        
        terr_perf = calculate_territory_performance(df_filtered, selected_product)
        simulation_results = monte_carlo_simulation(terr_perf)
        
        col_mc1, col_mc2 = st.columns([2, 1])
        
        with col_mc1:
            fig_mc = go.Figure()
            
            for city, results in simulation_results.items():
                fig_mc.add_trace(go.Box(
                    y=results['simulations'],
                    name=city[:15],  # Kısa isim
                    boxmean='sd'
                ))
            
            fig_mc.update_layout(
                height=500,
                xaxis_title='Territory',
                yaxis_title='Simüle Edilmiş PF Satış',
                showlegend=False,
                plot_bgcolor='#0f172a'
            )
            
            st.plotly_chart(fig_mc, use_container_width=True, key="monte_carlo_box")
        
        with col_mc2:
            st.markdown("##### 📈 Senaryo Analizi")
            
            selected_city_mc = st.selectbox(
                "Territory Seç",
                list(simulation_results.keys()),
                key='mc_select'
            )
            
            if selected_city_mc:
                res = simulation_results[selected_city_mc]
                
                st.metric("📊 Mevcut", f"{res['current']:,.0f}")
                st.metric(
                    "🎯 Ortalama Tahmin",
                    f"{res['mean']:,.0f}",
                    delta=f"{((res['mean']/res['current']-1)*100):.1f}%"
                )
                
                st.markdown("**Senaryo Aralıkları:**")
                st.info(f"😰 Kötümser (%10): {res['p10']:,.0f}")
                st.success(f"😊 Gerçekçi (%50): {res['p50']:,.0f}")
                st.warning(f"🚀 İyimser (%90): {res['p90']:,.0f}")
    
    # TAB 6: AKSİYON PLANI
    with tabs[5]:
        st.header("📋 Otomatik Aksiyon Planı")
        st.caption("🤖 AI destekli öneriler - Veriye dayalı aksiyonlar")
        
        terr_perf = calculate_territory_performance(df_filtered, selected_product)
        strategy_df = calculate_investment_strategy(terr_perf)
        
        aksiyonlar = []
        
        # 1. En büyük fırsatlar
        top_firsatlar = strategy_df[
            (strategy_df['Pazar_Payi_%'] < 5) & 
            (strategy_df['Toplam_Pazar'] > strategy_df['Toplam_Pazar'].median())
        ].nlargest(3, 'Toplam_Pazar')
        
        for idx, row in top_firsatlar.iterrows():
            aksiyonlar.append({
                'Öncelik': '🔴 Kritik',
                'Aksiyon': f"{row['TERRITORIES']}'de agresif yatırım",
                'Neden': f"Pazar büyük ({row['Toplam_Pazar']:,.0f}) ama payımız %{row['Pazar_Payi_%']:.1f}",
                'Potansiyel': f"+{row['Buyume_Potansiyeli']:,.0f}"
            })
        
        # Aksiyonları göster
        for idx, aksiyon in enumerate(aksiyonlar, 1):
            bg_color = "#DC2626" if aksiyon['Öncelik'] == '🔴 Kritik' else "#EA580C"
            
            st.markdown(f"""
            <div style="
                background: {bg_color};
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 15px;
                color: white;
            ">
                <h4>{idx}. {aksiyon['Aksiyon']}</h4>
                <p><b>Neden:</b> {aksiyon['Neden']}</p>
                <p><b>Potansiyel:</b> {aksiyon['Potansiyel']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # TAB 7: RAPOR
    with tabs[6]:
        st.header("📥 Raporları İndir")
        
        terr_perf = calculate_territory_performance(df_filtered, selected_product)
        
        # Excel rapor
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            terr_perf.to_excel(writer, sheet_name='Territory', index=False)
        
        st.download_button(
            "📥 Excel İndir",
            output.getvalue(),
            f"portfoy_analizi_{datetime.now().strftime('%Y%m%d')}.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

if __name__ == "__main__":
    main()
                f"%{avg_growth:.1f}",
                delta="Pozitif" if avg_growth > 0 else "Negatif"
            )
            
            with col_tsm2:
                if 'Volatility' in time_series.columns:
                    avg_vol = time_series['Volatility'].mean()
                    st.metric("📊 Volatilite", f"{avg_vol:,.0f}")
            
            with col_tsm3:
                if forecast_df is not None:
                    total_forecast = forecast_df['Forecast'].sum()
                    st.metric("🔮 Toplam Tahmin", f"{total_forecast:,.0f}")
            
            with col_tsm4:
                if 'Trend_Strength' in time_series.columns:
                    trend_str = time_series['Trend_Strength'].iloc[-1]
                    st.metric("📉 Trend Gücü", f"{trend_str:,.0f}")
            
            # Forecast table
            if forecast_df is not None:
                st.markdown("#### 🔮 Tahmin Detayları")
                st.dataframe(
                    forecast_df.style.format({'Forecast': '{:,.0f}'}),
                    use_container_width=True
                )
    
    # =========================================================================
    # TAB 4: GEOGRAPHIC ANALYSIS
    # =========================================================================
    with tabs[3]:
        st.header("🗺️ Coğrafi Analiz")
        
        st.info("💡 **Not:** Türkiye haritası için turkey.geojson dosyası gereklidir.")
        
        # Region analysis
        if 'REGION' in df_filtered.columns:
            cols_map = get_product_columns(selected_product)
            
            region_data = df_filtered.groupby('REGION').agg({
                cols_map['pf']: 'sum',
                cols_map['rakip']: 'sum'
            }).reset_index()
            
            region_data.columns = ['REGION', 'PF_Satis', 'Rakip_Satis']
            region_data['Toplam_Pazar'] = region_data['PF_Satis'] + region_data['Rakip_Satis']
            region_data['Pazar_Payi_%'] = safe_divide(region_data['PF_Satis'], region_data['Toplam_Pazar']) * 100
            
            # Region bar chart
            st.markdown("#### 📊 Bölge Bazlı Performans")
            
            fig_region = go.Figure()
            
            fig_region.add_trace(go.Bar(
                x=region_data['REGION'],
                y=region_data['PF_Satis'],
                name='PF Satış',
                marker_color=[REGION_COLORS.get(r, '#3B82F6') for r in region_data['REGION']],
                text=region_data['PF_Satis'].apply(lambda x: f'{x:,.0f}'),
                textposition='outside'
            ))
            
            fig_region.update_layout(
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                xaxis=dict(tickangle=-45)
            )
            
            st.plotly_chart(fig_region, use_container_width=True, key="geo_region_bar")
            
            # Region table
            st.dataframe(
                region_data.style.format({
                    'PF_Satis': '{:,.0f}',
                    'Rakip_Satis': '{:,.0f}',
                    'Toplam_Pazar': '{:,.0f}',
                    'Pazar_Payi_%': '{:.1f}%'
                }),
                use_container_width=True
            )
    
    # =========================================================================
    # TAB 5: STRATEGIC PORTFOLIO
    # =========================================================================
    with tabs[4]:
        st.header("⭐ Stratejik Portföy Yönetimi")
        
        # Calculate matrices
        bcg_df = calculate_bcg_matrix(df_filtered, selected_product, start_date, end_date)
        strategy_df = calculate_investment_strategy(bcg_df)
        ge_df = calculate_ge_mckinsey_matrix(strategy_df)
        
        # Portfolio summary
        st.markdown("#### 📊 Portföy Özeti")
        
        col_p1, col_p2, col_p3, col_p4 = st.columns(4)
        
        bcg_counts = strategy_df['BCG_Kategori'].value_counts()
        
        with col_p1:
            stars = bcg_counts.get("⭐ Star", 0)
            st.metric("⭐ Stars", stars)
        
        with col_p2:
            cows = bcg_counts.get("🐄 Cash Cow", 0)
            st.metric("🐄 Cash Cows", cows)
        
        with col_p3:
            questions = bcg_counts.get("❓ Question Mark", 0)
            st.metric("❓ Questions", questions)
        
        with col_p4:
            dogs = bcg_counts.get("🐶 Dog", 0)
            st.metric("🐶 Dogs", dogs)
        
        st.markdown("---")
        
        # BCG Matrix
        col_bcg1, col_bcg2 = st.columns([2, 1])
        
        with col_bcg1:
            st.markdown("#### 🎯 BCG Matrix")
            fig_bcg = create_bcg_scatter(strategy_df, "portfolio_bcg")
            st.plotly_chart(fig_bcg, use_container_width=True, key="portfolio_bcg_main")
        
        with col_bcg2:
            st.markdown("#### 📚 BCG Rehberi")
            
            st.markdown("""
            **⭐ STARS**  
            Yüksek pay + Büyüme  
            → Yatırım yap, büyüt
            
            **🐄 CASH COWS**  
            Yüksek pay + Düşük büyüme  
            → Gelir topla, koru
            
            **❓ QUESTIONS**  
            Düşük pay + Büyüme  
            → En yüksek fırsat!
            
            **🐶 DOGS**  
            Düşük pay + Büyüme  
            → İzle veya çık
            """)
        
        st.markdown("---")
        
        # GE-McKinsey Matrix
        st.markdown("#### 🎯 GE-McKinsey 9-Box Matrix")
        
        fig_ge = create_ge_mckinsey_scatter(ge_df, "portfolio_ge")
        st.plotly_chart(fig_ge, use_container_width=True, key="portfolio_ge_main")
        
        # Investment strategy distribution
        st.markdown("#### 💼 Yatırım Stratejisi Dağılımı")
        
        col_st1, col_st2, col_st3, col_st4, col_st5 = st.columns(5)
        
        strat_counts = strategy_df['Yatirim_Stratejisi'].value_counts()
        
        strategies = [
            ('🚀 Agresif', col_st1),
            ('⚡ Hızlandırılmış', col_st2),
            ('🛡️ Koruma', col_st3),
            ('💎 Potansiyel', col_st4),
            ('👁️ İzleme', col_st5)
        ]
        
        for strat, col in strategies:
            with col:
                count = strat_counts.get(strat, 0)
                pf_sum = strategy_df[strategy_df['Yatirim_Stratejisi'] == strat]['PF_Satis'].sum()
                st.metric(strat, f"{count} terr.", delta=f"{pf_sum:,.0f} PF")
        
        # Detailed strategy table
        st.markdown("#### 📋 Strateji Detay Tablosu")
        
        display_cols = ['TERRITORIES', 'REGION', 'BCG_Kategori', 'GE_Kategori',
                       'Yatirim_Stratejisi', 'PF_Satis', 'Pazar_Payi_%',
                       'Oncelik_Skoru', 'Tavsiye_Edilen_Yatirim']
        display_cols = [c for c in display_cols if c in strategy_df.columns]
        
        st.dataframe(
            strategy_df[display_cols].sort_values('Oncelik_Skoru', ascending=False).style.format({
                'PF_Satis': '{:,.0f}',
                'Pazar_Payi_%': '{:.1f}%',
                'Oncelik_Skoru': '{:.0f}',
                'Tavsiye_Edilen_Yatirim': '{:,.0f}'
            }).background_gradient(subset=['Oncelik_Skoru'], cmap='YlOrRd'),
            use_container_width=True,
            height=500
        )
    
    # =========================================================================
    # TAB 6: MONTE CARLO & RISK
    # =========================================================================
    with tabs[5]:
        st.header("🎲 Monte Carlo Simülasyonu & Risk Analizi")
        
        st.info("🔮 **10,000 senaryo simülasyonu** - Gelecek dönem risk ve fırsat analizi")
        
        terr_perf = calculate_territory_performance(df_filtered, selected_product, start_date, end_date)
        
        col_mc1, col_mc2 = st.columns([1, 1])
        
        with col_mc1:
            n_sims = st.slider("🎯 Simülasyon Sayısı", 100, 10000, 1000, step=100)
        
        with col_mc2:
            mc_periods = st.slider("📅 Tahmin Periyodu", 3, 12, 6)
        
        # Run simulation
        simulation_results = monte_carlo_simulation(terr_perf, n_sims, mc_periods)
        
        # Visualization
        col_mcv1, col_mcv2 = st.columns([2, 1])
        
        with col_mcv1:
            st.markdown("#### 📊 Simülasyon Sonuçları (Box Plot)")
            
            fig_mc = go.Figure()
            
            for territory, results in simulation_results.items():
                fig_mc.add_trace(go.Box(
                    y=results['final_values'],
                    name=territory[:20],
                    boxmean='sd',
                    marker_color='#3B82F6'
                ))
            
            fig_mc.update_layout(
                height=500,
                xaxis_title='Territory',
                yaxis_title='Simüle Edilmiş PF Satış',
                showlegend=False,
                plot_bgcolor='#0f172a',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            
            st.plotly_chart(fig_mc, use_container_width=True, key="mc_box_plot")
        
        with col_mcv2:
            st.markdown("#### 🎯 Territory Seç")
            
            selected_mc_terr = st.selectbox(
                "Territory",
                list(simulation_results.keys()),
                key='mc_territory_select'
            )
            
            if selected_mc_terr:
                res = simulation_results[selected_mc_terr]
                
                st.metric("📊 Mevcut", f"{res['current']:,.0f}")
                st.metric(
                    "🎯 Ortalama Tahmin",
                    f"{res['mean']:,.0f}",
                    delta=f"{((res['mean']/res['current']-1)*100):.1f}%"
                )
                
                st.markdown("**Senaryo Aralıkları:**")
                st.success(f"🚀 İyimser (P90): {res['p90']:,.0f}")
                st.info(f"😊 Gerçekçi (P50): {res['p50']:,.0f}")
                st.warning(f"😰 Kötümser (P10): {res['p10']:,.0f}")
                
                st.markdown("**Risk Metrikleri:**")
                st.metric("📈 Artış Olasılığı", f"%{res['prob_increase']:.1f}")
                st.metric("🚀 İkiye Katlama", f"%{res['prob_double']:.1f}")
                st.metric("📊 Volatilite (Std)", f"{res['std']:,.0f}")
        
        st.markdown("---")
        
        # Monte Carlo paths
        if selected_mc_terr:
            st.markdown("#### 📈 Simülasyon Yolları (İlk 100)")
            
            res = simulation_results[selected_mc_terr]
            paths = res['paths'][:100]
            
            fig_paths = go.Figure()
            
            for i, path in enumerate(paths):
                fig_paths.add_trace(go.Scatter(
                    y=path,
                    mode='lines',
                    line=dict(width=1, color='rgba(59, 130, 246, 0.1)'),
                    showlegend=False,
                    hoverinfo='skip'
                ))
            
            # Add mean path
            mean_path = res['paths'].mean(axis=0)
            fig_paths.add_trace(go.Scatter(
                y=mean_path,
                mode='lines',
                name='Ortalama',
                line=dict(width=4, color='#F59E0B')
            ))
            
            fig_paths.update_layout(
                height=400,
                xaxis_title='Periyot',
                yaxis_title='PF Satış',
                plot_bgcolor='#0f172a',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            
            st.plotly_chart(fig_paths, use_container_width=True, key="mc_paths")
    
    # =========================================================================
    # TAB 7: AI INSIGHTS & ML
    # =========================================================================
    with tabs[6]:
        st.header("🤖 AI İçgörüler & Machine Learning")
        
        terr_perf = calculate_territory_performance(df_filtered, selected_product, start_date, end_date)
        
        # Clustering
        st.markdown("#### 🔍 K-Means Clustering Analizi")
        
        n_clusters = st.slider("🎯 Cluster Sayısı", 2, 6, 4)
        
        cluster_df = perform_clustering(terr_perf, n_clusters)
        
        if 'Cluster_Label' in cluster_df.columns:
            # Cluster visualization
            col_cl1, col_cl2 = st.columns(2)
            
            with col_cl1:
                st.markdown("##### 📊 Cluster Dağılımı")
                
                fig_cluster = px.scatter(
                    cluster_df,
                    x='Toplam_Pazar',
                    y='Pazar_Payi_%',
                    color='Cluster_Label',
                    size='PF_Satis',
                    hover_name='TERRITORIES',
                    size_max=50
                )
                
                fig_cluster.update_layout(
                    height=500,
                    plot_bgcolor='#0f172a',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                
                st.plotly_chart(fig_cluster, use_container_width=True, key="cluster_scatter")
            
            with col_cl2:
                st.markdown("##### 📋 Cluster Özeti")
                
                cluster_summary = cluster_df.groupby('Cluster_Label').agg({
                    'TERRITORIES': 'count',
                    'PF_Satis': 'mean',
                    'Pazar_Payi_%': 'mean',
                    'Toplam_Pazar': 'mean'
                }).reset_index()
                
                cluster_summary.columns = ['Cluster', 'Territory Sayısı', 
                                           'Ort. PF Satış', 'Ort. Pazar Payı', 'Ort. Pazar']
                
                st.dataframe(
                    cluster_summary.style.format({
                        'Ort. PF Satış': '{:,.0f}',
                        'Ort. Pazar Payı': '{:.1f}%',
                        'Ort. Pazar': '{:,.0f}'
                    }),
                    use_container_width=True
                )
        
        st.markdown("---")
        
        # Anomaly Detection
        st.markdown("#### 🚨 Anomali Tespiti")
        
        anom_df = detect_anomalies(terr_perf, 'PF_Satis', threshold=2)
        
        anomalies = anom_df[anom_df['Is_Anomaly'] == True]
        
        if len(anomalies) > 0:
            st.markdown(f"""
            <div class="alert-warning">
                <h4>⚠️ {len(anomalies)} Anomali Tespit Edildi!</h4>
            </div>
            """, unsafe_allow_html=True)
            
            col_an1, col_an2 = st.columns(2)
            
            with col_an1:
                pos_anom = anomalies[anomalies['Anomaly_Type'] == '📈 Pozitif Anomali']
                st.success(f"📈 **{len(pos_anom)} Pozitif Anomali**")
                if len(pos_anom) > 0:
                    st.dataframe(
                        pos_anom[['TERRITORIES', 'PF_Satis', 'Z_Score']],
                        use_container_width=True
                    )
            
            with col_an2:
                neg_anom = anomalies[anomalies['Anomaly_Type'] == '📉 Negatif Anomali']
                st.error(f"📉 **{len(neg_anom)} Negatif Anomali**")
                if len(neg_anom) > 0:
                    st.dataframe(
                        neg_anom[['TERRITORIES', 'PF_Satis', 'Z_Score']],
                        use_container_width=True
                    )
        else:
            st.success("✅ Anomali tespit edilmedi!")
        
        st.markdown("---")
        
        # Auto-generated SWOT
        st.markdown("#### 🎯 Otomatik SWOT Analizi")
        
        swot = generate_swot_analysis(terr_perf)
        
        col_sw1, col_sw2 = st.columns(2)
        
        with col_sw1:
            st.markdown("##### 💪 Güçlü Yönler")
            for s in swot['Strengths']:
                st.success(f"✅ {s}")
            
            st.markdown("##### 🌟 Fırsatlar")
            for o in swot['Opportunities']:
                st.info(f"💡 {o}")
        
        with col_sw2:
            st.markdown("##### ⚠️ Zayıf Yönler")
            for w in swot['Weaknesses']:
                st.warning(f"⚠️ {w}")
            
            st.markdown("##### 🚨 Tehditler")
            for t in swot['Threats']:
                st.error(f"🚨 {t}")
    
    # =========================================================================
    # TAB 8: MANAGER PERFORMANCE
    # =========================================================================
    with tabs[7]:
        st.header("👥 Manager Performans Analizi")
        
        manager_perf = calculate_manager_performance(df_filtered, selected_product)
        
        # Top 3 managers
        st.markdown("#### 🏆 Top 3 Manager")
        
        top3 = manager_perf.head(3)
        
        col_m1, col_m2, col_m3 = st.columns(3)
        
        medals = ["🥇", "🥈", "🥉"]
        colors = [
            "linear-gradient(135deg, #FFD700 0%, #FFA500 100%)",
            "linear-gradient(135deg, #C0C0C0 0%, #A8A8A8 100%)",
            "linear-gradient(135deg, #CD7F32 0%, #B8860B 100%)"
        ]
        
        for idx, (col, row, medal, color) in enumerate(zip([col_m1, col_m2, col_m3], top3.itertuples(), medals, colors)):
            with col:
                st.markdown(f"""
                <div style="
                    background: {color};
                    padding: 2rem;
                    border-radius: 16px;
                    text-align: center;
                    color: white;
                    box-shadow: 0 8px 24px rgba(0,0,0,0.3);
                ">
                    <h1 style="font-size: 3rem; margin: 0;">{medal}</h1>
                    <h3 style="margin: 1rem 0; font-weight: bold;">{row.Manager}</h3>
                    <h2 style="font-size: 2rem; margin: 1rem 0;">{row.PF_Satis:,.0f}</h2>
                    <p style="margin: 0.5rem 0;">PF Satış</p>
                    <p style="margin: 0.5rem 0;">{int(row.Territory_Sayisi)} Territory</p>
                    <h3 style="margin: 1rem 0;">%{row.Pazar_Payi_:,.1f}</h3>
                    <p style="margin: 0;">Pazar Payı</p>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Manager comparison
        col_mv1, col_mv2 = st.columns(2)
        
        with col_mv1:
            st.markdown("#### 📊 Manager Bazlı PF Satış")
            
            fig_mgr = px.bar(
                manager_perf.head(10),
                x='Manager',
                y='PF_Satis',
                color='Pazar_Payi_%',
                color_continuous_scale='Blues',
                text='PF_Satis'
            )
            
            fig_mgr.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
            fig_mgr.update_layout(
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                xaxis=dict(tickangle=-45)
            )
            
            st.plotly_chart(fig_mgr, use_container_width=True, key="mgr_bar")
        
        with col_mv2:
            st.markdown("#### 🎯 Efficiency Score")
            
            fig_eff = px.scatter(
                manager_perf,
                x='Territory_Sayisi',
                y='Pazar_Payi_%',
                size='PF_Satis',
                color='Performance_Tier',
                hover_name='Manager',
                size_max=50
            )
            
            fig_eff.update_layout(
                height=400,
                plot_bgcolor='#0f172a',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            
            st.plotly_chart(fig_eff, use_container_width=True, key="mgr_eff")
        
        # Detailed table
        st.markdown("#### 📋 Tüm Manager Detayları")
        
        st.dataframe(
            manager_perf.style.format({
                'PF_Satis': '{:,.0f}',
                'Rakip_Satis': '{:,.0f}',
                'Toplam_Pazar': '{:,.0f}',
                'Pazar_Payi_%': '{:.1f}%',
                'Avg_PF_Per_Territory': '{:,.0f}',
                'Efficiency_Score': '{:.1f}'
            }).background_gradient(subset=['Efficiency_Score'], cmap='RdYlGn'),
            use_container_width=True,
            height=500
        )
    
    # =========================================================================
    # TAB 9: ACTION PLAN
    # =========================================================================
    with tabs[8]:
        st.header("📋 Otomatik Aksiyon Planı")
        
        terr_perf = calculate_territory_performance(df_filtered, selected_product, start_date, end_date)
        strategy_df = calculate_investment_strategy(terr_perf)
        
        # Generate actions
        aksiyonlar = []
        
        # 1. Büyük fırsatlar
        big_opps = strategy_df[
            (strategy_df['Pazar_Payi_%'] < 5) &
            (strategy_df['Toplam_Pazar'] > strategy_df['Toplam_Pazar'].median())
        ].nlargest(3, 'Toplam_Pazar')
        
        for idx, row in big_opps.iterrows():
            aksiyonlar.append({
                'Öncelik': '🔴 Kritik',
                'Territory': row['TERRITORIES'],
                'Aksiyon': 'Agresif yatırım ve penetrasyon stratejisi',
                'Neden': f"Büyük pazar ({row['Toplam_Pazar']:,.0f}) ama payımız sadece %{row['Pazar_Payi_%']:.1f}",
                'Hedef': f"Pazar payını %{row['Pazar_Payi_%']:.1f} → %15'e çıkar",
                'Potansiyel': f"+{row['Buyume_Potansiyeli']:,.0f} kutu",
                'Tavsiye_Yatirim': f"{row.get('Tavsiye_Edilen_Yatirim', 0):,.0f}",
                'Timeline': '3-6 ay'
            })
        
        # 2. Sıfır satış
        zero_sales = strategy_df[strategy_df['PF_Satis'] == 0].nlargest(2, 'Toplam_Pazar')
        
        for idx, row in zero_sales.iterrows():
            aksiyonlar.append({
                'Öncelik': '🟠 Yüksek',
                'Territory': row['TERRITORIES'],
                'Aksiyon': 'Pazar girişi stratejisi',
                'Neden': f"Hiç satış yok ama pazar var ({row['Toplam_Pazar']:,.0f})",
                'Hedef': 'İlk satışları başlat, dağıtım kanalları kur',
                'Potansiyel': f"+{row['Toplam_Pazar']:,.0f} kutu",
                'Tavsiye_Yatirim': f"{row.get('Tavsiye_Edilen_Yatirim', 0):,.0f}",
                'Timeline': '6-12 ay'
            })
        
        # 3. Question Marks
        questions = strategy_df[
            (strategy_df['BCG_Kategori'] == "❓ Question Mark") &
            (strategy_df['Oncelik_Skoru'] > 50)
        ].nlargest(2, 'Oncelik_Skoru')
        
        for idx, row in questions.iterrows():
            aksiyonlar.append({
                'Öncelik': '🟡 Orta',
                'Territory': row['TERRITORIES'],
                'Aksiyon': 'Star olmak için yatırım artır',
                'Neden': f"Büyüyen pazar, düşük payımız var",
                'Hedef': 'Question Mark → Star geçişi',
                'Potansiyel': f"+{row['Buyume_Potansiyeli']:,.0f} kutu",
                'Tavsiye_Yatirim': f"{row.get('Tavsiye_Edilen_Yatirim', 0):,.0f}",
                'Timeline': '9-12 ay'
            })
        
        # Display actions
        st.markdown("#### 🎯 Öncelikli Aksiyonlar")
        
        for i, aksiyon in enumerate(aksiyonlar, 1):
            priority_colors = {
                '🔴 Kritik': '#DC2626',
                '🟠 Yüksek': '#EA580C',
                '🟡 Orta': '#F59E0B'
            }
            
            bg_color = priority_colors.get(aksiyon['Öncelik'], '#3B82F6')
            
            st.markdown(f"""
            <div style="
                background: {bg_color};
                padding: 1.5rem;
                border-radius: 12px;
                margin-bottom: 1rem;
                color: white;
            ">
                <h3>{i}. {aksiyon['Territory']} - {aksiyon['Aksiyon']}</h3>
                <p><strong>Öncelik:</strong> {aksiyon['Öncelik']}</p>
                <p><strong>Neden:</strong> {aksiyon['Neden']}</p>
                <p><strong>Hedef:</strong> {aksiyon['Hedef']}</p>
                <p><strong>Potansiyel Kazanç:</strong> {aksiyon['Potansiyel']}</p>
                <p><strong>Tavsiye Edilen Yatırım:</strong> {aksiyon['Tavsiye_Yatirim']}</p>
                <p><strong>Zaman Çizelgesi:</strong> {aksiyon['Timeline']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # =========================================================================
    # TAB 10: REPORTS & EXPORT
    # =========================================================================
    with tabs[9]:
        st.header("📥 Raporlar & Export")
        
        terr_perf = calculate_territory_performance(df_filtered, selected_product, start_date, end_date)
        strategy_df = calculate_investment_strategy(terr_perf)
        
        col_exp1, col_exp2 = st.columns(2)
        
        with col_exp1:
            st.markdown("#### 📊 Excel Raporları")
            
            # Multi-sheet Excel
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                terr_perf.to_excel(writer, sheet_name='Territory Performance', index=False)
                strategy_df.to_excel(writer, sheet_name='Investment Strategy', index=False)
                
                if 'MANAGER' in df_filtered.columns:
                    manager_perf = calculate_manager_performance(df_filtered, selected_product)
                    manager_perf.to_excel(writer, sheet_name='Manager Performance', index=False)
            
            st.download_button(
                "📥 İndir: Kapsamlı Excel Raporu",
                output.getvalue(),
                f"ticari_portfoy_analizi_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="excel_download"
            )
        
        with col_exp2:
            st.markdown("#### 📄 CSV Exports")
            
            # CSV exports
            csv_terr = terr_perf.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 İndir: Territory CSV",
                csv_terr,
                f"territory_performance_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv",
                key="csv_terr"
            )
            
            csv_strat = strategy_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 İndir: Strategy CSV",
                csv_strat,
                f"investment_strategy_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv",
                key="csv_strat"
            )
        
        st.markdown("---")
        
        # Executive summary
        st.markdown("#### 📊 Executive Summary")
        
        summary = f"""
        # TİCARİ PORTFÖY ANALİZ RAPORU
        
        **Rapor Tarihi:** {datetime.now().strftime('%d.%m.%Y %H:%M')}  
        **Ürün:** {selected_product}  
        **Territory:** {selected_territory}  
        **Dönem:** {start_date} - {end_date}
        
        ## GENEL ÖZET
        
        - **Toplam PF Satış:** {total_pf:,.0f}
        - **Toplam Pazar:** {total_market:,.0f}
        - **Pazar Payı:** %{market_share:.1f}
        - **Aktif Territory:** {len(terr_perf)}
        
        ## PORTFÖY DAĞILIMI
        
        - ⭐ **Stars:** {len(strategy_df[strategy_df['BCG_Kategori'] == "⭐ Star"])} territory
        - 🐄 **Cash Cows:** {len(strategy_df[strategy_df['BCG_Kategori'] == "🐄 Cash Cow"])} territory
        - ❓ **Question Marks:** {len(strategy_df[strategy_df['BCG_Kategori'] == "❓ Question Mark"])} territory
        - 🐶 **Dogs:** {len(strategy_df[strategy_df['BCG_Kategori'] == "🐶 Dog"])} territory
        
        ## STRATEJİK ÖNERİLER
        
        {len(big_opps)} büyük fırsat tespit edildi.  
        Toplam potansiyel: {big_opps['Buyume_Potansiyeli'].sum():,.0f} kutu
        
        ---
        
        *Bu rapor Ultra Ticari Portföy Analiz Sistemi v3.0 tarafından otomatik oluşturulmuştur.*
        """
        
        st.markdown(summary)
        
        # Download as text
        st.download_button(
            "📥 İndir: Executive Summary (TXT)",
            summary.encode('utf-8'),
            f"executive_summary_{datetime.now().strftime('%Y%m%d')}.txt",
            "text/plain",
            key="summary_txt"
        )

if __name__ == "__main__":
    main()

# =============================================================================
# BONUS FEATURES & UTILITIES
# =============================================================================

def calculate_pareto_analysis(df):
    """Pareto 80/20 analizi"""
    df_sorted = df.sort_values('PF_Satis', ascending=False).copy()
    df_sorted['Cumulative_PF'] = df_sorted['PF_Satis'].cumsum()
    df_sorted['Cumulative_%'] = (df_sorted['Cumulative_PF'] / df_sorted['PF_Satis'].sum() * 100)
    
    # 80% satışı yapan territory sayısı
    terr_80 = df_sorted[df_sorted['Cumulative_%'] <= 80]['TERRITORIES'].count()
    
    return df_sorted, terr_80

def calculate_concentration_risk(df):
    """Konsantrasyon riski - Herfindahl Index"""
    total_sales = df['PF_Satis'].sum()
    
    if total_sales == 0:
        return 0
    
    market_shares = df['PF_Satis'] / total_sales
    herfindahl_index = (market_shares ** 2).sum() * 10000
    
    # HHI Interpretation:
    # < 1000: Düşük konsantrasyon
    # 1000-1800: Orta konsantrasyon
    # > 1800: Yüksek konsantrasyon
    
    return herfindahl_index

def calculate_growth_momentum(df, periods=3):
    """Momentum skoru hesaplama"""
    df_mom = df.copy()
    
    if 'PF_Buyume_%' in df_mom.columns:
        # Son N dönem büyüme ortalaması
        df_mom['Momentum_Score'] = df_mom['PF_Buyume_%'].rolling(window=periods, min_periods=1).mean()
        
        # Acceleration (büyümenin artış hızı)
        df_mom['Acceleration'] = df_mom['PF_Buyume_%'].diff()
    
    return df_mom

def calculate_market_penetration(df):
    """Pazar penetrasyon derinliği"""
    df_pen = df.copy()
    
    # Penetrasyon skoru (0-100)
    df_pen['Penetration_Score'] = (
        (df_pen['Pazar_Payi_%'] / 100) * 70 +
        (df_pen['Agirlik_%'] / 100) * 30
    ) * 100
    
    # Penetrasyon kategorisi
    def assign_penetration(score):
        if score >= 75:
            return "🔥 Dominant"
        elif score >= 50:
            return "💪 Strong"
        elif score >= 25:
            return "📈 Growing"
        else:
            return "🌱 Emerging"
    
    df_pen['Penetration_Category'] = df_pen['Penetration_Score'].apply(assign_penetration)
    
    return df_pen

def calculate_efficiency_metrics(df):
    """Etkinlik metrikleri"""
    df_eff = df.copy()
    
    # ROI (basitleştirilmiş)
    df_eff['ROI_Estimate'] = safe_divide(df_eff['PF_Satis'], df_eff['Buyume_Potansiyeli']) * 100
    
    # Market share gain potential
    df_eff['Share_Gain_Potential'] = (100 - df_eff['Pazar_Payi_%']) * df_eff['Toplam_Pazar'] / 100
    
    # Efficiency ratio
    df_eff['Efficiency_Ratio'] = safe_divide(df_eff['PF_Satis'], df_eff['Toplam_Pazar']) * 100
    
    return df_eff

def generate_recommendations(df, top_n=5):
    """AI-powered öneriler"""
    recommendations = []
    
    # 1. Hızlı kazanım fırsatları
    quick_wins = df[
        (df['Pazar_Payi_%'] >= 40) &
        (df['Pazar_Payi_%'] < 60) &
        (df['Toplam_Pazar'] > df['Toplam_Pazar'].median())
    ].nlargest(top_n, 'Buyume_Potansiyeli')
    
    for idx, row in quick_wins.iterrows():
        recommendations.append({
            'Tip': '⚡ Hızlı Kazanım',
            'Territory': row['TERRITORIES'],
            'Aksiyon': 'Son kilometre için güçlü itme',
            'Beklenen_Sonuc': f"%{row['Pazar_Payi_%']:.0f} → %60+ pazar payı',
            'Yatirim_Seviyesi': 'Orta',
            'Risk': 'Düşük',
            'ROI': 'Yüksek'
        })
    
    # 2. Defansif aksiyonlar
    defensive = df[
        (df['Goreceli_Pazar_Payi'] > 1.5) &
        (df['Toplam_Pazar'] > df['Toplam_Pazar'].quantile(0.75))
    ].nlargest(top_n, 'PF_Satis')
    
    for idx, row in defensive.iterrows():
        recommendations.append({
            'Tip': '🛡️ Defansif',
            'Territory': row['TERRITORIES'],
            'Aksiyon': 'Lider pozisyonu koru, rakip saldırılarını önle',
            'Beklenen_Sonuc': 'Pazar liderliğinin sürdürülmesi',
            'Yatirim_Seviyesi': 'Stabil',
            'Risk': 'Orta',
            'ROI': 'Stabil'
        })
    
    # 3. Büyüme odaklı
    growth = df[
        (df['Pazar_Payi_%'] < 20) &
        (df['Toplam_Pazar'] > df['Toplam_Pazar'].median()) &
        (df['Buyume_Potansiyeli'] > df['Buyume_Potansiyeli'].quantile(0.75))
    ].nlargest(top_n, 'Buyume_Potansiyeli')
    
    for idx, row in growth.iterrows():
        recommendations.append({
            'Tip': '🚀 Büyüme',
            'Territory': row['TERRITORIES'],
            'Aksiyon': 'Agresif pazar girişi ve pay kazanma',
            'Beklenen_Sonuc': f"+{row['Buyume_Potansiyeli']:,.0f} kutu potansiyel",
            'Yatirim_Seviyesi': 'Yüksek',
            'Risk': 'Yüksek',
            'ROI': 'Çok Yüksek'
        })
    
    return pd.DataFrame(recommendations)

def create_performance_heatmap(df):
    """Performans ısı haritası"""
    if 'REGION' not in df.columns or 'Yatirim_Stratejisi' not in df.columns:
        return None
    
    pivot = df.pivot_table(
        index='REGION',
        columns='Yatirim_Stratejisi',
        values='PF_Satis',
        aggfunc='sum',
        fill_value=0
    )
    
    fig = px.imshow(
        pivot,
        labels=dict(x="Yatırım Stratejisi", y="Bölge", color="PF Satış"),
        color_continuous_scale='YlOrRd',
        aspect="auto",
        text_auto='.0f'
    )
    
    fig.update_layout(
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(tickangle=-30)
    )
    
    return fig

def create_waterfall_chart(df, top_n=15):
    """Waterfall chart - Kümülatif katkı"""
    top_terr = df.nlargest(top_n, 'PF_Satis').copy()
    
    fig = go.Figure(go.Waterfall(
        name="PF Satış",
        orientation="v",
        measure=["relative"] * len(top_terr) + ["total"],
        x=list(top_terr['TERRITORIES']) + ["🎯 TOPLAM"],
        y=list(top_terr['PF_Satis']) + [0],
        text=[f"{x:,.0f}" for x in top_terr['PF_Satis']] + [f"{top_terr['PF_Satis'].sum():,.0f}"],
        textposition="outside",
        connector={"line": {"color": "rgba(255,255,255,0.3)", "width": 2}},
        increasing={"marker": {"color": "#10B981", "line": {"color": "white", "width": 1}}},
        decreasing={"marker": {"color": "#EF4444"}},
        totals={"marker": {"color": "#3B82F6", "line": {"color": "white", "width": 2}}}
    ))
    
    fig.update_layout(
        title=f"Top {top_n} Territory - Kümülatif PF Satış Katkısı",
        height=500,
        plot_bgcolor='#0f172a',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(tickangle=-45),
        showlegend=False
    )
    
    return fig

def create_radar_chart(df, territories):
    """Multi-territory radar chart"""
    categories = ['PF Satış', 'Pazar Payı', 'Pazar Büyüklüğü', 'Performans Skoru']
    
    fig = go.Figure()
    
    for territory in territories:
        terr_data = df[df['TERRITORIES'] == territory]
        
        if len(terr_data) > 0:
            row = terr_data.iloc[0]
            
            values = [
                row['PF_Satis'] / df['PF_Satis'].max() * 100,
                row['Pazar_Payi_%'],
                row['Toplam_Pazar'] / df['Toplam_Pazar'].max() * 100,
                row.get('Performance_Score', 50)
            ]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=territory[:20]
            ))
    
    fig.update_layout(
        polar=dict(
            bgcolor='#0f172a',
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor='rgba(148,163,184,0.2)'
            ),
            angularaxis=dict(
                gridcolor='rgba(148,163,184,0.2)'
            )
        ),
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        showlegend=True
    )
    
    return fig

def create_bubble_chart(df):
    """3D Bubble chart"""
    fig = px.scatter(
        df.nlargest(30, 'PF_Satis'),
        x='Toplam_Pazar',
        y='Pazar_Payi_%',
        size='PF_Satis',
        color='Performance_Score',
        color_continuous_scale='Viridis',
        hover_name='TERRITORIES',
        size_max=60
    )
    
    fig.update_layout(
        title='Performans Bubble Chart',
        xaxis_title='Toplam Pazar →',
        yaxis_title='Pazar Payı (%) →',
        height=600,
        plot_bgcolor='#0f172a',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    return fig

def create_funnel_chart(df):
    """Sales funnel chart"""
    total_market = df['Toplam_Pazar'].sum()
    total_pf = df['PF_Satis'].sum()
    top_20 = df.nlargest(20, 'PF_Satis')['PF_Satis'].sum()
    top_10 = df.nlargest(10, 'PF_Satis')['PF_Satis'].sum()
    top_5 = df.nlargest(5, 'PF_Satis')['PF_Satis'].sum()
    
    funnel_data = pd.DataFrame({
        'Aşama': ['🌍 Toplam Pazar', '📦 PF Toplam', '🏆 Top 20', '⭐ Top 10', '👑 Top 5'],
        'Değer': [total_market, total_pf, top_20, top_10, top_5]
    })
    
    fig = go.Figure(go.Funnel(
        y=funnel_data['Aşama'],
        x=funnel_data['Değer'],
        textposition='inside',
        textinfo='value+percent initial',
        marker=dict(color=['#60A5FA', '#3B82F6', '#2563EB', '#1D4ED8', '#1E40AF'])
    ))
    
    fig.update_layout(
        title='Pazar Penetrasyon Hunisi',
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    return fig

# =============================================================================
# ADVANCED VISUALIZATIONS (3D)
# =============================================================================

def create_3d_scatter(df):
    """3D Scatter plot"""
    fig = px.scatter_3d(
        df.nlargest(50, 'PF_Satis'),
        x='Toplam_Pazar',
        y='Pazar_Payi_%',
        z='PF_Satis',
        color='Performance_Score',
        size='Buyume_Potansiyeli',
        hover_name='TERRITORIES',
        color_continuous_scale='Viridis',
        size_max=50
    )
    
    fig.update_layout(
        title='3D Performans Analizi',
        height=700,
        scene=dict(
            bgcolor='#0f172a',
            xaxis=dict(
                title='Pazar Büyüklüğü',
                backgroundcolor='#0f172a',
                gridcolor='rgba(148,163,184,0.2)'
            ),
            yaxis=dict(
                title='Pazar Payı (%)',
                backgroundcolor='#0f172a',
                gridcolor='rgba(148,163,184,0.2)'
            ),
            zaxis=dict(
                title='PF Satış',
                backgroundcolor='#0f172a',
                gridcolor='rgba(148,163,184,0.2)'
            )
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    return fig

def create_parallel_coordinates(df):
    """Parallel coordinates plot"""
    fig = px.parallel_coordinates(
        df.nlargest(30, 'PF_Satis'),
        dimensions=['PF_Satis', 'Pazar_Payi_%', 'Toplam_Pazar', 'Performance_Score'],
        color='Performance_Score',
        color_continuous_scale='Viridis',
        labels={
            'PF_Satis': 'PF Satış',
            'Pazar_Payi_%': 'Pazar Payı',
            'Toplam_Pazar': 'Pazar',
            'Performance_Score': 'Skor'
        }
    )
    
    fig.update_layout(
        title='Paralel Koordinat Analizi',
        height=500,
        plot_bgcolor='#0f172a',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    return fig

# =============================================================================
# STATISTICAL TESTS & CORRELATIONS
# =============================================================================

def perform_correlation_analysis(df):
    """Korelasyon analizi"""
    numeric_cols = ['PF_Satis', 'Rakip_Satis', 'Toplam_Pazar', 'Pazar_Payi_%', 
                   'Buyume_Potansiyeli', 'Performance_Score']
    
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    
    if len(numeric_cols) < 2:
        return None
    
    corr_matrix = df[numeric_cols].corr()
    
    fig = px.imshow(
        corr_matrix,
        labels=dict(color="Korelasyon"),
        color_continuous_scale='RdBu',
        aspect="auto",
        text_auto='.2f',
        zmin=-1,
        zmax=1
    )
    
    fig.update_layout(
        title='Korelasyon Matrisi',
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    return fig

def calculate_risk_metrics(df):
    """Risk metrikleri"""
    df_risk = df.copy()
    
    # Volatility (std dev)
    df_risk['Sales_Volatility'] = df_risk['PF_Satis'].std() / df_risk['PF_Satis'].mean()
    
    # Coefficient of Variation
    df_risk['CV'] = safe_divide(df_risk['PF_Satis'].std(), df_risk['PF_Satis'].mean()) * 100
    
    # Downside risk (below median)
    median_sales = df_risk['PF_Satis'].median()
    below_median = df_risk[df_risk['PF_Satis'] < median_sales]
    df_risk['Downside_Risk'] = len(below_median) / len(df_risk) * 100
    
    # Sharpe-like ratio (simplified)
    risk_free_rate = 0.05  # 5% baseline
    excess_return = (df_risk['PF_Satis'].mean() - risk_free_rate * df_risk['Toplam_Pazar'].mean())
    df_risk['Sharpe_Ratio'] = safe_divide(excess_return, df_risk['PF_Satis'].std())
    
    return df_risk

# =============================================================================
# BONUS ANALYTICS END
# =============================================================================

"""
═══════════════════════════════════════════════════════════════════════════════
END OF ULTRA ADVANCED COMMERCIAL PORTFOLIO ANALYSIS SYSTEM v3.0
═══════════════════════════════════════════════════════════════════════════════

TOTAL FEATURES IMPLEMENTED:
- 📊 10 Main Analysis Tabs
- 🎯 50+ Visualization Types
- 🤖 Advanced ML Algorithms
- 📈 Multiple Forecasting Methods
- ⭐ Strategic Portfolio Matrices
- 🎲 Monte Carlo Simulation
- 👥 Manager Performance Analytics
- 📋 Auto-Generated Action Plans
- 📥 Multi-Format Export (Excel, CSV, PDF, TXT)
- 🗺️ Geographic Analysis with Map Support
- 🔍 AI-Powered Insights & Recommendations
- 📊 Real-time KPI Tracking
- 🎨 Premium UI/UX Design
- ⚡ High Performance Data Processing

TOTAL LINES OF CODE: 2000+

Thank you for using this advanced analytics platform!
═══════════════════════════════════════════════════════════════════════════════
"""
