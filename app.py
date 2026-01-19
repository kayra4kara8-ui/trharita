"""
🎯 GELİŞMİŞ TİCARİ PORTFÖY ANALİZ SİSTEMİ
Territory Bazlı Performans, ML Tahminleme, Türkiye Haritası ve Rekabet Analizi

Özellikler:
- 🗺️ Türkiye il bazlı harita görselleştirme (GELİŞTİRİLMİŞ VERSİYON)
- 🤖 GERÇEK Machine Learning (Linear Regression, Ridge, Random Forest)
- 📊 Aylık/Yıllık dönem seçimi
- 📈 Gelişmiş rakip analizi ve trend karşılaştırması
- 🎯 Dinamik zaman aralığı filtreleme
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
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Ticari Portföy Analizi",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CSS
# =============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f1729 0%, #1a1f2e 50%, #242837 100%);
        background-attachment: fixed;
    }
    
    .main-header {
        font-size: 3.5rem;
        font-weight: 900;
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #3B82F6 0%, #10B981 50%, #F59E0B 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 50px rgba(59, 130, 246, 0.2);
        letter-spacing: -0.5px;
        margin-bottom: 1rem;
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #3B82F6 0%, #10B981 50%, #F59E0B 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    div[data-testid="metric-container"] {
        background: rgba(30, 41, 59, 0.85);
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid rgba(59, 130, 246, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25);
        backdrop-filter: blur(12px);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-8px);
        box-shadow: 0 16px 48px rgba(59, 130, 246, 0.25);
        border-color: rgba(59, 130, 246, 0.4);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        padding: 0.5rem;
        background: rgba(30, 41, 59, 0.7);
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #94a3b8;
        font-weight: 600;
        padding: 1rem 2rem;
        background: rgba(30, 41, 59, 0.5);
        border-radius: 10px;
        margin: 0;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border: 1px solid transparent;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(59, 130, 246, 0.15);
        color: #e0e7ff;
        border-color: rgba(59, 130, 246, 0.3);
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #3B82F6 0%, #10B981 50%, #F59E0B 100%);
        color: white;
        box-shadow: 0 8px 24px rgba(59, 130, 246, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
        transform: scale(1.02);
    }
    
    h1, h2, h3 {
        color: #f8fafc !important;
        font-weight: 700;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    h1 {
        font-size: 2.5rem;
        margin-top: 0;
    }
    
    h2 {
        font-size: 2rem;
        margin-top: 0;
    }
    
    h3 {
        font-size: 1.5rem;
    }
    
    p, span, div, label {
        color: #cbd5e1;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #3B82F6 0%, #10B981 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 12px;
        font-weight: 600;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 16px rgba(59, 130, 246, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .stButton>button:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 28px rgba(59, 130, 246, 0.4);
    }
    
    .stButton>button:active {
        transform: translateY(-1px);
    }
    
    .stButton>button::after {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        transition: left 0.7s;
    }
    
    .stButton>button:hover::after {
        left: 100%;
    }
    
    .dataframe {
        border-radius: 12px;
        overflow: hidden;
    }
    
    .stDataFrame {
        border-radius: 12px;
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(30, 41, 59, 0.5);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #3B82F6 0%, #10B981 100%);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #10B981 0%, #F59E0B 100%);
    }
    
    /* Card styling for visualizations */
    .plotly-graph-div {
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
    }
    
    /* Sidebar improvements */
    [data-testid="stSidebar"] {
        background: rgba(15, 23, 41, 0.95);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(59, 130, 246, 0.1);
    }
    
    /* Input field styling */
    .stSelectbox, .stSlider, .stRadio {
        background: rgba(30, 41, 59, 0.7);
        padding: 8px;
        border-radius: 10px;
        border: 1px solid rgba(59, 130, 246, 0.2);
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #3B82F6 0%, #10B981 50%, #F59E0B 100%);
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SADE RENK PALETİ
# =============================================================================
# Sade ve profesyonel bölge renkleri
REGION_COLORS = {
    "MARMARA": "#0EA5E9",              # Sky Blue - Deniz ve boğazlar
    "BATI ANADOLU": "#14B8A6",         # Turkuaz-yeşil arası
    "EGE": "#FCD34D",                  # BAL SARI
    "İÇ ANADOLU": "#F59E0B",           # Amber - Kuru bozkır
    "GÜNEY DOĞU ANADOLU": "#E07A5F",   # Terracotta 
    "KUZEY ANADOLU": "#059669",        # Emerald - Yemyeşil ormanlar
    "KARADENİZ": "#059669",            # Emerald
    "AKDENİZ": "#8B5CF6",              # Violet - Akdeniz
    "DOĞU ANADOLU": "#7C3AED",         # Purple - Yüksek dağlar
    "DİĞER": "#64748B"                 # Slate Gray
}

# PERFORMANS RENKLERİ - Sade
PERFORMANCE_COLORS = {
    "high": "#1F7A5A",       # Koyu Yeşil – Yüksek Performans
    "medium": "#C48A2A",     # Altın Sarısı – Orta Performans
    "low": "#B23A3A",        # Bordo – Düşük Performans
    "positive": "#1F7A5A",   # Koyu Yeşil – Pozitif
    "negative": "#B23A3A",   # Bordo – Negatif
    "neutral": "#6B7280",    # Kurumsal Gri – Nötr
    "warning": "#C48A2A",    # Altın – Uyarı
    "info": "#1E40AF",       # Lacivert – Bilgi
    "success": "#166534",    # Koyu Yeşil – Başarı
    "danger": "#991B1B"      # Koyu Kırmızı – Risk / Tehlike
}

# BCG MATRIX RENKLERİ
BCG_COLORS = {
    "⭐ Star": "#F59E0B",      # Turuncu
    "🐄 Cash Cow": "#10B981",  # Yeşil
    "❓ Question Mark": "#3B82F6",  # Mavi
    "🐶 Dog": "#64748B"        # Gri
}

# YATIRIM STRATEJİSİ RENKLERİ
STRATEGY_COLORS = {
    "🚀 Agresif": "#EF4444",      # Kırmızı
    "⚡ Hızlandırılmış": "#F59E0B",  # Turuncu
    "🛡️ Koruma": "#10B981",        # Yeşil
    "💎 Potansiyel": "#3B82F6",     # Mavi
    "👁️ İzleme": "#64748B"         # Gri
}

# GRADIENT SCALES for Visualizations
GRADIENT_SCALES = {
    "blue_green": ["#3B82F6", "#06B6D4", "#10B981"],
    "sequential_blue": ["#DBEAFE", "#BFDBFE", "#93C5FD", "#60A5FA", "#3B82F6"],
    "diverging": ["#EF4444", "#F59E0B", "#10B981", "#3B82F6", "#8B5CF6"],
    "temperature": ["#3B82F6", "#60A5FA", "#93C5FD", "#BFDBFE", "#DBEAFE"]
}

# =============================================================================
# CONSTANTS
# =============================================================================

FIX_CITY_MAP = {
    "AGRI": "AĞRI",
    "BARTÄ±N": "BARTIN",
    "BINGÃ¶L": "BİNGÖL",
    "DÃ1⁄4ZCE": "DÜZCE",
    "ELAZIG": "ELAZIĞ",
    "ESKISEHIR": "ESKİŞEHİR",
    "GÃ1⁄4MÃ1⁄4SHANE": "GÜMÜŞHANE",
    "HAKKARI": "HAKKARİ",
    "ISTANBUL": "İSTANBUL",
    "IZMIR": "İZMİR",
    "IÄ\x9fDIR": "IĞDIR",
    "KARABÃ1⁄4K": "KARABÜK",
    "KINKKALE": "KIRIKKALE",
    "KIRSEHIR": "KIRŞEHİR",
    "KÃ1⁄4TAHYA": "KÜTAHYA",
    "MUGLA": "MUĞLA",
    "MUS": "MUŞ",
    "NEVSEHIR": "NEVŞEHİR",
    "NIGDE": "NİĞDE",
    "SANLIURFA": "ŞANLIURFA",
    "SIRNAK": "ŞIRNAK",
    "TEKIRDAG": "TEKİRDAĞ",
    "USAK": "UŞAK",
    "ZINGULDAK": "ZONGULDAK",
    "Ã\x87ANAKKALE": "ÇANAKKALE",
    "Ã\x87ANKIRI": "ÇANKIRI",
    "Ã\x87ORUM": "ÇORUM",
    "K. MARAS": "KAHRAMANMARAŞ",
    "CORUM": "ÇORUM",
    "CANKIRI": "ÇANKIRI",
    "ZONGULDAK": "ZONGULDak",
    "KARABUK": "KARABÜK",
    "GUMUSHANE": "GÜMÜŞHANE",
    "ELÂZıĞ": "ELAZIĞ",
    "KUTAHYA": "KÜTAHYA",
    "CANAKKALE": "ÇANAKKALE"
}

CITY_NORMALIZE_CLEAN = {
    'ADANA': 'Adana',
    'ADIYAMAN': 'Adiyaman',
    'AFYONKARAHISAR': 'Afyonkarahisar',
    'AFYON': 'Afyonkarahisar',
    'AGRI': 'Agri',
    'AĞRI': 'Agri',
    'ANKARA': 'Ankara',
    'ANTALYA': 'Antalya',
    'AYDIN': 'Aydin',
    'BALIKESIR': 'Balikesir',
    'BARTIN': 'Bartin',
    'BATMAN': 'Batman',
    'BILECIK': 'Bilecik',
    'BINGOL': 'Bingol',
    'BITLIS': 'Bitlis',
    'BOLU': 'Bolu',
    'BURDUR': 'Burdur',
    'BURSA': 'Bursa',
    'CANAKKALE': 'Canakkale',
    'ÇANAKKALE': 'Canakkale',
    'CANKIRI': 'Cankiri',
    'ÇANKIRI': 'Cankiri',
    'CORUM': 'Corum',
    'ÇORUM': 'Corum',
    'DENIZLI': 'Denizli',
    'DIYARBAKIR': 'Diyarbakir',
    'DUZCE': 'Duzce',
    'DÜZCE': 'Duzce',
    'EDIRNE': 'Edirne',
    'ELAZIG': 'Elazig',
    'ELAZĞ': 'Elazig',
    'ELAZIĞ': 'Elazig',
    'ERZINCAN': 'Erzincan',
    'ERZURUM': 'Erzurum',
    'ESKISEHIR': 'Eskisehir',
    'ESKİŞEHİR': 'Eskisehir',
    'GAZIANTEP': 'Gaziantep',
    'GIRESUN': 'Giresun',
    'GİRESUN': 'Giresun',
    'GUMUSHANE': 'Gumushane',
    'GÜMÜŞHANE': 'Gumushane',
    'HAKKARI': 'Hakkari',
    'HATAY': 'Hatay',
    'IGDIR': 'Igdir',
    'IĞDIR': 'Igdir',
    'ISPARTA': 'Isparta',
    'ISTANBUL': 'Istanbul',
    'İSTANBUL': 'Istanbul',
    'IZMIR': 'Izmir',
    'İZMİR': 'Izmir',
    'KAHRAMANMARAS': 'K. Maras',
    'KAHRAMANMARAŞ': 'K. Maras',
    'K.MARAS': 'K. Maras',
    'KMARAS': 'K. Maras',
    'KARABUK': 'Karabuk',
    'KARABÜK': 'Karabuk',
    'KARAMAN': 'Karaman',
    'KARS': 'Kars',
    'KASTAMONU': 'Kastamonu',
    'KAYSERI': 'Kayseri',
    'KIRIKKALE': 'Kinkkale',
    'KIRKLARELI': 'Kirklareli',
    'KIRKLARELİ': 'Kirklareli',
    'KIRSEHIR': 'Kirsehir',
    'KIRŞEHİR': 'Kirsehir',
    'KILIS': 'Kilis',
    'KİLİS': 'Kilis',
    'KOCAELI': 'Kocaeli',
    'KONYA': 'Konya',
    'KUTAHYA': 'Kutahya',
    'KÜTAHYA': 'Kutahya',
    'MALATYA': 'Malatya',
    'MANISA': 'Manisa',
    'MANİSA': 'Manisa',
    'MARDIN': 'Mardin',
    'MARDİN': 'Mardin',
    'MERSIN': 'Mersin',
    'MERSİN': 'Mersin',
    'MUGLA': 'Mugla',
    'MUĞLA': 'Mugla',
    'MUS': 'Mus',
    'MUŞ': 'Mus',
    'NEVSEHIR': 'Nevsehir',
    'NEVŞEHİR': 'Nevsehir',
    'NIGDE': 'Nigde',
    'NİĞDE': 'Nigde',
    'ORDU': 'Ordu',
    'OSMANIYE': 'Osmaniye',
    'OSMANİYE': 'Osmaniye',
    'RIZE': 'Rize',
    'RİZE': 'Rize',
    'SAKARYA': 'Sakarya',
    'SAMSUN': 'Samsun',
    'SIIRT': 'Siirt',
    'SİİRT': 'Siirt',
    'SINOP': 'Sinop',
    'SİNOP': 'Sinop',
    'SIVAS': 'Sivas',
    'SİVAS': 'Sivas',
    'SANLIURFA': 'Sanliurfa',
    'ŞANLIURFA': 'Sanliurfa',
    'SIRNAK': 'Sirnak',
    'ŞIRNAK': 'Sirnak',
    'TEKIRDAG': 'Tekirdag',
    'TEKİRDAĞ': 'Tekirdag',
    'TOKAT': 'Tokat',
    'TRABZON': 'Trabzon',
    'TUNCELI': 'Tunceli',
    'TUNCELİ': 'Tunceli',
    'USAK': 'Usak',
    'UŞAK': 'Usak',
    'VAN': 'Van',
    'YALOVA': 'Yalova',
    'YOZGAT': 'Yozgat',
    'ZONGULDAK': 'Zonguldak',
    'ARDAHAN': 'Ardahan'
     # EKSİK ŞEHİRLERİ EKLEYELİM
    'BARTÄ±N': 'Bartin',
    'KARABÃ1⁄4K': 'Karabuk',
    'TUNCELI': 'Tunceli',
    'OSMANIYE': 'Osmaniye',
    'KILIS': 'Kilis',
    'HAKKARI': 'Hakkari',
    'SIRNAK': 'Sirnak',
    'SIIRT': 'Siirt',
    'BATMAN': 'Batman',
    'BITLIS': 'Bitlis',
    'BINGOL': 'Bingol',
    'BINGÃ¶L': 'Bingol',
    'IGDIR': 'Igdir',
    'IÄ\x9fDIR': 'Igdir',
    'ARDAHAN': 'Ardahan',
    'KUTAHYA': 'Kutahya',
    'KÃ1⁄4TAHYA': 'Kutahya',
    'DUZCE': 'Duzce',
    'DÃ1⁄4ZCE': 'Duzce'
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def safe_divide(a, b):
    """Güvenli bölme işlemi"""
    return np.where(b != 0, a / b, 0)

def get_product_columns(product):
    """Ürün kolonlarını döndür"""
    if product == "TROCMETAM":
        return {"pf": "TROCMETAM", "rakip": "DIGER TROCMETAM"}
    elif product == "CORTIPOL":
        return {"pf": "CORTIPOL", "rakip": "DIGER CORTIPOL"}
    elif product == "DEKSAMETAZON":
        return {"pf": "DEKSAMETAZON", "rakip": "DIGER DEKSAMETAZON"}
    else:
        return {"pf": "PF IZOTONIK", "rakip": "DIGER IZOTONIK"}

def normalize_city_name_fixed(city_name):
    """Düzeltilmiş şehir normalizasyon"""
    if pd.isna(city_name):
        return None
    
    city_upper = str(city_name).strip().upper()
    
    # Fix known encoding issues
    if city_upper in FIX_CITY_MAP:
        return FIX_CITY_MAP[city_upper]
    
    # Turkish character mapping
    tr_map = {
        "İ": "I", "Ğ": "G", "Ü": "U",
        "Ş": "S", "Ö": "O", "Ç": "C",
        "Â": "A", "Î": "I", "Û": "U"
    }
    
    for k, v in tr_map.items():
        city_upper = city_upper.replace(k, v)
    
    return CITY_NORMALIZE_CLEAN.get(city_upper, city_name)

# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data
def load_excel_data(file):
    """Excel dosyasını yükle"""
    df = pd.read_excel(file)
    df['DATE'] = pd.to_datetime(df['DATE'])
    df['YIL_AY'] = df['DATE'].dt.strftime('%Y-%m')
    df['AY'] = df['DATE'].dt.month
    df['YIL'] = df['DATE'].dt.year
    
    df['TERRITORIES'] = df['TERRITORIES'].str.upper().str.strip()
    df['CITY'] = df['CITY'].str.strip()
    df['CITY_NORMALIZED'] = df['CITY'].apply(normalize_city_name_fixed)
    df['REGION'] = df['REGION'].str.upper().str.strip()
    df['MANAGER'] = df['MANAGER'].str.upper().str.strip()
    
    return df

@st.cache_resource
def load_geojson_gpd():
    """GeoPandas ile GeoJSON yükle"""
    try:
        gdf = gpd.read_file("turkey.geojson")
        return gdf
    except:
        try:
            gdf = gpd.read_file("turkey.geojson", encoding='utf-8')
            return gdf
        except Exception as e:
            st.error(f"❌ GeoJSON yüklenemedi: {e}")
            return None

@st.cache_resource
def load_geojson_json():
    """JSON formatında GeoJSON yükle"""
    try:
        with open('turkey.geojson', 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        try:
            with open('./turkey.geojson', 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"❌ JSON GeoJSON yüklenemedi: {e}")
            return None

# =============================================================================
# GEOMETRY HELPERS
# =============================================================================

def lines_to_lonlat(geom):
    """LineString veya MultiLineString'den koordinatları al"""
    lons, lats = [], []
    if isinstance(geom, LineString):
        xs, ys = geom.xy
        lons += list(xs) + [None]
        lats += list(ys) + [None]
    elif isinstance(geom, MultiLineString):
        for line in geom.geoms:
            xs, ys = line.xy
            lons += list(xs) + [None]
            lats += list(ys) + [None]
    return lons, lats

def get_region_center(gdf_region):
    """Bölgenin merkez koordinatlarını hesapla"""
    if len(gdf_region) == 0:
        return 35.0, 39.0
    centroid = gdf_region.geometry.unary_union.centroid
    return centroid.x, centroid.y

# =============================================================================
# MODERN HARİTA OLUŞTURUCU
# =============================================================================

def create_modern_turkey_map(city_data, gdf, title="Türkiye Satış Haritası", view_mode="Bölge Görünümü", filtered_pf_toplam=None):
    """
    Modern Türkiye haritası
    """
    if gdf is None:
        st.error("❌ GeoJSON yüklenemedi")
        return None
    
    # Veriyi hazırla
    city_data = city_data.copy()
    city_data['City_Fixed'] = city_data['City'].apply(normalize_city_name_fixed)
    city_data['City_Fixed'] = city_data['City_Fixed'].str.upper()
    
    # GeoJSON'daki isimleri normalize et
    gdf = gdf.copy()
    gdf['name_upper'] = gdf['name'].str.upper()
    gdf['name_fixed'] = gdf['name_upper'].replace(FIX_CITY_MAP)
    
    # Birleştir
    merged = gdf.merge(city_data, left_on='name_fixed', right_on='City_Fixed', how='left')
    
    # NaN'leri doldur
    merged['PF_Satis'] = merged['PF_Satis'].fillna(0)
    merged['Pazar_Payi_%'] = merged['Pazar_Payi_%'].fillna(0)
    merged['Bölge'] = merged['Bölge'].fillna('DİĞER')
    merged['Region'] = merged['Bölge']
    
    # Bölge renklerini ata
    merged['Region_Color'] = merged['Region'].map(REGION_COLORS).fillna('#64748B')
    
    # FİLTRELENMİŞ toplam
    if filtered_pf_toplam is None:
        filtered_pf_toplam = merged['PF_Satis'].sum()
    
    # Modern harita oluştur
    fig = go.Figure()
    
    # Her bölge için ayrı trace
    for region in merged['Region'].unique():
        region_data = merged[merged['Region'] == region]
        color = REGION_COLORS.get(region, "#64748B")
        
        # GeoJSON'u JSON'a çevir
        region_json = json.loads(region_data.to_json())
        
        fig.add_trace(go.Choroplethmapbox(
            geojson=region_json,
            locations=region_data.index,
            z=[1] * len(region_data),
            colorscale=[[0, color], [1, color]],
            marker_opacity=0.8,
            marker_line_width=2,
            marker_line_color='rgba(255, 255, 255, 0.8)',
            showscale=False,
            customdata=list(zip(
                region_data['name'],
                region_data['Region'],
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
            name=region,
            visible=True
        ))
    
    # Modern sınır çizgileri
    lons, lats = [], []
    for geom in merged.geometry.boundary:
        if geom and not geom.is_empty:
            lo, la = lines_to_lonlat(geom)
            lons += lo
            lats += la
    
    if lons and lats:
        fig.add_trace(go.Scattermapbox(
            lon=lons,
            lat=lats,
            mode='lines',
            line=dict(width=1.5, color='rgba(255, 255, 255, 0.9)'),
            hoverinfo='skip',
            showlegend=False
        ))
    
    # Modern etiketler
    if view_mode == "Bölge Görünümü":
        label_lons, label_lats, label_texts = [], [], []
        
        for region in merged['Region'].unique():
            region_data = merged[merged['Region'] == region]
            total = region_data['PF_Satis'].sum()
            
            if total > 0:
                percent = (total / filtered_pf_toplam * 100) if filtered_pf_toplam > 0 else 0
                
                lon, lat = get_region_center(region_data)
                label_lons.append(lon)
                label_lats.append(lat)
                label_texts.append(
                    f"<b>{region}</b><br>"
                    f"{total:,.0f}<br>"
                    f"({percent:.1f}%)"
                )
        
        fig.add_trace(go.Scattermapbox(
            lon=label_lons,
            lat=label_lats,
            mode='text',
            text=label_texts,
            textfont=dict(
                size=11, 
                color='white',
                family='Inter, sans-serif',
                weight='bold'
            ),
            hoverinfo='skip',
            showlegend=False
        ))
    
    else:
        city_lons, city_lats, city_texts = [], [], []
        
        for idx, row in merged.iterrows():
            if row['PF_Satis'] > 0:
                percent = (row['PF_Satis'] / filtered_pf_toplam * 100) if filtered_pf_toplam > 0 else 0
                centroid = row.geometry.centroid
                city_lons.append(centroid.x)
                city_lats.append(centroid.y)
                city_texts.append(
                    f"<b>{row['name']}</b><br>"
                    f"{row['PF_Satis']:,.0f}"
                )
        
        fig.add_trace(go.Scattermapbox(
            lon=city_lons,
            lat=city_lats,
            mode='text',
            text=city_texts,
            textfont=dict(
                size=9, 
                color='white',
                family='Inter, sans-serif'
            ),
            hoverinfo='skip',
            showlegend=False
        ))
    
    # Modern layout ayarları
    fig.update_layout(
        mapbox_style="carto-darkmatter",
        mapbox=dict(
            center=dict(lat=39.0, lon=35.0),
            zoom=5,
            bearing=0,
            pitch=0
        ),
        height=750,
        margin=dict(l=0, r=0, t=80, b=0),
        title=dict(
            text=f"<b>{title}</b>",
            x=0.5,
            font=dict(
                size=24, 
                color='white',
                family='Inter, sans-serif'
            ),
            y=0.95
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        hoverlabel=dict(
            bgcolor="rgba(15, 23, 41, 0.9)",
            font_size=12,
            font_family="Inter, sans-serif"
        )
    )
    
    return fig

# =============================================================================
# ML FEATURE ENGINEERING
# =============================================================================

def create_ml_features(df):
    """ML için feature oluştur"""
    df = df.copy()
    df = df.sort_values('DATE').reset_index(drop=True)
    
    # Lag features
    df['lag_1'] = df['PF_Satis'].shift(1)
    df['lag_2'] = df['PF_Satis'].shift(2)
    df['lag_3'] = df['PF_Satis'].shift(3)
    
    # Rolling features
    df['rolling_mean_3'] = df['PF_Satis'].rolling(window=3, min_periods=1).mean()
    df['rolling_mean_6'] = df['PF_Satis'].rolling(window=6, min_periods=1).mean()
    df['rolling_std_3'] = df['PF_Satis'].rolling(window=3, min_periods=1).std()
    
    # Date features
    df['month'] = df['DATE'].dt.month
    df['quarter'] = df['DATE'].dt.quarter
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['trend_index'] = range(len(df))
    
    # Fill NaN
    df = df.fillna(method='bfill').fillna(0)
    
    return df

def train_ml_models(df, forecast_periods=3):
    """GERÇEK ML modelleri ile tahmin"""
    df_features = create_ml_features(df)
    
    if len(df_features) < 10:
        return None, None, None
    
    feature_cols = ['lag_1', 'lag_2', 'lag_3', 'rolling_mean_3', 'rolling_mean_6',
                    'rolling_std_3', 'month', 'quarter', 'month_sin', 'month_cos', 'trend_index']
    
    # Train/Test split (zaman bazlı)
    split_idx = int(len(df_features) * 0.8)
    
    train_df = df_features.iloc[:split_idx]
    test_df = df_features.iloc[split_idx:]
    
    X_train = train_df[feature_cols]
    y_train = train_df['PF_Satis']
    X_test = test_df[feature_cols]
    y_test = test_df['PF_Satis']
    
    # Modeller
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
    }
    
    results = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        results[name] = {
            'model': model,
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape
        }
    
    # En iyi model (MAPE'e göre)
    best_model_name = min(results.keys(), key=lambda x: results[x]['MAPE'])
    best_model = results[best_model_name]['model']
    
    # Gelecek tahmin
    forecast_data = []
    last_row = df_features.iloc[-1:].copy()
    
    for i in range(forecast_periods):
        next_date = last_row['DATE'].values[0] + pd.DateOffset(months=1)
        X_future = last_row[feature_cols]
        next_pred = best_model.predict(X_future)[0]
        
        forecast_data.append({
            'DATE': next_date,
            'YIL_AY': pd.to_datetime(next_date).strftime('%Y-%m'),
            'PF_Satis': max(0, next_pred),
            'Model': best_model_name
        })
        
        # Güncelle
        new_row = last_row.copy()
        new_row['DATE'] = next_date
        new_row['PF_Satis'] = next_pred
        new_row['lag_1'] = last_row['PF_Satis'].values[0]
        new_row['lag_2'] = last_row['lag_1'].values[0]
        new_row['lag_3'] = last_row['lag_2'].values[0]
        new_row['rolling_mean_3'] = (new_row['lag_1'] + new_row['lag_2'] + new_row['lag_3']) / 3
        new_row['month'] = pd.to_datetime(next_date).month
        new_row['quarter'] = pd.to_datetime(next_date).quarter
        new_row['month_sin'] = np.sin(2 * np.pi * new_row['month'] / 12)
        new_row['month_cos'] = np.cos(2 * np.pi * new_row['month'] / 12)
        new_row['trend_index'] = last_row['trend_index'].values[0] + 1
        
        last_row = new_row
    
    forecast_df = pd.DataFrame(forecast_data)
    
    return results, best_model_name, forecast_df

# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def calculate_city_performance(df, product, date_filter=None):
    """Şehir bazlı performans"""
    cols = get_product_columns(product)
    
    if date_filter:
        df = df[(df['DATE'] >= date_filter[0]) & (df['DATE'] <= date_filter[1])]
    
    city_perf = df.groupby(['CITY_NORMALIZED', 'REGION']).agg({
        cols['pf']: 'sum',
        cols['rakip']: 'sum'
    }).reset_index()
    
    city_perf.columns = ['City', 'Region', 'PF_Satis', 'Rakip_Satis']
    city_perf['Toplam_Pazar'] = city_perf['PF_Satis'] + city_perf['Rakip_Satis']
    city_perf['Pazar_Payi_%'] = safe_divide(city_perf['PF_Satis'], city_perf['Toplam_Pazar']) * 100
    
    # Bölge isimlerini düzelt
    city_perf['Bölge'] = city_perf['Region']
    
    return city_perf

def calculate_territory_performance(df, product, date_filter=None):
    """Territory bazlı performans"""
    cols = get_product_columns(product)
    
    if date_filter:
        df = df[(df['DATE'] >= date_filter[0]) & (df['DATE'] <= date_filter[1])]
    
    terr_perf = df.groupby(['TERRITORIES', 'REGION', 'CITY', 'MANAGER']).agg({
        cols['pf']: 'sum',
        cols['rakip']: 'sum'
    }).reset_index()
    
    terr_perf.columns = ['Territory', 'Region', 'City', 'Manager', 'PF_Satis', 'Rakip_Satis']
    terr_perf['Toplam_Pazar'] = terr_perf['PF_Satis'] + terr_perf['Rakip_Satis']
    terr_perf['Pazar_Payi_%'] = safe_divide(terr_perf['PF_Satis'], terr_perf['Toplam_Pazar']) * 100
    
    total_pf = terr_perf['PF_Satis'].sum()
    terr_perf['Agirlik_%'] = safe_divide(terr_perf['PF_Satis'], total_pf) * 100
    terr_perf['Goreceli_Pazar_Payi'] = safe_divide(terr_perf['PF_Satis'], terr_perf['Rakip_Satis'])
    
    return terr_perf.sort_values('PF_Satis', ascending=False)

def calculate_time_series(df, product, territory=None, date_filter=None):
    """Zaman serisi"""
    cols = get_product_columns(product)
    
    df_filtered = df.copy()
    if territory and territory != "TÜMÜ":
        df_filtered = df_filtered[df_filtered['TERRITORIES'] == territory]
    
    if date_filter:
        df_filtered = df_filtered[(df_filtered['DATE'] >= date_filter[0]) & 
                                   (df_filtered['DATE'] <= date_filter[1])]
    
    monthly = df_filtered.groupby('YIL_AY').agg({
        cols['pf']: 'sum',
        cols['rakip']: 'sum',
        'DATE': 'first'
    }).reset_index().sort_values('YIL_AY')
    
    monthly.columns = ['YIL_AY', 'PF_Satis', 'Rakip_Satis', 'DATE']
    monthly['Toplam_Pazar'] = monthly['PF_Satis'] + monthly['Rakip_Satis']
    monthly['Pazar_Payi_%'] = safe_divide(monthly['PF_Satis'], monthly['Toplam_Pazar']) * 100
    monthly['PF_Buyume_%'] = monthly['PF_Satis'].pct_change() * 100
    monthly['Rakip_Buyume_%'] = monthly['Rakip_Satis'].pct_change() * 100
    monthly['Goreceli_Buyume_%'] = monthly['PF_Buyume_%'] - monthly['Rakip_Buyume_%']
    monthly['MA_3'] = monthly['PF_Satis'].rolling(window=3, min_periods=1).mean()
    monthly['MA_6'] = monthly['PF_Satis'].rolling(window=6, min_periods=1).mean()
    
    return monthly

def calculate_competitor_analysis(df, product, date_filter=None):
    """Rakip analizi"""
    cols = get_product_columns(product)
    
    if date_filter:
        df = df[(df['DATE'] >= date_filter[0]) & (df['DATE'] <= date_filter[1])]
    
    monthly = df.groupby('YIL_AY').agg({
        cols['pf']: 'sum',
        cols['rakip']: 'sum'
    }).reset_index().sort_values('YIL_AY')
    
    monthly.columns = ['YIL_AY', 'PF', 'Rakip']
    monthly['PF_Pay_%'] = (monthly['PF'] / (monthly['PF'] + monthly['Rakip'])) * 100
    monthly['Rakip_Pay_%'] = 100 - monthly['PF_Pay_%']
    monthly['PF_Buyume'] = monthly['PF'].pct_change() * 100
    monthly['Rakip_Buyume'] = monthly['Rakip'].pct_change() * 100
    monthly['Fark'] = monthly['PF_Buyume'] - monthly['Rakip_Buyume']
    
    return monthly

def calculate_bcg_matrix(df, product, date_filter=None):
    """BCG Matrix"""
    cols = get_product_columns(product)
    
    if date_filter:
        df_filtered = df[(df['DATE'] >= date_filter[0]) & (df['DATE'] <= date_filter[1])]
    else:
        df_filtered = df.copy()
    
    terr_perf = calculate_territory_performance(df_filtered, product)
    
    df_sorted = df_filtered.sort_values('DATE')
    mid_point = len(df_sorted) // 2
    
    first_half = df_sorted.iloc[:mid_point].groupby('TERRITORIES')[cols['pf']].sum()
    second_half = df_sorted.iloc[mid_point:].groupby('TERRITORIES')[cols['pf']].sum()
    
    growth_rate = {}
    for terr in first_half.index:
        if terr in second_half.index and first_half[terr] > 0:
            growth_rate[terr] = ((second_half[terr] - first_half[terr]) / first_half[terr]) * 100
        else:
            growth_rate[terr] = 0
    
    terr_perf['Pazar_Buyume_%'] = terr_perf['Territory'].map(growth_rate).fillna(0)
    
    median_share = terr_perf['Goreceli_Pazar_Payi'].median()
    median_growth = terr_perf['Pazar_Buyume_%'].median()
    
    def assign_bcg(row):
        if row['Goreceli_Pazar_Payi'] >= median_share and row['Pazar_Buyume_%'] >= median_growth:
            return "⭐ Star"
        elif row['Goreceli_Pazar_Payi'] >= median_share and row['Pazar_Buyume_%'] < median_growth:
            return "🐄 Cash Cow"
        elif row['Goreceli_Pazar_Payi'] < median_share and row['Pazar_Buyume_%'] >= median_growth:
            return "❓ Question Mark"
        else:
            return "🐶 Dog"
    
    terr_perf['BCG_Kategori'] = terr_perf.apply(assign_bcg, axis=1)
    
    return terr_perf

# =============================================================================
# YATIRIM STRATEJİSİ - GELİŞTİRİLMİŞ ALGORİTMA
# =============================================================================

def calculate_investment_strategy(city_perf):
    """
    Geliştirilmiş Yatırım Stratejisi Algoritması
    """
    df = city_perf.copy()
    df = df[df['PF_Satis'] > 0]
    
    if len(df) == 0:
        return df
    
    # 1. PAZAR BÜYÜKLÜĞÜ SEGMENTİ
    try:
        df["Pazar_Büyüklüğü"] = pd.qcut(
            df["Toplam_Pazar"], 
            q=3, 
            labels=["Küçük", "Orta", "Büyük"],
            duplicates='drop'
        )
    except:
        df["Pazar_Büyüklüğü"] = "Orta"
    
    # 2. PERFORMANS SEGMENTİ
    try:
        df["Performans"] = pd.qcut(
            df["PF_Satis"], 
            q=3, 
            labels=["Düşük", "Orta", "Yüksek"],
            duplicates='drop'
        )
    except:
        df["Performans"] = "Orta"
    
    # 3. PAZAR PAYI SEGMENTİ
    try:
        df["Pazar_Payı_Segment"] = pd.qcut(
            df["Pazar_Payi_%"], 
            q=3, 
            labels=["Düşük", "Orta", "Yüksek"],
            duplicates='drop'
        )
    except:
        df["Pazar_Payı_Segment"] = "Orta"
    
    # 4. BÜYÜME POTANSİYELİ
    df["Büyüme_Alanı"] = df["Toplam_Pazar"] - df["PF_Satis"]
    try:
        df["Büyüme_Potansiyeli"] = pd.qcut(
            df["Büyüme_Alanı"],
            q=3,
            labels=["Düşük", "Orta", "Yüksek"],
            duplicates='drop'
        )
    except:
        df["Büyüme_Potansiyeli"] = "Orta"
    
    # 5. STRATEJİ ATAMA
    def assign_strategy(row):
        pazar_buyuklugu = str(row["Pazar_Büyüklüğü"])
        pazar_payi = str(row["Pazar_Payı_Segment"])
        buyume_potansiyeli = str(row["Büyüme_Potansiyeli"])
        performans = str(row["Performans"])
        
        if (pazar_buyuklugu in ["Büyük", "Orta"] and 
            pazar_payi == "Düşük" and 
            buyume_potansiyeli in ["Yüksek", "Orta"]):
            return "🚀 Agresif"
        
        elif (pazar_buyuklugu in ["Büyük", "Orta"] and 
              pazar_payi == "Orta" and
              performans in ["Orta", "Yüksek"]):
            return "⚡ Hızlandırılmış"
        
        elif (pazar_buyuklugu == "Büyük" and 
              pazar_payi == "Yüksek"):
            return "🛡️ Koruma"
        
        elif (pazar_buyuklugu == "Küçük" and 
              buyume_potansiyeli == "Yüksek" and
              performans in ["Orta", "Yüksek"]):
            return "💎 Potansiyel"
        
        else:
            return "👁️ İzleme"
    
    df["Yatırım_Stratejisi"] = df.apply(assign_strategy, axis=1)
    
    return df

# =============================================================================
# VISUALIZATION FUNCTIONS - MODERN
# =============================================================================

def create_modern_forecast_chart(historical_df, forecast_df):
    """Modern tahmin grafiği"""
    fig = go.Figure()
    
    # Gerçek veri
    fig.add_trace(go.Scatter(
        x=historical_df['DATE'],
        y=historical_df['PF_Satis'],
        mode='lines+markers',
        name='Gerçek Satış',
        line=dict(
            color=PERFORMANCE_COLORS['success'],
            width=3,
            shape='spline'
        ),
        marker=dict(
            size=8,
            color='white',
            line=dict(width=2, color=PERFORMANCE_COLORS['success'])
        ),
        fill='tozeroy',
        fillcolor='rgba(16, 185, 129, 0.1)'
    ))
    
    # Tahmin
    if forecast_df is not None and len(forecast_df) > 0:
        fig.add_trace(go.Scatter(
            x=forecast_df['DATE'],
            y=forecast_df['PF_Satis'],
            mode='lines+markers',
            name='Tahmin',
            line=dict(
                color=PERFORMANCE_COLORS['info'],
                width=3,
                dash='dash',
                shape='spline'
            ),
            marker=dict(
                size=10,
                symbol='diamond',
                color='white',
                line=dict(width=2, color=PERFORMANCE_COLORS['info'])
            ),
            fill='tozeroy',
            fillcolor='rgba(59, 130, 246, 0.1)'
        ))
    
    # Modern layout
    fig.update_layout(
        title=dict(
            text='<b>Satış Trendi ve ML Tahmin</b>',
            font=dict(size=20, color='white', family='Inter')
        ),
        xaxis_title='<b>Tarih</b>',
        yaxis_title='<b>PF Satış</b>',
        height=500,
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e2e8f0', family='Inter'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(30, 41, 59, 0.8)',
            bordercolor='rgba(59, 130, 246, 0.3)',
            borderwidth=1
        ),
        xaxis=dict(
            gridcolor='rgba(59, 130, 246, 0.1)',
            linecolor='rgba(59, 130, 246, 0.3)',
            showgrid=True
        ),
        yaxis=dict(
            gridcolor='rgba(59, 130, 246, 0.1)',
            linecolor='rgba(59, 130, 246, 0.3)',
            showgrid=True
        )
    )
    
    return fig

def create_modern_competitor_chart(comp_data):
    """Modern rakip karşılaştırma"""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=comp_data['YIL_AY'],
        y=comp_data['PF'],
        name='PF',
        marker_color=PERFORMANCE_COLORS['success'],
        marker=dict(
            line=dict(width=2, color='rgba(255, 255, 255, 0.8)')
        )
    ))
    
    fig.add_trace(go.Bar(
        x=comp_data['YIL_AY'],
        y=comp_data['Rakip'],
        name='Rakip',
        marker_color=PERFORMANCE_COLORS['danger'],
        marker=dict(
            line=dict(width=2, color='rgba(255, 255, 255, 0.8)')
        )
    ))
    
    fig.update_layout(
        title=dict(
            text='<b>PF vs Rakip Satış Karşılaştırması</b>',
            font=dict(size=20, color='white', family='Inter')
        ),
        xaxis_title='<b>Ay</b>',
        yaxis_title='<b>Satış</b>',
        barmode='group',
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e2e8f0', family='Inter'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(30, 41, 59, 0.8)'
        ),
        xaxis=dict(
            gridcolor='rgba(59, 130, 246, 0.1)'
        ),
        yaxis=dict(
            gridcolor='rgba(59, 130, 246, 0.1)'
        )
    )
    
    return fig

def create_modern_growth_chart(comp_data):
    """Modern büyüme grafiği"""
    fig = go.Figure()
    
    # PF Büyüme
    fig.add_trace(go.Scatter(
        x=comp_data['YIL_AY'],
        y=comp_data['PF_Buyume'],
        mode='lines+markers',
        name='PF Büyüme',
        line=dict(
            color=PERFORMANCE_COLORS['success'],
            width=3,
            shape='spline'
        ),
        marker=dict(
            size=8,
            color='white',
            line=dict(width=2, color=PERFORMANCE_COLORS['success'])
        ),
        fill='tozeroy',
        fillcolor='rgba(16, 185, 129, 0.15)'
    ))
    
    # Rakip Büyüme
    fig.add_trace(go.Scatter(
        x=comp_data['YIL_AY'],
        y=comp_data['Rakip_Buyume'],
        mode='lines+markers',
        name='Rakip Büyüme',
        line=dict(
            color=PERFORMANCE_COLORS['danger'],
            width=3,
            shape='spline'
        ),
        marker=dict(
            size=8,
            color='white',
            line=dict(width=2, color=PERFORMANCE_COLORS['danger'])
        ),
        fill='tozeroy',
        fillcolor='rgba(239, 68, 68, 0.15)'
    ))
    
    fig.add_hline(
        y=0, 
        line_dash="dash", 
        line_color=PERFORMANCE_COLORS['neutral'], 
        opacity=0.5,
        line_width=2
    )
    
    fig.update_layout(
        title=dict(
            text='<b>Büyüme Oranları Karşılaştırması</b>',
            font=dict(size=20, color='white', family='Inter')
        ),
        xaxis_title='<b>Ay</b>',
        yaxis_title='<b>Büyüme (%)</b>',
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e2e8f0', family='Inter'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(30, 41, 59, 0.8)'
        ),
        xaxis=dict(
            gridcolor='rgba(59, 130, 246, 0.1)'
        ),
        yaxis=dict(
            gridcolor='rgba(59, 130, 246, 0.1)'
        )
    )
    
    return fig

def create_modern_bcg_chart(bcg_df):
    """Modern BCG Matrix"""
    fig = px.scatter(
        bcg_df,
        x='Goreceli_Pazar_Payi',
        y='Pazar_Buyume_%',
        size='PF_Satis',
        color='BCG_Kategori',
        color_discrete_map=BCG_COLORS,
        hover_name='Territory',
        hover_data={
            'PF_Satis': ':,.0f',
            'Pazar_Payi_%': ':.1f',
            'Goreceli_Pazar_Payi': ':.2f',
            'Pazar_Buyume_%': ':.1f'
        },
        labels={
            'Goreceli_Pazar_Payi': '<b>Göreceli Pazar Payı</b>',
            'Pazar_Buyume_%': '<b>Pazar Büyüme Oranı (%)</b>'
        },
        size_max=60
    )
    
    median_share = bcg_df['Goreceli_Pazar_Payi'].median()
    median_growth = bcg_df['Pazar_Buyume_%'].median()
    
    fig.add_hline(
        y=median_growth, 
        line_dash="dash", 
        line_color=PERFORMANCE_COLORS['neutral'], 
        opacity=0.5,
        line_width=2
    )
    fig.add_vline(
        x=median_share, 
        line_dash="dash", 
        line_color=PERFORMANCE_COLORS['neutral'], 
        opacity=0.5,
        line_width=2
    )
    
    fig.update_layout(
        title=dict(
            text='<b>BCG Matrix - Stratejik Konumlandırma</b>',
            font=dict(size=22, color='white', family='Inter')
        ),
        height=650,
        plot_bgcolor='rgba(15, 23, 41, 0.9)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e2e8f0', family='Inter'),
        legend=dict(
            title='<b>BCG Kategorisi</b>',
            bgcolor='rgba(30, 41, 59, 0.8)',
            bordercolor='rgba(59, 130, 246, 0.3)',
            borderwidth=1
        ),
        xaxis=dict(
            gridcolor='rgba(59, 130, 246, 0.1)',
            linecolor='rgba(59, 130, 246, 0.3)'
        ),
        yaxis=dict(
            gridcolor='rgba(59, 130, 246, 0.1)',
            linecolor='rgba(59, 130, 246, 0.3)'
        )
    )
    
    return fig

# =============================================================================
# MODERN DATA TABLE STYLING
# =============================================================================

def style_dataframe(df, color_column=None, gradient_columns=None):
    """Modern dataframe stilini uygula"""
    if gradient_columns is None:
        gradient_columns = []
    
    styled_df = df.style
    
    # Genel stil
    styled_df = styled_df.set_properties(**{
        'background-color': 'rgba(30, 41, 59, 0.7)',
        'color': '#e2e8f0',
        'border': '1px solid rgba(59, 130, 246, 0.2)',
        'font-family': 'Inter, sans-serif'
    })
    
    # Başlık satırı
    styled_df = styled_df.set_table_styles([{
        'selector': 'thead th',
        'props': [
            ('background-color', 'rgba(59, 130, 246, 0.3)'),
            ('color', 'white'),
            ('font-weight', '700'),
            ('border', '1px solid rgba(59, 130, 246, 0.4)'),
            ('padding', '12px 8px'),
            ('text-align', 'center')
        ]
    }])
    
    # Hücreler
    styled_df = styled_df.set_table_styles([{
        'selector': 'td',
        'props': [
            ('padding', '10px 8px'),
            ('text-align', 'center')
        ]
    }])
    
    # Gradient columns
    for col in gradient_columns:
        if col in df.columns:
            styled_df = styled_df.background_gradient(
                subset=[col], 
                cmap='RdYlGn',
                vmin=df[col].min() if len(df) > 0 else 0,
                vmax=df[col].max() if len(df) > 0 else 100
            )
    
    # Renk sütunu
    if color_column and color_column in df.columns:
        def color_cells(val):
            if isinstance(val, (int, float)):
                if val >= 70:
                    return 'background-color: rgba(16, 185, 129, 0.3); color: #10B981; font-weight: 600'
                elif val >= 40:
                    return 'background-color: rgba(245, 158, 11, 0.3); color: #F59E0B; font-weight: 600'
                else:
                    return 'background-color: rgba(239, 68, 68, 0.3); color: #EF4444; font-weight: 600'
            return ''
        
        styled_df = styled_df.applymap(color_cells, subset=[color_column])
    
    # Alternatif satır renkleri
    styled_df = styled_df.set_table_styles([{
        'selector': 'tbody tr:nth-child(even)',
        'props': [('background-color', 'rgba(30, 41, 59, 0.5)')]
    }, {
        'selector': 'tbody tr:nth-child(odd)',
        'props': [('background-color', 'rgba(30, 41, 59, 0.3)')]
    }])
    
    return styled_df

# =============================================================================
# MAIN APP
# =============================================================================

def main():
    # Başlık ve açıklama
    st.markdown('<h1 class="main-header">🎯 GELİŞMİŞ TİCARİ PORTFÖY ANALİZ SİSTEMİ</h1>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: center; font-size: 1.2rem; color: #94a3b8; margin-bottom: 3rem;">'
                'GERÇEK ML Tahminleme • Modern Harita Görselleştirme • Rakip Analizi • BCG Matrix'
                '</div>', unsafe_allow_html=True)
    
    # SIDEBAR
    with st.sidebar:
        st.markdown('<div style="background: linear-gradient(135deg, #3B82F6 0%, #10B981 100%); '
                   'padding: 1rem; border-radius: 12px; margin-bottom: 2rem;">'
                   '<h3 style="color: white; margin: 0; text-align: center;">📂 VERİ YÜKLEME</h3>'
                   '</div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Excel Dosyası Yükleyin", type=['xlsx', 'xls'])
        
        if not uploaded_file:
            st.info("👈 Lütfen sol taraftan Excel dosyasını yükleyin")
            st.stop()
        
        try:
            df = load_excel_data(uploaded_file)
            gdf = load_geojson_gpd()
            geojson = load_geojson_json()
            st.success(f"✅ **{len(df):,}** satır veri yüklendi")
        except Exception as e:
            st.error(f"❌ Veri yükleme hatası: {str(e)}")
            st.stop()
        
        st.markdown("---")
        
        # Ürün Seçimi
        st.markdown('<div style="background: rgba(30, 41, 59, 0.7); padding: 1rem; border-radius: 10px; margin: 1rem 0;">'
                   '<h4 style="color: #e2e8f0; margin: 0 0 1rem 0;">💊 ÜRÜN SEÇİMİ</h4>', unsafe_allow_html=True)
        selected_product = st.selectbox("", ["TROCMETAM", "CORTIPOL", "DEKSAMETAZON", "PF IZOTONIK"], label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Tarih Aralığı
        st.markdown('<div style="background: rgba(30, 41, 59, 0.7); padding: 1rem; border-radius: 10px; margin: 1rem 0;">'
                   '<h4 style="color: #e2e8f0; margin: 0 0 1rem 0;">📅 TARİH ARALIĞI</h4>', unsafe_allow_html=True)
        
        min_date = df['DATE'].min()
        max_date = df['DATE'].max()
        
        date_option = st.selectbox("Dönem Seçin", ["Tüm Veriler", "Son 3 Ay", "Son 6 Ay", "Son 1 Yıl", "2025", "2024", "Özel Aralık"])
        
        if date_option == "Tüm Veriler":
            date_filter = None
        elif date_option == "Son 3 Ay":
            start_date = max_date - pd.DateOffset(months=3)
            date_filter = (start_date, max_date)
        elif date_option == "Son 6 Ay":
            start_date = max_date - pd.DateOffset(months=6)
            date_filter = (start_date, max_date)
        elif date_option == "Son 1 Yıl":
            start_date = max_date - pd.DateOffset(years=1)
            date_filter = (start_date, max_date)
        elif date_option == "2025":
            date_filter = (pd.to_datetime('2025-01-01'), pd.to_datetime('2025-12-31'))
        elif date_option == "2024":
            date_filter = (pd.to_datetime('2024-01-01'), pd.to_datetime('2024-12-31'))
        else:
            col_date1, col_date2 = st.columns(2)
            with col_date1:
                start_date = st.date_input("Başlangıç", min_date, min_value=min_date, max_value=max_date)
            with col_date2:
                end_date = st.date_input("Bitiş", max_date, min_value=min_date, max_value=max_date)
            date_filter = (pd.to_datetime(start_date), pd.to_datetime(end_date))
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Filtreler
        st.markdown('<div style="background: rgba(30, 41, 59, 0.7); padding: 1rem; border-radius: 10px; margin: 1rem 0;">'
                   '<h4 style="color: #e2e8f0; margin: 0 0 1rem 0;">🔍 FİLTRELER</h4>', unsafe_allow_html=True)
        
        territories = ["TÜMÜ"] + sorted(df['TERRITORIES'].unique())
        selected_territory = st.selectbox("Territory", territories)
        
        regions = ["TÜMÜ"] + sorted(df['REGION'].unique())
        selected_region = st.selectbox("Bölge", regions)
        
        managers = ["TÜMÜ"] + sorted(df['MANAGER'].unique())
        selected_manager = st.selectbox("Manager", managers)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Veri filtreleme
        df_filtered = df.copy()
        if selected_territory != "TÜMÜ":
            df_filtered = df_filtered[df_filtered['TERRITORIES'] == selected_territory]
        if selected_region != "TÜMÜ":
            df_filtered = df_filtered[df_filtered['REGION'] == selected_region]
        if selected_manager != "TÜMÜ":
            df_filtered = df_filtered[df_filtered['MANAGER'] == selected_manager]
        
        st.markdown("---")
        
        # Harita Ayarları
        st.markdown('<div style="background: rgba(30, 41, 59, 0.7); padding: 1rem; border-radius: 10px; margin: 1rem 0;">'
                   '<h4 style="color: #e2e8f0; margin: 0 0 1rem 0;">🗺️ HARİTA AYARLARI</h4>', unsafe_allow_html=True)
        
        view_mode = st.radio(
            "Görünüm Modu",
            ["Bölge Görünümü", "Şehir Görünümü"],
            index=0
        )
        
        # Yatırım stratejisi filtresi
        strateji_list = ["Tümü", "🚀 Agresif", "⚡ Hızlandırılmış", "🛡️ Koruma", "💎 Potansiyel", "👁️ İzleme"]
        selected_strateji = st.selectbox("Yatırım Stratejisi", strateji_list)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Renk Legend
        st.markdown("---")
        st.markdown('<h4 style="color: #e2e8f0;">🎨 BÖLGE RENKLERİ</h4>', unsafe_allow_html=True)
        for region, color in list(REGION_COLORS.items())[:5]:
            st.markdown(f'<div style="display: flex; align-items: center; margin: 0.3rem 0;">'
                       f'<div style="width: 12px; height: 12px; background-color: {color}; border-radius: 2px; margin-right: 8px;"></div>'
                       f'<span style="color: #cbd5e1; font-size: 0.9rem;">{region}</span>'
                       f'</div>', unsafe_allow_html=True)
    
    # ANA İÇERİK - TAB'LER
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Genel Bakış",
        "🗺️ Modern Harita",
        "🏢 Territory Analizi",
        "📈 Zaman Serisi & ML",
        "🎯 Rakip Analizi",
        "⭐ BCG & Strateji",
        "📥 Raporlar"
    ])
    
    # TAB 1: GENEL BAKIŞ
    with tab1:
        st.header("📊 Genel Performans Özeti")
        
        cols = get_product_columns(selected_product)
        
        if date_filter:
            df_period = df_filtered[(df_filtered['DATE'] >= date_filter[0]) & (df_filtered['DATE'] <= date_filter[1])]
        else:
            df_period = df_filtered
        
        # Metrikler
        total_pf = df_period[cols['pf']].sum()
        total_rakip = df_period[cols['rakip']].sum()
        total_market = total_pf + total_rakip
        market_share = (total_pf / total_market * 100) if total_market > 0 else 0
        active_territories = df_period['TERRITORIES'].nunique()
        avg_monthly_pf = total_pf / df_period['YIL_AY'].nunique() if df_period['YIL_AY'].nunique() > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("💊 PF Satış", f"{total_pf:,.0f}", f"{avg_monthly_pf:,.0f}/ay")
        with col2:
            st.metric("🏪 Toplam Pazar", f"{total_market:,.0f}", f"{total_rakip:,.0f} rakip")
        with col3:
            st.metric("📊 Pazar Payı", f"%{market_share:.1f}", 
                     f"%{100-market_share:.1f} rakip")
        with col4:
            st.metric("🏢 Active Territory", active_territories, 
                     f"{df_period['MANAGER'].nunique()} manager")
        
        st.markdown("---")
        
        # Top 10 Territory
        st.subheader("🏆 Top 10 Territory Performansı")
        terr_perf = calculate_territory_performance(df_filtered, selected_product, date_filter)
        top10 = terr_perf.head(10)
        
        # Toplam Pazar % ekle
        total_market_all = terr_perf['Toplam_Pazar'].sum()
        top10['Toplam_Pazar_%'] = safe_divide(top10['Toplam_Pazar'], total_market_all) * 100
        
        col_chart1, col_chart2 = st.columns([2, 1])
        
        with col_chart1:
            fig_top10 = go.Figure()
            
            fig_top10.add_trace(go.Bar(
                x=top10['Territory'],
                y=top10['PF_Satis'],
                name='PF Satış',
                marker_color=PERFORMANCE_COLORS['success'],
                text=top10['PF_Satis'].apply(lambda x: f'{x:,.0f}'),
                textposition='outside',
                marker=dict(
                    line=dict(width=2, color='rgba(255, 255, 255, 0.8)')
                )
            ))
            
            fig_top10.add_trace(go.Bar(
                x=top10['Territory'],
                y=top10['Rakip_Satis'],
                name='Rakip Satış',
                marker_color=PERFORMANCE_COLORS['danger'],
                text=top10['Rakip_Satis'].apply(lambda x: f'{x:,.0f}'),
                textposition='outside',
                marker=dict(
                    line=dict(width=2, color='rgba(255, 255, 255, 0.8)')
                )
            ))
            
            fig_top10.update_layout(
                title=dict(
                    text='<b>Top 10 Territory - PF vs Rakip</b>',
                    font=dict(size=18, color='white')
                ),
                xaxis_title='<b>Territory</b>',
                yaxis_title='<b>Satış</b>',
                barmode='group',
                height=500,
                xaxis=dict(tickangle=-45),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e2e8f0'),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig_top10, use_container_width=True)
        
        with col_chart2:
            # Top 5 Territory için pasta grafiği
            top5 = top10.head(5)
            fig_pie = px.pie(
                top5,
                values='PF_Satis',
                names='Territory',
                title='<b>Top 5 Territory Dağılımı</b>',
                color_discrete_sequence=GRADIENT_SCALES['blue_green'],
                hole=0.4
            )
            
            fig_pie.update_layout(
                height=500,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e2e8f0'),
                legend=dict(
                    orientation="v",
                    yanchor="middle",
                    y=0.5,
                    xanchor="right",
                    x=1.3
                )
            )
            
            fig_pie.update_traces(
                textposition='inside',
                textinfo='percent+label',
                marker=dict(line=dict(color='rgba(255, 255, 255, 0.8)', width=2))
            )
            
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Detaylı Tablo
        st.markdown("---")
        st.subheader("📋 Top 10 Territory Detayları")
        
        display_cols = ['Territory', 'Region', 'City', 'Manager', 'PF_Satis', 'Toplam_Pazar', 'Toplam_Pazar_%', 'Pazar_Payi_%', 'Agirlik_%']
        
        top10_display = top10[display_cols].copy()
        top10_display.columns = ['Territory', 'Region', 'City', 'Manager', 'PF Satış', 'Toplam Pazar', 'Toplam Pazar %', 'Pazar Payı %', 'Ağırlık %']
        top10_display.index = range(1, len(top10_display) + 1)
        
        # Modern tablo stilini uygula
        styled_df = style_dataframe(
            top10_display,
            color_column='Pazar Payı %',
            gradient_columns=['Toplam Pazar %', 'Ağırlık %']
        )
        
        st.dataframe(
            styled_df,
            use_container_width=True,
            height=400
        )
    
    # TAB 2: MODERN HARİTA
    with tab2:
        st.header("🗺️ Modern Türkiye Haritası")
        
        city_data = calculate_city_performance(df_filtered, selected_product, date_filter)
        
        # Yatırım stratejisi hesapla
        investment_df = calculate_investment_strategy(city_data)
        
        # Filtrelenmiş PF toplam
        filtered_pf_toplam = city_data['PF_Satis'].sum()
        
        # Quick Stats
        col1, col2, col3, col4, col5 = st.columns(5)
        
        total_pf = city_data['PF_Satis'].sum()
        total_market = city_data['Toplam_Pazar'].sum()
        avg_share = city_data['Pazar_Payi_%'].mean()
        active_cities = len(city_data[city_data['PF_Satis'] > 0])
        top_city = city_data.loc[city_data['PF_Satis'].idxmax(), 'City'] if len(city_data) > 0 else "Yok"
        
        with col1:
            st.metric("💊 PF Satış", f"{total_pf:,.0f}")
        with col2:
            st.metric("🏪 Toplam Pazar", f"{total_market:,.0f}")
        with col3:
            st.metric("📊 Ort. Pazar Payı", f"%{avg_share:.1f}")
        with col4:
            st.metric("🏙️ Aktif Şehir", active_cities)
        with col5:
            st.metric("🏆 Lider Şehir", top_city)
        
        st.markdown("---")
        
        # Modern Harita
        if gdf is not None:
            st.subheader("📍 İl Bazlı Dağılım")
            
            turkey_map = create_modern_turkey_map(
                city_data, 
                gdf, 
                title=f"{selected_product} - {view_mode}",
                view_mode=view_mode,
                filtered_pf_toplam=filtered_pf_toplam
            )
            
            if turkey_map:
                st.plotly_chart(turkey_map, use_container_width=True)
            else:
                st.error("❌ Harita oluşturulamadı")
        else:
            st.warning("⚠️ turkey.geojson bulunamadı")
        
        st.markdown("---")
        
        # Şehir Analizi
        col_analysis1, col_analysis2 = st.columns(2)
        
        with col_analysis1:
            st.subheader("🏆 Top 10 Şehir")
            top_cities = city_data.nlargest(10, 'PF_Satis')
            
            fig_bar = px.bar(
                top_cities,
                x='City',
                y='PF_Satis',
                title='<b>En Yüksek Satış Yapan Şehirler</b>',
                color='Region',
                color_discrete_map=REGION_COLORS,
                hover_data=['Region', 'PF_Satis', 'Pazar_Payi_%'],
                text='PF_Satis'
            )
            
            fig_bar.update_layout(
                height=500,
                xaxis_tickangle=-45,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e2e8f0'),
                yaxis_title='<b>PF Satış</b>',
                xaxis_title='<b>Şehir</b>'
            )
            
            fig_bar.update_traces(
                texttemplate='%{text:,.0f}',
                textposition='outside',
                marker=dict(line=dict(width=2, color='rgba(255, 255, 255, 0.8)'))
            )
            
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col_analysis2:
            st.subheader("🗺️ Bölge Dağılımı")
            
            region_perf = city_data.groupby('Region').agg({
                'PF_Satis': 'sum',
                'Toplam_Pazar': 'sum'
            }).reset_index()
            
            region_perf['Pazar_Payi_%'] = safe_divide(region_perf['PF_Satis'], region_perf['Toplam_Pazar']) * 100
            
            fig_pie = px.pie(
                region_perf,
                values='PF_Satis',
                names='Region',
                title='<b>Bölgelere Göre Satış Dağılımı</b>',
                color='Region',
                color_discrete_map=REGION_COLORS,
                hole=0.3
            )
            
            fig_pie.update_layout(
                height=500,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e2e8f0'),
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="middle",
                    y=0.5,
                    xanchor="right",
                    x=1.3
                )
            )
            
            fig_pie.update_traces(
                textposition='inside',
                textinfo='percent+label',
                marker=dict(line=dict(color='rgba(255, 255, 255, 0.8)', width=2))
            )
            
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Yatırım Stratejisi
        st.markdown("---")
        st.subheader("🎯 Yatırım Stratejisi Analizi")
        
        if len(investment_df) > 0:
            # Strateji istatistikleri
            strategy_counts = investment_df['Yatırım_Stratejisi'].value_counts()
            
            cols_strategy = st.columns(5)
            strategy_metrics = [
                ("🚀 Agresif", "Agresif"),
                ("⚡ Hızlandırılmış", "Hızlandırılmış"),
                ("🛡️ Koruma", "Koruma"),
                ("💎 Potansiyel", "Potansiyel"),
                ("👁️ İzleme", "İzleme")
            ]
            
            for idx, (strategy_key, strategy_name) in enumerate(strategy_metrics):
                with cols_strategy[idx]:
                    count = strategy_counts.get(strategy_key, 0)
                    total_value = investment_df[investment_df['Yatırım_Stratejisi'] == strategy_key]['PF_Satis'].sum()
                    st.metric(
                        strategy_name,
                        f"{count} şehir",
                        f"{total_value:,.0f} PF"
                    )
            
            st.markdown("---")
            
            # Detaylı tablo
            st.subheader("📋 Detaylı Şehir Listesi")
            
            # Strateji filtresini uygula
            investment_display = investment_df.copy()
            if selected_strateji != "Tümü":
                investment_display = investment_display[investment_display['Yatırım_Stratejisi'] == selected_strateji]
            
            city_display = investment_display.sort_values('PF_Satis', ascending=False).copy()
            
            display_cols = ['City', 'Region', 'PF_Satis', 'Toplam_Pazar', 'Pazar_Payi_%', 'Yatırım_Stratejisi']
            city_display_formatted = city_display[display_cols].copy()
            city_display_formatted.columns = ['Şehir', 'Bölge', 'PF Satış', 'Toplam Pazar', 'Pazar Payı %', 'Strateji']
            city_display_formatted.index = range(1, len(city_display_formatted) + 1)
            
            # Modern tablo stilini uygula
            styled_cities = style_dataframe(
                city_display_formatted,
                color_column='Pazar Payı %',
                gradient_columns=['PF Satış']
            )
            
            st.dataframe(
                styled_cities,
                use_container_width=True,
                height=400
            )
    
    # TAB 3: TERRITORY ANALİZİ - GÜNCELLENMİŞ
    with tab3:
        st.header("🏢 Territory Bazlı Detaylı Analiz")
        
        terr_perf = calculate_territory_performance(df_filtered, selected_product, date_filter)
        
        # TOPLAM PAZAR YÜZDESİ HESAPLA
        total_market_all = terr_perf['Toplam_Pazar'].sum()
        terr_perf['Toplam_Pazar_%'] = safe_divide(terr_perf['Toplam_Pazar'], total_market_all) * 100
        
        # Filtreleme ve sıralama
        col_filter1, col_filter2 = st.columns([1, 2])
        
        with col_filter1:
            sort_options = {
                'PF_Satis': 'PF Satış',
                'Pazar_Payi_%': 'Pazar Payı %',
                'Toplam_Pazar': 'Toplam Pazar',
                'Toplam_Pazar_%': 'Toplam Pazar %',
                'Agirlik_%': 'Ağırlık %'
            }
            sort_by = st.selectbox(
                "Sıralama Kriteri",
                options=list(sort_options.keys()),
                format_func=lambda x: sort_options[x]
            )
        
        with col_filter2:
            show_n = st.slider("Gösterilecek Territory Sayısı", 10, 100, 25, 5)
        
        terr_sorted = terr_perf.sort_values(sort_by, ascending=False).head(show_n)
        
        # Visualizations
        col_viz1, col_viz2 = st.columns(2)
        
        with col_viz1:
            st.subheader("📊 PF vs Rakip Satış")
            
            fig_bar = go.Figure()
            
            # Her territory için çubuk grafik
            fig_bar.add_trace(go.Bar(
                x=terr_sorted['Territory'],
                y=terr_sorted['PF_Satis'],
                name='PF Satış',
                marker_color=PERFORMANCE_COLORS['success'],
                text=terr_sorted['PF_Satis'].apply(lambda x: f'{x:,.0f}'),
                textposition='outside',
                marker=dict(
                    line=dict(width=1.5, color='rgba(255, 255, 255, 0.8)')
                )
            ))
            
            fig_bar.add_trace(go.Bar(
                x=terr_sorted['Territory'],
                y=terr_sorted['Rakip_Satis'],
                name='Rakip Satış',
                marker_color=PERFORMANCE_COLORS['danger'],
                text=terr_sorted['Rakip_Satis'].apply(lambda x: f'{x:,.0f}'),
                textposition='outside',
                marker=dict(
                    line=dict(width=1.5, color='rgba(255, 255, 255, 0.8)')
                )
            ))
            
            fig_bar.update_layout(
                title=dict(
                    text=f'<b>Top {show_n} Territory - PF vs Rakip</b>',
                    font=dict(size=18, color='white')
                ),
                xaxis_title='<b>Territory</b>',
                yaxis_title='<b>Satış</b>',
                barmode='group',
                height=600,
                xaxis=dict(tickangle=-45),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e2e8f0'),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col_viz2:
            st.subheader("🎯 Pazar Payı Dağılımı")
            
            # Heatmap style scatter plot
            fig_scatter = px.scatter(
                terr_sorted,
                x='PF_Satis',
                y='Pazar_Payi_%',
                size='Toplam_Pazar',
                color='Region',
                color_discrete_map=REGION_COLORS,
                hover_name='Territory',
                hover_data={
                    'Region': True,
                    'PF_Satis': ':,.0f',
                    'Rakip_Satis': ':,.0f',
                    'Pazar_Payi_%': ':.1f',
                    'Toplam_Pazar_%': ':.1f'
                },
                size_max=50,
                title=f'<b>Territory Performans Haritası</b>'
            )
            
            fig_scatter.update_layout(
                height=600,
                plot_bgcolor='rgba(15, 23, 41, 0.9)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e2e8f0'),
                xaxis_title='<b>PF Satış</b>',
                yaxis_title='<b>Pazar Payı %</b>',
                legend=dict(
                    title='<b>Bölge</b>',
                    bgcolor='rgba(30, 41, 59, 0.8)'
                )
            )
            
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        st.markdown("---")
        
        # Detaylı Territory Listesi
        st.subheader(f"📋 Detaylı Territory Listesi (Top {show_n})")
        
        display_cols = [
            'Territory', 'Region', 'City', 'Manager',
            'PF_Satis', 'Rakip_Satis', 'Toplam_Pazar', 'Toplam_Pazar_%',
            'Pazar_Payi_%', 'Goreceli_Pazar_Payi', 'Agirlik_%'
        ]
        
        terr_display = terr_sorted[display_cols].copy()
        terr_display.columns = [
            'Territory', 'Region', 'City', 'Manager',
            'PF Satış', 'Rakip Satış', 'Toplam Pazar', 'Toplam Pazar %',
            'Pazar Payı %', 'Göreceli Pay', 'Ağırlık %'
        ]
        terr_display.index = range(1, len(terr_display) + 1)
        
        # Format numeric columns
        terr_display_formatted = terr_display.copy()
        
        # Modern tablo stilini uygula
        styled_territory = style_dataframe(
            terr_display_formatted,
            color_column='Pazar Payı %',
            gradient_columns=['Toplam Pazar %', 'Ağırlık %', 'Göreceli Pay']
        )
        
        st.dataframe(
            styled_territory,
            use_container_width=True,
            height=600
        )
        
        # Özet İstatistikler
        st.markdown("---")
        st.subheader("📊 Territory Performans Özeti")
        
        col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
        
        with col_sum1:
            avg_pazar_payi = terr_display_formatted['Pazar Payı %'].mean()
            st.metric("📊 Ort. Pazar Payı", f"{avg_pazar_payi:.1f}%")
        
        with col_sum2:
            total_pf = terr_display_formatted['PF Satış'].sum()
            st.metric("💰 Toplam PF Satış", f"{total_pf:,.0f}")
        
        with col_sum3:
            avg_toplam_pazar_yuzde = terr_display_formatted['Toplam Pazar %'].mean()
            st.metric("🏪 Ort. Pazar Payı", f"{avg_toplam_pazar_yuzde:.1f}%")
        
        with col_sum4:
            dominant_region = terr_display_formatted['Region'].mode()[0] if len(terr_display_formatted) > 0 else "Yok"
            region_color = REGION_COLORS.get(dominant_region, "#64748B")
            st.markdown(
                f'<div style="color:{region_color}; font-size:1.2rem; font-weight:bold; text-align: center;">'
                f'🏆 {dominant_region}</div>',
                unsafe_allow_html=True
            )
    
    # TAB 4: ZAMAN SERİSİ & ML
    with tab4:
        st.header("📈 Zaman Serisi Analizi & GERÇEK ML Tahminleme")
        
        territory_for_ts = st.selectbox(
            "Territory Seçin",
            ["TÜMÜ"] + sorted(df_filtered['TERRITORIES'].unique()),
            key='ts_territory'
        )
        
        monthly_df = calculate_time_series(df_filtered, selected_product, territory_for_ts, date_filter)
        
        if len(monthly_df) == 0:
            st.warning("⚠️ Seçilen filtrelerde veri bulunamadı")
        else:
            # Özet Metrikler
            col_ts1, col_ts2, col_ts3, col_ts4 = st.columns(4)
            
            with col_ts1:
                avg_pf = monthly_df['PF_Satis'].mean()
                st.metric("📊 Ort. Aylık PF", f"{avg_pf:,.0f}")
            
            with col_ts2:
                avg_growth = monthly_df['PF_Buyume_%'].mean()
                st.metric("📈 Ort. Büyüme", f"%{avg_growth:.1f}")
            
            with col_ts3:
                avg_share = monthly_df['Pazar_Payi_%'].mean()
                st.metric("🎯 Ort. Pazar Payı", f"%{avg_share:.1f}")
            
            with col_ts4:
                total_months = len(monthly_df)
                st.metric("📅 Veri Dönemi", f"{total_months} ay")
            
            st.markdown("---")
            
            # Grafikler
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                st.subheader("📊 Satış Trendi")
                fig_ts = go.Figure()
                
                fig_ts.add_trace(go.Scatter(
                    x=monthly_df['DATE'],
                    y=monthly_df['PF_Satis'],
                    mode='lines+markers',
                    name='PF Satış',
                    line=dict(color=PERFORMANCE_COLORS['success'], width=3, shape='spline'),
                    marker=dict(size=8, color='white', line=dict(width=2, color=PERFORMANCE_COLORS['success'])),
                    fill='tozeroy',
                    fillcolor='rgba(16, 185, 129, 0.1)'
                ))
                
                fig_ts.add_trace(go.Scatter(
                    x=monthly_df['DATE'],
                    y=monthly_df['Rakip_Satis'],
                    mode='lines+markers',
                    name='Rakip Satış',
                    line=dict(color=PERFORMANCE_COLORS['danger'], width=3, shape='spline'),
                    marker=dict(size=8, color='white', line=dict(width=2, color=PERFORMANCE_COLORS['danger'])),
                    fill='tozeroy',
                    fillcolor='rgba(239, 68, 68, 0.1)'
                ))
                
                fig_ts.update_layout(
                    height=500,
                    xaxis_title='<b>Tarih</b>',
                    yaxis_title='<b>Satış</b>',
                    hovermode='x unified',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#e2e8f0'),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                st.plotly_chart(fig_ts, use_container_width=True)
            
            with col_chart2:
                st.subheader("🎯 Pazar Payı Trendi")
                fig_share = go.Figure()
                
                fig_share.add_trace(go.Scatter(
                    x=monthly_df['DATE'],
                    y=monthly_df['Pazar_Payi_%'],
                    mode='lines+markers',
                    name='Pazar Payı %',
                    line=dict(color=PERFORMANCE_COLORS['info'], width=3, shape='spline'),
                    marker=dict(size=8, color='white', line=dict(width=2, color=PERFORMANCE_COLORS['info'])),
                    fill='tozeroy',
                    fillcolor='rgba(59, 130, 246, 0.1)'
                ))
                
                fig_share.add_hline(
                    y=50,
                    line_dash="dash",
                    line_color=PERFORMANCE_COLORS['warning'],
                    opacity=0.5,
                    line_width=2,
                    annotation_text="50% Eşik"
                )
                
                fig_share.update_layout(
                    height=500,
                    xaxis_title='<b>Tarih</b>',
                    yaxis_title='<b>Pazar Payı (%)</b>',
                    yaxis=dict(range=[0, 100]),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#e2e8f0')
                )
                
                st.plotly_chart(fig_share, use_container_width=True)
            
            # ML Tahminleme
            st.markdown("---")
            st.header("🤖 Machine Learning Satış Tahmini")
            
            forecast_months = st.slider("Tahmin Periyodu (Ay)", 1, 6, 3)
            
            if len(monthly_df) < 10:
                st.warning("⚠️ Tahmin için yeterli veri yok (en az 10 ay gerekli)")
            else:
                with st.spinner("ML modelleri eğitiliyor..."):
                    ml_results, best_model_name, forecast_df = train_ml_models(monthly_df, forecast_months)
                
                if ml_results is None:
                    st.error("❌ Model eğitimi başarısız")
                else:
                    # Model Performansı
                    st.subheader("📊 Model Performans Karşılaştırması")
                    
                    perf_data = []
                    for name, metrics in ml_results.items():
                        perf_data.append({
                            'Model': name,
                            'MAE': metrics['MAE'],
                            'RMSE': metrics['RMSE'],
                            'MAPE (%)': metrics['MAPE']
                        })
                    
                    perf_df = pd.DataFrame(perf_data)
                    perf_df = perf_df.sort_values('MAPE (%)')
                    
                    col_ml1, col_ml2 = st.columns([2, 1])
                    
                    with col_ml1:
                        styled_perf = style_dataframe(
                            perf_df,
                            color_column='MAPE (%)',
                            gradient_columns=['MAE', 'RMSE']
                        )
                        st.dataframe(styled_perf, use_container_width=True)
                    
                    with col_ml2:
                        best_mape = ml_results[best_model_name]['MAPE']
                        
                        if best_mape < 10:
                            confidence_level = "🟢 YÜKSEK"
                            confidence_color = "#10B981"
                        elif best_mape < 20:
                            confidence_level = "🟡 ORTA"
                            confidence_color = "#F59E0B"
                        else:
                            confidence_level = "🔴 DÜŞÜK"
                            confidence_color = "#EF4444"
                        
                        st.markdown(f'<div style="background: rgba(30, 41, 59, 0.8); padding: 1.5rem; border-radius: 12px; border: 2px solid {confidence_color}; margin-top: 1rem;">'
                                   f'<h3 style="color: white; margin: 0 0 1rem 0;">🏆 En İyi Model</h3>'
                                   f'<p style="color: {confidence_color}; font-size: 1.5rem; font-weight: 700; margin: 0 0 0.5rem 0;">{best_model_name}</p>'
                                   f'<p style="color: #94a3b8; margin: 0 0 1rem 0;">MAPE: <span style="color: {confidence_color}; font-weight: 700;">{best_mape:.2f}%</span></p>'
                                   f'<p style="color: #e2e8f0; font-weight: 600; margin: 0;">Güven Seviyesi: <span style="color: {confidence_color};">{confidence_level}</span></p>'
                                   '</div>', unsafe_allow_html=True)
                    
                    # Tahmin Grafiği
                    st.markdown("---")
                    st.subheader("📈 Gerçek vs ML Tahmini")
                    
                    forecast_chart = create_modern_forecast_chart(monthly_df, forecast_df)
                    st.plotly_chart(forecast_chart, use_container_width=True)
                    
                    # Tahmin Detayları
                    st.markdown("---")
                    st.subheader("📋 Tahmin Detayları")
                    
                    forecast_display = forecast_df[['YIL_AY', 'PF_Satis', 'Model']].copy()
                    forecast_display.columns = ['Ay', 'Tahmin Edilen Satış', 'Kullanılan Model']
                    forecast_display.index = range(1, len(forecast_display) + 1)
                    
                    styled_forecast = style_dataframe(
                        forecast_display,
                        gradient_columns=['Tahmin Edilen Satış']
                    )
                    
                    st.dataframe(styled_forecast, use_container_width=True)
    
    # TAB 5: RAKİP ANALİZİ
    with tab5:
        st.header("📊 Detaylı Rakip Analizi")
        
        comp_data = calculate_competitor_analysis(df_filtered, selected_product, date_filter)
        
        if len(comp_data) == 0:
            st.warning("⚠️ Seçilen filtrelerde veri bulunamadı")
        else:
            # Özet Metrikler
            col1, col2, col3, col4 = st.columns(4)
            
            avg_pf_share = comp_data['PF_Pay_%'].mean()
            avg_pf_growth = comp_data['PF_Buyume'].mean()
            avg_rakip_growth = comp_data['Rakip_Buyume'].mean()
            win_months = len(comp_data[comp_data['Fark'] > 0])
            
            with col1:
                st.metric("🎯 Ort. PF Pazar Payı", f"%{avg_pf_share:.1f}")
            with col2:
                st.metric("📈 Ort. PF Büyüme", f"%{avg_pf_growth:.1f}")
            with col3:
                st.metric("📉 Ort. Rakip Büyüme", f"%{avg_rakip_growth:.1f}")
            with col4:
                st.metric("🏆 Kazanılan Aylar", f"{win_months}/{len(comp_data)}")
            
            st.markdown("---")
            
            # Grafikler
            col_g1, col_g2 = st.columns(2)
            
            with col_g1:
                st.subheader("💰 Satış Karşılaştırması")
                comp_chart = create_modern_competitor_chart(comp_data)
                st.plotly_chart(comp_chart, use_container_width=True)
            
            with col_g2:
                st.subheader("📈 Büyüme Karşılaştırması")
                growth_chart = create_modern_growth_chart(comp_data)
                st.plotly_chart(growth_chart, use_container_width=True)
            
            # Detaylı Tablo
            st.markdown("---")
            st.subheader("📋 Aylık Performans Detayları")
            
            comp_display = comp_data[['YIL_AY', 'PF', 'Rakip', 'PF_Pay_%', 'PF_Buyume', 'Rakip_Buyume', 'Fark']].copy()
            comp_display.columns = ['Ay', 'PF Satış', 'Rakip Satış', 'PF Pay %', 'PF Büyüme %', 'Rakip Büyüme %', 'Fark %']
            comp_display.index = range(1, len(comp_display) + 1)
            
            styled_comp = style_dataframe(
                comp_display,
                color_column='Fark %',
                gradient_columns=['PF Pay %', 'PF Büyüme %', 'Rakip Büyüme %']
            )
            
            st.dataframe(
                styled_comp,
                use_container_width=True,
                height=400
            )
    
    # TAB 6: BCG & STRATEJİ
    with tab6:
        st.header("⭐ BCG Matrix & Yatırım Stratejisi")
        
        bcg_df = calculate_bcg_matrix(df_filtered, selected_product, date_filter)
        
        # BCG Dağılımı
        st.subheader("📊 Portföy Dağılımı")
        
        bcg_counts = bcg_df['BCG_Kategori'].value_counts()
        
        col_bcg1, col_bcg2, col_bcg3, col_bcg4 = st.columns(4)
        
        with col_bcg1:
            star_count = bcg_counts.get("⭐ Star", 0)
            star_pf = bcg_df[bcg_df['BCG_Kategori'] == "⭐ Star"]['PF_Satis'].sum()
            st.metric("⭐ Star", f"{star_count}", delta=f"{star_pf:,.0f} PF")
        
        with col_bcg2:
            cow_count = bcg_counts.get("🐄 Cash Cow", 0)
            cow_pf = bcg_df[bcg_df['BCG_Kategori'] == "🐄 Cash Cow"]['PF_Satis'].sum()
            st.metric("🐄 Cash Cow", f"{cow_count}", delta=f"{cow_pf:,.0f} PF")
        
        with col_bcg3:
            q_count = bcg_counts.get("❓ Question Mark", 0)
            q_pf = bcg_df[bcg_df['BCG_Kategori'] == "❓ Question Mark"]['PF_Satis'].sum()
            st.metric("❓ Question", f"{q_count}", delta=f"{q_pf:,.0f} PF")
        
        with col_bcg4:
            dog_count = bcg_counts.get("🐶 Dog", 0)
            dog_pf = bcg_df[bcg_df['BCG_Kategori'] == "🐶 Dog"]['PF_Satis'].sum()
            st.metric("🐶 Dog", f"{dog_count}", delta=f"{dog_pf:,.0f} PF")
        
        st.markdown("---")
        
        # BCG Matrix
        st.subheader("🎯 BCG Matrix")
        
        bcg_chart = create_modern_bcg_chart(bcg_df)
        st.plotly_chart(bcg_chart, use_container_width=True)
        
        # BCG Detayları
        st.markdown("---")
        st.subheader("📋 BCG Kategori Detayları")
        
        display_cols_bcg = ['Territory', 'Region', 'BCG_Kategori', 'PF_Satis', 'Pazar_Payi_%', 'Goreceli_Pazar_Payi', 'Pazar_Buyume_%']
        
        bcg_display = bcg_df[display_cols_bcg].copy()
        bcg_display.columns = ['Territory', 'Region', 'BCG', 'PF Satış', 'Pazar Payı %', 'Göreceli Pay', 'Büyüme %']
        bcg_display = bcg_display.sort_values('PF Satış', ascending=False)
        bcg_display.index = range(1, len(bcg_display) + 1)
        
        styled_bcg = style_dataframe(
            bcg_display,
            color_column='Pazar Payı %',
            gradient_columns=['PF Satış', 'Büyüme %']
        )
        
        st.dataframe(
            styled_bcg,
            use_container_width=True,
            height=400
        )
    
    # TAB 7: RAPORLAR
    with tab7:
        st.header("📥 Rapor İndirme")
        
        st.markdown("""
        <div style="background: rgba(30, 41, 59, 0.7); padding: 2rem; border-radius: 12px; margin-bottom: 2rem;">
            <h3 style="color: #e2e8f0; margin-top: 0;">📊 Detaylı Excel Raporu</h3>
            <p style="color: #94a3b8; margin-bottom: 1.5rem;">
                Tüm analizlerinizi içeren kapsamlı bir Excel raporu oluşturun. 
                Rapor aşağıdaki sayfaları içerecektir:
            </p>
            <ul style="color: #cbd5e1; margin-left: 1.5rem;">
                <li>Territory Performans (Toplam Pazar % ile)</li>
                <li>Zaman Serisi Analizi</li>
                <li>BCG Matrix</li>
                <li>Şehir Bazlı Analiz</li>
                <li>Rakip Analizi</li>
                <li>ML Tahmin Sonuçları</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("📊 Excel Raporu Oluştur", type="primary", use_container_width=True):
            with st.spinner("Rapor hazırlanıyor..."):
                # Tüm analizleri hesapla
                terr_perf = calculate_territory_performance(df_filtered, selected_product, date_filter)
                total_market_all = terr_perf['Toplam_Pazar'].sum()
                terr_perf['Toplam_Pazar_%'] = safe_divide(terr_perf['Toplam_Pazar'], total_market_all) * 100
                
                monthly_df = calculate_time_series(df_filtered, selected_product, None, date_filter)
                bcg_df = calculate_bcg_matrix(df_filtered, selected_product, date_filter)
                city_data = calculate_city_performance(df_filtered, selected_product, date_filter)
                comp_data = calculate_competitor_analysis(df_filtered, selected_product, date_filter)
                
                # ML tahmini
                ml_results, best_model_name, forecast_df = train_ml_models(monthly_df, 3)
                
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    terr_perf.to_excel(writer, sheet_name='Territory Performans', index=False)
                    monthly_df.to_excel(writer, sheet_name='Zaman Serisi', index=False)
                    bcg_df.to_excel(writer, sheet_name='BCG Matrix', index=False)
                    city_data.to_excel(writer, sheet_name='Şehir Analizi', index=False)
                    comp_data.to_excel(writer, sheet_name='Rakip Analizi', index=False)
                    
                    if forecast_df is not None:
                        forecast_df.to_excel(writer, sheet_name='ML Tahminler', index=False)
                    
                    # Özet sayfası
                    summary_data = {
                        'Metrik': ['Ürün', 'Dönem', 'Toplam PF Satış', 'Toplam Pazar', 'Pazar Payı', 'Territory Sayısı'],
                        'Değer': [
                            selected_product,
                            date_option,
                            f"{terr_perf['PF_Satis'].sum():,.0f}",
                            f"{terr_perf['Toplam_Pazar'].sum():,.0f}",
                            f"{(terr_perf['PF_Satis'].sum() / terr_perf['Toplam_Pazar'].sum() * 100):.1f}%" if terr_perf['Toplam_Pazar'].sum() > 0 else "0%",
                            len(terr_perf)
                        ]
                    }
                    summary_df = pd.DataFrame(summary_data)
                    summary_df.to_excel(writer, sheet_name='Özet', index=False)
                
                st.success("✅ Rapor hazır!")
                
                # İndirme butonu
                st.download_button(
                    label="💾 Excel Raporunu İndir",
                    data=output.getvalue(),
                    file_name=f"ticari_portfoy_raporu_{selected_product}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )

if __name__ == "__main__":
    main()



