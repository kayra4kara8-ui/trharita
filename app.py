"""
🎯 McKINSEY STİLİ TİCARİ PORTFÖY ANALİZ SİSTEMİ
Professional Analytics with McKinsey Design Language

Özellikler:
- 🎨 McKinsey renk paleti ve görsel kimlik
- 🗺️ Türkiye il bazlı harita görselleştirme
- 🤖 Machine Learning (Linear Regression, Ridge, Random Forest)
- 📊 McKinsey tarzı grafikler ve analizler
- 📈 Profesyonel raporlama ve insight'lar
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

warnings.filterwarnings("ignore")

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="McKinsey Portfolio Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# McKINSEY COLOR PALETTE
# =============================================================================
MCKINSEY_COLORS = {
    # Primary McKinsey Colors
    "navy": "#002856",           # McKinsey Navy (Primary)
    "blue": "#0066CC",           # McKinsey Blue
    "light_blue": "#4A90E2",     # Light Blue
    "teal": "#00A3A1",           # Teal
    "yellow": "#FFB81C",         # McKinsey Yellow
    "gold": "#F7A800",           # Gold
    
    # Secondary Colors
    "green": "#39B54A",          # Green (positive)
    "red": "#E31937",            # Red (negative/alert)
    "orange": "#FF6900",         # Orange (warning)
    "purple": "#663399",         # Purple
    "gray": "#666666",           # Dark Gray
    "light_gray": "#999999",     # Light Gray
    "background": "#F5F5F5",     # Background Gray
    
    # Chart Colors (McKinsey Style)
    "chart_1": "#002856",        # Navy
    "chart_2": "#0066CC",        # Blue
    "chart_3": "#00A3A1",        # Teal
    "chart_4": "#FFB81C",        # Yellow
    "chart_5": "#39B54A",        # Green
    "chart_6": "#E31937",        # Red
    "chart_7": "#663399",        # Purple
    "chart_8": "#FF6900",        # Orange
}

# McKinsey Chart Color Sequence
MCKINSEY_CHART_COLORS = [
    MCKINSEY_COLORS["navy"],
    MCKINSEY_COLORS["blue"],
    MCKINSEY_COLORS["teal"],
    MCKINSEY_COLORS["yellow"],
    MCKINSEY_COLORS["green"],
    MCKINSEY_COLORS["orange"],
    MCKINSEY_COLORS["purple"],
    MCKINSEY_COLORS["red"]
]

# Performance Colors (McKinsey)
PERFORMANCE_COLORS = {
    "excellent": MCKINSEY_COLORS["green"],
    "good": MCKINSEY_COLORS["teal"],
    "average": MCKINSEY_COLORS["yellow"],
    "below": MCKINSEY_COLORS["orange"],
    "poor": MCKINSEY_COLORS["red"],
    "positive": MCKINSEY_COLORS["green"],
    "negative": MCKINSEY_COLORS["red"],
}

# Region Colors (McKinsey Palette)
REGION_COLORS = {
    "MARMARA": MCKINSEY_COLORS["navy"],
    "BATI ANADOLU": MCKINSEY_COLORS["blue"],
    "EGE": MCKINSEY_COLORS["teal"],
    "İÇ ANADOLU": MCKINSEY_COLORS["yellow"],
    "GÜNEY DOĞU ANADOLU": MCKINSEY_COLORS["orange"],
    "KUZEY ANADOLU": MCKINSEY_COLORS["green"],
    "KARADENİZ": MCKINSEY_COLORS["purple"],
    "AKDENİZ": MCKINSEY_COLORS["light_blue"],
    "DOĞU ANADOLU": MCKINSEY_COLORS["red"],
    "DİĞER": MCKINSEY_COLORS["gray"]
}

# BCG Matrix Colors (McKinsey Style)
BCG_COLORS = {
    "⭐ Star": MCKINSEY_COLORS["green"],
    "🐄 Cash Cow": MCKINSEY_COLORS["navy"],
    "❓ Question Mark": MCKINSEY_COLORS["yellow"],
    "🐶 Dog": MCKINSEY_COLORS["red"]
}

# Investment Strategy Colors
STRATEGY_COLORS = {
    "🚀 Agresif": MCKINSEY_COLORS["red"],
    "⚡ Hızlandırılmış": MCKINSEY_COLORS["orange"],
    "🛡️ Koruma": MCKINSEY_COLORS["navy"],
    "💎 Potansiyel": MCKINSEY_COLORS["teal"],
    "👁️ İzleme": MCKINSEY_COLORS["gray"]
}

# =============================================================================
# McKINSEY CSS STYLING
# =============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Lato:wght@300;400;700;900&display=swap');
    
    * {
        font-family: 'Lato', sans-serif;
    }
    
    .stApp {
        background-color: #FFFFFF;
    }
    
    /* McKinsey Header */
    .mckinsey-header {
        background: linear-gradient(135deg, #002856 0%, #0066CC 100%);
        padding: 2.5rem 2rem;
        border-radius: 0;
        margin: -1rem -1rem 2rem -1rem;
        box-shadow: 0 4px 6px rgba(0, 40, 86, 0.1);
    }
    
    .mckinsey-title {
        font-size: 2.8rem;
        font-weight: 900;
        color: white;
        margin: 0 0 0.5rem 0;
        letter-spacing: -0.5px;
    }
    
    .mckinsey-subtitle {
        font-size: 1.1rem;
        font-weight: 300;
        color: rgba(255, 255, 255, 0.9);
        margin: 0;
    }
    
    /* Metric Cards - McKinsey Style */
    div[data-testid="metric-container"] {
        background: white;
        padding: 1.5rem;
        border-radius: 4px;
        border-left: 4px solid #002856;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.08);
        transition: all 0.3s ease;
    }
    
    div[data-testid="metric-container"]:hover {
        box-shadow: 0 4px 12px rgba(0, 40, 86, 0.15);
        transform: translateY(-2px);
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 2.5rem;
        font-weight: 700;
        color: #002856;
    }
    
    div[data-testid="stMetricLabel"] {
        font-size: 0.9rem;
        font-weight: 600;
        color: #666666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    div[data-testid="stMetricDelta"] {
        font-size: 0.9rem;
        font-weight: 600;
    }
    
    /* Tabs - McKinsey Style */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: white;
        border-bottom: 2px solid #E0E0E0;
        padding: 0;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #666666;
        font-weight: 600;
        font-size: 1rem;
        padding: 1rem 2rem;
        background: transparent;
        border: none;
        border-bottom: 3px solid transparent;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: #002856;
        background: rgba(0, 40, 86, 0.05);
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #002856;
        background: transparent;
        border-bottom-color: #002856;
    }
    
    /* Headers */
    h1, h2, h3, h4 {
        color: #002856;
        font-weight: 700;
    }
    
    h1 {
        font-size: 2.2rem;
        margin-top: 0;
        border-bottom: 3px solid #FFB81C;
        padding-bottom: 0.5rem;
        margin-bottom: 1.5rem;
    }
    
    h2 {
        font-size: 1.8rem;
        margin-top: 2rem;
        color: #0066CC;
    }
    
    h3 {
        font-size: 1.4rem;
        color: #002856;
    }
    
    /* Buttons - McKinsey Style */
    .stButton>button {
        background: #002856;
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 4px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton>button:hover {
        background: #0066CC;
        box-shadow: 0 4px 12px rgba(0, 40, 86, 0.25);
        transform: translateY(-2px);
    }
    
    .stButton>button:active {
        transform: translateY(0);
    }
    
    /* Sidebar - McKinsey Style */
    [data-testid="stSidebar"] {
        background: #F5F5F5;
        border-right: 1px solid #E0E0E0;
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #002856;
    }
    
    /* Section Dividers */
    hr {
        border: none;
        border-top: 2px solid #E0E0E0;
        margin: 2rem 0;
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 4px;
        border-left: 4px solid #0066CC;
    }
    
    /* Insight Box - McKinsey Style */
    .mckinsey-insight {
        background: linear-gradient(135deg, #002856 0%, #0066CC 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 4px;
        margin: 1.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 40, 86, 0.1);
    }
    
    .mckinsey-insight h3 {
        color: white;
        margin-top: 0;
        margin-bottom: 1rem;
        font-size: 1.3rem;
    }
    
    .mckinsey-insight p {
        color: rgba(255, 255, 255, 0.95);
        margin: 0.5rem 0;
        line-height: 1.6;
    }
    
    /* Chart containers */
    .plotly-graph-div {
        border-radius: 4px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.08);
    }
    
    /* Data tables */
    .dataframe {
        font-size: 0.9rem;
    }
    
    .dataframe thead tr th {
        background-color: #002856 !important;
        color: white !important;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 0.85rem;
        letter-spacing: 0.5px;
        padding: 1rem 0.5rem !important;
    }
    
    .dataframe tbody tr:nth-child(even) {
        background-color: #F5F5F5;
    }
    
    .dataframe tbody tr:hover {
        background-color: rgba(0, 102, 204, 0.1);
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #002856 0%, #0066CC 100%);
    }
    
    /* Select boxes */
    .stSelectbox label, .stSlider label, .stRadio label {
        color: #002856;
        font-weight: 600;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        border: 2px dashed #0066CC;
        border-radius: 4px;
        padding: 1rem;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #F5F5F5;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #002856;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #0066CC;
    }
    
    /* Section cards */
    .mckinsey-card {
        background: white;
        padding: 1.5rem;
        border-radius: 4px;
        border-left: 4px solid #FFB81C;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.08);
        margin: 1rem 0;
    }
    
    .mckinsey-card h4 {
        color: #002856;
        margin-top: 0;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CONSTANTS & DATA MAPPINGS
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
    "ZONGULDAK": "ZONGULDAK",
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
    """Şehir normalizasyon"""
    if pd.isna(city_name):
        return None
    
    city_upper = str(city_name).strip().upper()
    
    if city_upper in FIX_CITY_MAP:
        return FIX_CITY_MAP[city_upper]
    
    tr_map = {
        "İ": "I", "Ğ": "G", "Ü": "U",
        "Ş": "S", "Ö": "O", "Ç": "C",
        "Â": "A", "Î": "I", "Û": "U"
    }
    
    for k, v in tr_map.items():
        city_upper = city_upper.replace(k, v)
    
    return CITY_NORMALIZE_CLEAN.get(city_upper, city_name)

def format_number(num):
    """Format number with K, M suffixes (McKinsey style)"""
    if num >= 1_000_000:
        return f"{num/1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num/1_000:.1f}K"
    else:
        return f"{num:.0f}"

def generate_insight(metric_name, value, benchmark=None, trend=None):
    """Generate McKinsey-style insight text"""
    insights = []
    
    if benchmark:
        if value > benchmark * 1.1:
            insights.append(f"✓ {metric_name} exceeds benchmark by {((value/benchmark - 1) * 100):.0f}%")
        elif value < benchmark * 0.9:
            insights.append(f"⚠ {metric_name} below benchmark by {((1 - value/benchmark) * 100):.0f}%")
    
    if trend:
        if trend > 0:
            insights.append(f"↗ Positive momentum with {trend:.1f}% growth")
        elif trend < 0:
            insights.append(f"↘ Declining trend at {abs(trend):.1f}%")
    
    return " | ".join(insights) if insights else "Performance within expected range"

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
# McKINSEY STYLE TURKEY MAP
# =============================================================================

def create_mckinsey_turkey_map(city_data, gdf, title="Turkey Sales Map", 
                               view_mode="Region View", filtered_pf_total=None):
    """
    McKinsey tarzında Türkiye haritası - GELİŞTİRİLMİŞ
    """
    if gdf is None:
        st.error("❌ GeoJSON could not be loaded")
        return None
    
    # Prepare data
    city_data = city_data.copy()
    city_data['City_Fixed'] = city_data['City'].apply(normalize_city_name_fixed)
    city_data['City_Fixed'] = city_data['City_Fixed'].str.upper()
    
    # Normalize GeoJSON names
    gdf = gdf.copy()
    gdf['name_upper'] = gdf['name'].str.upper()
    gdf['name_fixed'] = gdf['name_upper'].replace(FIX_CITY_MAP)
    
    # Merge
    merged = gdf.merge(city_data, left_on='name_fixed', right_on='City_Fixed', how='left')
    
    # Fill NaN
    merged['PF_Satis'] = merged['PF_Satis'].fillna(0)
    merged['Pazar_Payi_%'] = merged['Pazar_Payi_%'].fillna(0)
    merged['Bölge'] = merged['Bölge'].fillna('DİĞER')
    merged['Region'] = merged['Bölge']
    
    # Assign region colors
    merged['Region_Color'] = merged['Region'].map(REGION_COLORS).fillna(MCKINSEY_COLORS["gray"])
    
    # Filtered total
    if filtered_pf_total is None:
        filtered_pf_total = merged['PF_Satis'].sum()
    
    # Create McKinsey style map
    fig = go.Figure()
    
    # Separate trace for each region
    for region in sorted(merged['Region'].unique()):
        region_data = merged[merged['Region'] == region]
        color = REGION_COLORS.get(region, MCKINSEY_COLORS["gray"])
        
        # Convert to JSON
        region_json = json.loads(region_data.to_json())
        
        fig.add_trace(go.Choroplethmapbox(
            geojson=region_json,
            locations=region_data.index,
            z=[1] * len(region_data),
            colorscale=[[0, color], [1, color]],
            marker_opacity=0.7,
            marker_line_width=1,
            marker_line_color='white',
            showscale=False,
            customdata=list(zip(
                region_data['name'],
                region_data['Region'],
                region_data['PF_Satis'],
                region_data['Pazar_Payi_%']
            )),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Region: %{customdata[1]}<br>"
                "PF Sales: %{customdata[2]:,.0f}<br>"
                "Market Share: %{customdata[3]:.1f}%"
                "<extra></extra>"
            ),
            name=region,
            visible=True
        ))
    
    # Boundary lines
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
            line=dict(width=0.8, color='white'),
            hoverinfo='skip',
            showlegend=False
        ))
    
    # Labels - GELİŞTİRİLMİŞ
    if view_mode == "Region View" or view_mode == "Bölge Görünümü":
        label_lons, label_lats, label_texts, label_colors = [], [], [], []
        
        for region in merged['Region'].unique():
            region_data = merged[merged['Region'] == region]
            total = region_data['PF_Satis'].sum()
            
            if total > 0:
                percent = (total / filtered_pf_total * 100) if filtered_pf_total > 0 else 0
                
                lon, lat = get_region_center(region_data)
                label_lons.append(lon)
                label_lats.append(lat)
                label_texts.append(
                    f"<b>{region}</b><br>"
                    f"{format_number(total)}<br>"
                    f"({percent:.1f}%)"
                )
                label_colors.append(MCKINSEY_COLORS["navy"])
        
        fig.add_trace(go.Scattermapbox(
            lon=label_lons,
            lat=label_lats,
            mode='text',
            text=label_texts,
            textfont=dict(
                size=11, 
                color=MCKINSEY_COLORS["navy"],
                family='Lato, sans-serif',
                weight='bold'
            ),
            hoverinfo='skip',
            showlegend=False
        ))
    
    else:
        # City View - TÜMÜNÜ GÖSTER
        city_lons, city_lats, city_texts, city_sizes = [], [], [], []
        
        for idx, row in merged.iterrows():
            if row['PF_Satis'] > 0:
                centroid = row.geometry.centroid
                city_lons.append(centroid.x)
                city_lats.append(centroid.y)
                city_texts.append(f"<b>{row['name']}</b><br>{format_number(row['PF_Satis'])}")
                city_sizes.append(8)  # Sabit boyut
        
        # Text markers ekle
        fig.add_trace(go.Scattermapbox(
            lon=city_lons,
            lat=city_lats,
            mode='markers+text',
            marker=dict(
                size=city_sizes,
                color=MCKINSEY_COLORS["navy"],
                opacity=0.6
            ),
            text=city_texts,
            textposition='top center',
            textfont=dict(
                size=8, 
                color=MCKINSEY_COLORS["navy"],
                family='Lato, sans-serif'
            ),
            hoverinfo='text',
            hovertext=city_texts,
            showlegend=False
        ))
    
    # Layout - McKinsey style
    fig.update_layout(
        mapbox_style="light",
        mapbox=dict(
            center=dict(lat=39.0, lon=35.0),
            zoom=5.3
        ),
        height=700,
        margin=dict(l=0, r=0, t=60, b=0),
        title=dict(
            text=f"<b>{title}</b>",
            x=0.5,
            font=dict(
                size=20, 
                color=MCKINSEY_COLORS["navy"],
                family='Lato, sans-serif'
            ),
            y=0.97
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=False,
        hoverlabel=dict(
            bgcolor="white",
            font_size=11,
            font_family="Lato, sans-serif",
            font_color=MCKINSEY_COLORS["navy"]
        )
    )
    
    return fig

# =============================================================================
# ML FEATURES & TRAINING
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
    """Train ML models"""
    df_features = create_ml_features(df)
    
    if len(df_features) < 10:
        return None, None, None
    
    feature_cols = ['lag_1', 'lag_2', 'lag_3', 'rolling_mean_3', 'rolling_mean_6',
                    'rolling_std_3', 'month', 'quarter', 'month_sin', 'month_cos', 'trend_index']
    
    # Train/Test split
    split_idx = int(len(df_features) * 0.8)
    
    train_df = df_features.iloc[:split_idx]
    test_df = df_features.iloc[split_idx:]
    
    X_train = train_df[feature_cols]
    y_train = train_df['PF_Satis']
    X_test = test_df[feature_cols]
    y_test = test_df['PF_Satis']
    
    # Models
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
        mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-10))) * 100
        
        results[name] = {
            'model': model,
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape
        }
    
    # Best model
    best_model_name = min(results.keys(), key=lambda x: results[x]['MAPE'])
    best_model = results[best_model_name]['model']
    
    # Future forecast
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
        
        # Update
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
    
    total_market_all = terr_perf['Toplam_Pazar'].sum()
    terr_perf['Toplam_Pazar_%'] = safe_divide(terr_perf['Toplam_Pazar'], total_market_all) * 100
    
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
    monthly['PF_Buyume'] = monthly['PF'].pct_change() * 100
    monthly['Rakip_Buyume'] = monthly['Rakip'].pct_change() * 100
    
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

def calculate_investment_strategy(city_perf):
    """Yatırım stratejisi"""
    df = city_perf.copy()
    df = df[df['PF_Satis'] > 0]
    
    if len(df) == 0:
        return df
    
    try:
        df["Pazar_Büyüklüğü"] = pd.qcut(df["Toplam_Pazar"], q=3, labels=["Küçük", "Orta", "Büyük"], duplicates='drop')
    except:
        df["Pazar_Büyüklüğü"] = "Orta"
    
    try:
        df["Performans"] = pd.qcut(df["PF_Satis"], q=3, labels=["Düşük", "Orta", "Yüksek"], duplicates='drop')
    except:
        df["Performans"] = "Orta"
    
    try:
        df["Pazar_Payı_Segment"] = pd.qcut(df["Pazar_Payi_%"], q=3, labels=["Düşük", "Orta", "Yüksek"], duplicates='drop')
    except:
        df["Pazar_Payı_Segment"] = "Orta"
    
    df["Büyüme_Alanı"] = df["Toplam_Pazar"] - df["PF_Satis"]
    try:
        df["Büyüme_Potansiyeli"] = pd.qcut(df["Büyüme_Alanı"], q=3, labels=["Düşük", "Orta", "Yüksek"], duplicates='drop')
    except:
        df["Büyüme_Potansiyeli"] = "Orta"
    
    def assign_strategy(row):
        pazar_buyuklugu = str(row["Pazar_Büyüklüğü"])
        pazar_payi = str(row["Pazar_Payı_Segment"])
        buyume_potansiyeli = str(row["Büyüme_Potansiyeli"])
        performans = str(row["Performans"])
        
        if pazar_buyuklugu in ["Büyük", "Orta"] and pazar_payi == "Düşük" and buyume_potansiyeli in ["Yüksek", "Orta"]:
            return "🚀 Agresif"
        elif pazar_buyuklugu in ["Büyük", "Orta"] and pazar_payi == "Orta" and performans in ["Orta", "Yüksek"]:
            return "⚡ Hızlandırılmış"
        elif pazar_buyuklugu == "Büyük" and pazar_payi == "Yüksek":
            return "🛡️ Koruma"
        elif pazar_buyuklugu == "Küçük" and buyume_potansiyeli == "Yüksek" and performans in ["Orta", "Yüksek"]:
            return "💎 Potansiyel"
        else:
            return "👁️ İzleme"
    
    df["Yatırım_Stratejisi"] = df.apply(assign_strategy, axis=1)
    
    return df

# =============================================================================
# McKINSEY STYLE CHARTS
# =============================================================================

def create_mckinsey_bar_chart(data, x, y, title, color_col=None, orientation='v'):
    """McKinsey style bar chart"""
    if color_col:
        colors = [REGION_COLORS.get(val, MCKINSEY_COLORS["navy"]) for val in data[color_col]]
    else:
        colors = MCKINSEY_COLORS["navy"]
    
    if orientation == 'v':
        fig = go.Figure(go.Bar(
            x=data[x],
            y=data[y],
            marker_color=colors,
            marker_line_color='white',
            marker_line_width=1.5,
            text=data[y].apply(lambda v: format_number(v)),
            textposition='outside',
            textfont=dict(size=11, family='Lato', color=MCKINSEY_COLORS["navy"])
        ))
        fig.update_layout(xaxis_title=x.upper(), yaxis_title=y.upper())
    else:
        fig = go.Figure(go.Bar(
            x=data[y],
            y=data[x],
            orientation='h',
            marker_color=colors,
            marker_line_color='white',
            marker_line_width=1.5,
            text=data[y].apply(lambda v: format_number(v)),
            textposition='outside',
            textfont=dict(size=11, family='Lato', color=MCKINSEY_COLORS["navy"])
        ))
        fig.update_layout(xaxis_title=y.upper(), yaxis_title=x.upper())
    
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(size=18, color=MCKINSEY_COLORS["navy"], family='Lato'),
            x=0.5,
            xanchor='center'
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=500,
        font=dict(family='Lato', color=MCKINSEY_COLORS["gray"]),
        xaxis=dict(showgrid=True, gridcolor='#E0E0E0', gridwidth=0.5),
        yaxis=dict(showgrid=True, gridcolor='#E0E0E0', gridwidth=0.5),
        margin=dict(l=80, r=40, t=80, b=60)
    )
    
    return fig

def create_mckinsey_line_chart(data, x, y_cols, title, y_names=None):
    """McKinsey style line chart"""
    fig = go.Figure()
    
    if y_names is None:
        y_names = y_cols
    
    for idx, (y_col, y_name) in enumerate(zip(y_cols, y_names)):
        color = MCKINSEY_CHART_COLORS[idx % len(MCKINSEY_CHART_COLORS)]
        
        fig.add_trace(go.Scatter(
            x=data[x],
            y=data[y_col],
            mode='lines+markers',
            name=y_name,
            line=dict(color=color, width=3),
            marker=dict(size=7, color='white', line=dict(width=2, color=color))
        ))
    
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(size=18, color=MCKINSEY_COLORS["navy"], family='Lato'),
            x=0.5,
            xanchor='center'
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=500,
        font=dict(family='Lato', color=MCKINSEY_COLORS["gray"]),
        xaxis=dict(
            showgrid=True, 
            gridcolor='#E0E0E0', 
            gridwidth=0.5,
            title=dict(text=x.upper(), font=dict(size=12, color=MCKINSEY_COLORS["gray"]))
        ),
        yaxis=dict(
            showgrid=True, 
            gridcolor='#E0E0E0', 
            gridwidth=0.5,
            title=dict(text="VALUE", font=dict(size=12, color=MCKINSEY_COLORS["gray"]))
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='white',
            bordercolor='#E0E0E0',
            borderwidth=1
        ),
        hovermode='x unified',
        margin=dict(l=80, r=40, t=100, b=60)
    )
    
    return fig

def create_mckinsey_scatter(data, x, y, size=None, color=None, title=""):
    """McKinsey style scatter plot"""
    if color:
        color_map = BCG_COLORS if color == 'BCG_Kategori' else REGION_COLORS
        fig = px.scatter(
            data,
            x=x,
            y=y,
            size=size,
            color=color,
            color_discrete_map=color_map,
            hover_name=data.columns[0] if len(data.columns) > 0 else None,
            size_max=40
        )
    else:
        fig = px.scatter(data, x=x, y=y, size=size, size_max=40)
        fig.update_traces(marker=dict(color=MCKINSEY_COLORS["navy"]))
    
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(size=18, color=MCKINSEY_COLORS["navy"], family='Lato'),
            x=0.5,
            xanchor='center'
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=550,
        font=dict(family='Lato', color=MCKINSEY_COLORS["gray"]),
        xaxis=dict(showgrid=True, gridcolor='#E0E0E0', gridwidth=0.5),
        yaxis=dict(showgrid=True, gridcolor='#E0E0E0', gridwidth=0.5),
        margin=dict(l=80, r=40, t=80, b=60)
    )
    
    return fig

# =============================================================================
# MAIN APP
# =============================================================================

def main():
    # McKinsey Header
    st.markdown("""
    <div class="mckinsey-header">
        <div class="mckinsey-title">McKinsey Portfolio Analytics</div>
        <div class="mckinsey-subtitle">Advanced Commercial Analytics & Strategic Insights</div>
    </div>
    """, unsafe_allow_html=True)
    
    # SIDEBAR
    with st.sidebar:
        st.markdown("### 📁 DATA UPLOAD")
        st.markdown("---")
        
        uploaded_file = st.file_uploader("Upload Excel File", type=['xlsx', 'xls'])
        
        if not uploaded_file:
            st.info("👈 Please upload an Excel file")
            st.stop()
        
        try:
            df = load_excel_data(uploaded_file)
            gdf = load_geojson_gpd()
            st.success(f"✅ **{len(df):,}** rows loaded")
        except Exception as e:
            st.error(f"❌ Data loading error: {str(e)}")
            st.stop()
        
        st.markdown("---")
        st.markdown("### 💊 PRODUCT SELECTION")
        selected_product = st.selectbox("Select Product", 
            ["TROCMETAM", "CORTIPOL", "DEKSAMETAZON", "PF IZOTONIK"])
        
        st.markdown("---")
        st.markdown("### 📅 DATE RANGE")
        
        min_date = df['DATE'].min()
        max_date = df['DATE'].max()
        
        date_option = st.selectbox("Period", 
            ["All Data", "Last 3 Months", "Last 6 Months", "Last 1 Year", "2025", "2024", "Custom Range"])
        
        if date_option == "All Data":
            date_filter = None
        elif date_option == "Last 3 Months":
            date_filter = (max_date - pd.DateOffset(months=3), max_date)
        elif date_option == "Last 6 Months":
            date_filter = (max_date - pd.DateOffset(months=6), max_date)
        elif date_option == "Last 1 Year":
            date_filter = (max_date - pd.DateOffset(years=1), max_date)
        elif date_option == "2025":
            date_filter = (pd.to_datetime('2025-01-01'), pd.to_datetime('2025-12-31'))
        elif date_option == "2024":
            date_filter = (pd.to_datetime('2024-01-01'), pd.to_datetime('2024-12-31'))
        else:
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start", min_date, min_value=min_date, max_value=max_date)
            with col2:
                end_date = st.date_input("End", max_date, min_value=min_date, max_value=max_date)
            date_filter = (pd.to_datetime(start_date), pd.to_datetime(end_date))
        
        st.markdown("---")
        st.markdown("### 🔍 FILTERS")
        
        territories = ["TÜMÜ"] + sorted(df['TERRITORIES'].unique())
        selected_territory = st.selectbox("Territory", territories)
        
        regions = ["TÜMÜ"] + sorted(df['REGION'].unique())
        selected_region = st.selectbox("Region", regions)
        
        managers = ["TÜMÜ"] + sorted(df['MANAGER'].unique())
        selected_manager = st.selectbox("Manager", managers)
        
        df_filtered = df.copy()
        if selected_territory != "TÜMÜ":
            df_filtered = df_filtered[df_filtered['TERRITORIES'] == selected_territory]
        if selected_region != "TÜMÜ":
            df_filtered = df_filtered[df_filtered['REGION'] == selected_region]
        if selected_manager != "TÜMÜ":
            df_filtered = df_filtered[df_filtered['MANAGER'] == selected_manager]
        
        st.markdown("---")
        st.markdown("### 🗺️ MAP SETTINGS")
        view_mode = st.radio("View Mode", ["Region View", "City View"])
    
    # MAIN CONTENT - TABS
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Executive Summary",
        "🗺️ Geographic Analysis",
        "🏢 Territory Performance",
        "📈 Time Series & ML",
        "🎯 Competitive Analysis",
        "⭐ Portfolio Strategy"
    ])
    
    # TAB 1: EXECUTIVE SUMMARY
    with tab1:
        st.markdown("## Executive Summary")
        
        cols = get_product_columns(selected_product)
        
        if date_filter:
            df_period = df_filtered[(df_filtered['DATE'] >= date_filter[0]) & (df_filtered['DATE'] <= date_filter[1])]
        else:
            df_period = df_filtered
        
        # Key Metrics
        total_pf = df_period[cols['pf']].sum()
        total_rakip = df_period[cols['rakip']].sum()
        total_market = total_pf + total_rakip
        market_share = (total_pf / total_market * 100) if total_market > 0 else 0
        active_territories = df_period['TERRITORIES'].nunique()
        avg_monthly = total_pf / df_period['YIL_AY'].nunique() if df_period['YIL_AY'].nunique() > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("PF SALES", format_number(total_pf), f"{format_number(avg_monthly)}/month")
        with col2:
            st.metric("TOTAL MARKET", format_number(total_market), f"{format_number(total_rakip)} competitor")
        with col3:
            delta_color = "normal" if market_share >= 50 else "inverse"
            st.metric("MARKET SHARE", f"{market_share:.1f}%", f"{100-market_share:.1f}% competitor")
        with col4:
            st.metric("ACTIVE TERRITORIES", active_territories, f"{df_period['MANAGER'].nunique()} managers")
        
        st.markdown("---")
        
        # McKinsey Insight Box
        avg_growth = df_period.groupby('YIL_AY')[cols['pf']].sum().pct_change().mean() * 100
        insight_text = generate_insight("Market Share", market_share, 50, avg_growth)
        
        st.markdown(f"""
        <div class="mckinsey-insight">
            <h3>💡 Key Insights</h3>
            <p>{insight_text}</p>
            <p>• Portfolio spans {active_territories} territories with {format_number(total_pf)} in PF sales</p>
            <p>• Market positioning {'strong' if market_share > 50 else 'requires attention'} at {market_share:.1f}% share</p>
            <p>• Average monthly performance trending {'positively' if avg_growth > 0 else 'negatively'} at {avg_growth:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Top Performers
        st.markdown("### 🏆 Top 10 Territory Performance")
        
        terr_perf = calculate_territory_performance(df_filtered, selected_product, date_filter)
        top10 = terr_perf.head(10)
        
        col_chart1, col_chart2 = st.columns([2, 1])
        
        with col_chart1:
            fig = create_mckinsey_bar_chart(
                top10, 'Territory', 'PF_Satis',
                'Top 10 Territories by PF Sales',
                color_col='Region'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col_chart2:
            fig_pie = px.pie(
                top10.head(5),
                values='PF_Satis',
                names='Territory',
                title='<b>Top 5 Territory Distribution</b>',
                color_discrete_sequence=MCKINSEY_CHART_COLORS
            )
            fig_pie.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                height=500,
                font=dict(family='Lato', color=MCKINSEY_COLORS["gray"])
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Detailed Table
        st.markdown("---")
        st.markdown("### 📋 Territory Performance Details")
        
        display_df = top10[['Territory', 'Region', 'City', 'Manager', 'PF_Satis', 
                            'Toplam_Pazar', 'Pazar_Payi_%', 'Toplam_Pazar_%']].copy()
        display_df.columns = ['Territory', 'Region', 'City', 'Manager', 'PF Sales', 
                              'Total Market', 'Market Share %', 'Market Size %']
        
        st.dataframe(display_df.style.format({
            'PF Sales': '{:,.0f}',
            'Total Market': '{:,.0f}',
            'Market Share %': '{:.1f}%',
            'Market Size %': '{:.1f}%'
        }).background_gradient(subset=['Market Share %'], cmap='RdYlGn'), 
        use_container_width=True, height=400)
    
    # TAB 2: GEOGRAPHIC ANALYSIS
    with tab2:
        st.markdown("## Geographic Performance Analysis")
        
        city_data = calculate_city_performance(df_filtered, selected_product, date_filter)
        filtered_pf_total = city_data['PF_Satis'].sum()
        
        # Quick Stats
        col1, col2, col3, col4 = st.columns(4)
        
        total_cities = len(city_data[city_data['PF_Satis'] > 0])
        top_city = city_data.loc[city_data['PF_Satis'].idxmax(), 'City'] if len(city_data) > 0 else "N/A"
        avg_city_sales = city_data['PF_Satis'].mean()
        
        with col1:
            st.metric("ACTIVE CITIES", total_cities)
        with col2:
            st.metric("TOP CITY", top_city)
        with col3:
            st.metric("AVG CITY SALES", format_number(avg_city_sales))
        with col4:
            avg_share = city_data['Pazar_Payi_%'].mean()
            st.metric("AVG MARKET SHARE", f"{avg_share:.1f}%")
        
        st.markdown("---")
        
        # Turkey Map
        if gdf is not None:
            st.markdown("### 📍 Geographic Distribution")
            turkey_map = create_mckinsey_turkey_map(
                city_data, gdf,
                title=f"{selected_product} - {view_mode}",
                view_mode=view_mode,
                filtered_pf_total=filtered_pf_total
            )
            if turkey_map:
                st.plotly_chart(turkey_map, use_container_width=True)
        
        st.markdown("---")
        
        col_a1, col_a2 = st.columns(2)
        
        with col_a1:
            st.markdown("### 🏆 Top 10 Cities")
            top_cities = city_data.nlargest(10, 'PF_Satis')
            fig = create_mckinsey_bar_chart(
                top_cities, 'City', 'PF_Satis',
                'Highest Performing Cities',
                color_col='Region', orientation='h'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col_a2:
            st.markdown("### 🗺️ Regional Distribution")
            region_perf = city_data.groupby('Region').agg({
                'PF_Satis': 'sum',
                'Toplam_Pazar': 'sum'
            }).reset_index()
            
            fig_region = px.pie(
                region_perf,
                values='PF_Satis',
                names='Region',
                title='<b>Sales by Region</b>',
                color='Region',
                color_discrete_map=REGION_COLORS
            )
            fig_region.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                height=500,
                font=dict(family='Lato', color=MCKINSEY_COLORS["gray"])
            )
            st.plotly_chart(fig_region, use_container_width=True)
    
    # TAB 3: TERRITORY PERFORMANCE
    with tab3:
        st.markdown("## Territory Performance Analysis")
        
        terr_perf = calculate_territory_performance(df_filtered, selected_product, date_filter)
        
        col_f1, col_f2 = st.columns([1, 2])
        
        with col_f1:
            sort_by = st.selectbox("Sort By", 
                ['PF_Satis', 'Pazar_Payi_%', 'Toplam_Pazar', 'Toplam_Pazar_%'],
                format_func=lambda x: x.replace('_', ' ').title())
        
        with col_f2:
            show_n = st.slider("Number of Territories", 10, 100, 25, 5)
        
        terr_sorted = terr_perf.sort_values(sort_by, ascending=False).head(show_n)
        
        # Visualizations
        col_v1, col_v2 = st.columns(2)
        
        with col_v1:
            st.markdown("### 📊 PF vs Competitor Sales")
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=terr_sorted['Territory'],
                y=terr_sorted['PF_Satis'],
                name='PF Sales',
                marker_color=MCKINSEY_COLORS["navy"],
                text=terr_sorted['PF_Satis'].apply(format_number),
                textposition='outside'
            ))
            fig.add_trace(go.Bar(
                x=terr_sorted['Territory'],
                y=terr_sorted['Rakip_Satis'],
                name='Competitor',
                marker_color=MCKINSEY_COLORS["red"],
                text=terr_sorted['Rakip_Satis'].apply(format_number),
                textposition='outside'
            ))
            
            fig.update_layout(
                title=dict(text=f'<b>Top {show_n} Territories</b>', font=dict(size=18, color=MCKINSEY_COLORS["navy"])),
                barmode='group',
                height=550,
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family='Lato'),
                xaxis=dict(tickangle=-45, showgrid=True, gridcolor='#E0E0E0'),
                yaxis=dict(showgrid=True, gridcolor='#E0E0E0'),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col_v2:
            st.markdown("### 🎯 Market Share Distribution")
            fig = create_mckinsey_scatter(
                terr_sorted, 'PF_Satis', 'Pazar_Payi_%',
                size='Toplam_Pazar', color='Region',
                title='Territory Performance Map'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.markdown(f"### 📋 Territory Details (Top {show_n})")
        
        display_cols = ['Territory', 'Region', 'City', 'Manager', 'PF_Satis', 
                       'Rakip_Satis', 'Toplam_Pazar', 'Pazar_Payi_%', 'Toplam_Pazar_%']
        
        display_df = terr_sorted[display_cols].copy()
        display_df.columns = ['Territory', 'Region', 'City', 'Manager', 'PF Sales',
                             'Competitor', 'Total Market', 'Market Share %', 'Market Size %']
        
        st.dataframe(display_df.style.format({
            'PF Sales': '{:,.0f}',
            'Competitor': '{:,.0f}',
            'Total Market': '{:,.0f}',
            'Market Share %': '{:.1f}%',
            'Market Size %': '{:.1f}%'
        }).background_gradient(subset=['Market Share %'], cmap='RdYlGn'),
        use_container_width=True, height=500)
    
    # TAB 4: TIME SERIES & ML
    with tab4:
        st.markdown("## Time Series Analysis & Machine Learning Forecast")
        
        territory_ts = st.selectbox("Select Territory", 
            ["TÜMÜ"] + sorted(df_filtered['TERRITORIES'].unique()), key='ts')
        
        monthly_df = calculate_time_series(df_filtered, selected_product, territory_ts, date_filter)
        
        if len(monthly_df) == 0:
            st.warning("⚠️ No data for selected filters")
        else:
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("AVG MONTHLY PF", format_number(monthly_df['PF_Satis'].mean()))
            with col2:
                st.metric("AVG GROWTH", f"{monthly_df['PF_Buyume_%'].mean():.1f}%")
            with col3:
                st.metric("AVG SHARE", f"{monthly_df['Pazar_Payi_%'].mean():.1f}%")
            with col4:
                st.metric("PERIODS", f"{len(monthly_df)} months")
            
            st.markdown("---")
            
            col_c1, col_c2 = st.columns(2)
            
            with col_c1:
                st.markdown("### 📊 Sales Trend")
                fig = create_mckinsey_line_chart(
                    monthly_df, 'DATE', 
                    ['PF_Satis', 'Rakip_Satis'],
                    'Sales Trend Over Time',
                    ['PF Sales', 'Competitor']
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col_c2:
                st.markdown("### 🎯 Market Share Trend")
                fig = create_mckinsey_line_chart(
                    monthly_df, 'DATE',
                    ['Pazar_Payi_%'],
                    'Market Share Evolution',
                    ['Market Share %']
                )
                fig.add_hline(y=50, line_dash="dash", line_color=MCKINSEY_COLORS["red"], 
                             opacity=0.5, annotation_text="50% Threshold")
                st.plotly_chart(fig, use_container_width=True)
            
            # ML Forecast
            st.markdown("---")
            st.markdown("## 🤖 Machine Learning Forecast")
            
            forecast_months = st.slider("Forecast Period (Months)", 1, 6, 3)
            
            if len(monthly_df) >= 10:
                with st.spinner("Training ML models..."):
                    ml_results, best_model, forecast_df = train_ml_models(monthly_df, forecast_months)
                
                if ml_results:
                    col_ml1, col_ml2 = st.columns([2, 1])
                    
                    with col_ml1:
                        st.markdown("### 📊 Model Performance")
                        perf_data = []
                        for name, metrics in ml_results.items():
                            perf_data.append({
                                'Model': name,
                                'MAE': f"{metrics['MAE']:,.0f}",
                                'RMSE': f"{metrics['RMSE']:,.0f}",
                                'MAPE': f"{metrics['MAPE']:.2f}%"
                            })
                        perf_df = pd.DataFrame(perf_data)
                        st.dataframe(perf_df, use_container_width=True)
                    
                    with col_ml2:
                        best_mape = ml_results[best_model]['MAPE']
                        confidence = "HIGH" if best_mape < 10 else "MEDIUM" if best_mape < 20 else "LOW"
                        color = MCKINSEY_COLORS["green"] if best_mape < 10 else MCKINSEY_COLORS["yellow"] if best_mape < 20 else MCKINSEY_COLORS["red"]
                        
                        st.markdown(f"""
                        <div class="mckinsey-card">
                            <h4>🏆 Best Model</h4>
                            <p style="font-size: 1.3rem; color: {color}; font-weight: 700;">{best_model}</p>
                            <p>MAPE: <span style="color: {color}; font-weight: 600;">{best_mape:.2f}%</span></p>
                            <p>Confidence: <span style="color: {color}; font-weight: 600;">{confidence}</span></p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Forecast Chart
                    st.markdown("### 📈 Forecast vs Actual")
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=monthly_df['DATE'],
                        y=monthly_df['PF_Satis'],
                        mode='lines+markers',
                        name='Actual',
                        line=dict(color=MCKINSEY_COLORS["navy"], width=3),
                        marker=dict(size=7, color='white', line=dict(width=2, color=MCKINSEY_COLORS["navy"]))
                    ))
                    
                    if forecast_df is not None:
                        fig.add_trace(go.Scatter(
                            x=forecast_df['DATE'],
                            y=forecast_df['PF_Satis'],
                            mode='lines+markers',
                            name='Forecast',
                            line=dict(color=MCKINSEY_COLORS["teal"], width=3, dash='dash'),
                            marker=dict(size=9, symbol='diamond', color='white', 
                                      line=dict(width=2, color=MCKINSEY_COLORS["teal"]))
                        ))
                    
                    fig.update_layout(
                        title=dict(text='<b>Sales Forecast</b>', font=dict(size=18, color=MCKINSEY_COLORS["navy"])),
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        height=500,
                        font=dict(family='Lato'),
                        xaxis=dict(showgrid=True, gridcolor='#E0E0E0'),
                        yaxis=dict(showgrid=True, gridcolor='#E0E0E0'),
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ Insufficient data for ML forecasting (minimum 10 months required)")
    
    # TAB 5: COMPETITIVE ANALYSIS
    with tab5:
        st.markdown("## Competitive Intelligence")
        
        comp_data = calculate_competitor_analysis(df_filtered, selected_product, date_filter)
        
        if len(comp_data) > 0:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("AVG PF SHARE", f"{comp_data['PF_Pay_%'].mean():.1f}%")
            with col2:
                st.metric("AVG PF GROWTH", f"{comp_data['PF_Buyume'].mean():.1f}%")
            with col3:
                st.metric("AVG COMP GROWTH", f"{comp_data['Rakip_Buyume'].mean():.1f}%")
            with col4:
                wins = len(comp_data[comp_data['PF_Buyume'] > comp_data['Rakip_Buyume']])
                st.metric("WINNING MONTHS", f"{wins}/{len(comp_data)}")
            
            st.markdown("---")
            
            col_c1, col_c2 = st.columns(2)
            
            with col_c1:
                st.markdown("### 💰 Sales Comparison")
                fig = go.Figure()
                fig.add_trace(go.Bar(x=comp_data['YIL_AY'], y=comp_data['PF'], 
                                    name='PF', marker_color=MCKINSEY_COLORS["navy"]))
                fig.add_trace(go.Bar(x=comp_data['YIL_AY'], y=comp_data['Rakip'], 
                                    name='Competitor', marker_color=MCKINSEY_COLORS["red"]))
                fig.update_layout(
                    title='<b>PF vs Competitor Sales</b>',
                    barmode='group',
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    height=500,
                    font=dict(family='Lato', color=MCKINSEY_COLORS["gray"]),
                    xaxis=dict(showgrid=True, gridcolor='#E0E0E0'),
                    yaxis=dict(showgrid=True, gridcolor='#E0E0E0')
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col_c2:
                st.markdown("### 📈 Growth Comparison")
                fig = create_mckinsey_line_chart(
                    comp_data, 'YIL_AY',
                    ['PF_Buyume', 'Rakip_Buyume'],
                    'Growth Rate Comparison',
                    ['PF Growth', 'Competitor Growth']
                )
                fig.add_hline(y=0, line_dash="dash", line_color=MCKINSEY_COLORS["gray"], opacity=0.5)
                st.plotly_chart(fig, use_container_width=True)
    
    # TAB 6: PORTFOLIO STRATEGY
    with tab6:
        st.markdown("## Portfolio Strategy & BCG Matrix")
        
        bcg_df = calculate_bcg_matrix(df_filtered, selected_product, date_filter)
        
        # BCG Distribution
        st.markdown("### 📊 Portfolio Distribution")
        
        bcg_counts = bcg_df['BCG_Kategori'].value_counts()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            star_count = bcg_counts.get("⭐ Star", 0)
            star_val = bcg_df[bcg_df['BCG_Kategori'] == "⭐ Star"]['PF_Satis'].sum()
            st.metric("⭐ STARS", star_count, format_number(star_val))
        
        with col2:
            cow_count = bcg_counts.get("🐄 Cash Cow", 0)
            cow_val = bcg_df[bcg_df['BCG_Kategori'] == "🐄 Cash Cow"]['PF_Satis'].sum()
            st.metric("🐄 CASH COWS", cow_count, format_number(cow_val))
        
        with col3:
            q_count = bcg_counts.get("❓ Question Mark", 0)
            q_val = bcg_df[bcg_df['BCG_Kategori'] == "❓ Question Mark"]['PF_Satis'].sum()
            st.metric("❓ QUESTIONS", q_count, format_number(q_val))
        
        with col4:
            dog_count = bcg_counts.get("🐶 Dog", 0)
            dog_val = bcg_df[bcg_df['BCG_Kategori'] == "🐶 Dog"]['PF_Satis'].sum()
            st.metric("🐶 DOGS", dog_count, format_number(dog_val))
        
        st.markdown("---")
        
        # BCG Matrix Chart
        st.markdown("### 🎯 BCG Matrix")
        
        fig = create_mckinsey_scatter(
            bcg_df, 'Goreceli_Pazar_Payi', 'Pazar_Buyume_%',
            size='PF_Satis', color='BCG_Kategori',
            title='Strategic Portfolio Positioning'
        )
        
        median_share = bcg_df['Goreceli_Pazar_Payi'].median()
        median_growth = bcg_df['Pazar_Buyume_%'].median()
        
        fig.add_hline(y=median_growth, line_dash="dash", line_color=MCKINSEY_COLORS["gray"], opacity=0.5)
        fig.add_vline(x=median_share, line_dash="dash", line_color=MCKINSEY_COLORS["gray"], opacity=0.5)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Strategic Recommendations
        st.markdown("---")
        st.markdown("### 💡 Strategic Recommendations")
        
        st.markdown("""
        <div class="mckinsey-insight">
            <h3>McKinsey BCG Framework Insights</h3>
            <p><strong>⭐ Stars:</strong> Maintain market leadership through continued investment and innovation</p>
            <p><strong>🐄 Cash Cows:</strong> Optimize operations and harvest cash for portfolio reinvestment</p>
            <p><strong>❓ Question Marks:</strong> Selective investment in high-potential opportunities</p>
            <p><strong>🐶 Dogs:</strong> Consider divestment or repositioning strategies</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Detailed Table
        st.markdown("---")
        st.markdown("### 📋 Portfolio Details")
        
        display_bcg = bcg_df[['Territory', 'Region', 'BCG_Kategori', 'PF_Satis', 
                              'Pazar_Payi_%', 'Pazar_Buyume_%']].copy()
        display_bcg.columns = ['Territory', 'Region', 'BCG Category', 'PF Sales',
                              'Market Share %', 'Growth %']
        
        st.dataframe(display_bcg.sort_values('PF Sales', ascending=False).style.format({
            'PF Sales': '{:,.0f}',
            'Market Share %': '{:.1f}%',
            'Growth %': '{:.1f}%'
        }).background_gradient(subset=['Growth %'], cmap='RdYlGn'),
        use_container_width=True, height=500)

if __name__ == "__main__":
    main()
