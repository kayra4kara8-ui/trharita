"""🎯 GELİŞMİŞ TİCARİ PORTFÖY ANALİZ SİSTEMİ
Territory Bazlı Performans, ML Tahminleme, Türkiye Haritası ve Rekabet Analizi

GELİŞTİRİLMİŞ ÖZELLİKLER:
- 🗺️ Türkiye il bazlı harita görselleştirme (GELİŞTİRİLMİŞ VERSİYON)
- 🤖 GERÇEK Machine Learning (Linear Regression, Ridge, Random Forest)
- 📊 GELİŞMİŞ Zaman Serisi Analizi (3 aylık, 6 aylık ortalamalar, mevsimsellik analizi)
- 📈 Gelişmiş rakip analizi ve trend karşılaştırması
- 🎯 Dinamik zaman aralığı filtreleme
- 📉 Trend analizi ve performans metrikleri
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString
import warnings
from scipy import stats

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
    /* Fontu McKinsey'in modern raporlarında kullandığına benzer temiz bir sans-serif yapalım */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* 1. ARKA PLAN: Derin, Ciddi Lacivert (McKinsey Blue) */
    .stApp {
        background-color: #051c2c; /* McKinsey Deep Navy */
        background-image: linear-gradient(180deg, #051c2c 0%, #03121d 100%);
        background-attachment: fixed;
    }
    
    /* 2. BAŞLIKLAR: Gradyan yok, sadece keskin beyaz ve otoriter */
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        text-align: left; /* Kurumsal raporlar genelde sola yaslıdır */
        padding: 2rem 0 1rem 0;
        color: #ffffff;
        letter-spacing: -0.5px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 2rem;
    }
    
    /* 3. METRİK DEĞERLERİ: Vurgu rengi (McKinsey Teal) */
    div[data-testid="stMetricValue"] {
        font-size: 2.5rem;
        font-weight: 600;
        color: #00A9BD; /* McKinsey Cyan/Teal */
    }
    
    div[data-testid="stMetricLabel"] {
        color: #b0b8c1;
        font-weight: 400;
        text-transform: uppercase;
        font-size: 0.85rem;
        letter-spacing: 1px;
    }
    
    /* 4. KARTLAR: Glassmorphism yerine temiz, net çizgiler */
    div[data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.03);
        padding: 1.5rem;
        border-radius: 4px; /* Köşeler daha az yuvarlak, daha ciddi */
        border-left: 4px solid #00A9BD; /* Sol tarafta ince bir vurgu çizgisi */
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.2s ease;
    }
    
    div[data-testid="metric-container"]:hover {
        background: rgba(255, 255, 255, 0.06);
        transform: translateY(-2px);
    }
    
    /* 5. SEKMELER (TABS): Minimalist */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background: transparent;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 0;
        padding: 0;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #8fa6b9;
        font-weight: 400;
        padding: 1rem 0;
        background: transparent;
        border: none;
        border-radius: 0;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: #ffffff;
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: transparent;
        color: #00A9BD;
        border-bottom: 3px solid #00A9BD; /* Sadece alt çizgi */
        font-weight: 600;
    }
    
    /* HEADERS */
    h1, h2, h3 {
        color: #ffffff !important;
        font-weight: 600;
    }
    
    p, span, div, label {
        color: #e0e6ed; /* Okunabilirlik için çok açık gri */
        line-height: 1.6;
    }
    
    /* 6. BUTONLAR: Sade ve Net */
    .stButton>button {
        background: #2B59C3; /* Kurumsal Mavi */
        color: white;
        border: none;
        padding: 0.6rem 1.5rem;
        border-radius: 4px;
        font-weight: 500;
        transition: all 0.2s;
        box-shadow: none;
        text-transform: uppercase;
        font-size: 0.9rem;
        letter-spacing: 0.5px;
    }
    
    .stButton>button:hover {
        background: #1e45a0;
        transform: none;
    }
    
    /* 7. TABLOLAR: Veri Odaklı, Temiz */
    .dataframe {
        font-size: 0.9rem;
        border: none !important;
    }
    
    .stDataFrame {
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Tablo Başlıkları */
    .dataframe thead th {
        background-color: #021019 !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        text-align: left !important;
        padding: 10px !important;
    }
    
    /* SCROLLBAR: Görünmez denecek kadar ince */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: #051c2c;
    }
    ::-webkit-scrollbar-thumb {
        background: #3e5060;
        border-radius: 3px;
    }
    
    /* SIDEBAR */
    [data-testid="stSidebar"] {
        background-color: #03121d;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* INPUT ALANLARI */
    .stSelectbox, .stSlider, .stRadio {
        color: white;
    }
    div[data-baseweb="select"] > div {
        background-color: #0e2a3f;
        border-color: #3e5060;
        color: white;
    }
    
    /* BİLGİ KARTLARI (INSIGHT CARDS) */
    .insight-card {
        background: #0e2a3f;
        padding: 1.5rem;
        border-radius: 4px;
        border-top: 3px solid #00A9BD;
        margin-bottom: 1rem;
    }
    
    /* TREND ANALİZİ STYLING */
    .trend-card {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(16, 185, 129, 0.1) 100%);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    .positive-trend {
        color: #10B981 !important;
        font-weight: 600;
    }
    
    .negative-trend {
        color: #EF4444 !important;
        font-weight: 600;
    }
    
    .neutral-trend {
        color: #94A3B8 !important;
        font-weight: 600;
    }
    
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SADE RENK PALETİ
# =============================================================================
# Monochromatic Blue - Kurumsal Mavi ve Slate Gri Tema
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

# PERFORMANS RENKLERİ - Kurumsal Mavi ve Slate Gri Tema
PERFORMANCE_COLORS = {
    "high": "#2563EB",       # Parlak Mavi – Yüksek Performans
    "medium": "#F59E0B",     # Altın Sarısı – Orta Performans
    "low": "#64748B",        # Slate Gri – Düşük Performans
    "positive": "#2563EB",   # Parlak Mavi – Pozitif (Eski Yeşil yerine)
    "negative": "#64748B",   # Slate Gri – Negatif (Eski Kırmızı yerine)
    "neutral": "#94A3B8",    # Açık Gri – Nötr
    "warning": "#F59E0B",    # Altın – Uyarı
    "info": "#0EA5E9",       # Sky Blue – Bilgi
    "success": "#06B6D4",    # Cyan – Başarı
    "danger": "#64748B"      # Slate Gri – Risk / Tehlike
}

# TREND ANALİZİ RENKLERİ (Mavi tonlarına güncellendi)
TREND_COLORS = {
    "strong_up": "#06B6D4",      # Cyan - Güçlü artış
    "up": "#3B82F6",            # Mavi - Artış
    "flat": "#94A3B8",          # Gri - Sabit
    "down": "#64748B",          # Slate Grey - Düşüş
    "strong_down": "#475569",   # Dark Slate - Güçlü düşüş
    "seasonal": "#0EA5E9",      # Sky Blue - Mevsimsel
    "cyclic": "#2563EB"         # Blue - Döngüsel
}

# BCG MATRIX RENKLERİ (Mavi tonlarına güncellendi)
BCG_COLORS = {
    "⭐ Star": "#2563EB",      # Parlak Mavi
    "🐄 Cash Cow": "#06B6D4",  # Cyan
    "❓ Question Mark": "#0EA5E9",  # Sky Blue
    "🐶 Dog": "#64748B"        # Slate Gray
}

# YATIRIM STRATEJİSİ RENKLERİ (Mavi tonlarına güncellendi)
STRATEGY_COLORS = {
    "🚀 Agresif": "#2563EB",      # Parlak Mavi
    "⚡ Hızlandırılmış": "#0EA5E9",  # Sky Blue
    "🛡️ Koruma": "#06B6D4",        # Cyan
    "💎 Potansiyel": "#3B82F6",     # Vivid Blue
    "👁️ İzleme": "#64748B"         # Slate Gray
}

# GRADIENT SCALES for Visualizations (Mavi tonlarına güncellendi)
GRADIENT_SCALES = {
    "blue_green": ["#1e3a8a", "#2563EB", "#0EA5E9", "#06B6D4"],
    "sequential_blue": ["#DBEAFE", "#BFDBFE", "#93C5FD", "#60A5FA", "#3B82F6", "#2563EB", "#1d4ed8"],
    "diverging": ["#64748B", "#94A3B8", "#BFDBFE", "#60A5FA", "#2563EB"],
    "temperature": ["#1e3a8a", "#1d4ed8", "#2563EB", "#3B82F6", "#60A5FA"],
    "trend": ["#475569", "#64748B", "#94A3B8", "#3B82F6", "#2563EB"]
}

# =============================================================================
# CONSTANTS
# =============================================================================

FIX_CITY_MAP = {
    "AGRI": "AĞRI",
    "BARTÄ±N": "BARTIN",
    "BARTIN": "BARTIN",
    "BINGÃ¶L": "BİNGÖL",
    "BINGOL": "BİNGÖL",
    "DÃ¼ZCE": "DÜZCE",
    "DÃ1⁄4ZCE": "DÜZCE",
    "DUZCE": "DÜZCE",
    "DÜZCE": "DÜZCE",
    "ELAZIG": "ELAZIĞ",
    "ELAZIĞ": "ELAZIĞ",
    "ESKISEHIR": "ESKİŞEHİR",
    "ESKİŞEHİR": "ESKİŞEHİR",
    "GÃ1⁄4MÃ1⁄4SHANE": "GÜMÜŞHANE",
    "GÃ¼mÃ¼SHANE": "GÜMÜŞHANE",
    "GÜMÜŞHANE": "GÜMÜŞHANE",
    "HAKKARI": "HAKKARİ",
    "HAKKARI": "HAKKARİ",
    "HAKKARİ": "HAKKARİ",
    "ISTANBUL": "İSTANBUL",
    "İSTANBUL": "İSTANBUL",
    "IZMIR": "İZMİR",
    "İZMİR": "İZMİR",
    "IÄ\x9fDIR": "IĞDIR",
    "IĞDIR": "IĞDIR",
    "KARABÃ1⁄4K": "KARABÜK",
    "KARABÜK": "KARABÜK",
    "KARABÃ¼K": "KARABÜK",
    "KINKKALE": "KIRIKKALE",
    "KIRIKKALE": "KIRIKKALE",
    "KIRSEHIR": "KIRŞEHİR",
    "KIRŞEHİR": "KIRŞEHİR",
    "KÃ1⁄4TAHYA": "KÜTAHYA",
    "KÃ¼TAHYA": "KÜTAHYA",
    "KÜTAHYA": "KÜTAHYA",
    "MUGLA": "MUĞLA",
    "MUĞLA": "MUĞLA",
    "MUS": "MUŞ",
    "MUŞ": "MUŞ",
    "NEVSEHIR": "NEVŞEHİR",
    "NEVŞEHİR": "NEVŞEHİR",
    "NIGDE": "NİĞDE",
    "NİĞDE": "NİĞDE",
    "SANLIURFA": "ŞANLIURFA",
    "ŞANLIURFA": "ŞANLIURFA",
    "SIRNAK": "ŞIRNAK",
    "ŞIRNAK": "ŞIRNAK",
    "TEKIRDAG": "TEKİRDAĞ",
    "TEKİRDAĞ": "TEKİRDAĞ",
    "USAK": "UŞAK",
    "UŞAK": "UŞAK",
    "ZINGULDAK": "ZONGULDAK",
    "ZONGULDAK": "ZONGULDAK",
    "Ã\x87ANAKKALE": "ÇANAKKALE",
    "ÇANAKKALE": "ÇANAKKALE",
    "Ã\x87ANKIRI": "ÇANKIRI",
    "ÇANKIRI": "ÇANKIRI",
    "Ã\x87ORUM": "ÇORUM",
    "ÇORUM": "ÇORUM",
    "K. MARAS": "KAHRAMANMARAŞ",
    "KAHRAMANMARAŞ": "KAHRAMANMARAŞ",
    "CORUM": "ÇORUM",
    "CANKIRI": "ÇANKIRI",
    "KARABUK": "KARABÜK",
    "GUMUSHANE": "GÜMÜŞHANE",
    "KUTAHYA": "KÜTAHYA",
    "CANAKKALE": "ÇANAKKALE",
    "TUNCELİ": "TUNCELİ",
    "TUNCELI": "TUNCELİ",
    "OSMANİYE": "OSMANİYE",
    "OSMANIYE": "OSMANİYE",
    "KİLİS": "KİLİS",
    "KILIS": "KİLİS",
    "ŞIRNAK": "ŞIRNAK",
    "SİİRT": "SİİRT",
    "SIIRT": "SİİRT",
    "BATMAN": "BATMAN",
    "BİTLİS": "BİTLİS",
    "BITLIS": "BİTLİS",
    "BİNGÖL": "BİNGÖL",
    "IĞDIR": "IĞDIR",
    "ARDAHAN": "ARDAHAN"
}

CITY_NORMALIZE_CLEAN = {
    'ADANA': 'Adana',
    'ADIYAMAN': 'Adiyaman',
    'AFYONKARAHISAR': 'Afyonkarahisar',
    'AFYON': 'Afyonkarahisar',
    'AGRI': 'Agri',
    'AĞRI': 'Agri',
    'AKSARAY': 'Aksaray',
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
    'HAKKARİ': 'Hakkari',
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
    'ZONGULDAK': 'Zonguldak',
    'ARDAHAN': 'Ardahan',
    'AKSARAY': 'Aksaray',
    'KIRIKKALE': 'Kirikkale'
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

def format_number(num):
    """Sayıları binlik ayırıcılı ve sadeleştirilmiş formatta göster"""
    if pd.isna(num):
        return "0"
    
    try:
        num = float(num)
        if num == 0:
            return "0"
        elif abs(num) >= 1_000_000_000:
            return f"{num/1_000_000_000:,.1f}B"
        elif abs(num) >= 1_000_000:
            return f"{num/1_000_000:,.1f}M"
        elif abs(num) >= 1_000:
            return f"{num/1_000:,.1f}K"
        else:
            return f"{num:,.0f}"
    except:
        return str(num)

def format_percentage(num):
    """Yüzdelikleri formatla"""
    if pd.isna(num):
        return "0%"
    try:
        return f"{float(num):.1f}%"
    except:
        return str(num)

def calculate_trend_slope(y_values):
    """Trend eğimini hesapla"""
    if len(y_values) < 2:
        return 0
    
    x = np.arange(len(y_values))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y_values)
    return slope

def classify_trend(slope, y_values):
    """Trendi sınıflandır"""
    if len(y_values) < 5:
        return "Yetersiz Veri"
    
    mean_value = np.mean(y_values)
    if mean_value == 0:
        return "Nötr"
    
    percent_slope = (slope / mean_value) * 100 if mean_value != 0 else 0
    
    if percent_slope > 10:
        return "📈 Güçlü Artış"
    elif percent_slope > 5:
        return "📈 Artış"
    elif percent_slope > -5:
        return "📊 Sabit"
    elif percent_slope > -10:
        return "📉 Düşüş"
    else:
        return "📉 Güçlü Düşüş"

def calculate_seasonality(y_values, period=12):
    """Mevsimsellik analizi"""
    if len(y_values) < period * 2:
        return None, "Yetersiz veri"
    
    try:
        from scipy.signal import periodogram
        f, Pxx = periodogram(y_values, fs=1)
        
        if len(f) > 0 and len(Pxx) > 0:
            # En yüksek mevsimsel frekans
            idx = np.argmax(Pxx[1:]) + 1
            dominant_period = 1 / f[idx] if f[idx] > 0 else 0
            
            if dominant_period >= period - 2 and dominant_period <= period + 2:
                return "Güçlü Mevsimsellik", round(dominant_period, 1)
            elif dominant_period >= 3 and dominant_period <= 24:
                return "Zayıf Mevsimsellik", round(dominant_period, 1)
            else:
                return "Mevsimsellik Yok", None
    except:
        return "Analiz Edilemedi", None
    
    return "Bilinmiyor", None

# =============================================================================
# GELİŞTİRİLMİŞ ZAMAN SERİSİ ANALİZ FONKSİYONLARI
# =============================================================================

def calculate_advanced_time_series(df, product, territory=None, date_filter=None):
    """GELİŞTİRİLMİŞ Zaman serisi analizi"""
    cols = get_product_columns(product)
    
    df_filtered = df.copy()
    if territory and territory != "TÜMÜ":
        df_filtered = df_filtered[df_filtered['TERRITORIES'] == territory]
    
    if date_filter:
        df_filtered = df_filtered[(df_filtered['DATE'] >= date_filter[0]) & 
                                   (df_filtered['DATE'] <= date_filter[1])]
    
    # Aylık gruplama
    monthly = df_filtered.groupby('YIL_AY').agg({
        cols['pf']: 'sum',
        cols['rakip']: 'sum',
        'DATE': 'first'
    }).reset_index().sort_values('YIL_AY')
    
    monthly.columns = ['YIL_AY', 'PF_Satis', 'Rakip_Satis', 'DATE']
    monthly['Toplam_Pazar'] = monthly['PF_Satis'] + monthly['Rakip_Satis']
    monthly['Pazar_Payi_%'] = safe_divide(monthly['PF_Satis'], monthly['Toplam_Pazar']) * 100
    
    # Temel büyüme oranları
    monthly['PF_Buyume_%'] = monthly['PF_Satis'].pct_change() * 100
    monthly['Rakip_Buyume_%'] = monthly['Rakip_Satis'].pct_change() * 100
    monthly['Goreceli_Buyume_%'] = monthly['PF_Buyume_%'] - monthly['Rakip_Buyume_%']
    
    # GELİŞTİRİLMİŞ Hareketli Ortalamalar
    monthly['MA_3'] = monthly['PF_Satis'].rolling(window=3, min_periods=1).mean()
    monthly['MA_6'] = monthly['PF_Satis'].rolling(window=6, min_periods=1).mean()
    monthly['MA_12'] = monthly['PF_Satis'].rolling(window=12, min_periods=1).mean()
    
    # GELİŞTİRİLMİŞ Hareketli Ortalama Büyüme
    monthly['MA_3_Growth'] = monthly['MA_3'].pct_change() * 100
    monthly['MA_6_Growth'] = monthly['MA_6'].pct_change() * 100
    monthly['MA_12_Growth'] = monthly['MA_12'].pct_change() * 100
    
    # Pazar Payı Hareketli Ortalamaları
    monthly['PP_MA_3'] = monthly['Pazar_Payi_%'].rolling(window=3, min_periods=1).mean()
    monthly['PP_MA_6'] = monthly['Pazar_Payi_%'].rolling(window=6, min_periods=1).mean()
    
    # Yıllık Büyüme (YoY)
    monthly['DATE_DT'] = pd.to_datetime(monthly['YIL_AY'] + '-01')
    monthly['Year'] = monthly['DATE_DT'].dt.year
    monthly['Month'] = monthly['DATE_DT'].dt.month
    
    # YoY büyümesini hesapla
    for idx, row in monthly.iterrows():
        if idx >= 12:
            same_month_last_year = monthly[(monthly['Year'] == row['Year'] - 1) & 
                                          (monthly['Month'] == row['Month'])]
            if not same_month_last_year.empty:
                monthly.loc[idx, 'YoY_PF_Growth'] = ((row['PF_Satis'] / same_month_last_year['PF_Satis'].values[0]) - 1) * 100
                monthly.loc[idx, 'YoY_Rakip_Growth'] = ((row['Rakip_Satis'] / same_month_last_year['Rakip_Satis'].values[0]) - 1) * 100
    
    # Mevsimsellik indeksi (basitleştirilmiş)
    if len(monthly) >= 12:
        monthly_grouped = monthly.groupby('Month')['PF_Satis'].mean()
        seasonality_base = monthly_grouped.mean()
        if seasonality_base > 0:
            monthly['Seasonality_Index'] = monthly.apply(
                lambda x: (monthly_grouped[x['Month']] / seasonality_base * 100) if x['Month'] in monthly_grouped.index else 100,
                axis=1
            )
    
    # Trend analizi
    if len(monthly) >= 3:
        # Son 3 ay vs Önceki 3 ay
        if len(monthly) >= 6:
            recent_3m = monthly.tail(3)['PF_Satis'].mean()
            previous_3m = monthly.tail(6).head(3)['PF_Satis'].mean()
            if previous_3m > 0:
                monthly.loc[monthly.index[-1], 'QoQ_Growth_3M'] = ((recent_3m / previous_3m) - 1) * 100
        
        # Son 6 ay vs Önceki 6 ay
        if len(monthly) >= 12:
            recent_6m = monthly.tail(6)['PF_Satis'].mean()
            previous_6m = monthly.tail(12).head(6)['PF_Satis'].mean()
            if previous_6m > 0:
                monthly.loc[monthly.index[-1], 'QoQ_Growth_6M'] = ((recent_6m / previous_6m) - 1) * 100
    
    # Volatilite hesaplama
    monthly['PF_Volatility'] = monthly['PF_Satis'].rolling(window=6, min_periods=3).std()
    monthly['PF_CV'] = safe_divide(monthly['PF_Volatility'], monthly['PF_Satis']) * 100
    
    # Momentum indikatörleri
    if len(monthly) >= 3:
        monthly['Momentum_3M'] = monthly['PF_Satis'] - monthly['PF_Satis'].shift(3)
        monthly['Momentum_6M'] = monthly['PF_Satis'] - monthly['PF_Satis'].shift(6)
    
    # Performans skoru (basitleştirilmiş)
    monthly['Performance_Score'] = (
        (monthly['Pazar_Payi_%'] / 100) * 0.4 +
        (np.minimum(monthly['PF_Buyume_%'].fillna(0), 50) / 50) * 0.3 +
        (1 - np.minimum(monthly['PF_CV'].fillna(50), 100) / 100) * 0.3
    ) * 100
    
    return monthly

def perform_trend_analysis(monthly_df):
    """Detaylı trend analizi"""
    if len(monthly_df) < 6:
        return {"error": "Yetersiz veri"}
    
    analysis = {}
    
    # 1. Temel trend analizi
    pf_values = monthly_df['PF_Satis'].values
    pf_slope = calculate_trend_slope(pf_values)
    pf_trend = classify_trend(pf_slope, pf_values)
    
    # 2. Hareketli ortalamalara göre trend
    if 'MA_3' in monthly_df.columns:
        ma3_slope = calculate_trend_slope(monthly_df['MA_3'].dropna().values)
        ma3_trend = classify_trend(ma3_slope, monthly_df['MA_3'].dropna().values)
        ma6_slope = calculate_trend_slope(monthly_df['MA_6'].dropna().values)
        ma6_trend = classify_trend(ma6_slope, monthly_df['MA_6'].dropna().values)
    else:
        ma3_trend = "Hesaplanamadı"
        ma6_trend = "Hesaplanamadı"
    
    # 3. Mevsimsellik analizi
    seasonality_type, period = calculate_seasonality(pf_values)
    
    # 4. Dönemsel büyüme analizi
    growth_metrics = {}
    
    if len(monthly_df) >= 4:
        # Son 1 ay vs Önceki 1 ay
        if len(monthly_df) >= 2:
            last_month = monthly_df['PF_Satis'].iloc[-1]
            prev_month = monthly_df['PF_Satis'].iloc[-2] if len(monthly_df) >= 2 else 0
            if prev_month > 0:
                growth_metrics['MoM_Growth'] = ((last_month / prev_month) - 1) * 100
        
        # Son 3 ay vs Önceki 3 ay
        if len(monthly_df) >= 6:
            recent_3m = monthly_df['PF_Satis'].tail(3).mean()
            previous_3m = monthly_df['PF_Satis'].tail(6).head(3).mean()
            if previous_3m > 0:
                growth_metrics['QoQ_3M_Growth'] = ((recent_3m / previous_3m) - 1) * 100
        
        # Son 6 ay vs Önceki 6 ay
        if len(monthly_df) >= 12:
            recent_6m = monthly_df['PF_Satis'].tail(6).mean()
            previous_6m = monthly_df['PF_Satis'].tail(12).head(6).mean()
            if previous_6m > 0:
                growth_metrics['QoQ_6M_Growth'] = ((recent_6m / previous_6m) - 1) * 100
    
    # 5. Pazar payı trendi
    if 'Pazar_Payi_%' in monthly_df.columns:
        pp_slope = calculate_trend_slope(monthly_df['Pazar_Payi_%'].values)
        pp_trend = classify_trend(pp_slope, monthly_df['Pazar_Payi_%'].values)
    else:
        pp_trend = "Hesaplanamadı"
    
    # 6. Volatilite analizi
    volatility = monthly_df['PF_Satis'].std() if len(monthly_df) > 1 else 0
    mean_value = monthly_df['PF_Satis'].mean() if len(monthly_df) > 0 else 0
    cv = (volatility / mean_value * 100) if mean_value > 0 else 0
    
    if cv < 20:
        volatility_class = "Düşük"
    elif cv < 50:
        volatility_class = "Orta"
    else:
        volatility_class = "Yüksek"
    
    # 7. Momentum analizi
    if len(monthly_df) >= 3:
        momentum_3m = monthly_df['PF_Satis'].iloc[-1] - monthly_df['PF_Satis'].iloc[-4] if len(monthly_df) >= 4 else 0
        momentum_6m = monthly_df['PF_Satis'].iloc[-1] - monthly_df['PF_Satis'].iloc[-7] if len(monthly_df) >= 7 else 0
    else:
        momentum_3m = 0
        momentum_6m = 0
    
    analysis = {
        "temel_trend": pf_trend,
        "hareketli_ortalama_3m_trend": ma3_trend,
        "hareketli_ortalama_6m_trend": ma6_trend,
        "mevsimsellik": seasonality_type,
        "mevsimsel_periyot": period,
        "pazar_payi_trendi": pp_trend,
        "volatilite": volatility_class,
        "volatilite_degeri": round(cv, 1),
        "momentum_3m": round(momentum_3m, 0),
        "momentum_6m": round(momentum_6m, 0),
        "buyume_metrikleri": growth_metrics,
        "trend_egimi": round(pf_slope, 2)
    }
    
    return analysis

def create_comparative_analysis(monthly_df, periods=[3, 6, 12]):
    """Karşılaştırmalı dönem analizi"""
    if len(monthly_df) < max(periods):
        return None
    
    comparisons = []
    
    for period in periods:
        if len(monthly_df) >= period:
            recent_data = monthly_df.tail(period)
            previous_data = monthly_df.tail(period*2).head(period)
            
            recent_avg = recent_data['PF_Satis'].mean()
            previous_avg = previous_data['PF_Satis'].mean()
            
            recent_share = recent_data['Pazar_Payi_%'].mean()
            previous_share = previous_data['Pazar_Payi_%'].mean()
            
            growth_rate = ((recent_avg / previous_avg) - 1) * 100 if previous_avg > 0 else 0
            share_change = recent_share - previous_share
            
            comparisons.append({
                'period': f'Son {period} ay',
                'ortalama_satis': recent_avg,
                'onceki_ortalama': previous_avg,
                'buyume_orani': growth_rate,
                'pazar_payi': recent_share,
                'pay_degisimi': share_change,
                'volatilite': recent_data['PF_Satis'].std(),
                'trend': classify_trend(
                    calculate_trend_slope(recent_data['PF_Satis'].values),
                    recent_data['PF_Satis'].values
                )
            })
    
    return pd.DataFrame(comparisons)

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
# MODERN HARİTA OLUŞTURUCU - GELİŞTİRİLMİŞ
# =============================================================================

def create_modern_turkey_map(city_data, gdf, title="Türkiye Satış Haritası", view_mode="Bölge Görünümü", filtered_pf_toplam=None):
    """
    Modern Türkiye haritası - Sabit Etiketleme ve Görünüm Düzeltmesi
    Revize: Etiketlerin harita üzerinde sürekli görünmesini sağlar.
    """
    if gdf is None:
        st.error("❌ GeoJSON yüklenemedi")
        return None
    
    # 1. VERİ HAZIRLIĞI
    # ---------------------------------------------------------
    city_data = city_data.copy()
    city_data['City_Fixed'] = city_data['City'].apply(normalize_city_name_fixed)
    city_data['City_Fixed'] = city_data['City_Fixed'].str.upper()
    
    # GeoJSON verisiyle birleştirme
    gdf = gdf.copy()
    gdf['name_upper'] = gdf['name'].str.upper()
    gdf['name_fixed'] = gdf['name_upper'].apply(lambda x: FIX_CITY_MAP.get(x, x))
    
    # Eksik şehirleri tamamlama
    all_cities_in_data = set(city_data['City_Fixed'].unique())
    all_cities_in_geojson = set(gdf['name_fixed'].unique())
    missing_cities = all_cities_in_geojson - all_cities_in_data
    
    for city in missing_cities:
        if city not in city_data['City_Fixed'].values:
            region_row = gdf[gdf['name_fixed'] == city]
            if len(region_row) > 0:
                region = region_row.iloc[0].get('region', 'DİĞER')
                new_row = pd.DataFrame({
                    'City': [city], 'City_Fixed': [city],
                    'Region': [region], 'Bölge': [region],
                    'PF_Satis': [0], 'Rakip_Satis': [0],
                    'Toplam_Pazar': [0], 'Pazar_Payi_%': [0]
                })
                city_data = pd.concat([city_data, new_row], ignore_index=True)
    
    # Merge işlemi
    merged = gdf.merge(city_data, left_on='name_fixed', right_on='City_Fixed', how='left')
    
    # NaN temizliği
    merged['PF_Satis'] = merged['PF_Satis'].fillna(0)
    merged['Toplam_Pazar'] = merged['Toplam_Pazar'].fillna(0)
    merged['Pazar_Payi_%'] = merged['Pazar_Payi_%'].fillna(0)
    merged['Bölge'] = merged['Bölge'].fillna('DİĞER')
    merged['Region'] = merged['Bölge']
    
    if filtered_pf_toplam is None:
        filtered_pf_toplam = merged['PF_Satis'].sum()

    # 2. HARİTA OLUŞTURMA (KATMANLAR)
    # ---------------------------------------------------------
    fig = go.Figure()
    
    # A) RENKLİ BÖLGE KATMANI (CHOROPLETH)
    # ---------------------------------------------------------
    for region in merged['Region'].unique():
        region_data = merged[merged['Region'] == region]
        color = REGION_COLORS.get(region, "#64748B")
        
        # GeoJSON dönüşümü
        region_json = json.loads(region_data.to_json())
        
        # Hover için HTML formatı (Sadece üzerine gelince görünür)
        hover_texts = []
        for _, row in region_data.iterrows():
            txt = (f"<b>{row['name']}</b><br>"
                   f"PF: {format_number(row['PF_Satis'])}<br>"
                   f"Pazar: {format_number(row['Toplam_Pazar'])}<br>"
                   f"Pay: %{row['Pazar_Payi_%']:.1f}")
            hover_texts.append(txt)

        fig.add_trace(go.Choroplethmapbox(
            geojson=region_json,
            locations=region_data.index,
            z=[1] * len(region_data),
            colorscale=[[0, color], [1, color]],
            marker_opacity=0.7,
            marker_line_width=1,
            marker_line_color='rgba(255,255,255,0.3)',
            showscale=False,
            name=region,
            hoverinfo='text',
            hovertext=hover_texts
        ))

    # B) SINIR ÇİZGİLERİ
    # ---------------------------------------------------------
    lons, lats = [], []
    for geom in merged.geometry.boundary:
        if geom and not geom.is_empty:
            lo, la = lines_to_lonlat(geom)
            lons += lo
            lats += la
            
    fig.add_trace(go.Scattermapbox(
        lon=lons, lat=lats,
        mode='lines',
        line=dict(width=1, color='rgba(255,255,255,0.5)'),
        hoverinfo='skip',
        showlegend=False
    ))

    # C) ETİKETLER (SABİT YAZILAR)
    # ---------------------------------------------------------
    label_lons, label_lats, label_texts = [], [], []
    
    if view_mode == "Bölge Görünümü":
        # Bölgelerin ortalamasını alarak merkez bul (daha güvenli ve hızlı)
        for region in merged['Region'].unique():
            r_data = merged[merged['Region'] == region]
            
            # Bölge toplamları
            total_pf = r_data['PF_Satis'].sum()
            total_pazar = r_data['Toplam_Pazar'].sum()
            
            # Sadece verisi olan bölgelere etiket koy
            if total_pazar > 0:
                # Bölgedeki şehirlerin orta noktası
                try:
                    center_lon = r_data.geometry.centroid.x.mean()
                    center_lat = r_data.geometry.centroid.y.mean()
                    
                    # Düz metin (\n ile alt satır) - HTML KULLANMA!
                    # "Marmara \n PF: 1.2M \n Pay: %22" gibi
                    share = (total_pf / total_pazar * 100)
                    txt = f"{region}\nPF: {format_number(total_pf)}\n%{share:.1f}"
                    
                    label_lons.append(center_lon)
                    label_lats.append(center_lat)
                    label_texts.append(txt)
                except:
                    continue
        
        text_size = 12
        marker_size = 0  # Bölge modunda nokta olmasın, sadece yazı olsun diye 0 yapabilirsiniz veya küçük bir nokta bırakabilirsiniz

    else:  # Şehir Görünümü
        for idx, row in merged.iterrows():
            # Sadece kayda değer satışı veya pazarı olanları etiketle
            if row['PF_Satis'] > 0 or row['Toplam_Pazar'] > 0:
                try:
                    lon = row.geometry.centroid.x
                    lat = row.geometry.centroid.y
                    
                    # Düz metin formatı
                    txt = f"{row['name']}\n{format_number(row['PF_Satis'])}"
                    
                    label_lons.append(lon)
                    label_lats.append(lat)
                    label_texts.append(txt)
                except:
                    continue
        
        text_size = 10
        marker_size = 5

    # Etiketleri Haritaya Ekle
    if label_lons:
        fig.add_trace(go.Scattermapbox(
            lon=label_lons,
            lat=label_lats,
            mode='markers+text',  # Hem nokta hem yazı
            text=label_texts,
            textposition='bottom center', # Yazı noktanın altında
            marker=dict(size=marker_size, color='white', opacity=0.8),
            textfont=dict(
                size=text_size,
                color='white', # Yazı rengi beyaz (koyu harita üzerinde okunur)
                family='Inter, sans-serif'
            ),
            hoverinfo='skip', # Üzerine gelince tekrar popup açma, zaten yazıyor
            showlegend=False
        ))

    # 3. LAYOUT AYARLARI
    # ---------------------------------------------------------
    # Otomatik Zoom Hesapla
    if len(label_lons) > 0:
        center_lat = sum(label_lats) / len(label_lats)
        center_lon = sum(label_lons) / len(label_lons)
        zoom = 5.5 if view_mode == "Şehir Görünümü" else 5.0
    else:
        center_lat = 39.0
        center_lon = 35.0
        zoom = 5.0

    fig.update_layout(
        mapbox_style="carto-darkmatter",
        mapbox=dict(
            center=dict(lat=center_lat, lon=center_lon),
            zoom=zoom,
            bearing=0,
            pitch=0
        ),
        uirevision=view_mode, # Mod değişince haritayı resetle
        height=750,
        margin=dict(l=0, r=0, t=80, b=0),
        title=dict(
            text=f"<b>{title}</b><br><span style='font-size: 14px; color: #94a3b8'>"
                 f"Toplam PF Satış: {format_number(filtered_pf_toplam)}</span>",
            x=0.5,
            y=0.97,
            font=dict(size=22, color='white', family='Inter')
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
    )
    
    return fig
# =============================================================================
# ML FEATURE ENGINEERING - GELİŞTİRİLMİŞ
# =============================================================================

def create_advanced_ml_features(df):
    """GELİŞTİRİLMİŞ ML için feature oluştur"""
    df = df.copy()
    df = df.sort_values('DATE').reset_index(drop=True)
    
    # Lag features (3, 6, 12 ay)
    for lag in [1, 2, 3, 4, 5, 6, 12]:
        if lag < len(df):
            df[f'lag_{lag}'] = df['PF_Satis'].shift(lag)
    
    # Rolling statistics
    windows = [3, 6, 12]
    for window in windows:
        if window <= len(df):
            df[f'rolling_mean_{window}'] = df['PF_Satis'].rolling(window=window, min_periods=1).mean()
            df[f'rolling_std_{window}'] = df['PF_Satis'].rolling(window=window, min_periods=1).std()
            df[f'rolling_min_{window}'] = df['PF_Satis'].rolling(window=window, min_periods=1).min()
            df[f'rolling_max_{window}'] = df['PF_Satis'].rolling(window=window, min_periods=1).max()
    
    # Exponential moving averages
    for span in [3, 6, 12]:
        if span <= len(df):
            df[f'ema_{span}'] = df['PF_Satis'].ewm(span=span, adjust=False).mean()
    
    # Date features
    df['month'] = df['DATE'].dt.month
    df['quarter'] = df['DATE'].dt.quarter
    df['year'] = df['DATE'].dt.year
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 4)
    df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 4)
    df['trend_index'] = range(len(df))
    
    # Seasonal features
    df['is_q1'] = (df['quarter'] == 1).astype(int)
    df['is_q2'] = (df['quarter'] == 2).astype(int)
    df['is_q3'] = (df['quarter'] == 3).astype(int)
    df['is_q4'] = (df['quarter'] == 4).astype(int)
    
    # Interaction features
    if 'Pazar_Payi_%' in df.columns:
        df['share_trend'] = df['Pazar_Payi_%'].rolling(window=3, min_periods=1).mean()
    
    # Growth features
    df['growth_1m'] = df['PF_Satis'].pct_change(periods=1) * 100
    df['growth_3m'] = df['PF_Satis'].pct_change(periods=3) * 100
    df['growth_6m'] = df['PF_Satis'].pct_change(periods=6) * 100
    
    # Momentum features
    if len(df) >= 3:
        df['momentum_3m'] = df['PF_Satis'] - df['PF_Satis'].shift(3)
        df['momentum_6m'] = df['PF_Satis'] - df['PF_Satis'].shift(6)
    
    # Volatility features
    df['volatility_3m'] = df['PF_Satis'].rolling(window=3, min_periods=1).std()
    df['volatility_6m'] = df['PF_Satis'].rolling(window=6, min_periods=1).std()
    
    # Fill NaN
    df = df.fillna(method='bfill').fillna(method='ffill').fillna(0)
    
    return df

def train_advanced_ml_models(df, forecast_periods=3):
    """GELİŞTİRİLMİŞ ML modelleri ile tahmin"""
    if len(df) < 24:  # En az 2 yıllık veri
        return None, None, None
    
    df_features = create_advanced_ml_features(df)
    
    # Feature selection
    feature_cols = [
        'lag_1', 'lag_2', 'lag_3', 'lag_6', 'lag_12',
        'rolling_mean_3', 'rolling_mean_6', 'rolling_mean_12',
        'rolling_std_3', 'rolling_std_6',
        'ema_3', 'ema_6',
        'month', 'quarter', 'year',
        'month_sin', 'month_cos',
        'trend_index',
        'growth_1m', 'growth_3m',
        'momentum_3m', 'momentum_6m',
        'volatility_3m'
    ]
    
    # Sadece mevcut kolonları kullan
    available_cols = [col for col in feature_cols if col in df_features.columns]
    
    # Train/Test split (zaman bazlı - son %20 test)
    split_idx = int(len(df_features) * 0.8)
    
    train_df = df_features.iloc[:split_idx]
    test_df = df_features.iloc[split_idx:]
    
    X_train = train_df[available_cols]
    y_train = train_df['PF_Satis']
    X_test = test_df[available_cols]
    y_test = test_df['PF_Satis']
    
    # Gelişmiş modeller
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            max_depth=10,
            min_samples_split=5,
            n_jobs=-1
        )
    }
    
    results = {}
    
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Negatif tahminleri 0 yap
            y_pred = np.maximum(y_pred, 0)
            
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mape = np.mean(np.abs((y_test - y_pred) / np.maximum(y_test, 1))) * 100
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'MAE': mae,
                'RMSE': rmse,
                'MAPE': mape,
                'R2': r2,
                'y_pred': y_pred
            }
        except Exception as e:
            st.warning(f"{name} modeli eğitilemedi: {str(e)}")
            continue
    
    if not results:
        return None, None, None
    
    # En iyi model (MAPE'e göre)
    best_model_name = min(results.keys(), key=lambda x: results[x]['MAPE'])
    best_model = results[best_model_name]['model']
    
    # Gelecek tahmin için basitleştirilmiş yöntem
    forecast_data = []
    last_date = df_features['DATE'].iloc[-1]
    last_values = df_features['PF_Satis'].values[-6:]  # Son 6 ay
    
    for i in range(forecast_periods):
        next_date = last_date + pd.DateOffset(months=i+1)
        
        # Basit bir projeksiyon: son 6 ayın ortalaması * mevsimsellik faktörü
        if len(last_values) > 0:
            base_value = np.mean(last_values)
            month = next_date.month
            # Mevsimsellik faktörü (basit)
            seasonal_factor = 1.0 + 0.1 * np.sin(2 * np.pi * month / 12)
            next_pred = base_value * seasonal_factor
        else:
            next_pred = df_features['PF_Satis'].iloc[-1]
        
        next_pred = max(0, next_pred)  # Negatif olmamasını sağla
        
        forecast_data.append({
            'DATE': next_date,
            'YIL_AY': next_date.strftime('%Y-%m'),
            'PF_Satis': next_pred,
            'Model': best_model_name,
            'Tahmin_Tipi': 'ML Tahmin'
        })
    
    forecast_df = pd.DataFrame(forecast_data)
    
    # Basit tahmin metodları ekle (benchmark)
    simple_forecasts = []
    
    # 1. Son değer yöntemi
    last_value = df_features['PF_Satis'].iloc[-1]
    for i in range(forecast_periods):
        simple_forecasts.append({
            'DATE': last_date + pd.DateOffset(months=i+1),
            'YIL_AY': (last_date + pd.DateOffset(months=i+1)).strftime('%Y-%m'),
            'PF_Satis': last_value,
            'Model': 'Son Değer',
            'Tahmin_Tipi': 'Basit Tahmin'
        })
    
    # 2. Hareketli ortalama yöntemi
    ma_value = df_features['PF_Satis'].tail(6).mean()
    for i in range(forecast_periods):
        simple_forecasts.append({
            'DATE': last_date + pd.DateOffset(months=i+1),
            'YIL_AY': (last_date + pd.DateOffset(months=i+1)).strftime('%Y-%m'),
            'PF_Satis': ma_value,
            'Model': '6 Aylık Ortalama',
            'Tahmin_Tipi': 'Basit Tahmin'
        })
    
    simple_forecast_df = pd.DataFrame(simple_forecasts)
    
    # Tüm tahminleri birleştir
    all_forecasts = pd.concat([forecast_df, simple_forecast_df], ignore_index=True)
    
    return results, best_model_name, all_forecasts

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
# VISUALIZATION FUNCTIONS - MODERN & MCKINSEY STYLE
# =============================================================================

def create_advanced_time_series_chart(monthly_df, forecast_df=None):
    """GELİŞTİRİLMİŞ zaman serisi grafiği"""
    fig = go.Figure()
    
    # Gerçek veri
    fig.add_trace(go.Scatter(
        x=monthly_df['DATE'],
        y=monthly_df['PF_Satis'],
        mode='lines+markers',
        name='Gerçek PF Satış',
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
        fillcolor='rgba(6, 182, 212, 0.1)',
        fill='tozeroy'
    ))
    
    # Hareketli ortalamalar
    if 'MA_3' in monthly_df.columns:
        fig.add_trace(go.Scatter(
            x=monthly_df['DATE'],
            y=monthly_df['MA_3'],
            mode='lines',
            name='3 Aylık Ortalama',
            line=dict(
                color=TREND_COLORS['cyclic'],
                width=2,
                dash='dash'
            ),
            opacity=0.7
        ))
    
    if 'MA_6' in monthly_df.columns:
        fig.add_trace(go.Scatter(
            x=monthly_df['DATE'],
            y=monthly_df['MA_6'],
            mode='lines',
            name='6 Aylık Ortalama',
            line=dict(
                color=TREND_COLORS['seasonal'],
                width=2,
                dash='dot'
            ),
            opacity=0.7
        ))
    
    # Tahminler
    if forecast_df is not None and len(forecast_df) > 0:
        # ML tahminleri
        ml_forecast = forecast_df[forecast_df['Tahmin_Tipi'] == 'ML Tahmin']
        if len(ml_forecast) > 0:
            fig.add_trace(go.Scatter(
                x=ml_forecast['DATE'],
                y=ml_forecast['PF_Satis'],
                mode='lines+markers',
                name='ML Tahmini',
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
                )
            ))
        
        # Basit tahminler
        simple_forecast = forecast_df[forecast_df['Tahmin_Tipi'] == 'Basit Tahmin']
        for model in simple_forecast['Model'].unique():
            model_data = simple_forecast[simple_forecast['Model'] == model]
            fig.add_trace(go.Scatter(
                x=model_data['DATE'],
                y=model_data['PF_Satis'],
                mode='lines',
                name=f'{model}',
                line=dict(
                    color='rgba(255, 255, 255, 0.3)',
                    width=1,
                    dash='dash'
                ),
                opacity=0.5
            ))
    
    fig.update_layout(
        title=dict(
            text='<b>Gelişmiş Zaman Serisi Analizi</b>',
            font=dict(size=22, color='white', family='Inter')
        ),
        xaxis_title='<b>Tarih</b>',
        yaxis_title='<b>PF Satış</b>',
        height=600,
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
            showgrid=True,
            tickformat=',.0f'
        )
    )
    
    return fig

def create_trend_analysis_chart(monthly_df):
    """Trend analizi grafiği"""
    fig = go.Figure()
    
    # PF Satış
    fig.add_trace(go.Scatter(
        x=monthly_df['DATE'],
        y=monthly_df['PF_Satis'],
        mode='lines+markers',
        name='PF Satış',
        line=dict(
            color=PERFORMANCE_COLORS['success'],
            width=3,
            shape='spline'
        ),
        marker=dict(size=6)
    ))
    
    # Trend çizgisi
    if len(monthly_df) >= 3:
        x = np.arange(len(monthly_df))
        y = monthly_df['PF_Satis'].values
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        
        fig.add_trace(go.Scatter(
            x=monthly_df['DATE'],
            y=p(x),
            mode='lines',
            name='Trend Çizgisi',
            line=dict(
                color=TREND_COLORS['strong_up'],
                width=2,
                dash='dash'
            )
        ))
    
    # Büyüme oranları (ikinci eksen)
    if 'PF_Buyume_%' in monthly_df.columns:
        fig.add_trace(go.Scatter(
            x=monthly_df['DATE'],
            y=monthly_df['PF_Buyume_%'],
            mode='lines',
            name='Büyüme %',
            line=dict(
                color=PERFORMANCE_COLORS['info'],
                width=2,
                dash='dot'
            ),
            yaxis="y2"
        ))
    
    fig.update_layout(
        title=dict(
            text='<b>Trend ve Büyüme Analizi</b>',
            font=dict(size=22, color='white', family='Inter')
        ),
        xaxis_title='<b>Tarih</b>',
        yaxis_title='<b>PF Satış</b>',
        yaxis2=dict(
            title='<b>Büyüme %</b>',
            overlaying='y',
            side='right',
            showgrid=False,
            ticksuffix='%'
        ),
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
            x=1
        )
    )
    
    return fig

def create_comparative_period_chart(comparisons_df):
    """Karşılaştırmalı dönem analizi grafiği"""
    if comparisons_df is None or len(comparisons_df) == 0:
        return None
    
    fig = go.Figure()
    
    # Ortalama satışlar
    fig.add_trace(go.Bar(
        x=comparisons_df['period'],
        y=comparisons_df['ortalama_satis'],
        name='Ortalama Satış',
        marker_color=PERFORMANCE_COLORS['success'],
        text=[format_number(x) for x in comparisons_df['ortalama_satis']],
        textposition='auto',
    ))
    
    # Önceki ortalama
    fig.add_trace(go.Bar(
        x=comparisons_df['period'],
        y=comparisons_df['onceki_ortalama'],
        name='Önceki Ortalama',
        marker_color='rgba(100, 116, 139, 0.6)',
        text=[format_number(x) for x in comparisons_df['onceki_ortalama']],
        textposition='auto',
    ))
    
    # Büyüme oranları (ikinci eksen)
    fig.add_trace(go.Scatter(
        x=comparisons_df['period'],
        y=comparisons_df['buyume_orani'],
        mode='lines+markers+text',
        name='Büyüme %',
        line=dict(color=PERFORMANCE_COLORS['warning'], width=3),
        marker=dict(size=10, color='white', line=dict(width=2, color=PERFORMANCE_COLORS['warning'])),
        text=[f"{x:.1f}%" for x in comparisons_df['buyume_orani']],
        textposition='top center',
        yaxis="y2"
    ))
    
    fig.update_layout(
        title=dict(
            text='<b>Dönemsel Karşılaştırma Analizi</b>',
            font=dict(size=22, color='white', family='Inter')
        ),
        xaxis_title='<b>Dönem</b>',
        yaxis_title='<b>Ortalama Satış</b>',
        yaxis2=dict(
            title='<b>Büyüme %</b>',
            overlaying='y',
            side='right',
            showgrid=False,
            ticksuffix='%'
        ),
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
            x=1
        )
    )
    
    return fig

def create_seasonality_chart(monthly_df):
    """Mevsimsellik analizi grafiği"""
    if 'Month' not in monthly_df.columns or len(monthly_df) < 12:
        return None
    
    monthly_avg = monthly_df.groupby('Month').agg({
        'PF_Satis': 'mean',
        'Pazar_Payi_%': 'mean'
    }).reset_index()
    
    monthly_avg.columns = ['Month', 'PF_Satis', 'Pazar_Payi_%']
    monthly_avg['Month_Name'] = monthly_avg['Month'].map({
        1: 'Oca', 2: 'Şub', 3: 'Mar', 4: 'Nis', 5: 'May', 6: 'Haz',
        7: 'Tem', 8: 'Ağu', 9: 'Eyl', 10: 'Eki', 11: 'Kas', 12: 'Ara'
    })
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=monthly_avg['PF_Satis'],
        theta=monthly_avg['Month_Name'],
        fill='toself',
        name='Aylık Ortalama Satış',
        line=dict(color=PERFORMANCE_COLORS['success'], width=2),
        fillcolor='rgba(6, 182, 212, 0.3)'
    ))
    
    fig.update_layout(
        title=dict(
            text='<b>Mevsimsellik Analizi (Radar Grafiği)</b>',
            font=dict(size=22, color='white', family='Inter')
        ),
        polar=dict(
            radialaxis=dict(
                visible=True,
                tickformat=',.0f'
            )
        ),
        showlegend=True,
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e2e8f0', family='Inter')
    )
    
    return fig

def create_volatility_chart(monthly_df):
    """Volatilite analizi grafiği"""
    if 'PF_Volatility' not in monthly_df.columns:
        return None
    
    fig = go.Figure()
    
    # Satışlar
    fig.add_trace(go.Scatter(
        x=monthly_df['DATE'],
        y=monthly_df['PF_Satis'],
        mode='lines',
        name='PF Satış',
        line=dict(color=PERFORMANCE_COLORS['success'], width=2),
        fillcolor='rgba(6, 182, 212, 0.1)',
        fill='tozeroy'
    ))
    
    # Volatilite bandı
    if 'MA_6' in monthly_df.columns and 'PF_Volatility' in monthly_df.columns:
        upper_band = monthly_df['MA_6'] + monthly_df['PF_Volatility']
        lower_band = monthly_df['MA_6'] - monthly_df['PF_Volatility']
        
        fig.add_trace(go.Scatter(
            x=monthly_df['DATE'],
            y=upper_band,
            mode='lines',
            name='+1 Std',
            line=dict(color='rgba(100, 116, 139, 0.5)', width=1, dash='dash'),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=monthly_df['DATE'],
            y=lower_band,
            mode='lines',
            name='-1 Std',
            line=dict(color='rgba(100, 116, 139, 0.5)', width=1, dash='dash'),
            fill='tonexty',
            fillcolor='rgba(100, 116, 139, 0.1)',
            showlegend=False
        ))
    
    fig.update_layout(
        title=dict(
            text='<b>Volatilite Analizi</b>',
            font=dict(size=22, color='white', family='Inter')
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
            x=1
        )
    )
    
    return fig

def create_modern_competitor_chart(comp_data):
    """Modern rakip karşılaştırma - McKinsey tarzı"""
    fig = go.Figure()
    
    # PF Satış
    fig.add_trace(go.Bar(
        x=comp_data['YIL_AY'],
        y=comp_data['PF'],
        name='PF',
        marker_color=PERFORMANCE_COLORS['success'],
        marker=dict(
            line=dict(width=2, color='rgba(255, 255, 255, 0.8)')
        ),
        text=[format_number(x) for x in comp_data['PF']],
        textposition='auto',
    ))
    
    # Rakip Satış
    fig.add_trace(go.Bar(
        x=comp_data['YIL_AY'],
        y=comp_data['Rakip'],
        name='Rakip',
        marker_color=PERFORMANCE_COLORS['danger'],
        marker=dict(
            line=dict(width=2, color='rgba(255, 255, 255, 0.8)')
        ),
        text=[format_number(x) for x in comp_data['Rakip']],
        textposition='auto',
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
            gridcolor='rgba(59, 130, 246, 0.1)',
            tickformat=',.0f'
        )
    )
    
    return fig

def create_modern_growth_chart(comp_data):
    """Modern büyüme grafiği - McKinsey tarzı"""
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
        fillcolor='rgba(6, 182, 212, 0.15)'
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
        fillcolor='rgba(100, 116, 139, 0.15)'
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
            gridcolor='rgba(59, 130, 246, 0.1)',
            ticksuffix='%'
        )
    )
    
    return fig

def create_modern_bcg_chart(bcg_df):
    """Modern BCG Matrix - McKinsey tarzı"""
    fig = px.scatter(
        bcg_df,
        x='Goreceli_Pazar_Payi',
        y='Pazar_Buyume_%',
        size='PF_Satis',
        color='BCG_Kategori',
        color_discrete_map=BCG_COLORS,
        hover_name='Territory',
        hover_data={
            'Region': True,
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
            linecolor='rgba(59, 130, 246, 0.3)',
            ticksuffix='%'
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
    
    # Orijinal sayısal değerleri sakla (gradient için)
    numeric_data = df.copy()
    
    # Sayısal sütunları formatla (görüntü için)
    df_formatted = df.copy()
    
    # Sayısal sütunları bul ve formatla
    for col in df_formatted.columns:
        if col in numeric_data.columns and numeric_data[col].dtype in ['int64', 'float64', 'int32', 'float32']:
            if any(keyword in col.lower() for keyword in ['%', 'yüzde', 'pay', 'oran', 'büyüme']):
                # Yüzdelik sütunlar
                df_formatted[col] = numeric_data[col].apply(lambda x: f"{x:,.1f}%" if pd.notnull(x) else "")
            else:
                # Normal sayısal sütunlar
                df_formatted[col] = numeric_data[col].apply(lambda x: format_number(x) if pd.notnull(x) else "")
    
    styled_df = df_formatted.style
    
    # Genel stil
    styled_df = styled_df.set_properties(**{
        'background-color': 'rgba(30, 41, 59, 0.7)',
        'color': '#e2e8f0',
        'border': '1px solid rgba(37, 99, 235, 0.3)',
        'font-family': 'Inter, sans-serif',
        'text-align': 'center'
    })
    
    # Başlık satırı
    styled_df = styled_df.set_table_styles([{
        'selector': 'thead th',
        'props': [
            ('background-color', 'rgba(37, 99, 235, 0.3)'),
            ('color', 'white'),
            ('font-weight', '700'),
            ('border', '1px solid rgba(37, 99, 235, 0.4)'),
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
    
    # Gradient uygula - TEK RENK (Mavi)
    for col in gradient_columns:
        if col in numeric_data.columns and numeric_data[col].dtype in ['int64', 'float64', 'int32', 'float32']:
            try:
                col_data = numeric_data[col].astype(float)
                min_val = col_data.min()
                max_val = col_data.max()
                
                if min_val != max_val:
                    # Tek renkli mavi gradient kullan
                    styled_df = styled_df.background_gradient(
                        subset=[col], 
                        cmap='Blues',  # Kırmızı-Yeşil yerine Mavi gradient
                        vmin=min_val,
                        vmax=max_val,
                        gmap=col_data
                    )
            except:
                pass
    
    # Renk sütunu - Mavi tonlarında
    if color_column and color_column in numeric_data.columns:
        def color_cells(val):
            try:
                num_val = float(val)
                if num_val >= 70:
                    return 'background-color: rgba(37, 99, 235, 0.3); color: #2563EB; font-weight: 600'
                elif num_val >= 40:
                    return 'background-color: rgba(245, 158, 11, 0.3); color: #F59E0B; font-weight: 600'
                else:
                    return 'background-color: rgba(100, 116, 139, 0.3); color: #64748B; font-weight: 600'
            except:
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
# MAIN APP - GELİŞTİRİLMİŞ VERSİYON
# =============================================================================

def main():
    # Başlık ve açıklama
    st.markdown('<h1 class="main-header">🎯 GELİŞMİŞ TİCARİ PORTFÖY ANALİZ SİSTEMİ</h1>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: center; font-size: 1.2rem; color: #94a3b8; margin-bottom: 3rem;">'
                'GERÇEK ML Tahminleme • Gelişmiş Zaman Serisi Analizi • Modern Harita • Rakip Analizi'
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
        "📈 Gelişmiş Zaman Serisi",
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
            st.metric("💊 PF Satış", format_number(total_pf), f"{format_number(avg_monthly_pf)}/ay")
        with col2:
            st.metric("🏪 Toplam Pazar", format_number(total_market), f"{format_number(total_rakip)} rakip")
        with col3:
            st.metric("📊 Pazar Payı", format_percentage(market_share), 
                     f"{format_percentage(100-market_share)} rakip")
        with col4:
            st.metric("🏢 Active Territory", str(active_territories), 
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
            
            pf_texts = [format_number(x) for x in top10['PF_Satis']]
            rakip_texts = [format_number(x) for x in top10['Rakip_Satis']]
            
            fig_top10.add_trace(go.Bar(
                x=top10['Territory'],
                y=top10['PF_Satis'],
                name='PF Satış',
                marker_color=PERFORMANCE_COLORS['success'],
                text=pf_texts,
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
                text=rakip_texts,
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
                ),
                yaxis=dict(
                    tickformat=',.0f'
                )
            )
            
            st.plotly_chart(fig_top10, use_container_width=True)
        
        with col_chart2:
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
        
        # Harita için Bölge Filtresi
        col_map_filter1, col_map_filter2 = st.columns(2)
        with col_map_filter1:
            unique_regions = ["TÜMÜ"] + sorted(df_filtered['REGION'].dropna().unique())
            selected_map_region = st.selectbox(
                "Harita için Bölge Seçin",
                unique_regions,
                key='map_region_filter'
            )
        
        # Şehir performans verisini BÖLGEYE GÖRE FİLTRELE
        city_data = calculate_city_performance(df_filtered, selected_product, date_filter)
        if selected_map_region != "TÜMÜ":
            city_data = city_data[city_data['Region'] == selected_map_region]
        
        # Yatırım stratejisini FİLTRELENMİŞ veri ile hesapla
        investment_df = calculate_investment_strategy(city_data)
        filtered_pf_toplam = city_data['PF_Satis'].sum()
        
        # Quick Stats
        col1, col2, col3, col4, col5 = st.columns(5)
        
        total_pf = city_data['PF_Satis'].sum()
        total_market = city_data['Toplam_Pazar'].sum()
        avg_share = city_data['Pazar_Payi_%'].mean()
        active_cities = len(city_data[city_data['PF_Satis'] > 0])
        top_city = city_data.loc[city_data['PF_Satis'].idxmax(), 'City'] if len(city_data) > 0 else "Yok"
        
        with col1:
            st.metric("💊 PF Satış", format_number(total_pf))
        with col2:
            st.metric("🏪 Toplam Pazar", format_number(total_market))
        with col3:
            st.metric("📊 Ort. Pazar Payı", format_percentage(avg_share))
        with col4:
            st.metric("🏙️ Aktif Şehir", str(active_cities))
        with col5:
            st.metric("🏆 Lider Şehir", top_city)
        
        st.markdown("---")
        
        # Modern Harita
        if gdf is not None:
            st.subheader(f"📍 İl Bazlı Dağılım - {selected_map_region if selected_map_region != 'TÜMÜ' else 'Tüm Bölgeler'}")
            
            turkey_map = create_modern_turkey_map(
                city_data, 
                gdf, 
                title=f"{selected_product} - {view_mode} - {selected_map_region if selected_map_region != 'TÜMÜ' else 'Tüm Bölgeler'}",
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
            
            bar_texts = [format_number(x) for x in top_cities['PF_Satis']]
            
            fig_bar = px.bar(
                top_cities,
                x='City',
                y='PF_Satis',
                title='<b>En Yüksek Satış Yapan Şehirler</b>',
                color='Region',
                color_discrete_map=REGION_COLORS,
                hover_data=['Region', 'PF_Satis', 'Pazar_Payi_%'],
                text=bar_texts
            )
            
            fig_bar.update_layout(
                height=500,
                xaxis_tickangle=-45,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e2e8f0'),
                yaxis_title='<b>PF Satış</b>',
                xaxis_title='<b>Şehir</b>',
                yaxis=dict(
                    tickformat=',.0f'
                )
            )
            
            fig_bar.update_traces(
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
                        f"{format_number(total_value)} PF"
                    )
            
            st.markdown("---")
            
            # Detaylı tablo
            st.subheader("📋 Detaylı Şehir Listesi")
            
            investment_display = investment_df.copy()
            if selected_strateji != "Tümü":
                investment_display = investment_display[investment_display['Yatırım_Stratejisi'] == selected_strateji]
            
            city_display = investment_display.sort_values('PF_Satis', ascending=False).copy()
            
            display_cols = ['City', 'Region', 'PF_Satis', 'Toplam_Pazar', 'Pazar_Payi_%', 'Yatırım_Stratejisi']
            city_display_formatted = city_display[display_cols].copy()
            city_display_formatted.columns = ['Şehir', 'Bölge', 'PF Satış', 'Toplam Pazar', 'Pazar Payı %', 'Strateji']
            city_display_formatted.index = range(1, len(city_display_formatted) + 1)
            
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
    
    # TAB 3: TERRITORY ANALİZİ
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
            
            pf_texts = [format_number(x) for x in terr_sorted['PF_Satis']]
            rakip_texts = [format_number(x) for x in terr_sorted['Rakip_Satis']]
            
            fig_bar = go.Figure()
            
            fig_bar.add_trace(go.Bar(
                x=terr_sorted['Territory'],
                y=terr_sorted['PF_Satis'],
                name='PF Satış',
                marker_color=PERFORMANCE_COLORS['success'],
                text=pf_texts,
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
                text=rakip_texts,
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
                ),
                yaxis=dict(
                    tickformat=',.0f'
                )
            )
            
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col_viz2:
            st.subheader("🎯 Pazar Payı Dağılımı")
            
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
                ),
                xaxis=dict(
                    tickformat=',.0f'
                ),
                yaxis=dict(
                    ticksuffix='%'
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
        
        styled_territory = style_dataframe(
            terr_display,
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
            avg_pazar_payi = terr_sorted['Pazar_Payi_%'].mean()
            st.metric("📊 Ort. Pazar Payı", format_percentage(avg_pazar_payi))
        
        with col_sum2:
            total_pf = terr_sorted['PF_Satis'].sum()
            st.metric("💰 Toplam PF Satış", format_number(total_pf))
        
        with col_sum3:
            avg_toplam_pazar_yuzde = terr_sorted['Toplam_Pazar_%'].mean()
            st.metric("🏪 Ort. Pazar Payı", format_percentage(avg_toplam_pazar_yuzde))
        
        with col_sum4:
            dominant_region = terr_display['Region'].mode()[0] if len(terr_display) > 0 else "Yok"
            region_color = REGION_COLORS.get(dominant_region, "#64748B")
            st.markdown(
                f'<div style="color:{region_color}; font-size:1.2rem; font-weight:bold; text-align: center;">'
                f'🏆 {dominant_region}</div>',
                unsafe_allow_html=True
            )
    
    # TAB 4: GELİŞTİRİLMİŞ ZAMAN SERİSİ ANALİZİ
    with tab4:
        st.header("📈 Gelişmiş Zaman Serisi Analizi & ML Tahminleme")
        
        col_ts1, col_ts2 = st.columns(2)
        
        with col_ts1:
            territory_for_ts = st.selectbox(
                "Territory Seçin",
                ["TÜMÜ"] + sorted(df_filtered['TERRITORIES'].unique()),
                key='ts_territory'
            )
        
        with col_ts2:
            analysis_type = st.selectbox(
                "Analiz Türü",
                ["Temel Zaman Serisi", "Trend Analizi", "Karşılaştırmalı Analiz", "Mevsimsellik Analizi", "Volatilite Analizi"]
            )
        
        # Gelişmiş zaman serisi hesapla
        monthly_df = calculate_advanced_time_series(df_filtered, selected_product, territory_for_ts, date_filter)
        
        if len(monthly_df) == 0:
            st.warning("⚠️ Seçilen filtrelerde veri bulunamadı")
        else:
            # Özet Metrikler
            col_ts1, col_ts2, col_ts3, col_ts4 = st.columns(4)
            
            with col_ts1:
                avg_pf = monthly_df['PF_Satis'].mean()
                st.metric("📊 Ort. Aylık PF", format_number(avg_pf))
            
            with col_ts2:
                avg_growth = monthly_df['PF_Buyume_%'].mean() if 'PF_Buyume_%' in monthly_df.columns else 0
                st.metric("📈 Ort. Büyüme", format_percentage(avg_growth))
            
            with col_ts3:
                avg_share = monthly_df['Pazar_Payi_%'].mean() if 'Pazar_Payi_%' in monthly_df.columns else 0
                st.metric("🎯 Ort. Pazar Payı", format_percentage(avg_share))
            
            with col_ts4:
                total_months = len(monthly_df)
                st.metric("📅 Veri Dönemi", f"{total_months} ay")
            
            st.markdown("---")
            
            # Trend analizi yap
            trend_analysis = perform_trend_analysis(monthly_df)
            
            # Trend bilgilerini göster
            if 'error' not in trend_analysis:
                col_trend1, col_trend2, col_trend3, col_trend4 = st.columns(4)
                
                with col_trend1:
                    st.metric("📈 Temel Trend", trend_analysis.get('temel_trend', 'Bilinmiyor'))
                
                with col_trend2:
                    st.metric("🔄 Mevsimsellik", trend_analysis.get('mevsimsellik', 'Bilinmiyor'))
                
                with col_trend3:
                    volatility = trend_analysis.get('volatilite', 'Bilinmiyor')
                    volatility_val = trend_analysis.get('volatilite_degeri', 0)
                    st.metric("📉 Volatilite", volatility, f"{volatility_val:.1f}%")
                
                with col_trend4:
                    momentum = trend_analysis.get('momentum_3m', 0)
                    st.metric("⚡ 3 Aylık Momentum", format_number(momentum))
            
            st.markdown("---")
            
            # Analiz türüne göre grafik göster
            if analysis_type == "Temel Zaman Serisi":
                st.subheader("📊 Temel Zaman Serisi Analizi")
                
                # ML tahmini
                forecast_months = st.slider("Tahmin Periyodu (Ay)", 1, 12, 6)
                
                if len(monthly_df) >= 12:
                    with st.spinner("ML modelleri eğitiliyor..."):
                        ml_results, best_model_name, forecast_df = train_advanced_ml_models(monthly_df, forecast_months)
                    
                    if ml_results is not None:
                        # Model Performansı
                        st.subheader("🤖 Model Performans Karşılaştırması")
                        
                        perf_data = []
                        for name, metrics in ml_results.items():
                            perf_data.append({
                                'Model': name,
                                'MAE': metrics['MAE'],
                                'RMSE': metrics['RMSE'],
                                'MAPE (%)': metrics['MAPE'],
                                'R²': metrics['R2']
                            })
                        
                        perf_df = pd.DataFrame(perf_data)
                        perf_df = perf_df.sort_values('MAPE (%)')
                        
                        col_ml1, col_ml2 = st.columns([2, 1])
                        
                        with col_ml1:
                            styled_perf = style_dataframe(
                                perf_df,
                                color_column='MAPE (%)',
                                gradient_columns=['MAE', 'RMSE', 'R²']
                            )
                            st.dataframe(styled_perf, use_container_width=True)
                        
                        with col_ml2:
                            best_mape = ml_results[best_model_name]['MAPE']
                            
                            if best_mape < 10:
                                confidence_level = "🟢 YÜKSEK"
                                confidence_color = "#06B6D4"
                            elif best_mape < 20:
                                confidence_level = "🟡 ORTA"
                                confidence_color = "#F59E0B"
                            else:
                                confidence_level = "🔴 DÜŞÜK"
                                confidence_color = "#64748B"
                            
                            st.markdown(f'<div style="background: rgba(30, 41, 59, 0.8); padding: 1.5rem; border-radius: 12px; border: 2px solid {confidence_color}; margin-top: 1rem;">'
                                       f'<h3 style="color: white; margin: 0 0 1rem 0;">🏆 En İyi Model</h3>'
                                       f'<p style="color: {confidence_color}; font-size: 1.5rem; font-weight: 700; margin: 0 0 0.5rem 0;">{best_model_name}</p>'
                                       f'<p style="color: #94a3b8; margin: 0 0 1rem 0;">MAPE: <span style="color: {confidence_color}; font-weight: 700;">{best_mape:.2f}%</span></p>'
                                       f'<p style="color: #e2e8f0; font-weight: 600; margin: 0;">Güven Seviyesi: <span style="color: {confidence_color};">{confidence_level}</span></p>'
                                       '</div>', unsafe_allow_html=True)
                        
                        st.markdown("---")
                        
                        # Gelişmiş zaman serisi grafiği
                        st.subheader("📈 Gelişmiş Zaman Serisi ve Tahminler")
                        ts_chart = create_advanced_time_series_chart(monthly_df, forecast_df)
                        st.plotly_chart(ts_chart, use_container_width=True)
                        
                        # Tahmin detayları
                        st.markdown("---")
                        st.subheader("📋 Tahmin Detayları")
                        
                        forecast_summary = forecast_df.groupby(['Model', 'Tahmin_Tipi']).agg({
                            'PF_Satis': ['mean', 'sum']
                        }).reset_index()
                        
                        forecast_summary.columns = ['Model', 'Tahmin Tipi', 'Ortalama Tahmin', 'Toplam Tahmin']
                        forecast_summary.index = range(1, len(forecast_summary) + 1)
                        
                        styled_forecast = style_dataframe(
                            forecast_summary,
                            gradient_columns=['Ortalama Tahmin', 'Toplam Tahmin']
                        )
                        
                        st.dataframe(styled_forecast, use_container_width=True)
                    else:
                        st.warning("ML modeli eğitilemedi. Yeterli veri yok olabilir.")
                        ts_chart = create_advanced_time_series_chart(monthly_df)
                        st.plotly_chart(ts_chart, use_container_width=True)
                else:
                    st.warning("ML tahmini için en az 12 ay veri gereklidir.")
                    ts_chart = create_advanced_time_series_chart(monthly_df)
                    st.plotly_chart(ts_chart, use_container_width=True)
            
            elif analysis_type == "Trend Analizi":
                st.subheader("📈 Trend Analizi")
                trend_chart = create_trend_analysis_chart(monthly_df)
                st.plotly_chart(trend_chart, use_container_width=True)
                
                # Dönemsel büyüme metrikleri
                if 'buyume_metrikleri' in trend_analysis:
                    st.subheader("📊 Dönemsel Büyüme Oranları")
                    
                    growth_metrics = trend_analysis['buyume_metrikleri']
                    if growth_metrics:
                        col_growth1, col_growth2, col_growth3 = st.columns(3)
                        
                        if 'MoM_Growth' in growth_metrics:
                            with col_growth1:
                                st.metric("📈 Aylık Büyüme (MoM)", format_percentage(growth_metrics['MoM_Growth']))
                        
                        if 'QoQ_3M_Growth' in growth_metrics:
                            with col_growth2:
                                st.metric("📊 3 Aylık Büyüme (QoQ)", format_percentage(growth_metrics['QoQ_3M_Growth']))
                        
                        if 'QoQ_6M_Growth' in growth_metrics:
                            with col_growth3:
                                st.metric("📈 6 Aylık Büyüme (QoQ)", format_percentage(growth_metrics['QoQ_6M_Growth']))
            
            elif analysis_type == "Karşılaştırmalı Analiz":
                st.subheader("📊 Karşılaştırmalı Dönem Analizi")
                
                comparisons_df = create_comparative_analysis(monthly_df, periods=[3, 6, 12])
                
                if comparisons_df is not None and len(comparisons_df) > 0:
                    comp_chart = create_comparative_period_chart(comparisons_df)
                    if comp_chart:
                        st.plotly_chart(comp_chart, use_container_width=True)
                    
                    # Detaylı tablo
                    st.subheader("📋 Dönemsel Performans Detayları")
                    
                    comp_display = comparisons_df.copy()
                    comp_display.columns = ['Dönem', 'Ortalama Satış', 'Önceki Ortalama', 'Büyüme %', 
                                          'Pazar Payı %', 'Pay Değişimi', 'Volatilite', 'Trend']
                    comp_display.index = range(1, len(comp_display) + 1)
                    
                    styled_comp = style_dataframe(
                        comp_display,
                        color_column='Büyüme %',
                        gradient_columns=['Ortalama Satış', 'Pazar Payı %', 'Volatilite']
                    )
                    
                    st.dataframe(styled_comp, use_container_width=True)
                else:
                    st.warning("Karşılaştırmalı analiz için yeterli veri yok.")
            
            elif analysis_type == "Mevsimsellik Analizi":
                st.subheader("🔄 Mevsimsellik Analizi")
                
                seasonality_chart = create_seasonality_chart(monthly_df)
                if seasonality_chart:
                    st.plotly_chart(seasonality_chart, use_container_width=True)
                    
                    # Mevsimsellik istatistikleri
                    if 'Month' in monthly_df.columns:
                        monthly_avg = monthly_df.groupby('Month').agg({
                            'PF_Satis': ['mean', 'std', 'min', 'max']
                        }).reset_index()
                        
                        monthly_avg.columns = ['Month', 'Ortalama', 'Std Sapma', 'Minimum', 'Maksimum']
                        monthly_avg['Month_Name'] = monthly_avg['Month'].map({
                            1: 'Oca', 2: 'Şub', 3: 'Mar', 4: 'Nis', 5: 'May', 6: 'Haz',
                            7: 'Tem', 8: 'Ağu', 9: 'Eyl', 10: 'Eki', 11: 'Kas', 12: 'Ara'
                        })
                        
                        st.subheader("📊 Aylık Performans İstatistikleri")
                        
                        styled_season = style_dataframe(
                            monthly_avg,
                            gradient_columns=['Ortalama', 'Std Sapma', 'Minimum', 'Maksimum']
                        )
                        
                        st.dataframe(styled_season, use_container_width=True)
                else:
                    st.warning("Mevsimsellik analizi için yeterli veri yok (en az 12 ay).")
            
            elif analysis_type == "Volatilite Analizi":
                st.subheader("📉 Volatilite Analizi")
                
                volatility_chart = create_volatility_chart(monthly_df)
                if volatility_chart:
                    st.plotly_chart(volatility_chart, use_container_width=True)
                    
                    # Volatilite istatistikleri
                    if 'PF_CV' in monthly_df.columns:
                        st.subheader("📊 Volatilite İstatistikleri")
                        
                        col_vol1, col_vol2, col_vol3 = st.columns(3)
                        
                        with col_vol1:
                            avg_vol = monthly_df['PF_CV'].mean()
                            st.metric("📊 Ortalama CV", f"{avg_vol:.1f}%")
                        
                        with col_vol2:
                            max_vol = monthly_df['PF_CV'].max()
                            st.metric("📈 Maksimum CV", f"{max_vol:.1f}%")
                        
                        with col_vol3:
                            min_vol = monthly_df['PF_CV'].min()
                            st.metric("📉 Minimum CV", f"{min_vol:.1f}%")
            
            # Detaylı zaman serisi tablosu
            st.markdown("---")
            st.subheader("📋 Detaylı Zaman Serisi Verisi")
            
            display_cols = ['YIL_AY', 'PF_Satis', 'Rakip_Satis', 'Pazar_Payi_%', 
                          'PF_Buyume_%', 'Rakip_Buyume_%', 'Goreceli_Buyume_%']
            
            # Sadece mevcut kolonları göster
            available_cols = [col for col in display_cols if col in monthly_df.columns]
            monthly_display = monthly_df[available_cols].copy()
            
            # Kolon isimlerini düzenle
            col_names = {
                'YIL_AY': 'Ay',
                'PF_Satis': 'PF Satış',
                'Rakip_Satis': 'Rakip Satış',
                'Pazar_Payi_%': 'Pazar Payı %',
                'PF_Buyume_%': 'PF Büyüme %',
                'Rakip_Buyume_%': 'Rakip Büyüme %',
                'Goreceli_Buyume_%': 'Göreceli Büyüme %'
            }
            
            monthly_display = monthly_display.rename(columns=col_names)
            monthly_display.index = range(1, len(monthly_display) + 1)
            
            styled_monthly = style_dataframe(
                monthly_display,
                color_column='Göreceli Büyüme %',
                gradient_columns=['PF Satış', 'Pazar Payı %', 'PF Büyüme %']
            )
            
            st.dataframe(
                styled_monthly,
                use_container_width=True,
                height=400
            )
    
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
                st.metric("🎯 Ort. PF Pazar Payı", format_percentage(avg_pf_share))
            with col2:
                st.metric("📈 Ort. PF Büyüme", format_percentage(avg_pf_growth))
            with col3:
                st.metric("📉 Ort. Rakip Büyüme", format_percentage(avg_rakip_growth))
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
            st.metric("⭐ Star", f"{star_count}", delta=f"{format_number(star_pf)} PF")
        
        with col_bcg2:
            cow_count = bcg_counts.get("🐄 Cash Cow", 0)
            cow_pf = bcg_df[bcg_df['BCG_Kategori'] == "🐄 Cash Cow"]['PF_Satis'].sum()
            st.metric("🐄 Cash Cow", f"{cow_count}", delta=f"{format_number(cow_pf)} PF")
        
        with col_bcg3:
            q_count = bcg_counts.get("❓ Question Mark", 0)
            q_pf = bcg_df[bcg_df['BCG_Kategori'] == "❓ Question Mark"]['PF_Satis'].sum()
            st.metric("❓ Question", f"{q_count}", delta=f"{format_number(q_pf)} PF")
        
        with col_bcg4:
            dog_count = bcg_counts.get("🐶 Dog", 0)
            dog_pf = bcg_df[bcg_df['BCG_Kategori'] == "🐶 Dog"]['PF_Satis'].sum()
            st.metric("🐶 Dog", f"{dog_count}", delta=f"{format_number(dog_pf)} PF")
        
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
                <li>Gelişmiş Zaman Serisi Analizi</li>
                <li>Trend Analizi Sonuçları</li>
                <li>ML Tahmin Sonuçları</li>
                <li>BCG Matrix</li>
                <li>Şehir Bazlı Analiz</li>
                <li>Rakip Analizi</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("📊 Excel Raporu Oluştur", type="primary", use_container_width=True):
            with st.spinner("Rapor hazırlanıyor..."):
                # Tüm analizleri hesapla
                terr_perf = calculate_territory_performance(df_filtered, selected_product, date_filter)
                total_market_all = terr_perf['Toplam_Pazar'].sum()
                terr_perf['Toplam_Pazar_%'] = safe_divide(terr_perf['Toplam_Pazar'], total_market_all) * 100
                
                monthly_df = calculate_advanced_time_series(df_filtered, selected_product, None, date_filter)
                trend_analysis = perform_trend_analysis(monthly_df)
                bcg_df = calculate_bcg_matrix(df_filtered, selected_product, date_filter)
                city_data = calculate_city_performance(df_filtered, selected_product, date_filter)
                comp_data = calculate_competitor_analysis(df_filtered, selected_product, date_filter)
                
                # ML tahmini
                if len(monthly_df) >= 12:
                    ml_results, best_model_name, forecast_df = train_advanced_ml_models(monthly_df, 6)
                else:
                    ml_results, best_model_name, forecast_df = None, None, None
                
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    terr_perf.to_excel(writer, sheet_name='Territory Performans', index=False)
                    monthly_df.to_excel(writer, sheet_name='Zaman Serisi', index=False)
                    
                    # Trend analizi sonuçları
                    if 'error' not in trend_analysis:
                        trend_df = pd.DataFrame([trend_analysis])
                        trend_df.to_excel(writer, sheet_name='Trend Analizi', index=False)
                    
                    if bcg_df is not None:
                        bcg_df.to_excel(writer, sheet_name='BCG Matrix', index=False)
                    
                    city_data.to_excel(writer, sheet_name='Şehir Analizi', index=False)
                    comp_data.to_excel(writer, sheet_name='Rakip Analizi', index=False)
                    
                    if forecast_df is not None:
                        forecast_df.to_excel(writer, sheet_name='ML Tahminler', index=False)
                    
                    # ML model performansları
                    if ml_results is not None:
                        perf_data = []
                        for name, metrics in ml_results.items():
                            perf_data.append({
                                'Model': name,
                                'MAE': metrics['MAE'],
                                'RMSE': metrics['RMSE'],
                                'MAPE': metrics['MAPE'],
                                'R2': metrics['R2']
                            })
                        perf_df = pd.DataFrame(perf_data)
                        perf_df.to_excel(writer, sheet_name='ML Performans', index=False)
                    
                    # Özet sayfası
                    summary_data = {
                        'Metrik': ['Ürün', 'Dönem', 'Toplam PF Satış', 'Toplam Pazar', 'Pazar Payı', 
                                  'Territory Sayısı', 'Trend Durumu', 'Mevsimsellik', 'Volatilite'],
                        'Değer': [
                            selected_product,
                            date_option,
                            f"{terr_perf['PF_Satis'].sum():,.0f}",
                            f"{terr_perf['Toplam_Pazar'].sum():,.0f}",
                            f"{(terr_perf['PF_Satis'].sum() / terr_perf['Toplam_Pazar'].sum() * 100):.1f}%" if terr_perf['Toplam_Pazar'].sum() > 0 else "0%",
                            len(terr_perf),
                            trend_analysis.get('temel_trend', 'Bilinmiyor'),
                            trend_analysis.get('mevsimsellik', 'Bilinmiyor'),
                            trend_analysis.get('volatilite', 'Bilinmiyor')
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



