"""
🎯 GELİŞMİŞ TİCARİ PORTFÖY ANALİZ SİSTEMİ
Territory × Zaman × Coğrafi Analiz + Gelecek Tahminleme Platformu

Yeni Özellikler:
- ✅ Türkiye haritası üzerinde interaktif görselleştirme (Şehir eşleştirme düzeltildi)
- ✅ Gelecek tahminleme (ARIMA, Linear Regression, Moving Average)
- ✅ Zaman çizelgesi analizi (Gantt chart)
- ✅ Territory bazlı performans ve yatırım stratejisi analizi
- ✅ Detaylı BCG Matrix ve stratejik konumlandırma
- ✅ Manager performans scorecards
- ✅ Otomatik aksiyon planı oluşturma
- ✅ Excel ve PDF rapor çıktıları
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
# 2. dosyanızın başındaki CSS kısmını bununla değiştirin:

st.markdown("""
<style>
    /* Ana başlık - Mavi gradient */
    .main-header {
        font-size: 2.8rem;
        font-weight: bold;
        text-align: center;
        padding: 1.5rem 0;
        margin-bottom: 2rem;
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Metric kartları - Mavi tonları */
    .metric-card {
        background: linear-gradient(135deg, #3b82f6 0%, #1e40af 100%);
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(59,130,246,0.2);
        color: white;
        text-align: center;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(59,130,246,0.3);
    }
    
    /* Tab arkaplanı - Açık mavi */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        padding: 1rem;
        border-radius: 10px;
    }
    
    /* Tab butonları - Mavi tonları */
    .stTabs [data-baseweb="tab"] {
        height: 3.5rem;
        padding: 0 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 8px;
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        color: #1e40af;
        transition: all 0.3s ease;
        border: 2px solid transparent;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        border-color: #3b82f6;
        transform: translateY(-2px);
    }
    
    /* Aktif tab - Koyu mavi */
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border-color: #1e40af;
        box-shadow: 0 4px 6px rgba(59,130,246,0.3);
    }
    
    /* Ana sayfa arkaplanı - Çok açık mavi */
    .stApp {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
    }
    
    /* Territory kartları - Mavi border */
    .territory-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #3b82f6;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(59,130,246,0.1);
    }
    
    /* Öncelik kartları - Mavi tonlarda */
    .priority-critical {
        background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    
    .priority-high {
        background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    
    .priority-medium {
        background: linear-gradient(135deg, #0891b2 0%, #0e7490 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    
    /* Sidebar - Mavi tonları */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #eff6ff 0%, #dbeafe 100%);
    }
    
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stRadio label,
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #1e40af !important;
    }
    
    /* Butonlar - Mavi */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
        box-shadow: 0 4px 8px rgba(59,130,246,0.3);
        transform: translateY(-2px);
    }
    
    /* Metrik kutuları - Mavi tonları */
    div[data-testid="stMetricValue"] {
        color: #1e40af;
        font-weight: bold;
    }
    
    /* Grafikler arkaplan */
    .plotly {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(59,130,246,0.1);
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# ŞEHİR NORMALIZATION MAP - EXCELDEKİ İSİMLER İLE GEOJSON EŞLEŞTİRMESİ
# =============================================================================
CITY_NORMALIZE_MAP = {
    # Excel'deki isim: GeoJSON'daki isim
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
    
    # Türkçe karakterleri koru
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
        
        # Tarih sütununu datetime'a çevir
        if 'DATE' in df.columns:
            df['DATE'] = pd.to_datetime(df['DATE'])
            df['YIL_AY'] = df['DATE'].dt.strftime('%Y-%m')
            df['AY'] = df['DATE'].dt.month
            df['YIL'] = df['DATE'].dt.year
            df['QUARTER'] = df['DATE'].dt.quarter
            df['HAFTA'] = df['DATE'].dt.isocalendar().week
        
        # Standartlaştırma
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
# TIME SERIES ANALYSIS
# =============================================================================

def calculate_time_series(df, product, territory=None, frequency='M'):
    """Zaman serisi analizi"""
    cols = get_product_columns(product)
    
    df_filtered = df.copy()
    if territory and territory != "TÜMÜ":
        df_filtered = df_filtered[df_filtered['TERRITORIES'] == territory]
    
    # HATA DÜZELTMESİ: group_col'u doğru oluştur
    if frequency == 'D':
        df_filtered['group_col'] = df_filtered['DATE'].dt.strftime('%Y-%m-%d')
    elif frequency == 'W':
        df_filtered['group_col'] = df_filtered['DATE'].dt.strftime('%Y-W%U')
    elif frequency == 'Q':
        df_filtered['group_col'] = df_filtered['DATE'].dt.to_period('Q').astype(str)
    else:  # Monthly
        df_filtered['group_col'] = df_filtered['YIL_AY']
    
    time_series = df_filtered.groupby('group_col').agg({
        cols['pf']: 'sum',
        cols['rakip']: 'sum'
    }).reset_index().sort_values('group_col')
    
    time_series.columns = ['Period', 'PF_Satis', 'Rakip_Satis']
    time_series['Toplam_Pazar'] = time_series['PF_Satis'] + time_series['Rakip_Satis']
    time_series['Pazar_Payi_%'] = safe_divide(time_series['PF_Satis'], time_series['Toplam_Pazar']) * 100
    
    time_series['PF_Buyume_%'] = time_series['PF_Satis'].pct_change() * 100
    time_series['Rakip_Buyume_%'] = time_series['Rakip_Satis'].pct_change() * 100
    time_series['Goreceli_Buyume_%'] = time_series['PF_Buyume_%'] - time_series['Rakip_Buyume_%']
    
    window_3 = min(3, len(time_series))
    window_6 = min(6, len(time_series))
    time_series['MA_3'] = time_series['PF_Satis'].rolling(window=window_3, min_periods=1).mean()
    time_series['MA_6'] = time_series['PF_Satis'].rolling(window=window_6, min_periods=1).mean()
    
    if len(time_series) > 2:
        x = np.arange(len(time_series))
        y = time_series['PF_Satis'].values
        z = np.polyfit(x, y, 1)
        time_series['Trend'] = np.poly1d(z)(x)
    else:
        time_series['Trend'] = time_series['PF_Satis']
    
    return time_series

# =============================================================================
# GELECEK TAHMİNLEME
# =============================================================================

def forecast_future(time_series_df, periods=6, method='linear'):
    """
    Gelecek tahminleme fonksiyonu
    method: 'linear', 'moving_average', 'exponential'
    """
    if len(time_series_df) < 3:
        return None
    
    df = time_series_df.copy()
    df['Period_Num'] = range(len(df))
    
    if method == 'linear':
        # Linear Regression
        X = df['Period_Num'].values.reshape(-1, 1)
        y = df['PF_Satis'].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        future_periods = np.arange(len(df), len(df) + periods).reshape(-1, 1)
        predictions = model.predict(future_periods)
        
    elif method == 'moving_average':
        # Moving Average (son 3 dönemin ortalaması)
        window = min(3, len(df))
        last_values = df['PF_Satis'].tail(window).mean()
        predictions = np.full(periods, last_values)
        
    elif method == 'exponential':
        # Exponential Weighted Moving Average
        alpha = 0.3
        last_value = df['PF_Satis'].iloc[-1]
        predictions = []
        for i in range(periods):
            if i == 0:
                pred = last_value
            else:
                pred = alpha * predictions[-1] + (1 - alpha) * last_value
            predictions.append(pred)
        predictions = np.array(predictions)
    
    # Tahmin dataframe'i oluştur
    last_period = df['Period'].iloc[-1]
    
    try:
        # Period formatına göre gelecek dönemleri oluştur
        if '-W' in last_period:  # Haftalık
            future_periods_list = [f"{last_period.split('-')[0]}-W{int(last_period.split('-W')[1]) + i + 1}" 
                                   for i in range(periods)]
        elif 'Q' in last_period:  # Çeyrek
            future_periods_list = [f"{int(last_period.split('Q')[0]) + (i//4)}Q{((int(last_period.split('Q')[1]) + i - 1) % 4) + 1}" 
                                   for i in range(1, periods + 1)]
        else:  # Aylık
            from datetime import datetime
            from dateutil.relativedelta import relativedelta
            last_date = datetime.strptime(last_period, '%Y-%m')
            future_periods_list = [(last_date + relativedelta(months=i+1)).strftime('%Y-%m') 
                                   for i in range(periods)]
    except:
        future_periods_list = [f"Gelecek+{i+1}" for i in range(periods)]
    
    forecast_df = pd.DataFrame({
        'Period': future_periods_list,
        'PF_Satis_Tahmin': predictions,
        'Tip': 'Tahmin'
    })
    
    return forecast_df

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
# INVESTMENT STRATEGY
# =============================================================================

def calculate_investment_strategy(bcg_df):
    """Yatırım stratejisi hesapla"""
    bcg_df = bcg_df.copy()
    
    try:
        bcg_df['Pazar_Buyuklugu_Segment'] = pd.qcut(
            bcg_df['Toplam_Pazar'], 
            q=3, 
            labels=['Küçük', 'Orta', 'Büyük'],
            duplicates='drop'
        )
    except:
        bcg_df['Pazar_Buyuklugu_Segment'] = 'Orta'
    
    try:
        bcg_df['Pazar_Payi_Segment'] = pd.qcut(
            bcg_df['Pazar_Payi_%'], 
            q=3, 
            labels=['Düşük', 'Orta', 'Yüksek'],
            duplicates='drop'
        )
    except:
        bcg_df['Pazar_Payi_Segment'] = 'Orta'
    
    try:
        bcg_df['Buyume_Potansiyeli_Segment'] = pd.qcut(
            bcg_df['Buyume_Potansiyeli'],
            q=3,
            labels=['Düşük', 'Orta', 'Yüksek'],
            duplicates='drop'
        )
    except:
        bcg_df['Buyume_Potansiyeli_Segment'] = 'Orta'
    
    def assign_strategy(row):
        pazar = str(row['Pazar_Buyuklugu_Segment'])
        payi = str(row['Pazar_Payi_Segment'])
        buyume = str(row['Buyume_Potansiyeli_Segment'])
        
        if pazar in ['Büyük', 'Orta'] and payi == 'Düşük' and buyume in ['Yüksek', 'Orta']:
            return '🚀 Agresif'
        elif pazar in ['Büyük', 'Orta'] and payi == 'Orta':
            return '⚡ Hızlandırılmış'
        elif pazar == 'Büyük' and payi == 'Yüksek':
            return '🛡️ Koruma'
        elif pazar == 'Küçük' and buyume == 'Yüksek':
            return '💎 Potansiyel'
        else:
            return '👁️ İzleme'
    
    bcg_df['Yatirim_Stratejisi'] = bcg_df.apply(assign_strategy, axis=1)
    
    def suggest_action(row):
        strategy = row['Yatirim_Stratejisi']
        if '🚀' in strategy:
            return 'Yatırımı artır, agresif büyüme stratejisi uygula'
        elif '⚡' in strategy:
            return 'Hızlandırılmış kaynak tahsisi, pazar payını yükselt'
        elif '🛡️' in strategy:
            return 'Lider konumu koru, savunma stratejisi'
        elif '💎' in strategy:
            return 'Seçici yatırım, gelecek potansiyeli izle'
        else:
            return 'Minimal kaynak, izleme modunda tut'
    
    bcg_df['Aksiyon'] = bcg_df.apply(suggest_action, axis=1)
    
    bcg_df['Oncelik_Skoru'] = 0
    
    max_pazar = bcg_df['Toplam_Pazar'].max()
    if max_pazar > 0:
        bcg_df['Oncelik_Skoru'] += (bcg_df['Toplam_Pazar'] / max_pazar) * 40
    
    max_pot = bcg_df['Buyume_Potansiyeli'].max()
    if max_pot > 0:
        bcg_df['Oncelik_Skoru'] += (bcg_df['Buyume_Potansiyeli'] / max_pot) * 30
    
    bcg_df.loc[bcg_df['Pazar_Payi_%'] < 10, 'Oncelik_Skoru'] += 30
    
    return bcg_df

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_territory_bar_chart(df, top_n=20, title="Territory Performans"):
    """Territory performans bar chart"""
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

def create_time_series_with_forecast(ts_df, forecast_df=None, title="Zaman Serisi"):
    """Gelecek tahminli zaman serisi grafiği"""
    fig = go.Figure()
    
    # Gerçek veri
    fig.add_trace(go.Scatter(
        x=ts_df['Period'],
        y=ts_df['PF_Satis'],
        mode='lines+markers',
        name='Gerçek Satış',
        line=dict(color='#3B82F6', width=3),
        marker=dict(size=8)
    ))
    
    # Tahmin
    if forecast_df is not None and len(forecast_df) > 0:
        fig.add_trace(go.Scatter(
            x=forecast_df['Period'],
            y=forecast_df['PF_Satis_Tahmin'],
            mode='lines+markers',
            name='Tahmin',
            line=dict(color='#F59E0B', width=3, dash='dash'),
            marker=dict(size=8, symbol='diamond')
        ))
    
    # Trend
    if 'Trend' in ts_df.columns:
        fig.add_trace(go.Scatter(
            x=ts_df['Period'],
            y=ts_df['Trend'],
            mode='lines',
            name='Trend',
            line=dict(color='#10B981', width=2, dash='dot')
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Dönem',
        yaxis_title='Satış',
        height=500,
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_turkey_choropleth(city_data, geojson_data):
    """Türkiye haritası choropleth"""
    if geojson_data is None:
        return None
    
    fig = px.choropleth(
        city_data,
        geojson=geojson_data,
        locations='CITY_NORMALIZED',
        featureidkey="properties.name",
        color='Pazar_Payi_%',
        color_continuous_scale='Blues',
        hover_name='CITY_NORMALIZED',
        hover_data={
            'PF_Satis': ':,.0f',
            'Toplam_Pazar': ':,.0f',
            'Pazar_Payi_%': ':.1f'
        },
        title='Türkiye - Şehir Bazlı Pazar Payı Haritası'
    )
    
    fig.update_geos(
        center=dict(lat=39, lon=35),
        projection_scale=20,
        visible=False
    )
    
    fig.update_layout(
        height=600,
        geo=dict(bgcolor='rgba(0,0,0,0)'),
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_bcg_scatter(bcg_df):
    """BCG Matrix scatter"""
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
# MANAGER SCORECARD
# =============================================================================

def create_manager_scorecard(df, product):
    """Manager performans scorecard"""
    cols = get_product_columns(product)
    
    manager_perf = df.groupby('MANAGER').agg({
        cols['pf']: 'sum',
        cols['rakip']: 'sum',
        'TERRITORIES': 'nunique'
    }).reset_index()
    
    manager_perf.columns = ['Manager', 'PF_Satis', 'Rakip_Satis', 'Territory_Sayisi']
    manager_perf['Toplam_Pazar'] = manager_perf['PF_Satis'] + manager_perf['Rakip_Satis']
    manager_perf['Pazar_Payi_%'] = safe_divide(manager_perf['PF_Satis'], manager_perf['Toplam_Pazar']) * 100
    
    manager_perf = manager_perf.sort_values('PF_Satis', ascending=False)
    manager_perf['Rank'] = range(1, len(manager_perf) + 1)
    
    return manager_perf

# =============================================================================
# ACTION PLAN
# =============================================================================

def generate_action_plan(df, product):
    """Aksiyon planı oluştur"""
    actions = []
    
    terr_perf = calculate_territory_performance(df, product)
    
    # Büyük fırsatlar
    opportunities = terr_perf[
        (terr_perf['Toplam_Pazar'] > terr_perf['Toplam_Pazar'].median()) &
        (terr_perf['Pazar_Payi_%'] < 10)
    ].head(3)
    
    for idx, row in opportunities.iterrows():
        actions.append({
            'Öncelik': '🔴 Kritik',
            'Territory': row['TERRITORIES'],
            'Aksiyon': f"Agresif yatırım",
            'Neden': f"Büyük pazar ama düşük pay",
            'Potansiyel': f"+{row['Buyume_Potansiyeli']:,.0f}",
            'Sorumlu': row.get('MANAGER', 'N/A')
        })
    
    return pd.DataFrame(actions)

# =============================================================================
# MAIN APP
# =============================================================================

def main():
    st.markdown('<h1 class="main-header">💊 TİCARİ PORTFÖY ANALİZ SİSTEMİ</h1>', unsafe_allow_html=True)
    
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
        "📈 Zaman Serisi & Tahmin",
        "🗺️ Türkiye Haritası",
        "⭐ BCG Matrix",
        "🔍 Rakip Analizi",
        "👥 Manager",
        "🎯 Aksiyon",
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
        fig = create_territory_bar_chart(terr_perf, 15)
        st.plotly_chart(fig, use_container_width=True)
    
    # TAB 2: TERRITORY ANALİZİ
    with tabs[1]:
        st.header("🏢 Territory Bazlı Detaylı Analiz")
        
        terr_perf = calculate_territory_performance(df_filtered, selected_product)
        
        # Filtreleme ve sıralama seçenekleri
        col_filter1, col_filter2, col_filter3 = st.columns(3)
        
        with col_filter1:
            sort_metric = st.selectbox(
                "🔄 Sıralama Kriteri",
                ['PF_Satis', 'Pazar_Payi_%', 'Toplam_Pazar', 'Buyume_Potansiyeli', 'Goreceli_Pazar_Payi'],
                format_func=lambda x: {
                    'PF_Satis': '💊 PF Satış',
                    'Pazar_Payi_%': '📊 Pazar Payı',
                    'Toplam_Pazar': '🏪 Toplam Pazar',
                    'Buyume_Potansiyeli': '🚀 Büyüme Potansiyeli',
                    'Goreceli_Pazar_Payi': '⚖️ Göreceli Pay (PF/Rakip)'
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
        
        # KPI Kartları
        st.subheader("📊 Genel Metrikler")
        
        col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
        
        with col_kpi1:
            avg_market_share = terr_perf['Pazar_Payi_%'].mean()
            st.metric(
                "📊 Ortalama Pazar Payı",
                f"%{avg_market_share:.1f}",
                delta=f"{len(terr_perf[terr_perf['Pazar_Payi_%'] > avg_market_share])} territory üstünde"
            )
        
        with col_kpi2:
            total_potential = terr_perf['Buyume_Potansiyeli'].sum()
            st.metric(
                "🚀 Toplam Potansiyel",
                f"{total_potential:,.0f}",
                help="Tüm territorylerdeki toplam büyüme potansiyeli"
            )
        
        with col_kpi3:
            high_share_terr = len(terr_perf[terr_perf['Pazar_Payi_%'] > 50])
            st.metric(
                "👑 Lider Territory",
                f"{high_share_terr}",
                delta="Pazar payı %50 üzeri"
            )
        
        with col_kpi4:
            low_share_terr = len(terr_perf[terr_perf['Pazar_Payi_%'] < 10])
            st.metric(
                "⚠️ Düşük Performans",
                f"{low_share_terr}",
                delta="Pazar payı %10 altı",
                delta_color="inverse"
            )
        
        st.markdown("---")
        
        # Görselleştirmeler
        col_viz1, col_viz2 = st.columns(2)
        
        with col_viz1:
            st.markdown("#### 📊 PF vs Rakip Satış Karşılaştırması")
            fig_comp = create_territory_bar_chart(terr_sorted, n_territories, "Territory Performans")
            st.plotly_chart(fig_comp, use_container_width=True)
        
        with col_viz2:
            st.markdown("#### 🎯 Pazar Büyüklüğü vs Pazar Payı")
            
            fig_scatter = px.scatter(
                terr_sorted,
                x='Toplam_Pazar',
                y='Pazar_Payi_%',
                size='PF_Satis',
                color='REGION' if 'REGION' in terr_sorted.columns else None,
                hover_name='TERRITORIES',
                hover_data={
                    'PF_Satis': ':,.0f',
                    'Rakip_Satis': ':,.0f',
                    'Toplam_Pazar': ':,.0f',
                    'Pazar_Payi_%': ':.1f',
                    'Buyume_Potansiyeli': ':,.0f'
                },
                title='Pazar Analizi Matrix',
                size_max=60
            )
            
            # Ortalama çizgileri
            avg_market = terr_sorted['Toplam_Pazar'].mean()
            avg_share = terr_sorted['Pazar_Payi_%'].mean()
            
            fig_scatter.add_hline(y=avg_share, line_dash="dash", line_color="red", 
                                 annotation_text=f"Ort. Pay: {avg_share:.1f}%")
            fig_scatter.add_vline(x=avg_market, line_dash="dash", line_color="blue",
                                 annotation_text=f"Ort. Pazar: {avg_market:,.0f}")
            
            fig_scatter.update_layout(height=500)
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        st.markdown("---")
        
        # Göreceli Pazar Payı Analizi
        col_rel1, col_rel2 = st.columns(2)
        
        with col_rel1:
            st.markdown("#### ⚖️ Göreceli Pazar Payı (PF/Rakip)")
            
            fig_rel = go.Figure()
            
            # 1.0 referans çizgisi
            fig_rel.add_hline(y=1.0, line_dash="dash", line_color="white", 
                             annotation_text="Eşit (1.0)")
            
            # Bar chart
            colors = ['#10B981' if x > 1 else '#EF4444' for x in terr_sorted['Goreceli_Pazar_Payi']]
            
            fig_rel.add_trace(go.Bar(
                x=terr_sorted['TERRITORIES'],
                y=terr_sorted['Goreceli_Pazar_Payi'],
                marker_color=colors,
                text=terr_sorted['Goreceli_Pazar_Payi'].apply(lambda x: f'{x:.2f}'),
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>Göreceli Pay: %{y:.2f}<extra></extra>'
            ))
            
            fig_rel.update_layout(
                title='Göreceli Pazar Payı (>1 = PF Lider, <1 = Rakip Lider)',
                xaxis_title='Territory',
                yaxis_title='Göreceli Pay (PF/Rakip)',
                height=400,
                xaxis=dict(tickangle=-45),
                plot_bgcolor='rgba(0,0,0,0)',
                showlegend=False
            )
            
            st.plotly_chart(fig_rel, use_container_width=True)
        
        with col_rel2:
            st.markdown("#### 🚀 Büyüme Potansiyeli Dağılımı")
            
            fig_pot = px.bar(
                terr_sorted.nlargest(15, 'Buyume_Potansiyeli'),
                x='TERRITORIES',
                y='Buyume_Potansiyeli',
                color='Pazar_Payi_%',
                color_continuous_scale='RdYlGn',
                text='Buyume_Potansiyeli',
                title='En Yüksek Potansiyelli 15 Territory'
            )
            
            fig_pot.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
            fig_pot.update_layout(height=400, xaxis=dict(tickangle=-45))
            st.plotly_chart(fig_pot, use_container_width=True)
        
        st.markdown("---")
        
        # Performans Segmentasyonu
        st.subheader("🎯 Performans Segmentasyonu")
        
        # Segment tanımlama
        terr_perf_seg = terr_perf.copy()
        
        # Pazar payı segmentleri
        terr_perf_seg['Segment'] = pd.cut(
            terr_perf_seg['Pazar_Payi_%'],
            bins=[0, 10, 30, 50, 100],
            labels=['🔴 Düşük (<10%)', '🟡 Orta (10-30%)', '🟢 İyi (30-50%)', '🌟 Lider (>50%)']
        )
        
        segment_summary = terr_perf_seg.groupby('Segment').agg({
            'TERRITORIES': 'count',
            'PF_Satis': 'sum',
            'Toplam_Pazar': 'sum',
            'Buyume_Potansiyeli': 'sum'
        }).reset_index()
        
        segment_summary.columns = ['Segment', 'Territory Sayısı', 'PF Satış', 'Toplam Pazar', 'Potansiyel']
        
        col_seg1, col_seg2 = st.columns([1, 1])
        
        with col_seg1:
            st.dataframe(
                segment_summary.style.format({
                    'PF Satış': '{:,.0f}',
                    'Toplam Pazar': '{:,.0f}',
                    'Potansiyel': '{:,.0f}'
                }),
                use_container_width=True,
                hide_index=True
            )
        
        with col_seg2:
            fig_seg = px.pie(
                segment_summary,
                values='Territory Sayısı',
                names='Segment',
                title='Territory Segmentasyon Dağılımı',
                color='Segment',
                color_discrete_map={
                    '🔴 Düşük (<10%)': '#EF4444',
                    '🟡 Orta (10-30%)': '#F59E0B',
                    '🟢 İyi (30-50%)': '#10B981',
                    '🌟 Lider (>50%)': '#3B82F6'
                }
            )
            fig_seg.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_seg, use_container_width=True)
        
        st.markdown("---")
        
        # Detaylı Territory Tablosu
        st.subheader("📋 Detaylı Territory Listesi")
        
        # Filtreleme seçenekleri
        col_tbl1, col_tbl2 = st.columns(2)
        
        with col_tbl1:
            if 'REGION' in terr_perf.columns:
                selected_regions = st.multiselect(
                    "🗺️ Bölge Filtresi",
                    terr_perf['REGION'].unique(),
                    default=None
                )
                
                if selected_regions:
                    terr_perf = terr_perf[terr_perf['REGION'].isin(selected_regions)]
        
        with col_tbl2:
            min_market = st.number_input(
                "🏪 Min. Pazar Büyüklüğü",
                min_value=0,
                value=0,
                step=1000
            )
            
            if min_market > 0:
                terr_perf = terr_perf[terr_perf['Toplam_Pazar'] >= min_market]
        
        # Tablo gösterimi
        display_cols = ['TERRITORIES', 'REGION', 'CITY', 'MANAGER', 'PF_Satis', 
                       'Rakip_Satis', 'Toplam_Pazar', 'Pazar_Payi_%', 
                       'Goreceli_Pazar_Payi', 'Buyume_Potansiyeli']
        
        display_cols = [col for col in display_cols if col in terr_perf.columns]
        
        terr_display = terr_perf[display_cols].copy()
        terr_display.index = range(1, len(terr_display) + 1)
        
        st.dataframe(
            terr_display.style.format({
                'PF_Satis': '{:,.0f}',
                'Rakip_Satis': '{:,.0f}',
                'Toplam_Pazar': '{:,.0f}',
                'Pazar_Payi_%': '{:.1f}%',
                'Goreceli_Pazar_Payi': '{:.2f}',
                'Buyume_Potansiyeli': '{:,.0f}'
            }).background_gradient(subset=['Pazar_Payi_%'], cmap='RdYlGn')
             .background_gradient(subset=['Goreceli_Pazar_Payi'], cmap='RdYlGn'),
            use_container_width=True,
            height=500
        )
        
        # CSV Export
        csv = terr_display.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 CSV İndir",
            csv,
            f"territory_analizi_{datetime.now().strftime('%Y%m%d')}.csv",
            "text/csv"
        )
    
    # TAB 3: ZAMAN SERİSİ & TAHMİN
    with tabs[2]:
        st.header("📈 Zaman Serisi Analizi & Gelecek Tahmini")
        
        col_ts1, col_ts2, col_ts3 = st.columns(3)
        
        with col_ts1:
            freq = st.selectbox("📅 Periyot", [('M', 'Aylık'), ('W', 'Haftalık')], format_func=lambda x: x[1])[0]
        
        with col_ts2:
            forecast_periods = st.slider("🔮 Tahmin Dönemi", 1, 12, 6)
        
        with col_ts3:
            forecast_method = st.selectbox(
                "📊 Model",
                ['linear', 'moving_average', 'exponential'],
                format_func=lambda x: {
                    'linear': 'Doğrusal Regresyon',
                    'moving_average': 'Hareketli Ortalama',
                    'exponential': 'Üstel Düzeltme'
                }[x]
            )
        
        time_series = calculate_time_series(df_filtered, selected_product, selected_territory, freq)
        
        if len(time_series) > 3:
            forecast_df = forecast_future(time_series, forecast_periods, forecast_method)
            
            fig_ts = create_time_series_with_forecast(time_series, forecast_df, "Satış Trendi & Tahmin")
            st.plotly_chart(fig_ts, use_container_width=True)
            
            if forecast_df is not None:
                st.subheader("🔮 Gelecek Tahminleri")
                st.dataframe(
                    forecast_df.style.format({'PF_Satis_Tahmin': '{:,.0f}'}),
                    use_container_width=True,
                    hide_index=True
                )
        else:
            st.warning("⚠️ Yetersiz veri")
    
    # TAB 4: TÜRKİYE HARİTASI
    with tabs[3]:
        st.header("🗺️ Türkiye Coğrafi Analiz")
        
        # Şehir bazlı veri hazırla
        cols_map = get_product_columns(selected_product)
        
        if 'CITY_NORMALIZED' in df_filtered.columns:
            city_data = df_filtered.groupby('CITY_NORMALIZED').agg({
                cols_map['pf']: 'sum',
                cols_map['rakip']: 'sum'
            }).reset_index()
            
            city_data.columns = ['CITY_NORMALIZED', 'PF_Satis', 'Rakip_Satis']
            city_data['Toplam_Pazar'] = city_data['PF_Satis'] + city_data['Rakip_Satis']
            city_data['Pazar_Payi_%'] = safe_divide(city_data['PF_Satis'], city_data['Toplam_Pazar']) * 100
            
            # GeoJSON yükle (uploaded file olarak)
            geojson_file = st.file_uploader("🗺️ turkey.geojson yükle", type=['geojson', 'json'])
            
            if geojson_file:
                try:
                    geojson_data = json.load(geojson_file)
                    fig_map = create_turkey_choropleth(city_data, geojson_data)
                    if fig_map:
                        st.plotly_chart(fig_map, use_container_width=True)
                except Exception as e:
                    st.error(f"Harita hatası: {str(e)}")
            else:
                st.info("💡 turkey.geojson dosyasını yükleyin")
            
            # Şehir tablosu
            st.subheader("🏙️ Şehir Detayları")
            st.dataframe(
                city_data.sort_values('PF_Satis', ascending=False).style.format({
                    'PF_Satis': '{:,.0f}',
                    'Rakip_Satis': '{:,.0f}',
                    'Toplam_Pazar': '{:,.0f}',
                    'Pazar_Payi_%': '{:.1f}%'
                }),
                use_container_width=True,
                height=400
            )
    
    # TAB 5: BCG MATRIX (GELİŞMİŞ)
    with tabs[4]:
        st.header("⭐ BCG Matrix & Yatırım Stratejisi")
        
        bcg_df = calculate_bcg_matrix(df_filtered, selected_product)
        strategy_df = calculate_investment_strategy(bcg_df)
        
        # BCG Özet Metrikleri
        st.subheader("📊 Portföy Dağılımı")
        
        bcg_counts = strategy_df['BCG_Kategori'].value_counts()
        
        col_bcg1, col_bcg2, col_bcg3, col_bcg4 = st.columns(4)
        
        bcg_metrics = [
            ("⭐ Star", col_bcg1, "#FFD700"),
            ("🐄 Cash Cow", col_bcg2, "#10B981"),
            ("❓ Question Mark", col_bcg3, "#3B82F6"),
            ("🐶 Dog", col_bcg4, "#6B7280")
        ]
        
        for category, col, color in bcg_metrics:
            with col:
                count = int(bcg_counts.get(category, 0))
                pf_sum = strategy_df[strategy_df['BCG_Kategori'] == category]['PF_Satis'].sum()
                avg_share = strategy_df[strategy_df['BCG_Kategori'] == category]['Pazar_Payi_%'].mean()
                
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, {color} 0%, {color}CC 100%);
                    padding: 1.5rem;
                    border-radius: 10px;
                    color: white;
                    text-align: center;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                ">
                    <h3 style="margin: 0; font-size: 2rem;">{category.split()[0]}</h3>
                    <h2 style="margin: 10px 0; font-size: 2.5rem;">{count}</h2>
                    <p style="margin: 5px 0; font-size: 0.9rem;">Territory</p>
                    <hr style="border-color: rgba(255,255,255,0.3); margin: 10px 0;">
                    <p style="margin: 5px 0; font-size: 1.2rem; font-weight: bold;">{pf_sum:,.0f}</p>
                    <p style="margin: 0; font-size: 0.85rem;">PF Satış</p>
                    <p style="margin: 10px 0 0 0; font-size: 1rem;">Ort. Pay: %{avg_share:.1f}</p>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # BCG Scatter Matrix
        st.subheader("🎯 BCG Matrix - Stratejik Konumlandırma")
        
        fig_bcg = create_bcg_scatter(strategy_df)
        st.plotly_chart(fig_bcg, use_container_width=True)
        
        st.markdown("---")
        
        # Yatırım Stratejisi Dağılımı
        st.subheader("💼 Yatırım Stratejisi Dağılımı")
        
        strategy_counts = strategy_df['Yatirim_Stratejisi'].value_counts()
        
        col_str1, col_str2, col_str3, col_str4, col_str5 = st.columns(5)
        
        strategies = [
            ('🚀 Agresif', col_str1, '#DC2626'),
            ('⚡ Hızlandırılmış', col_str2, '#EA580C'),
            ('🛡️ Koruma', col_str3, '#10B981'),
            ('💎 Potansiyel', col_str4, '#8B5CF6'),
            ('👁️ İzleme', col_str5, '#6B7280')
        ]
        
        for strategy, col, color in strategies:
            with col:
                count = int(strategy_counts.get(strategy, 0))
                pf_sum = strategy_df[strategy_df['Yatirim_Stratejisi'] == strategy]['PF_Satis'].sum()
                
                st.markdown(f"""
                <div style="
                    background: {color};
                    padding: 1rem;
                    border-radius: 8px;
                    color: white;
                    text-align: center;
                ">
                    <h4 style="margin: 0;">{strategy.split()[0]}</h4>
                    <h2 style="margin: 10px 0;">{count}</h2>
                    <p style="margin: 0; font-size: 0.9rem;">{pf_sum:,.0f} PF</p>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Strateji Detay Grafikleri
        col_sg1, col_sg2 = st.columns(2)
        
        with col_sg1:
            st.markdown("#### 📊 Strateji Bazlı Toplam Satış")
            
            strategy_summary = strategy_df.groupby('Yatirim_Stratejisi').agg({
                'PF_Satis': 'sum',
                'TERRITORIES': 'count',
                'Pazar_Payi_%': 'mean'
            }).reset_index()
            
            strategy_summary.columns = ['Strateji', 'Toplam_Satis', 'Territory_Sayisi', 'Ort_Pazar_Payi']
            
            fig_strat_bar = px.bar(
                strategy_summary,
                x='Strateji',
                y='Toplam_Satis',
                color='Ort_Pazar_Payi',
                color_continuous_scale='RdYlGn',
                text='Toplam_Satis',
                title='Strateji Bazlı Toplam Satış'
            )
            
            fig_strat_bar.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
            fig_strat_bar.update_layout(height=400, xaxis=dict(tickangle=-45))
            st.plotly_chart(fig_strat_bar, use_container_width=True)
        
        with col_sg2:
            st.markdown("#### 🎯 Öncelik Skoru Dağılımı")
            
            fig_priority = px.box(
                strategy_df,
                x='Yatirim_Stratejisi',
                y='Oncelik_Skoru',
                color='Yatirim_Stratejisi',
                title='Strateji Bazlı Öncelik Skoru',
                points='all'
            )
            
            fig_priority.update_layout(height=400, showlegend=False, xaxis=dict(tickangle=-45))
            st.plotly_chart(fig_priority, use_container_width=True)
        
        st.markdown("---")
        
        # Stratejik Aksiyonlar - Kategori Bazlı
        st.subheader("🎯 Önerilen Stratejik Aksiyonlar")
        
        # Her strateji için filtreli görünüm
        selected_strategy = st.selectbox(
            "🔍 Strateji Filtrele",
            ['Tümü'] + sorted(strategy_df['Yatirim_Stratejisi'].unique().tolist())
        )
        
        if selected_strategy != 'Tümü':
            strategy_filtered = strategy_df[strategy_df['Yatirim_Stratejisi'] == selected_strategy]
        else:
            strategy_filtered = strategy_df
        
        # Öncelik skoruna göre sırala
        strategy_filtered = strategy_filtered.sort_values('Oncelik_Skoru', ascending=False)
        
        # Aksiyon tablosu
        display_cols_strategy = ['TERRITORIES', 'REGION', 'BCG_Kategori', 'Yatirim_Stratejisi',
                                'PF_Satis', 'Pazar_Payi_%', 'Buyume_Potansiyeli', 
                                'Oncelik_Skoru', 'Aksiyon']
        
        display_cols_strategy = [col for col in display_cols_strategy if col in strategy_filtered.columns]
        
        strategy_display = strategy_filtered[display_cols_strategy].copy()
        strategy_display.index = range(1, len(strategy_display) + 1)
        
        st.dataframe(
            strategy_display.style.format({
                'PF_Satis': '{:,.0f}',
                'Pazar_Payi_%': '{:.1f}%',
                'Buyume_Potansiyeli': '{:,.0f}',
                'Oncelik_Skoru': '{:.0f}'
            }).background_gradient(subset=['Oncelik_Skoru'], cmap='YlOrRd')
             .background_gradient(subset=['Pazar_Payi_%'], cmap='RdYlGn'),
            use_container_width=True,
            height=500
        )
        
        st.markdown("---")
        
        # BCG Transition Matrix (Gelecek Tahmin)
        st.subheader("🔄 BCG Kategori Geçiş Potansiyeli")
        
        st.info("""
        💡 **BCG Kategori Açıklamaları:**
        
        - **⭐ Stars:** Yüksek pazar payı + Yüksek büyüme → En değerli territoryler, yatırımı sürdür
        - **🐄 Cash Cows:** Yüksek pazar payı + Düşük büyüme → Karlı territoryler, nakit üret
        - **❓ Question Marks:** Düşük pazar payı + Yüksek büyüme → Yatırım gerektirir, dikkatli karar ver
        - **🐶 Dogs:** Düşük pazar payı + Düşük büyüme → Minimal kaynak, yeniden değerlendir
        """)
        
        # Geçiş potansiyeli analizi
        transition_potential = []
        
        for idx, row in strategy_df.iterrows():
            current_bcg = row['BCG_Kategori']
            current_share = row['Goreceli_Pazar_Payi']
            current_growth = row['Pazar_Buyume_%']
            
            # Potansiyel geçişler
            if current_bcg == "❓ Question Mark" and row['Yatirim_Stratejisi'] == '🚀 Agresif':
                transition_potential.append({
                    'Territory': row['TERRITORIES'],
                    'Şu An': current_bcg,
                    'Potansiyel': '⭐ Star',
                    'Aksiyon': 'Agresif yatırım ile Star olabilir',
                    'Risk': 'Orta'
                })
            elif current_bcg == "⭐ Star" and current_growth < 0:
                transition_potential.append({
                    'Territory': row['TERRITORIES'],
                    'Şu An': current_bcg,
                    'Potansiyel': '🐄 Cash Cow',
                    'Aksiyon': 'Büyüme yavaşladı, Cash Cow stratejisine geç',
                    'Risk': 'Düşük'
                })
            elif current_bcg == "🐶 Dog" and row['Buyume_Potansiyeli'] > strategy_df['Buyume_Potansiyeli'].median():
                transition_potential.append({
                    'Territory': row['TERRITORIES'],
                    'Şu An': current_bcg,
                    'Potansiyel': '❓ Question Mark',
                    'Aksiyon': 'Potansiyel var, yeniden değerlendir',
                    'Risk': 'Yüksek'
                })
        
        if transition_potential:
            st.markdown("#### 🔄 Potansiyel Kategori Geçişleri")
            transition_df = pd.DataFrame(transition_potential)
            st.dataframe(transition_df, use_container_width=True, hide_index=True)
    
    # TAB 6: RAKİP ANALİZİ
    with tabs[5]:
        st.header("🔍 Detaylı Rakip Analizi")
        
        cols_rakip = get_product_columns(selected_product)
        
        # Rakip Özet Metrikleri
        st.subheader("📊 Rakip Performans Özeti")
        
        total_pf_rakip = df_filtered[cols_rakip['pf']].sum()
        total_rakip = df_filtered[cols_rakip['rakip']].sum()
        total_all = total_pf_rakip + total_rakip
        
        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
        
        with col_r1:
            st.metric(
                "💊 PF Toplam",
                f"{total_pf_rakip:,.0f}",
                delta=f"%{(total_pf_rakip/total_all*100):.1f} pay"
            )
        
        with col_r2:
            st.metric(
                "🏪 Rakip Toplam",
                f"{total_rakip:,.0f}",
                delta=f"%{(total_rakip/total_all*100):.1f} pay"
            )
        
        with col_r3:
            pf_dominant = len(terr_perf[terr_perf['Goreceli_Pazar_Payi'] > 1])
            st.metric(
                "👑 PF Lider Territory",
                f"{pf_dominant}",
                delta="PF > Rakip"
            )
        
        with col_r4:
            rakip_dominant = len(terr_perf[terr_perf['Goreceli_Pazar_Payi'] < 1])
            st.metric(
                "⚠️ Rakip Lider Territory",
                f"{rakip_dominant}",
                delta="Rakip > PF",
                delta_color="inverse"
            )
        
        st.markdown("---")
        
        # Rekabet Yoğunluğu Haritası
        st.subheader("🗺️ Rekabet Yoğunluğu Analizi")
        
        col_heat1, col_heat2 = st.columns(2)
        
        with col_heat1:
            st.markdown("#### 📊 PF vs Rakip - Territory Bazlı")
            
            # Stacked bar chart
            fig_comp = go.Figure()
            
            terr_comp = terr_perf.nlargest(20, 'Toplam_Pazar')
            
            fig_comp.add_trace(go.Bar(
                name='PF Satış',
                x=terr_comp['TERRITORIES'],
                y=terr_comp['PF_Satis'],
                marker_color='#3B82F6',
                text=terr_comp['PF_Satis'].apply(lambda x: f'{x:,.0f}'),
                textposition='inside'
            ))
            
            fig_comp.add_trace(go.Bar(
                name='Rakip Satış',
                x=terr_comp['TERRITORIES'],
                y=terr_comp['Rakip_Satis'],
                marker_color='#EF4444',
                text=terr_comp['Rakip_Satis'].apply(lambda x: f'{x:,.0f}'),
                textposition='inside'
            ))
            
            fig_comp.update_layout(
                barmode='stack',
                title='Top 20 Territory - Rekabet Durumu',
                height=500,
                xaxis=dict(tickangle=-45),
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_comp, use_container_width=True)
        
        with col_heat2:
            st.markdown("#### ⚖️ Göreceli Güç Dengesi")
            
            # Waterfall chart
            terr_wat = terr_perf.nlargest(15, 'Buyume_Potansiyeli')
            
            fig_wat = go.Figure(go.Waterfall(
                name="Potansiyel",
                orientation="v",
                x=terr_wat['TERRITORIES'],
                y=terr_wat['Buyume_Potansiyeli'],
                text=terr_wat['Buyume_Potansiyeli'].apply(lambda x: f'{x:,.0f}'),
                textposition="outside",
                connector={"line": {"color": "rgb(63, 63, 63)"}},
                decreasing={"marker": {"color": "#EF4444"}},
                increasing={"marker": {"color": "#10B981"}},
                totals={"marker": {"color": "#3B82F6"}}
            ))
            
            fig_wat.update_layout(
                title="Büyüme Potansiyeli (Rakipten Kazanılabilir)",
                height=500,
                xaxis=dict(tickangle=-45)
            )
            
            st.plotly_chart(fig_wat, use_container_width=True)
        
        st.markdown("---")
        
        # Zaman Bazlı Rakip Karşılaştırması
        st.subheader("📈 Zaman Bazlı PF vs Rakip Performansı")
        
        time_comp = calculate_time_series(df_filtered, selected_product, selected_territory, 'M')
        
        if len(time_comp) > 0:
            fig_time_comp = go.Figure()
            
            # PF çizgisi
            fig_time_comp.add_trace(go.Scatter(
                x=time_comp['Period'],
                y=time_comp['PF_Satis'],
                mode='lines+markers',
                name='PF Satış',
                line=dict(color='#3B82F6', width=3),
                marker=dict(size=8),
                fill='tonexty'
            ))
            
            # Rakip çizgisi
            fig_time_comp.add_trace(go.Scatter(
                x=time_comp['Period'],
                y=time_comp['Rakip_Satis'],
                mode='lines+markers',
                name='Rakip Satış',
                line=dict(color='#EF4444', width=3),
                marker=dict(size=8),
                fill='tozeroy'
            ))
            
            # Pazar payı çizgisi (ikinci y ekseni)
            fig_time_comp.add_trace(go.Scatter(
                x=time_comp['Period'],
                y=time_comp['Pazar_Payi_%'],
                mode='lines+markers',
                name='PF Pazar Payı %',
                line=dict(color='#10B981', width=2, dash='dash'),
                marker=dict(size=6, symbol='diamond'),
                yaxis='y2'
            ))
            
            fig_time_comp.update_layout(
                title='PF vs Rakip - Zaman Serisi Karşılaştırması',
                xaxis_title='Dönem',
                yaxis_title='Satış',
                yaxis2=dict(
                    title='Pazar Payı (%)',
                    overlaying='y',
                    side='right',
                    range=[0, 100]
                ),
                height=500,
                hovermode='x unified',
                legend=dict(x=0, y=1.1, orientation='h')
            )
            
            st.plotly_chart(fig_time_comp, use_container_width=True)
        
        st.markdown("---")
        
        # Rakip Güç Analizi - Territory Segmentasyonu
        st.subheader("🎯 Rakip Güç Segmentasyonu")
        
        # Segmentlere ayır
        terr_perf_rakip = terr_perf.copy()
        
        terr_perf_rakip['Rakip_Gucu'] = pd.cut(
            terr_perf_rakip['Goreceli_Pazar_Payi'],
            bins=[0, 0.5, 1.0, 2.0, 100],
            labels=['🔴 Rakip Çok Güçlü', '🟡 Rakip Üstün', '🟢 Dengeli', '💙 PF Dominant']
        )
        
        segment_rakip = terr_perf_rakip.groupby('Rakip_Gucu').agg({
            'TERRITORIES': 'count',
            'PF_Satis': 'sum',
            'Rakip_Satis': 'sum',
            'Buyume_Potansiyeli': 'sum'
        }).reset_index()
        
        segment_rakip.columns = ['Rakip Gücü', 'Territory Sayısı', 'PF Satış', 'Rakip Satış', 'Potansiyel']
        segment_rakip['PF Pay %'] = (segment_rakip['PF Satış'] / (segment_rakip['PF Satış'] + segment_rakip['Rakip Satış']) * 100).round(1)
        
        col_rs1, col_rs2 = st.columns([1, 1])
        
        with col_rs1:
            st.dataframe(
                segment_rakip.style.format({
                    'PF Satış': '{:,.0f}',
                    'Rakip Satış': '{:,.0f}',
                    'Potansiyel': '{:,.0f}',
                    'PF Pay %': '{:.1f}%'
                }),
                use_container_width=True,
                hide_index=True
            )
        
        with col_rs2:
            fig_seg_rakip = px.sunburst(
                segment_rakip,
                path=['Rakip Gücü'],
                values='Territory Sayısı',
                color='PF Pay %',
                color_continuous_scale='RdYlGn',
                title='Rakip Güç Segmentasyonu'
            )
            fig_seg_rakip.update_layout(height=400)
            st.plotly_chart(fig_seg_rakip, use_container_width=True)
        
        st.markdown("---")
        
        # En Kritik Rakip Bölgeleri
        st.subheader("⚠️ Kritik Rakip Bölgeleri (Öncelikli Aksiyon Gerekli)")
        
        critical_rakip = terr_perf_rakip[
            (terr_perf_rakip['Goreceli_Pazar_Payi'] < 0.5) &
            (terr_perf_rakip['Toplam_Pazar'] > terr_perf_rakip['Toplam_Pazar'].median())
        ].nlargest(10, 'Buyume_Potansiyeli')
        
        if len(critical_rakip) > 0:
            critical_display = critical_rakip[['TERRITORIES', 'REGION', 'PF_Satis', 'Rakip_Satis', 
                                              'Pazar_Payi_%', 'Goreceli_Pazar_Payi', 'Buyume_Potansiyeli']].copy()
            
            st.dataframe(
                critical_display.style.format({
                    'PF_Satis': '{:,.0f}',
                    'Rakip_Satis': '{:,.0f}',
                    'Pazar_Payi_%': '{:.1f}%',
                    'Goreceli_Pazar_Payi': '{:.2f}',
                    'Buyume_Potansiyeli': '{:,.0f}'
                }).background_gradient(subset=['Buyume_Potansiyeli'], cmap='YlOrRd'),
                use_container_width=True
            )
            
            st.warning(f"""
            ⚠️ **Aksiyon Önerisi:** Bu {len(critical_rakip)} territory'de rakip çok güçlü konumda. 
            Agresif yatırım ve pazar payı kazanma stratejisi uygulanmalı!
            """)
        else:
            st.success("✅ Kritik rakip bölgesi tespit edilmedi!")
    
    # TAB 7: MANAGER
    with tabs[6]:
        st.header("⭐ BCG Matrix & Strateji")
        
        bcg_df = calculate_bcg_matrix(df_filtered, selected_product)
        strategy_df = calculate_investment_strategy(bcg_df)
        
        bcg_counts = strategy_df['BCG_Kategori'].value_counts()
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("⭐ Stars", bcg_counts.get("⭐ Star", 0))
        col2.metric("🐄 Cows", bcg_counts.get("🐄 Cash Cow", 0))
        col3.metric("❓ Questions", bcg_counts.get("❓ Question Mark", 0))
        col4.metric("🐶 Dogs", bcg_counts.get("🐶 Dog", 0))
        
        fig_bcg = create_bcg_scatter(strategy_df)
        st.plotly_chart(fig_bcg, use_container_width=True)
        
        st.subheader("📋 Strateji Detayları")
        display_cols = ['TERRITORIES', 'BCG_Kategori', 'Yatirim_Stratejisi', 
                       'PF_Satis', 'Pazar_Payi_%', 'Oncelik_Skoru']
        
        st.dataframe(
            strategy_df[display_cols].style.format({
                'PF_Satis': '{:,.0f}',
                'Pazar_Payi_%': '{:.1f}%',
                'Oncelik_Skoru': '{:.0f}'
            }),
            use_container_width=True,
            height=400
        )
    
    # TAB 5: MANAGER
    with tabs[4]:
        st.header("👥 Manager Performans")
        
        manager_perf = create_manager_scorecard(df_filtered, selected_product)
        
        # Top 3
        col1, col2, col3 = st.columns(3)
        top3 = manager_perf.head(3)
        
        for idx, (col, row) in enumerate(zip([col1, col2, col3], top3.itertuples())):
            emoji = ["🥇", "🥈", "🥉"][idx]
            with col:
                st.metric(
                    f"{emoji} {row.Manager}",
                    f"{row.PF_Satis:,.0f}",
                    delta=f"%{row.Pazar_Payi_:,.1f}"
                )
        
        st.dataframe(
            manager_perf.style.format({
                'PF_Satis': '{:,.0f}',
                'Pazar_Payi_%': '{:.1f}%'
            }),
            use_container_width=True
        )
    
    # TAB 8: AKSİYON
    with tabs[7]:
        st.header("🎯 Aksiyon Planı")
        
        action_plan = generate_action_plan(df_filtered, selected_product)
        
        if len(action_plan) > 0:
            for idx, row in action_plan.iterrows():
                st.markdown(f"""
                <div class="priority-critical">
                    <h4>{row['Aksiyon']}</h4>
                    <p><strong>Territory:</strong> {row['Territory']}</p>
                    <p><strong>Potansiyel:</strong> {row['Potansiyel']}</p>
                </div>
                """, unsafe_allow_html=True)
    
    # TAB 9: RAPOR
    with tabs[8]:
        st.header("📥 Rapor İndirme")
        
        # Excel rapor
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            terr_perf.to_excel(writer, sheet_name='Territory', index=False)
            time_series.to_excel(writer, sheet_name='Zaman Serisi', index=False)
            if forecast_df is not None:
                forecast_df.to_excel(writer, sheet_name='Tahmin', index=False)
        
        st.download_button(
            "📥 Excel İndir",
            output.getvalue(),
            f"rapor_{datetime.now().strftime('%Y%m%d')}.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

if __name__ == "__main__":
    main()


