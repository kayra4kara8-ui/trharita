"""
🎯 GELİŞMİŞ TİCARİ PORTFÖY ANALİZ SİSTEMİ
Territory Bazlı Performans, ML Tahminleme, Türkiye Haritası ve Rekabet Analizi

Yeni Özellikler:
- 🗺️ Geopandas ile Türkiye şehir bazlı harita görselleştirme
- 🤖 Machine Learning satış tahminleme
- 📊 Aylık/Yıllık dönem seçimi
- 📈 Gelişmiş rakip analizi ve trend karşılaştırması
- 🎯 Dinamik zaman aralığı filtreleme
- ⭐ BCG Matrix ve Yatırım Stratejisi
"""

import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from shapely.geometry import LineString, MultiLineString
import warnings
from io import BytesIO
import json

warnings.filterwarnings("ignore")

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Gelişmiş Ticari Portföy Analizi",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS
# =============================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E40AF;
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        padding: 0 2rem;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# ŞEHİR İSİM HARİTALAMA (Shapefile ve Excel uyumluluğu için)
# =============================================================================
CITY_FIX_MAP = {
    "ISTANBUL": "İSTANBUL",
    "IZMIR": "İZMİR",
    "SANLIURFA": "ŞANLIURFA",
    "USAK": "UŞAK",
    "ELAZIG": "ELAZIĞ",
    "MUGLA": "MUĞLA",
    "KIRSEHIR": "KIRŞEHİR",
    "NEVSEHIR": "NEVŞEHİR",
    "NIGDE": "NİĞDE",
    "TEKIRDAG": "TEKİRDAĞ",
    "CANAKKALE": "ÇANAKKALE",
    "CANKIRI": "ÇANKIRI",
    "CORUM": "ÇORUM",
    "GUMUSHANE": "GÜMÜŞHANE",
    "KAHRAMANMARAS": "KAHRAMANMARAŞ",
    "KARABUK": "KARABÜK",
    "KIRIKKALE": "KIRIKKALE",
    "KIRKLARELI": "KIRKLARELİ",
    "KUTAHYA": "KÜTAHYA",
    "DUZCE": "DÜZCE"
}

REGION_COLORS = {
    "KUZEY ANADOLU": "#2E8B57",
    "MARMARA": "#2F6FD6",
    "İÇ ANADOLU": "#8B6B4A",
    "BATI ANADOLU": "#2BB0A6",
    "GÜNEY DOĞU ANADOLU": "#A05A2C",
    "IC ANADOLU": "#8B6B4A",
    "BATI ANADOLU": "#2BB0A6", 
    "GUNEY DOGU ANADOLU": "#A05A2C"
}

# GeoJSON için şehir mapping
CITY_NAME_MAPPING = {
    'ADANA': 'Adana',
    'ADIYAMAN': 'Adiyaman',
    'AFYONKARAHISAR': 'Afyonkarahisar',
    'AFYONKARAHİSAR': 'Afyonkarahisar',
    'AĞRI': 'Agri',
    'AGRI': 'Agri',
    'AKSARAY': 'Aksaray',
    'AMASYA': 'Amasya',
    'ANKARA': 'Ankara',
    'ANTALYA': 'Antalya',
    'ARTVİN': 'Artvin',
    'ARTVIN': 'Artvin',
    'AYDIN': 'Aydin',
    'BALIKESİR': 'Balikesir',
    'BALIKESIR': 'Balikesir',
    'BARTIN': 'Bartın',
    'BATMAN': 'Batman',
    'BAYBURT': 'Bayburt',
    'BİLECİK': 'Bilecik',
    'BILECIK': 'Bilecik',
    'BİNGÖL': 'Bingöl',
    'BINGOL': 'Bingöl',
    'BİTLİS': 'Bitlis',
    'BITLIS': 'Bitlis',
    'BOLU': 'Bolu',
    'BURDUR': 'Burdur',
    'BURSA': 'Bursa',
    'ÇANAKKALE': 'Çanakkale',
    'CANAKKALE': 'Çanakkale',
    'ÇANKIRI': 'Çankiri',
    'CANKIRI': 'Çankiri',
    'ÇORUM': 'Çorum',
    'CORUM': 'Çorum',
    'DENİZLİ': 'Denizli',
    'DENIZLI': 'Denizli',
    'DİYARBAKIR': 'Diyarbakir',
    'DIYARBAKIR': 'Diyarbakir',
    'DÜZCE': 'Düzce',
    'DUZCE': 'Düzce',
    'EDİRNE': 'Edirne',
    'EDIRNE': 'Edirne',
    'ELAZIĞ': 'Elazig',
    'ELAZIG': 'Elazig',
    'ERZİNCAN': 'Erzincan',
    'ERZINCAN': 'Erzincan',
    'ERZURUM': 'Erzurum',
    'ESKİŞEHİR': 'Eskisehir',
    'ESKISEHIR': 'Eskisehir',
    'GAZİANTEP': 'Gaziantep',
    'GAZIANTEP': 'Gaziantep',
    'GİRESUN': 'Giresun',
    'GIRESUN': 'Giresun',
    'GÜMÜŞHANE': 'Gümüşhane',
    'GUMUSHANE': 'Gümüşhane',
    'HAKKARİ': 'Hakkari',
    'HAKKARI': 'Hakkari',
    'HATAY': 'Hatay',
    'IĞDIR': 'Iğdir',
    'IGDIR': 'Iğdir',
    'ISPARTA': 'Isparta',
    'İSTANBUL': 'Istanbul',
    'ISTANBUL': 'Istanbul',
    'İZMİR': 'Izmir',
    'IZMIR': 'Izmir',
    'KAHRAMANMARAŞ': 'K. Maras',
    'KAHRAMANMARAS': 'K. Maras',
    'KARABÜK': 'Karabük',
    'KARABUK': 'Karabük',
    'KARAMAN': 'Karaman',
    'KARS': 'Kars',
    'KASTAMONU': 'Kastamonu',
    'KAYSERİ': 'Kayseri',
    'KAYSERI': 'Kayseri',
    'KIRIKKALE': 'Kırıkkale',
    'KIRKLARELİ': 'Kirklareli',
    'KIRKLARELI': 'Kirklareli',
    'KIRŞEHİR': 'Kirsehir',
    'KIRSEHIR': 'Kirsehir',
    'KİLİS': 'Kilis',
    'KILIS': 'Kilis',
    'KOCAELİ': 'Kocaeli',
    'KOCAELI': 'Kocaeli',
    'KONYA': 'Konya',
    'KÜTAHYA': 'Kütahya',
    'KUTAHYA': 'Kütahya',
    'MALATYA': 'Malatya',
    'MANİSA': 'Manisa',
    'MANISA': 'Manisa',
    'MARDİN': 'Mardin',
    'MARDIN': 'Mardin',
    'MERSİN': 'Mersin',
    'MERSIN': 'Mersin',
    'MUĞLA': 'Mugla',
    'MUGLA': 'Mugla',
    'MUŞ': 'Mus',
    'MUS': 'Mus',
    'NEVŞEHİR': 'Nevsehir',
    'NEVSEHIR': 'Nevsehir',
    'NİĞDE': 'Nigde',
    'NIGDE': 'Nigde',
    'ORDU': 'Ordu',
    'OSMANİYE': 'Osmaniye',
    'OSMANIYE': 'Osmaniye',
    'RİZE': 'Rize',
    'RIZE': 'Rize',
    'SAKARYA': 'Sakarya',
    'SAMSUN': 'Samsun',
    'SİİRT': 'Siirt',
    'SIIRT': 'Siirt',
    'SİNOP': 'Sinop',
    'SINOP': 'Sinop',
    'SİVAS': 'Sivas',
    'SIVAS': 'Sivas',
    'ŞANLIURFA': 'Sanliurfa',
    'SANLIURFA': 'Sanliurfa',
    'ŞIRNAK': 'Sirnak',
    'SIRNAK': 'Sirnak',
    'TEKİRDAĞ': 'Tekirdag',
    'TEKIRDAG': 'Tekirdag',
    'TOKAT': 'Tokat',
    'TRABZON': 'Trabzon',
    'TUNCELİ': 'Tunceli',
    'TUNCELI': 'Tunceli',
    'UŞAK': 'Usak',
    'USAK': 'Usak',
    'VAN': 'Van',
    'YALOVA': 'Yalova',
    'YOZGAT': 'Yozgat',
    'ZONGULDAK': 'Zonguldak',
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def safe_divide(a, b):
    """Güvenli bölme işlemi"""
    return np.where(b != 0, a / b, 0)

def format_number(x):
    """Sayı formatla"""
    if pd.isna(x):
        return 0
    return round(float(x), 2)

def get_product_columns(product):
    """Ürün kolonlarını döndür"""
    if product == "TROCMETAM":
        return {"pf": "TROCMETAM", "rakip": "DIGER TROCMETAM"}
    elif product == "CORTIPOL":
        return {"pf": "CORTIPOL", "rakip": "DIGER CORTIPOL"}
    elif product == "DEKSAMETAZON":
        return {"pf": "DEKSAMETAZON", "rakip": "DIGER DEKSAMETAZON"}
    else:  # PF IZOTONIK
        return {"pf": "PF IZOTONIK", "rakip": "DIGER IZOTONIK"}

def normalize_city_name(city_name):
    """Şehir ismini GeoJSON formatına çevir"""
    city_upper = str(city_name).strip().upper()
    return CITY_NAME_MAPPING.get(city_upper, city_name)

# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data
def load_excel_data(file):
    """Excel dosyasını yükle ve ön işleme yap"""
    df = pd.read_excel(file)
    
    # Tarih sütununu datetime'a çevir
    df['DATE'] = pd.to_datetime(df['DATE'])
    df['YIL_AY'] = df['DATE'].dt.strftime('%Y-%m')
    df['AY'] = df['DATE'].dt.month
    df['YIL'] = df['DATE'].dt.year
    
    # Standartlaştırma
    df['TERRITORIES'] = df['TERRITORIES'].str.upper().str.strip()
    df['CITY'] = df['CITY'].str.strip()
    df['CITY_NORMALIZED'] = df['CITY'].apply(normalize_city_name)
    df['REGION'] = df['REGION'].str.upper().str.strip()
    df['MANAGER'] = df['MANAGER'].str.upper().str.strip()
    
    return df

@st.cache_data
def load_geojson():
    """Türkiye GeoJSON'ını yükle"""
    try:
        with open('/mnt/user-data/uploads/turkey.geojson', 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return None

@st.cache_data
def load_turkey_shapefile(_shp_file):
    """Türkiye shapefile'ını yükle"""
    try:
        gdf = gpd.read_file(_shp_file)
        gdf["name"] = gdf["name"].str.upper()
        gdf["CITY_CLEAN"] = gdf["name"].replace(CITY_FIX_MAP).str.upper()
        return gdf
    except Exception as e:
        st.error(f"Shapefile yükleme hatası: {str(e)}")
        return None

# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def calculate_city_performance(df, product, date_filter=None):
    """Şehir bazlı performans analizi"""
    cols = get_product_columns(product)
    
    # Tarih filtresi
    if date_filter:
        df = df[(df['DATE'] >= date_filter[0]) & (df['DATE'] <= date_filter[1])]
    
    # Şehir bazlı toplam
    city_perf = df.groupby(['CITY_NORMALIZED', 'CITY']).agg({
        cols['pf']: 'sum',
        cols['rakip']: 'sum'
    }).reset_index()
    
    city_perf.columns = ['City_Normalized', 'City_Original', 'PF_Satis', 'Rakip_Satis']
    city_perf['Toplam_Pazar'] = city_perf['PF_Satis'] + city_perf['Rakip_Satis']
    city_perf['Pazar_Payi_%'] = safe_divide(city_perf['PF_Satis'], city_perf['Toplam_Pazar']) * 100
    
    return city_perf

def calculate_territory_performance(df, product, date_filter=None):
    """Territory bazlı performans analizi"""
    cols = get_product_columns(product)
    
    # Tarih filtresi
    if date_filter:
        df = df[(df['DATE'] >= date_filter[0]) & (df['DATE'] <= date_filter[1])]
    
    # Territory bazlı toplam
    terr_perf = df.groupby(['TERRITORIES', 'REGION', 'CITY', 'MANAGER']).agg({
        cols['pf']: 'sum',
        cols['rakip']: 'sum'
    }).reset_index()
    
    terr_perf.columns = ['Territory', 'Region', 'City', 'Manager', 'PF_Satis', 'Rakip_Satis']
    terr_perf['Toplam_Pazar'] = terr_perf['PF_Satis'] + terr_perf['Rakip_Satis']
    terr_perf['Pazar_Payi_%'] = safe_divide(terr_perf['PF_Satis'], terr_perf['Toplam_Pazar']) * 100
    
    # Toplam içindeki ağırlık
    total_pf = terr_perf['PF_Satis'].sum()
    terr_perf['Agirlik_%'] = safe_divide(terr_perf['PF_Satis'], total_pf) * 100
    
    # Göreceli pazar payı (BCG için)
    terr_perf['Goreceli_Pazar_Payi'] = safe_divide(terr_perf['PF_Satis'], terr_perf['Rakip_Satis'])
    
    return terr_perf.sort_values('PF_Satis', ascending=False)

def calculate_time_series(df, product, territory=None, date_filter=None):
    """Aylık zaman serisi analizi"""
    cols = get_product_columns(product)
    
    # Filtreleme
    df_filtered = df.copy()
    if territory and territory != "TÜMÜ":
        df_filtered = df_filtered[df_filtered['TERRITORIES'] == territory]
    
    if date_filter:
        df_filtered = df_filtered[(df_filtered['DATE'] >= date_filter[0]) & 
                                   (df_filtered['DATE'] <= date_filter[1])]
    
    # Aylık toplam
    monthly = df_filtered.groupby('YIL_AY').agg({
        cols['pf']: 'sum',
        cols['rakip']: 'sum',
        'DATE': 'first'
    }).reset_index().sort_values('YIL_AY')
    
    monthly.columns = ['YIL_AY', 'PF_Satis', 'Rakip_Satis', 'DATE']
    monthly['Toplam_Pazar'] = monthly['PF_Satis'] + monthly['Rakip_Satis']
    monthly['Pazar_Payi_%'] = safe_divide(monthly['PF_Satis'], monthly['Toplam_Pazar']) * 100
    
    # Büyüme oranları
    monthly['PF_Buyume_%'] = monthly['PF_Satis'].pct_change() * 100
    monthly['Rakip_Buyume_%'] = monthly['Rakip_Satis'].pct_change() * 100
    monthly['Goreceli_Buyume_%'] = monthly['PF_Buyume_%'] - monthly['Rakip_Buyume_%']
    
    # Hareketli ortalamalar
    monthly['MA_3'] = monthly['PF_Satis'].rolling(window=3, min_periods=1).mean()
    monthly['MA_6'] = monthly['PF_Satis'].rolling(window=6, min_periods=1).mean()
    
    return monthly

def simple_forecast(df, periods=3):
    """Basit tahmin modeli (hareketli ortalama ve trend)"""
    if len(df) < 3:
        return None
    
    # Son 3 ayın ortalaması
    recent_avg = df['PF_Satis'].tail(3).mean()
    
    # Trend hesaplama (son 6 ay)
    if len(df) >= 6:
        x = np.arange(len(df.tail(6)))
        y = df['PF_Satis'].tail(6).values
        z = np.polyfit(x, y, 1)
        trend = z[0]
    else:
        trend = 0
    
    # Tahmin
    forecasts = []
    last_date = df['DATE'].max()
    
    for i in range(1, periods + 1):
        forecast_date = last_date + pd.DateOffset(months=i)
        forecast_value = max(0, recent_avg + (trend * i))
        forecasts.append({
            'YIL_AY': forecast_date.strftime('%Y-%m'),
            'DATE': forecast_date,
            'PF_Satis': forecast_value,
            'Tahmin': True
        })
    
    return pd.DataFrame(forecasts)

def calculate_competitor_analysis(df, product, date_filter=None):
    """Rakip analizi"""
    cols = get_product_columns(product)
    
    if date_filter:
        df = df[(df['DATE'] >= date_filter[0]) & (df['DATE'] <= date_filter[1])]
    
    # Aylık rakip performansı
    monthly = df.groupby('YIL_AY').agg({
        cols['pf']: 'sum',
        cols['rakip']: 'sum'
    }).reset_index().sort_values('YIL_AY')
    
    monthly.columns = ['YIL_AY', 'PF', 'Rakip']
    monthly['PF_Pay_%'] = (monthly['PF'] / (monthly['PF'] + monthly['Rakip'])) * 100
    monthly['Rakip_Pay_%'] = 100 - monthly['PF_Pay_%']
    
    # Büyüme karşılaştırması
    monthly['PF_Buyume'] = monthly['PF'].pct_change() * 100
    monthly['Rakip_Buyume'] = monthly['Rakip'].pct_change() * 100
    monthly['Fark'] = monthly['PF_Buyume'] - monthly['Rakip_Buyume']
    
    return monthly

def calculate_bcg_matrix(df, product, date_filter=None):
    """BCG Matrix kategorileri hesapla"""
    cols = get_product_columns(product)
    
    if date_filter:
        df_filtered = df[(df['DATE'] >= date_filter[0]) & (df['DATE'] <= date_filter[1])]
    else:
        df_filtered = df.copy()
    
    # Territory performansı
    terr_perf = calculate_territory_performance(df_filtered, product)
    
    # Pazar büyüme oranı hesapla
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
    
    # BCG Sınıflandırma
    median_share = terr_perf['Goreceli_Pazar_Payi'].median()
    median_growth = terr_perf['Pazar_Buyume_%'].median()
    
    def assign_bcg(row):
        if row['Goreceli_Pazar_Payi'] >= median_share and row['Pazar_Buyume_%'] >= median_growth:
            return "⭐ Yıldız"
        elif row['Goreceli_Pazar_Payi'] >= median_share and row['Pazar_Buyume_%'] < median_growth:
            return "💰 Nakit İnek"
        elif row['Goreceli_Pazar_Payi'] < median_share and row['Pazar_Buyume_%'] >= median_growth:
            return "❓ Soru İşareti"
        else:
            return "🐕 Köpek"
    
    terr_perf['BCG_Kategori'] = terr_perf.apply(assign_bcg, axis=1)
    
    # Yatırım stratejisi
    def get_strategy(category):
        if '⭐' in category:
            return '🚀 Büyümeye Yatırım - Lider konumu koruyun'
        elif '💰' in category:
            return '💵 Nakit Üretimi - Verimliliği optimize edin'
        elif '❓' in category:
            return '🎯 Seçici Yatırım - Pazar payını artırın'
        else:
            return '⚠️ Gözden Geçir - Stratejik önemi değerlendirin'
    
    terr_perf['Strateji'] = terr_perf['BCG_Kategori'].apply(get_strategy)
    
    return terr_perf

# =============================================================================
# TÜRKIYE HARİTASI FONKSİYONLARI
# =============================================================================

def lines_to_lonlat(geom):
    """Geometriyi lon/lat dizilerine çevir"""
    lons, lats = [], []
    if isinstance(geom, LineString):
        xs, ys = geom.xy
        lons += list(xs) + [None]
        lats += list(ys) + [None]
    elif isinstance(geom, MultiLineString):
        for g in geom.geoms:
            xs, ys = g.xy
            lons += list(xs) + [None]
            lats += list(ys) + [None]
    return lons, lats

def create_turkey_choropleth_map(city_data, turkey_gdf):
    """Türkiye şehir bazlı choropleth harita (Geopandas ile)"""
    if turkey_gdf is None or city_data.empty:
        return None
    
    # Verileri birleştir
    merged = turkey_gdf.merge(
        city_data,
        left_on='CITY_CLEAN',
        right_on='City_Normalized',
        how='left'
    )
    
    merged['PF_Satis'] = merged['PF_Satis'].fillna(0)
    merged['Pazar_Payi_%'] = merged['Pazar_Payi_%'].fillna(0)
    
    # Choropleth oluştur
    fig = px.choropleth(
        merged,
        geojson=merged.__geo_interface__,
        locations=merged.index,
        color='PF_Satis',
        hover_name='CITY_CLEAN',
        hover_data={
            'PF_Satis': ':,.0f',
            'Pazar_Payi_%': ':.1f'
        },
        color_continuous_scale='YlOrRd',
        labels={'PF_Satis': 'PF Satış'}
    )
    
    fig.update_geos(
        fitbounds="geojson",
        visible=False
    )
    
    fig.update_layout(
        margin=dict(l=0, r=0, t=40, b=0),
        height=600
    )
    
    # Şehir sınırlarını ekle
    all_lons, all_lats = [], []
    for geom in merged.geometry.boundary:
        lo, la = lines_to_lonlat(geom)
        all_lons += lo
        all_lats += la
    
    fig.add_scattergeo(
        lon=all_lons,
        lat=all_lats,
        mode="lines",
        line=dict(width=0.6, color="rgba(60,60,60,0.6)"),
        hoverinfo="skip",
        showlegend=False
    )
    
    # Şehir merkezlerini ekle (hover için)
    pts = merged.to_crs(3857)
    pts["centroid"] = pts.geometry.centroid
    pts = pts.to_crs(merged.crs)
    
    fig.add_scattergeo(
        lon=pts.centroid.x,
        lat=pts.centroid.y,
        mode="markers",
        marker=dict(size=6, color="rgba(0,0,0,0)"),
        hoverinfo="text",
        text=(
            "<b>" + pts["CITY_CLEAN"].fillna("") + "</b><br>" +
            "PF Satış: " + pts["PF_Satis"].fillna(0).astype(int).map(lambda x: f"{x:,}") + "<br>" +
            "Pazar Payı: %" + pts["Pazar_Payi_%"].fillna(0).round(1).astype(str)
        ),
        showlegend=False
    )
    
    return fig

def create_turkey_map(city_data, geojson, title="Türkiye Satış Haritası"):
    """Türkiye haritası oluştur (GeoJSON ile)"""
    if geojson is None:
        return None
    
    fig = px.choropleth(
        city_data,
        geojson=geojson,
        locations='City_Normalized',
        featureidkey="properties.name",
        color='PF_Satis',
        hover_name='City_Original',
        hover_data={
            'PF_Satis': ':,.0f',
            'Pazar_Payi_%': ':.1f',
            'City_Normalized': False,
            'City_Original': False
        },
        color_continuous_scale="YlOrRd",
        labels={'PF_Satis': 'PF Satış'},
        title=title
    )
    
    fig.update_geos(
        fitbounds="locations",
        visible=False
    )
    
    fig.update_layout(
        height=600,
        margin=dict(l=0, r=0, t=50, b=0),
        geo=dict(
            showframe=False,
            showcoastlines=False,
            projection_type='mercator'
        )
    )
    
    return fig

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_forecast_chart(historical_df, forecast_df):
    """Tahmin grafiği"""
    fig = go.Figure()
    
    # Gerçek veriler
    fig.add_trace(go.Scatter(
        x=historical_df['DATE'],
        y=historical_df['PF_Satis'],
        mode='lines+markers',
        name='Gerçek Satış',
        line=dict(color='#3B82F6', width=2),
        marker=dict(size=6)
    ))
    
    # Tahmin
    if forecast_df is not None and len(forecast_df) > 0:
        fig.add_trace(go.Scatter(
            x=forecast_df['DATE'],
            y=forecast_df['PF_Satis'],
            mode='lines+markers',
            name='Tahmin',
            line=dict(color='#EF4444', width=2, dash='dash'),
            marker=dict(size=6, symbol='diamond')
        ))
    
    fig.update_layout(
        title='Satış Trendi ve Tahmin',
        xaxis_title='Tarih',
        yaxis_title='PF Satış',
        height=400,
        hovermode='x unified'
    )
    
    return fig

def create_competitor_comparison_chart(comp_data):
    """Rakip karşılaştırma grafiği"""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=comp_data['YIL_AY'],
        y=comp_data['PF'],
        name='PF',
        marker_color='#3B82F6'
    ))
    
    fig.add_trace(go.Bar(
        x=comp_data['YIL_AY'],
        y=comp_data['Rakip'],
        name='Rakip',
        marker_color='#EF4444'
    ))
    
    fig.update_layout(
        title='PF vs Rakip Satış Karşılaştırması',
        xaxis_title='Ay',
        yaxis_title='Satış',
        barmode='group',
        height=400
    )
    
    return fig

def create_market_share_trend(comp_data):
    """Pazar payı trend grafiği"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=comp_data['YIL_AY'],
        y=comp_data['PF_Pay_%'],
        mode='lines+markers',
        name='PF Pazar Payı',
        fill='tozeroy',
        line=dict(color='#3B82F6', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=comp_data['YIL_AY'],
        y=comp_data['Rakip_Pay_%'],
        mode='lines+markers',
        name='Rakip Pazar Payı',
        fill='tozeroy',
        line=dict(color='#EF4444', width=2)
    ))
    
    fig.update_layout(
        title='Pazar Payı Trendi (%)',
        xaxis_title='Ay',
        yaxis_title='Pazar Payı (%)',
        height=400,
        yaxis=dict(range=[0, 100])
    )
    
    return fig

def create_growth_comparison(comp_data):
    """Büyüme karşılaştırma grafiği"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=comp_data['YIL_AY'],
        y=comp_data['PF_Buyume'],
        mode='lines+markers',
        name='PF Büyüme',
        line=dict(color='#3B82F6', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=comp_data['YIL_AY'],
        y=comp_data['Rakip_Buyume'],
        mode='lines+markers',
        name='Rakip Büyüme',
        line=dict(color='#EF4444', width=2)
    ))
    
    # Sıfır çizgisi
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title='Büyüme Oranları Karşılaştırması (%)',
        xaxis_title='Ay',
        yaxis_title='Büyüme (%)',
        height=400
    )
    
    return fig

# =============================================================================
# MAIN APP
# =============================================================================

def main():
    # Header
    st.markdown('<h1 class="main-header">🎯 GELİŞMİŞ TİCARİ PORTFÖY ANALİZ SİSTEMİ</h1>', unsafe_allow_html=True)
    st.markdown("**ML Tahminleme • Türkiye Haritası • Rakip Analizi • BCG Matrix • Yatırım Stratejisi**")
    
    # Sidebar
    st.sidebar.header("📂 Dosya Yükleme")
    uploaded_file = st.sidebar.file_uploader(
        "Excel Dosyası Yükleyin",
        type=['xlsx', 'xls'],
        help="Ticari Ürün 2025 verisi"
    )
    
    uploaded_shp = st.sidebar.file_uploader(
        "Türkiye Harita Dosyası (.shp)",
        type=['shp'],
        help="Türkiye şehir sınırları shapefile (opsiyonel)"
    )
    
    if not uploaded_file:
        st.info("👈 Lütfen sol taraftan Excel dosyasını yükleyin")
        st.stop()
    
    # Veriyi yükle
    try:
        df = load_excel_data(uploaded_file)
        geojson = load_geojson()
        
        # Shapefile yükle (eğer varsa)
        turkey_map = None
        if uploaded_shp:
            turkey_map = load_turkey_shapefile(uploaded_shp)
        
        st.sidebar.success(f"✅ {len(df)} satır veri yüklendi")
        if turkey_map is not None:
            st.sidebar.success(f"✅ Harita yüklendi: {len(turkey_map)} şehir")
    except Exception as e:
        st.error(f"❌ Veri yükleme hatası: {str(e)}")
        st.stop()
    
    # Ürün seçimi
    st.sidebar.markdown("---")
    st.sidebar.header("💊 Ürün Seçimi")
    selected_product = st.sidebar.selectbox(
        "Ürün",
        ["TROCMETAM", "CORTIPOL", "DEKSAMETAZON", "PF IZOTONIK"]
    )
    
    # Tarih aralığı seçimi
    st.sidebar.markdown("---")
    st.sidebar.header("📅 Tarih Aralığı")
    
    min_date = df['DATE'].min()
    max_date = df['DATE'].max()
    
    date_option = st.sidebar.selectbox(
        "Dönem Seçin",
        ["Tüm Veriler", "Son 3 Ay", "Son 6 Ay", "Son 1 Yıl", "2025", "2024", "Özel Aralık"]
    )
    
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
        col_date1, col_date2 = st.sidebar.columns(2)
        with col_date1:
            start_date = st.date_input("Başlangıç", min_date, min_value=min_date, max_value=max_date)
        with col_date2:
            end_date = st.date_input("Bitiş", max_date, min_value=min_date, max_value=max_date)
        date_filter = (pd.to_datetime(start_date), pd.to_datetime(end_date))
    
    # Filtreler
    st.sidebar.markdown("---")
    st.sidebar.header("🔍 Filtreler")
    
    territories = ["TÜMÜ"] + sorted(df['TERRITORIES'].unique())
    selected_territory = st.sidebar.selectbox("Territory", territories)
    
    regions = ["TÜMÜ"] + sorted(df['REGION'].unique())
    selected_region = st.sidebar.selectbox("Bölge", regions)
    
    managers = ["TÜMÜ"] + sorted(df['MANAGER'].unique())
    selected_manager = st.sidebar.selectbox("Manager", managers)
    
    # Veriyi filtrele
    df_filtered = df.copy()
    if selected_territory != "TÜMÜ":
        df_filtered = df_filtered[df_filtered['TERRITORIES'] == selected_territory]
    if selected_region != "TÜMÜ":
        df_filtered = df_filtered[df_filtered['REGION'] == selected_region]
    if selected_manager != "TÜMÜ":
        df_filtered = df_filtered[df_filtered['MANAGER'] == selected_manager]
    
    # ==========================================================================
    # TAB YAPISI
    # ==========================================================================
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Genel Bakış",
        "🗺️ Türkiye Haritası",
        "🏢 Territory Analizi", 
        "📈 Zaman Serisi & ML",
        "🎯 Rakip Analizi",
        "⭐ BCG & Strateji",
        "📥 Raporlar"
    ])
    
    # ==========================================================================
    # TAB 1: GENEL BAKIŞ
    # ==========================================================================
    with tab1:
        st.header("📊 Genel Performans Özeti")
        
        cols = get_product_columns(selected_product)
        
        if date_filter:
            df_period = df_filtered[(df_filtered['DATE'] >= date_filter[0]) & 
                                     (df_filtered['DATE'] <= date_filter[1])]
        else:
            df_period = df_filtered
        
        total_pf = df_period[cols['pf']].sum()
        total_rakip = df_period[cols['rakip']].sum()
        total_market = total_pf + total_rakip
        market_share = (total_pf / total_market * 100) if total_market > 0 else 0
        active_territories = df_period['TERRITORIES'].nunique()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("💊 PF Satış", f"{total_pf:,.0f}")
        with col2:
            st.metric("🏪 Toplam Pazar", f"{total_market:,.0f}")
        with col3:
            st.metric("📊 Pazar Payı", f"%{market_share:.1f}")
        with col4:
            st.metric("🏢 Territory Sayısı", active_territories)
        
        st.markdown("---")
        
        st.subheader("🏆 Top 10 Territory")
        terr_perf = calculate_territory_performance(df_filtered, selected_product, date_filter)
        top10 = terr_perf.head(10)
        
        fig_top10 = go.Figure()
        
        fig_top10.add_trace(go.Bar(
            x=top10['Territory'],
            y=top10['PF_Satis'],
            name='PF Satış',
            marker_color='#3B82F6',
            text=top10['PF_Satis'].apply(lambda x: f'{x:,.0f}'),
            textposition='outside'
        ))
        
        fig_top10.add_trace(go.Bar(
            x=top10['Territory'],
            y=top10['Rakip_Satis'],
            name='Rakip Satış',
            marker_color='#EF4444',
            text=top10['Rakip_Satis'].apply(lambda x: f'{x:,.0f}'),
            textposition='outside'
        ))
        
        fig_top10.update_layout(
            title='Top 10 Territory - PF vs Rakip',
            xaxis_title='Territory',
            yaxis_title='Satış',
            barmode='group',
            height=500,
            xaxis=dict(tickangle=-45)
        )
        
        st.plotly_chart(fig_top10, use_container_width=True)
    
    # ==========================================================================
    # TAB 2: TÜRKİYE HARİTASI
    # ==========================================================================
    with tab2:
        st.header("🗺️ Türkiye İl Bazlı Satış Haritası")
        
        city_data = calculate_city_performance(df_filtered, selected_product, date_filter)
        
        col1, col2, col3, col4 = st.columns(4)
        
        total_pf = city_data['PF_Satis'].sum()
        total_market = city_data['Toplam_Pazar'].sum()
        avg_share = city_data['Pazar_Payi_%'].mean()
        active_cities = len(city_data[city_data['PF_Satis'] > 0])
        
        with col1:
            st.metric("💊 Toplam PF Satış", f"{total_pf:,.0f}")
        with col2:
            st.metric("🏪 Toplam Pazar", f"{total_market:,.0f}")
        with col3:
            st.metric("📊 Ort. Pazar Payı", f"%{avg_share:.1f}")
        with col4:
            st.metric("🏙️ Aktif Şehir", active_cities)
        
        st.markdown("---")
        
        # Harita göster
        st.subheader("📍 Şehir Bazlı Satış Dağılımı")
        
        # Shapefile varsa onu kullan, yoksa GeoJSON
        if turkey_map is not None:
            st.info("🗺️ Geopandas Shapefile ile oluşturulan harita")
            turkey_fig = create_turkey_choropleth_map(city_data, turkey_map)
            if turkey_fig:
                st.plotly_chart(turkey_fig, use_container_width=True)
        elif geojson:
            st.info("🗺️ GeoJSON ile oluşturulan harita")
            turkey_fig = create_turkey_map(city_data, geojson, 
                                          f"{selected_product} - Şehir Bazlı Satış Dağılımı")
            if turkey_fig:
                st.plotly_chart(turkey_fig, use_container_width=True)
        else:
            st.warning("⚠️ Harita dosyası yüklenmedi. Lütfen sidebar'dan .shp veya GeoJSON yükleyin")
        
        st.markdown("---")
        
        # Top şehirler
        st.subheader("🏆 Top 10 Şehir")
        top_cities = city_data.nlargest(10, 'PF_Satis')
        
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            fig_bar = px.bar(
                top_cities,
                x='City_Original',
                y='PF_Satis',
                title='En Yüksek Satış Yapan Şehirler',
                color='Pazar_Payi_%',
                color_continuous_scale='Blues'
            )
            fig_bar.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col_chart2:
            fig_pie = px.pie(
                top_cities,
                values='PF_Satis',
                names='City_Original',
                title='Top 10 Şehir Satış Dağılımı'
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Detaylı tablo
        st.markdown("---")
        st.subheader("📋 Detaylı Şehir Listesi")
        
        city_display = city_data.sort_values('PF_Satis', ascending=False).copy()
        city_display = city_display[['City_Original', 'PF_Satis', 'Rakip_Satis', 
                                      'Toplam_Pazar', 'Pazar_Payi_%']]
        city_display.columns = ['Şehir', 'PF Satış', 'Rakip Satış', 'Toplam Pazar', 'Pazar Payı %']
        city_display.index = range(1, len(city_display) + 1)
        
        st.dataframe(
            city_display.style.format({
                'PF Satış': '{:,.0f}',
                'Rakip Satış': '{:,.0f}',
                'Toplam Pazar': '{:,.0f}',
                'Pazar Payı %': '{:.1f}'
            }).background_gradient(subset=['Pazar Payı %'], cmap='RdYlGn'),
            use_container_width=True,
            height=400
        )
    
    # ==========================================================================
    # TAB 3-7: Diğer tablar aynı kalacak (önceki koddan devam)
    # ==========================================================================
    # (Kalan tablar için önceki kodu kullan - çok uzun olduğu için kesiyorum)
    
    with tab3:
        st.header("🏢 Territory Bazlı Detaylı Analiz")
        terr_perf = calculate_territory_performance(df_filtered, selected_product, date_filter)
        
        col_f1, col_f2 = st.columns([1, 3])
        with col_f1:
            sort_by = st.selectbox(
                "Sıralama",
                ['PF_Satis', 'Pazar_Payi_%', 'Toplam_Pazar', 'Agirlik_%'],
                format_func=lambda x: {
                    'PF_Satis': 'PF Satış',
                    'Pazar_Payi_%': 'Pazar Payı %',
                    'Toplam_Pazar': 'Toplam Pazar',
                    'Agirlik_%': 'Ağırlık %'
                }[x]
            )
        with col_f2:
            show_n = st.slider("Gösterilecek Territory Sayısı", 10, 50, 20)
        
        terr_sorted = terr_perf.sort_values(sort_by, ascending=False).head(show_n)
        
        col_v1, col_v2 = st.columns(2)
        
        with col_v1:
            st.markdown("#### 📊 PF vs Rakip Satış")
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(x=terr_sorted['Territory'], y=terr_sorted['PF_Satis'], 
                                     name='PF Satış', marker_color='#3B82F6'))
            fig_bar.add_trace(go.Bar(x=terr_sorted['Territory'], y=terr_sorted['Rakip_Satis'], 
                                     name='Rakip Satış', marker_color='#EF4444'))
            fig_bar.update_layout(barmode='group', height=500, xaxis=dict(tickangle=-45))
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col_v2:
            st.markdown("#### 🎯 Pazar Payı Dağılımı")
            fig_pie = px.pie(terr_sorted.head(10), values='PF_Satis', names='Territory',
                            title='Top 10 Territory - PF Satış Dağılımı')
            fig_pie.update_layout(height=500)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        st.markdown("---")
        st.subheader("📋 Detaylı Territory Listesi")
        
        display_cols = ['Territory', 'Region', 'City', 'Manager', 'PF_Satis', 'Rakip_Satis', 
                       'Toplam_Pazar', 'Pazar_Payi_%', 'Goreceli_Pazar_Payi', 'Agirlik_%']
        terr_display = terr_sorted[display_cols].copy()
        terr_display.columns = ['Territory', 'Region', 'City', 'Manager', 'PF Satış', 'Rakip Satış',
                               'Toplam Pazar', 'Pazar Payı %', 'Göreceli Pay', 'Ağırlık %']
        terr_display.index = range(1, len(terr_display) + 1)
        
        st.dataframe(
            terr_display.style.format({
                'PF Satış': '{:,.0f}', 'Rakip Satış': '{:,.0f}', 'Toplam Pazar': '{:,.0f}',
                'Pazar Payı %': '{:.1f}', 'Göreceli Pay': '{:.2f}', 'Ağırlık %': '{:.1f}'
            }).background_gradient(subset=['Pazar Payı %'], cmap='RdYlGn'),
            use_container_width=True
        )

    with tab4:
        st.header("📈 Zaman Serisi Analizi & ML Tahminleme")
        
        territory_for_ts = st.selectbox(
            "Territory Seçin",
            ["TÜMÜ"] + sorted(df_filtered['TERRITORIES'].unique()),
            key='ts_territory'
        )
        
        monthly_df = calculate_time_series(df_filtered, selected_product, territory_for_ts, date_filter)
        
        if len(monthly_df) > 0:
            st.subheader("📊 Zaman Serisi Analizi")
            
            col_ts1, col_ts2, col_ts3, col_ts4 = st.columns(4)
            with col_ts1:
                st.metric("📊 Ort. Aylık PF", f"{monthly_df['PF_Satis'].mean():,.0f}")
            with col_ts2:
                st.metric("📈 Ort. Büyüme", f"%{monthly_df['PF_Buyume_%'].mean():.1f}")
            with col_ts3:
                st.metric("🎯 Ort. Pazar Payı", f"%{monthly_df['Pazar_Payi_%'].mean():.1f}")
            with col_ts4:
                st.metric("📅 Veri Dönemi", f"{len(monthly_df)} ay")
            
            st.markdown("---")
            
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                st.markdown("#### 📊 Satış Trendi")
                fig_ts = go.Figure()
                fig_ts.add_trace(go.Scatter(x=monthly_df['DATE'], y=monthly_df['PF_Satis'],
                                           mode='lines+markers', name='PF Satış',
                                           line=dict(color='#3B82F6', width=3), marker=dict(size=8)))
                fig_ts.add_trace(go.Scatter(x=monthly_df['DATE'], y=monthly_df['Rakip_Satis'],
                                           mode='lines+markers', name='Rakip Satış',
                                           line=dict(color='#EF4444', width=3), marker=dict(size=8)))
                fig_ts.add_trace(go.Scatter(x=monthly_df['DATE'], y=monthly_df['MA_3'],
                                           mode='lines', name='3 Aylık Ort.',
                                           line=dict(color='#10B981', width=2, dash='dash')))
                fig_ts.update_layout(xaxis_title='Tarih', yaxis_title='Satış', height=400)
                st.plotly_chart(fig_ts, use_container_width=True)
            
            with col_chart2:
                st.markdown("#### 🎯 Pazar Payı Trendi")
                fig_share = go.Figure()
                fig_share.add_trace(go.Scatter(x=monthly_df['DATE'], y=monthly_df['Pazar_Payi_%'],
                                              mode='lines+markers', fill='tozeroy',
                                              line=dict(color='#8B5CF6', width=2), marker=dict(size=8)))
                fig_share.update_layout(xaxis_title='Tarih', yaxis_title='Pazar Payı (%)', height=400)
                st.plotly_chart(fig_share, use_container_width=True)
            
            st.markdown("---")
            st.subheader("🤖 Machine Learning Satış Tahmini")
            
            forecast_months = st.slider("Tahmin Periyodu (Ay)", 1, 6, 3)
            
            if len(monthly_df) >= 3:
                forecast_df = simple_forecast(monthly_df, forecast_months)
                
                col_ml1, col_ml2, col_ml3 = st.columns(3)
                last_actual = monthly_df['PF_Satis'].iloc[-1]
                first_forecast = forecast_df['PF_Satis'].iloc[0] if forecast_df is not None else 0
                change = ((first_forecast - last_actual) / last_actual * 100) if last_actual > 0 else 0
                
                with col_ml1:
                    st.metric("📊 Son Gerçek Satış", f"{last_actual:,.0f}")
                with col_ml2:
                    st.metric("🔮 İlk Tahmin", f"{first_forecast:,.0f}", delta=f"%{change:.1f}")
                with col_ml3:
                    avg_forecast = forecast_df['PF_Satis'].mean() if forecast_df is not None else 0
                    st.metric("📈 Ort. Tahmin", f"{avg_forecast:,.0f}")
                
                st.markdown("---")
                forecast_chart = create_forecast_chart(monthly_df, forecast_df)
                st.plotly_chart(forecast_chart, use_container_width=True)
                
                if forecast_df is not None:
                    forecast_display = forecast_df[['YIL_AY', 'PF_Satis']].copy()
                    forecast_display.columns = ['Ay', 'Tahmin Edilen Satış']
                    forecast_display.index = range(1, len(forecast_display) + 1)
                    st.dataframe(forecast_display.style.format({'Tahmin Edilen Satış': '{:,.0f}'}),
                                use_container_width=True)

    with tab5:
        st.header("📊 Detaylı Rakip Analizi")
        
        comp_data = calculate_competitor_analysis(df_filtered, selected_product, date_filter)
        
        if len(comp_data) > 0:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("🎯 Ort. PF Pazar Payı", f"%{comp_data['PF_Pay_%'].mean():.1f}")
            with col2:
                st.metric("📈 Ort. PF Büyüme", f"%{comp_data['PF_Buyume'].mean():.1f}")
            with col3:
                st.metric("📉 Ort. Rakip Büyüme", f"%{comp_data['Rakip_Buyume'].mean():.1f}")
            with col4:
                win_months = len(comp_data[comp_data['Fark'] > 0])
                st.metric("🏆 Kazanılan Aylar", f"{win_months}/{len(comp_data)}")
            
            st.markdown("---")
            
            col_g1, col_g2 = st.columns(2)
            with col_g1:
                st.subheader("💰 Satış Karşılaştırması")
                st.plotly_chart(create_competitor_comparison_chart(comp_data), use_container_width=True)
            with col_g2:
                st.subheader("📊 Pazar Payı Trendi")
                st.plotly_chart(create_market_share_trend(comp_data), use_container_width=True)
            
            st.markdown("---")
            st.subheader("📈 Büyüme Karşılaştırması")
            st.plotly_chart(create_growth_comparison(comp_data), use_container_width=True)
            
            st.markdown("---")
            st.subheader("📋 Aylık Performans Detayları")
            
            comp_display = comp_data[['YIL_AY', 'PF', 'Rakip', 'PF_Pay_%', 'PF_Buyume', 
                                      'Rakip_Buyume', 'Fark']].copy()
            comp_display.columns = ['Ay', 'PF Satış', 'Rakip Satış', 'PF Pay %',
                                   'PF Büyüme %', 'Rakip Büyüme %', 'Fark %']
            
            def highlight_winner(row):
                if row['Fark %'] > 0:
                    return ['background-color: #d4edda'] * len(row)
                elif row['Fark %'] < 0:
                    return ['background-color: #f8d7da'] * len(row)
                else:
                    return [''] * len(row)
            
            st.dataframe(
                comp_display.style.format({
                    'PF Satış': '{:,.0f}', 'Rakip Satış': '{:,.0f}', 'PF Pay %': '{:.1f}',
                    'PF Büyüme %': '{:.1f}', 'Rakip Büyüme %': '{:.1f}', 'Fark %': '{:.1f}'
                }).apply(highlight_winner, axis=1),
                use_container_width=True, height=400
            )

    with tab6:
        st.header("⭐ BCG Matrix & Yatırım Stratejisi")
        
        bcg_df = calculate_bcg_matrix(df_filtered, selected_product, date_filter)
        
        st.subheader("📊 Portföy Dağılımı")
        bcg_counts = bcg_df['BCG_Kategori'].value_counts()
        
        col_bcg1, col_bcg2, col_bcg3, col_bcg4 = st.columns(4)
        
        categories = ["⭐ Yıldız", "💰 Nakit İnek", "❓ Soru İşareti", "🐕 Köpek"]
        cols = [col_bcg1, col_bcg2, col_bcg3, col_bcg4]
        
        for cat, col in zip(categories, cols):
            with col:
                count = bcg_counts.get(cat, 0)
                pf = bcg_df[bcg_df['BCG_Kategori'] == cat]['PF_Satis'].sum()
                st.metric(cat, f"{count}", delta=f"{pf:,.0f} PF")
        
        st.markdown("---")
        st.subheader("🎯 BCG Matrix")
        
        color_map = {
            "⭐ Yıldız": "#FFD700",
            "💰 Nakit İnek": "#10B981",
            "❓ Soru İşareti": "#3B82F6",
            "🐕 Köpek": "#9CA3AF"
        }
        
        fig_bcg = px.scatter(
            bcg_df, x='Goreceli_Pazar_Payi', y='Pazar_Buyume_%', size='PF_Satis',
            color='BCG_Kategori', color_discrete_map=color_map, hover_name='Territory',
            hover_data={'PF_Satis': ':,.0f', 'Pazar_Payi_%': ':.1f'}, size_max=50
        )
        
        median_share = bcg_df['Goreceli_Pazar_Payi'].median()
        median_growth = bcg_df['Pazar_Buyume_%'].median()
        fig_bcg.add_hline(y=median_growth, line_dash="dash", line_color="rgba(255,255,255,0.4)")
        fig_bcg.add_vline(x=median_share, line_dash="dash", line_color="rgba(255,255,255,0.4)")
        fig_bcg.update_layout(height=600)
        
        st.plotly_chart(fig_bcg, use_container_width=True)
        
        st.markdown("---")
        st.subheader("💡 Yatırım Stratejileri")
        
        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            st.info("**⭐ YILDIZLAR:** Yüksek büyüme + Yüksek pay → Yatırımı artır")
            st.success("**💰 NAKİT İNEKLERİ:** Düşük büyüme + Yüksek pay → Verimliliği optimize et")
        with col_exp2:
            st.warning("**❓ SORU İŞARETLERİ:** Yüksek büyüme + Düşük pay → Agresif yatırım yap")
            st.error("**🐕 KÖPEKLER:** Düşük büyüme + Düşük pay → Çıkışı değerlendir")
        
        st.markdown("---")
        st.subheader("📋 BCG Kategori Detayları")
        
        bcg_display = bcg_df[['Territory', 'Region', 'BCG_Kategori', 'Strateji', 'PF_Satis',
                              'Pazar_Payi_%', 'Goreceli_Pazar_Payi', 'Pazar_Buyume_%']].copy()
        bcg_display.columns = ['Territory', 'Region', 'BCG', 'Strateji', 'PF Satış',
                              'Pazar Payı %', 'Göreceli Pay', 'Büyüme %']
        bcg_display = bcg_display.sort_values('PF Satış', ascending=False)
        bcg_display.index = range(1, len(bcg_display) + 1)
        
        st.dataframe(
            bcg_display.style.format({
                'PF Satış': '{:,.0f}', 'Pazar Payı %': '{:.1f}',
                'Göreceli Pay': '{:.2f}', 'Büyüme %': '{:.1f}'
            }),
            use_container_width=True
        )

    with tab7:
        st.header("📥 Rapor İndirme")
        st.markdown("Detaylı analizlerin Excel raporlarını indirebilirsiniz.")
        
        if st.button("📥 Excel Raporu Oluştur", type="primary"):
            with st.spinner("Rapor hazırlanıyor..."):
                terr_perf = calculate_territory_performance(df_filtered, selected_product, date_filter)
                monthly_df = calculate_time_series(df_filtered, selected_product, None, date_filter)
                bcg_df = calculate_bcg_matrix(df_filtered, selected_product, date_filter)
                city_data = calculate_city_performance(df_filtered, selected_product, date_filter)
                comp_data = calculate_competitor_analysis(df_filtered, selected_product, date_filter)
                
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    terr_perf.to_excel(writer, sheet_name='Territory Performans', index=False)
                    monthly_df.to_excel(writer, sheet_name='Zaman Serisi', index=False)
                    bcg_df.to_excel(writer, sheet_name='BCG Matrix', index=False)
                    city_data.to_excel(writer, sheet_name='Şehir Analizi', index=False)
                    comp_data.to_excel(writer, sheet_name='Rakip Analizi', index=False)
                
                st.success("✅ Rapor hazır!")
                st.download_button(
                    label="💾 Excel Raporunu İndir",
                    data=output.getvalue(),
                    file_name=f"ticari_portfoy_raporu_{selected_product}_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

if __name__ == "__main__":
    main()
