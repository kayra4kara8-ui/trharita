"""
🎯 GELİŞMİŞ TİCARİ PORTFÖY ANALİZ SİSTEMİ
Territory Bazlı Performans, ML Tahminleme, Türkiye Haritası ve Rekabet Analizi

Özellikler:
- 🗺️ Geopandas ile Türkiye il bazlı harita görselleştirme  
- 🤖 Machine Learning satış tahminleme
- 📊 Aylık/Yıllık dönem seçimi
- 📈 Gelişmiş rakip analizi ve trend karşılaştırması
- 🎯 Dinamik zaman aralığı filtreleme
- 💼 BCG Matrix ve Yatırım Stratejisi
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
# ŞEHİR İSİM HARİTALAMA ve BÖLGE RENKLERİ
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
    "TEKIRDAG": "TEKİRDAĞ"
}

REGION_COLORS = {
    "KUZEY ANADOLU": "#2E8B57",
    "MARMARA": "#2F6FD6",
    "İÇ ANADOLU": "#8B6B4A",
    "BATI ANADOLU": "#2BB0A6",
    "GÜNEY DOĞU ANADOLU": "#A05A2C"
}

# =============================================================================
# HEADER
# =============================================================================
st.markdown('<h1 class="main-header">🎯 Gelişmiş Ticari Portföy Analiz Sistemi</h1>', 
            unsafe_allow_html=True)

# =============================================================================
# SIDEBAR - DOSYA YÜKLEME VE FİLTRELER
# =============================================================================
with st.sidebar:
    st.header("📁 Veri Yükleme")
    
    uploaded_file = st.file_uploader(
        "Excel Dosyası Yükleyin",
        type=['xlsx', 'xls'],
        help="Ticari ürün verilerin içeren Excel dosyasını yükleyin"
    )
    
    uploaded_shp = st.file_uploader(
        "Türkiye Harita Dosyası (.shp)",
        type=['shp'],
        help="Türkiye şehir sınırları shapefile dosyası"
    )
    
    if not uploaded_file:
        st.warning("⚠️ Lütfen Excel dosyasını yükleyin")
        st.stop()

# =============================================================================
# VERİ YÜKLEME
# =============================================================================
@st.cache_data
def load_excel_data(file):
    df = pd.read_excel(file)
    
    # Tarih sütunu oluştur
    if 'YIL_AY' in df.columns:
        df['DATE'] = pd.to_datetime(df['YIL_AY'].astype(str) + '01', format='%Y%m%d', errors='coerce')
    
    # Şehir isimlerini büyük harfe çevir
    if 'CITY' in df.columns:
        df['CITY'] = df['CITY'].str.upper()
    
    return df

@st.cache_data
def load_turkey_map(_shp_file):
    """Türkiye haritasını yükle"""
    try:
        gdf = gpd.read_file(_shp_file)
        gdf["name"] = gdf["name"].str.upper()
        gdf["CITY_CLEAN"] = gdf["name"].replace(CITY_FIX_MAP).str.upper()
        return gdf
    except:
        return None

df = load_excel_data(uploaded_file)

# Harita yükle
turkey_map = None
if uploaded_shp:
    turkey_map = load_turkey_map(uploaded_shp)

# =============================================================================
# PRODUCT SEÇİMİ VE FİLTRELER
# =============================================================================
with st.sidebar:
    st.markdown("---")
    st.header("🎯 Ürün Seçimi")
    
    available_products = []
    if 'TROCMETAM' in df.columns:
        available_products.append('TROCMETAM')
    if 'CORTIPOL' in df.columns:
        available_products.append('CORTIPOL')
    if 'DEKSAMETAZON' in df.columns:
        available_products.append('DEKSAMETAZON')
    if 'PF IZOTONIK' in df.columns:
        available_products.append('PF IZOTONIK')
    
    selected_product = st.selectbox(
        "Ürün Seçiniz",
        available_products,
        index=0 if available_products else None
    )
    
    st.markdown("---")
    st.header("📅 Tarih Aralığı")
    
    if 'DATE' in df.columns:
        min_date = df['DATE'].min()
        max_date = df['DATE'].max()
        
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            start_date = st.date_input("Başlangıç", value=min_date, min_value=min_date, max_value=max_date)
        with col_d2:
            end_date = st.date_input("Bitiş", value=max_date, min_value=min_date, max_value=max_date)
        
        date_filter = (pd.Timestamp(start_date), pd.Timestamp(end_date))
    else:
        date_filter = None
        st.info("Tarih sütunu bulunamadı")
    
    st.markdown("---")
    st.header("🔍 Filtreler")
    
    # Territory filtresi
    selected_territory = "TÜMÜ"
    if 'TERRITORIES' in df.columns:
        territories = ["TÜMÜ"] + sorted(df['TERRITORIES'].dropna().unique().tolist())
        selected_territory = st.selectbox("Territory", territories)
    
    # Region filtresi
    selected_region = "TÜMÜ"
    if 'REGION' in df.columns:
        regions = ["TÜMÜ"] + sorted(df['REGION'].dropna().unique().tolist())
        selected_region = st.selectbox("Region", regions)
    
    # Manager filtresi
    selected_manager = "TÜMÜ"
    if 'MANAGER' in df.columns:
        managers = ["TÜMÜ"] + sorted(df['MANAGER'].dropna().unique().tolist())
        selected_manager = st.selectbox("Manager", managers)

# =============================================================================
# YARDIMCI FONKSİYONLAR
# =============================================================================
def get_product_columns(product):
    """Ürüne göre sütun isimlerini döndür"""
    if product == 'TROCMETAM':
        return {'pf': 'TROCMETAM', 'rakip': 'TROPIKAL'}
    elif product == 'CORTIPOL':
        return {'pf': 'CORTIPOL', 'rakip': 'FENILEFRIN'}
    elif product == 'DEKSAMETAZON':
        return {'pf': 'DEKSAMETAZON', 'rakip': 'DEKSAMETAZON RAKIP'}
    elif product == 'PF IZOTONIK':
        return {'pf': 'PF IZOTONIK', 'rakip': 'IZOTONIK RAKIP'}
    return {}

def calculate_city_performance(df, product, date_filter=None):
    """Şehir bazlı performans hesapla"""
    cols = get_product_columns(product)
    if not cols or 'CITY' not in df.columns:
        return pd.DataFrame()
    
    df_calc = df.copy()
    if date_filter:
        df_calc = df_calc[(df_calc['DATE'] >= date_filter[0]) & (df_calc['DATE'] <= date_filter[1])]
    
    city_perf = df_calc.groupby('CITY').agg({
        cols['pf']: 'sum',
        cols['rakip']: 'sum'
    }).reset_index()
    
    city_perf.columns = ['City', 'PF_Satis', 'Rakip_Satis']
    city_perf['Toplam_Pazar'] = city_perf['PF_Satis'] + city_perf['Rakip_Satis']
    city_perf['Pazar_Payi_%'] = np.where(
        city_perf['Toplam_Pazar'] != 0,
        (city_perf['PF_Satis'] / city_perf['Toplam_Pazar']) * 100,
        0
    )
    
    return city_perf[city_perf['Toplam_Pazar'] > 0].sort_values('PF_Satis', ascending=False)

def calculate_territory_performance(df, product, date_filter=None):
    """Territory bazlı performans hesapla"""
    cols = get_product_columns(product)
    if not cols:
        return pd.DataFrame()
    
    df_calc = df.copy()
    if date_filter:
        df_calc = df_calc[(df_calc['DATE'] >= date_filter[0]) & (df_calc['DATE'] <= date_filter[1])]
    
    group_cols = ['TERRITORIES']
    if 'REGION' in df_calc.columns:
        group_cols.append('REGION')
    if 'CITY' in df_calc.columns:
        group_cols.append('CITY')
    if 'MANAGER' in df_calc.columns:
        group_cols.append('MANAGER')
    
    terr_perf = df_calc.groupby(group_cols).agg({
        cols['pf']: 'sum',
        cols['rakip']: 'sum'
    }).reset_index()
    
    terr_perf.columns = group_cols + ['PF_Satis', 'Rakip_Satis']
    terr_perf['Toplam_Pazar'] = terr_perf['PF_Satis'] + terr_perf['Rakip_Satis']
    terr_perf['Pazar_Payi_%'] = np.where(
        terr_perf['Toplam_Pazar'] != 0,
        (terr_perf['PF_Satis'] / terr_perf['Toplam_Pazar']) * 100,
        0
    )
    
    total_market = terr_perf['Toplam_Pazar'].sum()
    terr_perf['Agirlik_%'] = np.where(
        total_market != 0,
        (terr_perf['Toplam_Pazar'] / total_market) * 100,
        0
    )
    
    terr_perf.columns = ['Territory'] + list(terr_perf.columns[1:])
    
    return terr_perf[terr_perf['Toplam_Pazar'] > 0].sort_values('PF_Satis', ascending=False)

def calculate_time_series(df, product, territory="TÜMÜ", date_filter=None):
    """Zaman serisi analizi"""
    cols = get_product_columns(product)
    if not cols or 'YIL_AY' not in df.columns:
        return pd.DataFrame()
    
    df_calc = df.copy()
    
    if date_filter:
        df_calc = df_calc[(df_calc['DATE'] >= date_filter[0]) & (df_calc['DATE'] <= date_filter[1])]
    
    if territory != "TÜMÜ" and 'TERRITORIES' in df_calc.columns:
        df_calc = df_calc[df_calc['TERRITORIES'] == territory]
    
    ts = df_calc.groupby('YIL_AY').agg({
        cols['pf']: 'sum',
        cols['rakip']: 'sum'
    }).reset_index()
    
    ts.columns = ['YIL_AY', 'PF_Satis', 'Rakip_Satis']
    ts['Toplam_Pazar'] = ts['PF_Satis'] + ts['Rakip_Satis']
    ts['Pazar_Payi_%'] = np.where(
        ts['Toplam_Pazar'] != 0,
        (ts['PF_Satis'] / ts['Toplam_Pazar']) * 100,
        0
    )
    
    # Hareketli ortalamalar
    ts['MA_3'] = ts['PF_Satis'].rolling(window=3, min_periods=1).mean()
    ts['MA_6'] = ts['PF_Satis'].rolling(window=6, min_periods=1).mean()
    
    # Büyüme oranı
    ts['Buyume_%'] = ts['PF_Satis'].pct_change() * 100
    
    ts = ts.sort_values('YIL_AY')
    
    return ts

def simple_forecast(ts_data, months=3):
    """Basit tahminleme modeli"""
    if len(ts_data) < 3:
        return None
    
    # Son 6 ayın ortalaması ve trendi
    last_6 = ts_data['PF_Satis'].tail(6).values
    avg = np.mean(last_6)
    
    # Basit trend hesaplama
    if len(last_6) >= 2:
        trend = (last_6[-1] - last_6[0]) / len(last_6)
    else:
        trend = 0
    
    # Tahmin
    forecast = []
    for i in range(1, months + 1):
        pred = avg + (trend * i)
        forecast.append(max(0, pred))  # Negatif değerleri önle
    
    return forecast

def calculate_competitor_analysis(df, product, date_filter=None):
    """Rakip analizi hesapla"""
    cols = get_product_columns(product)
    if not cols or 'YIL_AY' not in df.columns:
        return pd.DataFrame()
    
    df_calc = df.copy()
    if date_filter:
        df_calc = df_calc[(df_calc['DATE'] >= date_filter[0]) & (df_calc['DATE'] <= date_filter[1])]
    
    monthly = df_calc.groupby('YIL_AY').agg({
        cols['pf']: 'sum',
        cols['rakip']: 'sum'
    }).reset_index()
    
    monthly.columns = ['YIL_AY', 'PF', 'Rakip']
    monthly['Toplam'] = monthly['PF'] + monthly['Rakip']
    monthly['PF_Pay_%'] = np.where(
        monthly['Toplam'] != 0,
        (monthly['PF'] / monthly['Toplam']) * 100,
        0
    )
    monthly['Rakip_Pay_%'] = np.where(
        monthly['Toplam'] != 0,
        (monthly['Rakip'] / monthly['Toplam']) * 100,
        0
    )
    monthly['Fark'] = monthly['PF'] - monthly['Rakip']
    monthly['Durum'] = monthly['Fark'].apply(lambda x: '🟢 Kazanç' if x > 0 else '🔴 Kayıp')
    
    # Büyüme oranları
    monthly['PF_Buyume_%'] = monthly['PF'].pct_change() * 100
    monthly['Rakip_Buyume_%'] = monthly['Rakip'].pct_change() * 100
    
    return monthly.sort_values('YIL_AY')

def calculate_bcg_matrix(df, product, date_filter=None):
    """BCG Matrix hesapla"""
    terr_perf = calculate_territory_performance(df, product, date_filter)
    
    if terr_perf.empty:
        return pd.DataFrame()
    
    # Median hesapla
    median_share = terr_perf['Pazar_Payi_%'].median()
    median_weight = terr_perf['Agirlik_%'].median()
    
    # BCG kategorisi belirle
    def get_bcg_category(row):
        if row['Pazar_Payi_%'] >= median_share and row['Agirlik_%'] >= median_weight:
            return '⭐ Yıldız'
        elif row['Pazar_Payi_%'] >= median_share and row['Agirlik_%'] < median_weight:
            return '❓ Soru İşareti'
        elif row['Pazar_Payi_%'] < median_share and row['Agirlik_%'] >= median_weight:
            return '💰 Nakit İnek'
        else:
            return '🐕 Köpek'
    
    terr_perf['BCG_Kategori'] = terr_perf.apply(get_bcg_category, axis=1)
    
    # Yatırım stratejisi
    def get_strategy(category):
        if '⭐' in category:
            return '🚀 Büyümeye Yatırım - Lider konumu koruyun'
        elif '❓' in category:
            return '🎯 Seçici Yatırım - Pazar payını artırın'
        elif '💰' in category:
            return '💵 Nakit Üretimi - Verimliliği optimize edin'
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

def create_turkey_choropleth_map(city_data, turkey_gdf, region_col='REGION'):
    """Türkiye bölge bazlı choropleth harita"""
    if turkey_gdf is None or city_data.empty:
        return None
    
    # Verileri birleştir
    merged = turkey_gdf.merge(
        city_data,
        left_on='CITY_CLEAN',
        right_on='City',
        how='left'
    )
    
    merged['PF_Satis'] = merged['PF_Satis'].fillna(0)
    
    # Eğer region bilgisi varsa, bölge bazlı topla
    if region_col in merged.columns:
        region_sum = merged.groupby(region_col, as_index=False)['PF_Satis'].sum()
        region_map = merged[[region_col, 'geometry']].dissolve(by=region_col).reset_index()
        region_map = region_map.merge(region_sum, on=region_col, how='left')
        
        # Choropleth oluştur
        fig = px.choropleth(
            region_map,
            geojson=region_map.__geo_interface__,
            locations=region_col,
            featureidkey=f'properties.{region_col}',
            color=region_col,
            color_discrete_map=REGION_COLORS,
            hover_name=region_col,
            hover_data={'PF_Satis': ':,'}
        )
    else:
        # Region yoksa şehir bazlı
        fig = px.choropleth(
            merged,
            geojson=merged.__geo_interface__,
            locations='CITY_CLEAN',
            featureidkey='properties.CITY_CLEAN',
            color='PF_Satis',
            color_continuous_scale='Blues',
            hover_name='CITY_CLEAN',
            hover_data={'PF_Satis': ':,'}
        )
    
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    
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
            "PF Satış: " + pts["PF_Satis"].fillna(0).astype(int).map(lambda x: f"{x:,}")
        ),
        showlegend=False
    )
    
    return fig

# =============================================================================
# VERİYİ FİLTRELE
# =============================================================================
df_filtered = df.copy()
if selected_territory != "TÜMÜ":
    df_filtered = df_filtered[df_filtered['TERRITORIES'] == selected_territory]
if selected_region != "TÜMÜ":
    df_filtered = df_filtered[df_filtered['REGION'] == selected_region]
if selected_manager != "TÜMÜ":
    df_filtered = df_filtered[df_filtered['MANAGER'] == selected_manager]

# =============================================================================
# TAB YAPISI
# =============================================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Genel Bakış",
    "🗺️ Türkiye Haritası",
    "🏢 Territory Analizi",
    "📈 Zaman Serisi & ML",
    "🎯 Rakip Analizi",
    "⭐ BCG & Strateji",
    "📥 Raporlar"
])

# =============================================================================
# TAB 1: GENEL BAKIŞ
# =============================================================================
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
    
    # Top 10 Territory
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

# =============================================================================
# TAB 2: TÜRKİYE HARİTASI
# =============================================================================
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
    
    # Harita
    if turkey_map is not None:
        st.subheader("📍 Bölge Bazlı Dağılım")
        
        # Harita oluştur
        turkey_fig = create_turkey_choropleth_map(city_data, turkey_map, 'REGION')
        
        if turkey_fig:
            st.plotly_chart(turkey_fig, use_container_width=True)
        else:
            st.warning("Harita oluşturulamadı")
    else:
        st.info("📌 Türkiye haritasını görmek için .shp dosyasını yükleyin")
    
    st.markdown("---")
    
    # Top şehirler
    st.subheader("🏆 Top 10 Şehir")
    top_cities = city_data.nlargest(10, 'PF_Satis')
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        fig_bar = px.bar(
            top_cities,
            x='City',
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
            names='City',
            title='Top 10 Şehir Satış Dağılımı'
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Detaylı tablo
    st.markdown("---")
    st.subheader("📋 Detaylı Şehir Listesi")
    
    city_display = city_data.sort_values('PF_Satis', ascending=False).copy()
    city_display.index = range(1, len(city_display) + 1)
    
    st.dataframe(
        city_display.style.format({
            'PF_Satis': '{:,.0f}',
            'Rakip_Satis': '{:,.0f}',
            'Toplam_Pazar': '{:,.0f}',
            'Pazar_Payi_%': '{:.1f}'
        }).background_gradient(subset=['Pazar_Payi_%'], cmap='RdYlGn'),
        use_container_width=True,
        height=400
    )

# =============================================================================
# TAB 3: TERRITORY ANALİZİ
# =============================================================================
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
    
    # Visualizations
    col_v1, col_v2 = st.columns(2)
    
    with col_v1:
        st.markdown("#### 📊 PF vs Rakip Satış")
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=terr_sorted['Territory'],
            y=terr_sorted['PF_Satis'],
            name='PF',
            marker_color='#3B82F6'
        ))
        fig_bar.add_trace(go.Bar(
            x=terr_sorted['Territory'],
            y=terr_sorted['Rakip_Satis'],
            name='Rakip',
            marker_color='#EF4444'
        ))
        fig_bar.update_layout(barmode='group', xaxis_tickangle=-45, height=400)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col_v2:
        st.markdown("#### 📈 Pazar Payı Dağılımı")
        fig_scatter = px.scatter(
            terr_sorted,
            x='Agirlik_%',
            y='Pazar_Payi_%',
            size='Toplam_Pazar',
            hover_name='Territory',
            color='Pazar_Payi_%',
            color_continuous_scale='RdYlGn'
        )
        fig_scatter.update_layout(height=400)
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Detaylı tablo
    st.markdown("---")
    st.subheader("📋 Detaylı Territory Listesi")
    
    terr_display = terr_sorted.copy()
    terr_display.index = range(1, len(terr_display) + 1)
    
    st.dataframe(
        terr_display.style.format({
            'PF_Satis': '{:,.0f}',
            'Rakip_Satis': '{:,.0f}',
            'Toplam_Pazar': '{:,.0f}',
            'Pazar_Payi_%': '{:.1f}',
            'Agirlik_%': '{:.1f}'
        }).background_gradient(subset=['Pazar_Payi_%'], cmap='RdYlGn'),
        use_container_width=True,
        height=400
    )

# =============================================================================
# TAB 4: ZAMAN SERİSİ & ML
# =============================================================================
with tab4:
    st.header("📈 Zaman Serisi Analizi & ML Tahminleme")
    
    territory_for_ts = st.selectbox(
        "Territory Seçin",
        ["TÜMÜ"] + sorted(df_filtered['TERRITORIES'].unique()),
        key='ts_territory'
    )
    
    ts_data = calculate_time_series(df_filtered, selected_product, territory_for_ts, date_filter)
    
    if not ts_data.empty:
        # Zaman serisi grafiği
        st.subheader("📊 Tarihsel Performans")
        
        fig_ts = go.Figure()
        
        fig_ts.add_trace(go.Scatter(
            x=ts_data['YIL_AY'],
            y=ts_data['PF_Satis'],
            name='PF Satış',
            line=dict(color='#3B82F6', width=2),
            mode='lines+markers'
        ))
        
        fig_ts.add_trace(go.Scatter(
            x=ts_data['YIL_AY'],
            y=ts_data['Rakip_Satis'],
            name='Rakip Satış',
            line=dict(color='#EF4444', width=2),
            mode='lines+markers'
        ))
        
        fig_ts.add_trace(go.Scatter(
            x=ts_data['YIL_AY'],
            y=ts_data['MA_3'],
            name='3 Aylık Ort.',
            line=dict(color='#10B981', width=1, dash='dash')
        ))
        
        fig_ts.update_layout(
            title='Aylık Satış Trendi',
            xaxis_title='Ay',
            yaxis_title='Satış',
            height=500
        )
        
        st.plotly_chart(fig_ts, use_container_width=True)
        
        # ML Tahminleme
        st.markdown("---")
        st.subheader("🤖 ML Tahminleme")
        
        col_ml1, col_ml2 = st.columns([1, 3])
        
        with col_ml1:
            forecast_months = st.slider("Tahmin Süresi (Ay)", 1, 6, 3)
        
        forecast = simple_forecast(ts_data, forecast_months)
        
        if forecast:
            # Tahmin grafiği
            last_date = ts_data['YIL_AY'].max()
            forecast_dates = []
            for i in range(1, forecast_months + 1):
                year = last_date // 100
                month = last_date % 100
                month += i
                if month > 12:
                    year += month // 12
                    month = month % 12
                    if month == 0:
                        month = 12
                        year -= 1
                forecast_dates.append(year * 100 + month)
            
            fig_forecast = go.Figure()
            
            # Geçmiş veri
            fig_forecast.add_trace(go.Scatter(
                x=ts_data['YIL_AY'],
                y=ts_data['PF_Satis'],
                name='Gerçek Satış',
                line=dict(color='#3B82F6', width=2),
                mode='lines+markers'
            ))
            
            # Tahmin
            fig_forecast.add_trace(go.Scatter(
                x=forecast_dates,
                y=forecast,
                name='Tahmin',
                line=dict(color='#10B981', width=2, dash='dash'),
                mode='lines+markers'
            ))
            
            fig_forecast.update_layout(
                title='Satış Tahmini',
                xaxis_title='Ay',
                yaxis_title='Satış',
                height=400
            )
            
            st.plotly_chart(fig_forecast, use_container_width=True)
            
            # Tahmin tablosu
            forecast_df = pd.DataFrame({
                'Ay': forecast_dates,
                'Tahmini Satış': forecast
            })
            
            st.dataframe(
                forecast_df.style.format({
                    'Tahmini Satış': '{:,.0f}'
                }),
                use_container_width=True
            )
        
        # Detaylı tablo
        st.markdown("---")
        st.subheader("📋 Aylık Performans Detayları")
        
        ts_display = ts_data.copy()
        ts_display.index = range(1, len(ts_display) + 1)
        
        st.dataframe(
            ts_display.style.format({
                'PF_Satis': '{:,.0f}',
                'Rakip_Satis': '{:,.0f}',
                'Toplam_Pazar': '{:,.0f}',
                'Pazar_Payi_%': '{:.1f}',
                'MA_3': '{:,.0f}',
                'MA_6': '{:,.0f}',
                'Buyume_%': '{:.1f}'
            }),
            use_container_width=True,
            height=400
        )

# =============================================================================
# TAB 5: RAKİP ANALİZİ
# =============================================================================
with tab5:
    st.header("🎯 Detaylı Rakip Analizi")
    
    comp_data = calculate_competitor_analysis(df_filtered, selected_product, date_filter)
    
    if not comp_data.empty:
        # Özet metrikler
        col1, col2, col3, col4 = st.columns(4)
        
        total_pf = comp_data['PF'].sum()
        total_rakip = comp_data['Rakip'].sum()
        avg_pf_share = comp_data['PF_Pay_%'].mean()
        win_months = len(comp_data[comp_data['Fark'] > 0])
        
        with col1:
            st.metric("💊 Toplam PF", f"{total_pf:,.0f}")
        with col2:
            st.metric("🏪 Toplam Rakip", f"{total_rakip:,.0f}")
        with col3:
            st.metric("📊 Ort. PF Payı", f"%{avg_pf_share:.1f}")
        with col4:
            st.metric("🏆 Kazanılan Ay", win_months)
        
        st.markdown("---")
        
        # Karşılaştırma grafiği
        st.subheader("📊 PF vs Rakip Trend Karşılaştırması")
        
        fig_comp = go.Figure()
        
        fig_comp.add_trace(go.Scatter(
            x=comp_data['YIL_AY'],
            y=comp_data['PF'],
            name='PF Satış',
            line=dict(color='#3B82F6', width=3),
            mode='lines+markers',
            fill='tonexty'
        ))
        
        fig_comp.add_trace(go.Scatter(
            x=comp_data['YIL_AY'],
            y=comp_data['Rakip'],
            name='Rakip Satış',
            line=dict(color='#EF4444', width=3),
            mode='lines+markers',
            fill='tozeroy'
        ))
        
        fig_comp.update_layout(
            title='Aylık Satış Karşılaştırması',
            xaxis_title='Ay',
            yaxis_title='Satış',
            height=500
        )
        
        st.plotly_chart(fig_comp, use_container_width=True)
        
        # Pazar payı trendi
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.markdown("#### 📈 Pazar Payı Trendi")
            fig_share = go.Figure()
            
            fig_share.add_trace(go.Scatter(
                x=comp_data['YIL_AY'],
                y=comp_data['PF_Pay_%'],
                name='PF Payı',
                line=dict(color='#3B82F6', width=2),
                mode='lines+markers'
            ))
            
            fig_share.add_trace(go.Scatter(
                x=comp_data['YIL_AY'],
                y=comp_data['Rakip_Pay_%'],
                name='Rakip Payı',
                line=dict(color='#EF4444', width=2),
                mode='lines+markers'
            ))
            
            fig_share.update_layout(height=400, yaxis_title='Pazar Payı %')
            st.plotly_chart(fig_share, use_container_width=True)
        
        with col_chart2:
            st.markdown("#### 📊 Büyüme Oranları")
            fig_growth = go.Figure()
            
            fig_growth.add_trace(go.Bar(
                x=comp_data['YIL_AY'],
                y=comp_data['PF_Buyume_%'],
                name='PF Büyüme',
                marker_color='#3B82F6'
            ))
            
            fig_growth.add_trace(go.Bar(
                x=comp_data['YIL_AY'],
                y=comp_data['Rakip_Buyume_%'],
                name='Rakip Büyüme',
                marker_color='#EF4444'
            ))
            
            fig_growth.update_layout(height=400, yaxis_title='Büyüme %')
            st.plotly_chart(fig_growth, use_container_width=True)
        
        # Detaylı tablo
        st.markdown("---")
        st.subheader("📋 Aylık Performans Detayları")
        
        comp_display = comp_data.copy()
        comp_display.index = range(1, len(comp_display) + 1)
        
        # Renk kodlaması için stil fonksiyonu
        def highlight_performance(row):
            if row['Fark'] > 0:
                return ['background-color: #D1FAE5'] * len(row)  # Açık yeşil
            else:
                return ['background-color: #FEE2E2'] * len(row)  # Açık kırmızı
        
        st.dataframe(
            comp_display.style.apply(highlight_performance, axis=1).format({
                'PF': '{:,.0f}',
                'Rakip': '{:,.0f}',
                'Toplam': '{:,.0f}',
                'PF_Pay_%': '{:.1f}',
                'Rakip_Pay_%': '{:.1f}',
                'Fark': '{:,.0f}',
                'PF_Buyume_%': '{:.1f}',
                'Rakip_Buyume_%': '{:.1f}'
            }),
            use_container_width=True,
            height=400
        )

# =============================================================================
# TAB 6: BCG MATRIX & STRATEJİ
# =============================================================================
with tab6:
    st.header("⭐ BCG Matrix ve Yatırım Stratejisi")
    
    bcg_data = calculate_bcg_matrix(df_filtered, selected_product, date_filter)
    
    if not bcg_data.empty:
        # BCG dağılımı
        st.subheader("📊 BCG Kategori Dağılımı")
        
        col_bcg1, col_bcg2 = st.columns([2, 3])
        
        with col_bcg1:
            bcg_counts = bcg_data['BCG_Kategori'].value_counts()
            
            fig_pie = px.pie(
                values=bcg_counts.values,
                names=bcg_counts.index,
                title='Territory Dağılımı',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col_bcg2:
            # BCG scatter plot
            fig_scatter = px.scatter(
                bcg_data,
                x='Agirlik_%',
                y='Pazar_Payi_%',
                size='PF_Satis',
                color='BCG_Kategori',
                hover_name='Territory',
                title='BCG Matrix',
                labels={
                    'Agirlik_%': 'Pazar Ağırlığı %',
                    'Pazar_Payi_%': 'Pazar Payı %'
                }
            )
            
            # Median çizgileri
            median_share = bcg_data['Pazar_Payi_%'].median()
            median_weight = bcg_data['Agirlik_%'].median()
            
            fig_scatter.add_hline(y=median_share, line_dash="dash", line_color="gray")
            fig_scatter.add_vline(x=median_weight, line_dash="dash", line_color="gray")
            
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Kategori bazlı analiz
        st.markdown("---")
        st.subheader("📈 Kategori Bazlı Performans")
        
        for category in bcg_data['BCG_Kategori'].unique():
            with st.expander(f"{category} - {len(bcg_data[bcg_data['BCG_Kategori'] == category])} Territory"):
                cat_data = bcg_data[bcg_data['BCG_Kategori'] == category].sort_values('PF_Satis', ascending=False)
                
                # Strateji önerisi
                strategy = cat_data['Strateji'].iloc[0]
                st.info(f"**Önerilen Strateji:** {strategy}")
                
                # Tablo
                cat_display = cat_data[['Territory', 'PF_Satis', 'Pazar_Payi_%', 'Agirlik_%']].copy()
                cat_display.index = range(1, len(cat_display) + 1)
                
                st.dataframe(
                    cat_display.style.format({
                        'PF_Satis': '{:,.0f}',
                        'Pazar_Payi_%': '{:.1f}',
                        'Agirlik_%': '{:.1f}'
                    }),
                    use_container_width=True
                )
        
        # Tam liste
        st.markdown("---")
        st.subheader("📋 Tüm Territory BCG Analizi")
        
        bcg_display = bcg_data.copy()
        bcg_display.index = range(1, len(bcg_display) + 1)
        
        st.dataframe(
            bcg_display[['Territory', 'BCG_Kategori', 'Strateji', 'PF_Satis', 
                         'Pazar_Payi_%', 'Agirlik_%']].style.format({
                'PF_Satis': '{:,.0f}',
                'Pazar_Payi_%': '{:.1f}',
                'Agirlik_%': '{:.1f}'
            }),
            use_container_width=True,
            height=400
        )

# =============================================================================
# TAB 7: RAPORLAR
# =============================================================================
with tab7:
    st.header("📥 Raporları İndir")
    
    st.info("Tüm analizleri Excel formatında indirebilirsiniz")
    
    if st.button("📊 Excel Raporu Oluştur", type="primary"):
        with st.spinner("Rapor hazırlanıyor..."):
            # Excel oluştur
            output = BytesIO()
            
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # Territory performans
                terr_perf = calculate_territory_performance(df_filtered, selected_product, date_filter)
                terr_perf.to_excel(writer, sheet_name='Territory Performans', index=False)
                
                # Zaman serisi
                ts_data = calculate_time_series(df_filtered, selected_product, "TÜMÜ", date_filter)
                ts_data.to_excel(writer, sheet_name='Zaman Serisi', index=False)
                
                # BCG Matrix
                bcg_data = calculate_bcg_matrix(df_filtered, selected_product, date_filter)
                bcg_data.to_excel(writer, sheet_name='BCG Matrix', index=False)
                
                # Şehir analizi
                city_data = calculate_city_performance(df_filtered, selected_product, date_filter)
                city_data.to_excel(writer, sheet_name='Şehir Analizi', index=False)
                
                # Rakip analizi
                comp_data = calculate_competitor_analysis(df_filtered, selected_product, date_filter)
                comp_data.to_excel(writer, sheet_name='Rakip Analizi', index=False)
            
            output.seek(0)
            
            st.download_button(
                label="⬇️ Excel Raporu İndir",
                data=output,
                file_name=f"ticari_portfoy_rapor_{selected_product}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
            st.success("✅ Rapor hazır!")

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🎯 Gelişmiş Ticari Portföy Analiz Sistemi v2.0</p>
    <p>Territory Bazlı Performans | ML Tahminleme | Türkiye Haritası | BCG Matrix | Rekabet Analizi</p>
</div>
""", unsafe_allow_html=True)
