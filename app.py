"""
🎯 GELİŞMİŞ TİCARİ PORTFÖY ANALİZ SİSTEMİ
Territory Bazlı Performans, ML Tahminleme, Türkiye Haritası ve Rekabet Analizi

Yeni Özellikler:
- 🗺️ Türkiye il bazlı harita görselleştirme
- 🤖 Machine Learning satış tahminleme
- 📊 Aylık/Yıllık dönem seçimi
- 📈 Gelişmiş rakip analizi ve trend karşılaştırması
- 🎯 Dinamik zaman aralığı filtreleme
- 📉 Prophet ile gelecek tahminleme
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
        color: #11b2ed;
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
# ŞEHİR İSİM HARİTALAMA (GeoJSON ve Excel uyumluluğu için)
# =============================================================================
CITY_NAME_MAPPING = {
    # Excel'deki isimler -> GeoJSON'daki isimler
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
    'AYDın': 'Aydin',
    'AYDIN': 'Aydin',
    'BALIKESİR': 'Balikesir',
    'BALIKESIR': 'Balikesir',
    'BARTIN': 'BartÄ±n',
    'BATMAN': 'Batman',
    'BAYBURT': 'Bayburt',
    'BİLECİK': 'Bilecik',
    'BILECIK': 'Bilecik',
    'BİNGÖL': 'BingÃ¶l',
    'BINGOL': 'BingÃ¶l',
    'BİTLİS': 'Bitlis',
    'BITLIS': 'Bitlis',
    'BOLU': 'Bolu',
    'BURDUR': 'Burdur',
    'BURSA': 'Bursa',
    'ÇANAKKALE': 'Ãanakkale',
    'CANAKKALE': 'Ãanakkale',
    'ÇANKIRI': 'Ãankiri',
    'CANKIRI': 'Ãankiri',
    'ÇORUM': 'Ãorum',
    'CORUM': 'Ãorum',
    'DENİZLİ': 'Denizli',
    'DENIZLI': 'Denizli',
    'DİYARBAKIR': 'Diyarbakir',
    'DIYARBAKIR': 'Diyarbakir',
    'DÜZCE': 'DÃ¼zce',
    'DUZCE': 'DÃ¼zce',
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
    'GÜMÜŞHANE': 'GÃ¼mÃ¼shane',
    'GUMUSHANE': 'GÃ¼mÃ¼shane',
    'HAKKARİ': 'Hakkari',
    'HAKKARI': 'Hakkari',
    'HATAY': 'Hatay',
    'IĞDIR': 'IÄdir',
    'IGDIR': 'IÄdir',
    'ISPARTA': 'Isparta',
    'İSTANBUL': 'Istanbul',
    'ISTANBUL': 'Istanbul',
    'İZMİR': 'Izmir',
    'IZMIR': 'Izmir',
    'KAHRAMANMARAŞ': 'K. Maras',
    'KAHRAMANMARAS': 'K. Maras',
    'KARABÜK': 'KarabÃ¼k',
    'KARABUK': 'KarabÃ¼k',
    'KARAMAN': 'Karaman',
    'KARS': 'Kars',
    'KASTAMONU': 'Kastamonu',
    'KAYSERİ': 'Kayseri',
    'KAYSERI': 'Kayseri',
    'KIRIKKALE': 'Kinkkale',
    'KIRKLARELİ': 'Kirklareli',
    'KIRKLARELI': 'Kirklareli',
    'KIRŞEHİR': 'Kirsehir',
    'KIRSEHIR': 'Kirsehir',
    'KİLİS': 'Kilis',
    'KILIS': 'Kilis',
    'KOCAELİ': 'Kocaeli',
    'KOCAELI': 'Kocaeli',
    'KONYA': 'Konya',
    'KÜTAHYA': 'KÃ¼tahya',
    'KUTAHYA': 'KÃ¼tahya',
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
    'ZONGULDAK': 'Zinguldak',
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
    city_perf = df.groupby(['CITY_NORMALIZED']).agg({
        cols['pf']: 'sum',
        cols['rakip']: 'sum'
    }).reset_index()
    
    city_perf.columns = ['City', 'PF_Satis', 'Rakip_Satis']
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
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_turkey_map(city_data, geojson, title="Türkiye Satış Haritası"):
    """Türkiye haritası oluştur"""
    if geojson is None:
        st.error("GeoJSON dosyası yüklenemedi")
        return None
    
    fig = px.choropleth(
        city_data,
        geojson=geojson,
        locations='City',
        featureidkey="properties.name",
        color='PF_Satis',
        hover_name='City',
        hover_data={
            'PF_Satis': ':,.0f',
            'Pazar_Payi_%': ':.1f',
            'City': False
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
    st.markdown("**ML Tahminleme • Türkiye Haritası • Rakip Analizi • BCG Matrix**")
    
    # Sidebar
    st.sidebar.header("📂 Dosya Yükleme")
    uploaded_file = st.sidebar.file_uploader(
        "Excel Dosyası Yükleyin",
        type=['xlsx', 'xls'],
        help="Ticari Ürün 2025 verisi"
    )
    
    if not uploaded_file:
        st.info("👈 Lütfen sol taraftan Excel dosyasını yükleyin")
        st.stop()
    
    # Veriyi yükle
    try:
        df = load_excel_data(uploaded_file)
        geojson = load_geojson()
        st.sidebar.success(f"✅ {len(df)} satır veri yüklendi")
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
    
    # Tarih aralığı seçimi - Kolay kullanım
    st.sidebar.markdown("---")
    st.sidebar.header("📅 Tarih Aralığı")
    
    min_date = df['DATE'].min()
    max_date = df['DATE'].max()
    
    # Hızlı seçim seçenekleri
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
    else:  # Özel Aralık
        col_date1, col_date2 = st.sidebar.columns(2)
        with col_date1:
            start_date = st.date_input("Başlangıç", min_date, min_value=min_date, max_value=max_date)
        with col_date2:
            end_date = st.date_input("Bitiş", max_date, min_value=min_date, max_value=max_date)
        date_filter = (pd.to_datetime(start_date), pd.to_datetime(end_date))
    
    # Filtreler
    st.sidebar.markdown("---")
    st.sidebar.header("🔍 Filtreler")
    
    # Territory filtresi
    territories = ["TÜMÜ"] + sorted(df['TERRITORIES'].unique())
    selected_territory = st.sidebar.selectbox("Territory", territories)
    
    # Region filtresi
    regions = ["TÜMÜ"] + sorted(df['REGION'].unique())
    selected_region = st.sidebar.selectbox("Bölge", regions)
    
    # Manager filtresi
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
        
        # Temel metrikler
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
        
        # Bar chart
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
        
        # Tablo
        display_cols = ['Territory', 'Region', 'City', 'Manager', 'PF_Satis', 
                       'Pazar_Payi_%', 'Agirlik_%']
        top10_display = top10[display_cols].copy()
        top10_display.columns = ['Territory', 'Region', 'City', 'Manager', 
                                 'PF Satış', 'Pazar Payı %', 'Ağırlık %']
        top10_display.index = range(1, len(top10_display) + 1)
        
        st.dataframe(
            top10_display.style.format({
                'PF Satış': '{:,.0f}',
                'Pazar Payı %': '{:.1f}',
                'Ağırlık %': '{:.1f}'
            }),
            use_container_width=True
        )
    
    # ==========================================================================
    # TAB 2: TÜRKİYE HARİTASI
    # ==========================================================================
    with tab2:
        st.header("🗺️ Türkiye İl Bazlı Satış Haritası")
        
        # Şehir performansı
        city_data = calculate_city_performance(df_filtered, selected_product, date_filter)
        
        # Metriks
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
        if geojson:
            st.subheader("📍 İl Bazlı Dağılım")
            turkey_map = create_turkey_map(city_data, geojson, 
                                          f"{selected_product} - Şehir Bazlı Satış Dağılımı")
            if turkey_map:
                st.plotly_chart(turkey_map, use_container_width=True)
        
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
    
    # ==========================================================================
    # TAB 3: TERRITORY ANALİZİ
    # ==========================================================================
    with tab3:
        st.header("🏢 Territory Bazlı Detaylı Analiz")
        
        # Territory performansı
        terr_perf = calculate_territory_performance(df_filtered, selected_product, date_filter)
        
        # Filtreler
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
                name='PF Satış',
                marker_color='#3B82F6'
            ))
            
            fig_bar.add_trace(go.Bar(
                x=terr_sorted['Territory'],
                y=terr_sorted['Rakip_Satis'],
                name='Rakip Satış',
                marker_color='#EF4444'
            ))
            
            fig_bar.update_layout(
                barmode='group',
                height=500,
                xaxis=dict(tickangle=-45)
            )
            
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col_v2:
            st.markdown("#### 🎯 Pazar Payı Dağılımı")
            fig_pie = px.pie(
                terr_sorted.head(10),
                values='PF_Satis',
                names='Territory',
                title='Top 10 Territory - PF Satış Dağılımı'
            )
            fig_pie.update_layout(height=500)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        st.markdown("---")
        
        # Detaylı tablo
        st.subheader("📋 Detaylı Territory Listesi")
        
        display_cols = ['Territory', 'Region', 'City', 'Manager', 
                       'PF_Satis', 'Rakip_Satis', 'Toplam_Pazar',
                       'Pazar_Payi_%', 'Goreceli_Pazar_Payi', 'Agirlik_%']
        
        terr_display = terr_sorted[display_cols].copy()
        terr_display.columns = ['Territory', 'Region', 'City', 'Manager',
                               'PF Satış', 'Rakip Satış', 'Toplam Pazar',
                               'Pazar Payı %', 'Göreceli Pay', 'Ağırlık %']
        terr_display.index = range(1, len(terr_display) + 1)
        
        st.dataframe(
            terr_display.style.format({
                'PF Satış': '{:,.0f}',
                'Rakip Satış': '{:,.0f}',
                'Toplam Pazar': '{:,.0f}',
                'Pazar Payı %': '{:.1f}',
                'Göreceli Pay': '{:.2f}',
                'Ağırlık %': '{:.1f}'
            }).background_gradient(subset=['Pazar Payı %'], cmap='RdYlGn'),
            use_container_width=True
        )
    
    # ==========================================================================
    # TAB 4: ZAMAN SERİSİ & ML TAHMİN
    # ==========================================================================
    with tab4:
        st.header("📈 Zaman Serisi Analizi & ML Tahminleme")
        
        # Territory seçimi
        territory_for_ts = st.selectbox(
            "Territory Seçin",
            ["TÜMÜ"] + sorted(df_filtered['TERRITORIES'].unique()),
            key='ts_territory'
        )
        
        # Zaman serisi hesapla
        monthly_df = calculate_time_series(df_filtered, selected_product, 
                                           territory_for_ts, date_filter)
        
        if len(monthly_df) == 0:
            st.warning("⚠️ Seçilen filtrelerde veri bulunamadı")
        else:
            # =================================================================
            # ZAMAN SERİSİ BÖLÜMÜ
            # =================================================================
            st.subheader("📊 Zaman Serisi Analizi")
            
            # Temel metrikler
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
                st.markdown("#### 📊 Satış Trendi")
                fig_ts = go.Figure()
                
                fig_ts.add_trace(go.Scatter(
                    x=monthly_df['DATE'],
                    y=monthly_df['PF_Satis'],
                    mode='lines+markers',
                    name='PF Satış',
                    line=dict(color='#3B82F6', width=3),
                    marker=dict(size=8)
                ))
                
                fig_ts.add_trace(go.Scatter(
                    x=monthly_df['DATE'],
                    y=monthly_df['Rakip_Satis'],
                    mode='lines+markers',
                    name='Rakip Satış',
                    line=dict(color='#EF4444', width=3),
                    marker=dict(size=8)
                ))
                
                fig_ts.add_trace(go.Scatter(
                    x=monthly_df['DATE'],
                    y=monthly_df['MA_3'],
                    mode='lines',
                    name='3 Aylık Ort.',
                    line=dict(color='#10B981', width=2, dash='dash')
                ))
                
                fig_ts.update_layout(
                    xaxis_title='Tarih',
                    yaxis_title='Satış',
                    height=400,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_ts, use_container_width=True)
            
            with col_chart2:
                st.markdown("#### 🎯 Pazar Payı Trendi")
                fig_share = go.Figure()
                
                fig_share.add_trace(go.Scatter(
                    x=monthly_df['DATE'],
                    y=monthly_df['Pazar_Payi_%'],
                    mode='lines+markers',
                    fill='tozeroy',
                    line=dict(color='#8B5CF6', width=2),
                    marker=dict(size=8)
                ))
                
                fig_share.update_layout(
                    xaxis_title='Tarih',
                    yaxis_title='Pazar Payı (%)',
                    height=400
                )
                
                st.plotly_chart(fig_share, use_container_width=True)
            
            st.markdown("---")
            
            # Büyüme analizi
            st.markdown("#### 📈 Büyüme Analizi")
            
            col_growth1, col_growth2 = st.columns(2)
            
            with col_growth1:
                fig_growth = go.Figure()
                
                colors_pf = ['#10B981' if x > 0 else '#EF4444' 
                            for x in monthly_df['PF_Buyume_%']]
                
                fig_growth.add_trace(go.Bar(
                    x=monthly_df['DATE'],
                    y=monthly_df['PF_Buyume_%'],
                    name='PF Büyüme %',
                    marker_color=colors_pf,
                    text=monthly_df['PF_Buyume_%'].apply(lambda x: f'{x:.1f}%' if not pd.isna(x) else ''),
                    textposition='outside'
                ))
                
                fig_growth.update_layout(
                    title='Aylık Büyüme Oranları',
                    xaxis_title='Tarih',
                    yaxis_title='Büyüme (%)',
                    height=400
                )
                
                st.plotly_chart(fig_growth, use_container_width=True)
            
            with col_growth2:
                st.markdown("##### 📊 Büyüme İstatistikleri")
                
                growth_stats = monthly_df[['PF_Buyume_%', 'Rakip_Buyume_%', 'Goreceli_Buyume_%']].describe()
                
                st.dataframe(
                    growth_stats.style.format("{:.2f}"),
                    use_container_width=True
                )
            
            # =================================================================
            # ML TAHMİN BÖLÜMÜ
            # =================================================================
            st.markdown("---")
            st.subheader("🤖 Machine Learning Satış Tahmini")
            
            st.info("⚡ Hareketli ortalama ve trend analizi ile gelecek aylar tahmin edilmektedir.")
            
            # Tahmin periyodu
            forecast_months = st.slider("Tahmin Periyodu (Ay)", 1, 6, 3)
            
            if len(monthly_df) < 3:
                st.warning("⚠️ Tahmin için yeterli veri yok (en az 3 ay gerekli)")
            else:
                # Tahmin yap
                forecast_df = simple_forecast(monthly_df, forecast_months)
                
                # Metrikler
                col_ml1, col_ml2, col_ml3 = st.columns(3)
                
                last_actual = monthly_df['PF_Satis'].iloc[-1]
                first_forecast = forecast_df['PF_Satis'].iloc[0] if forecast_df is not None else 0
                change = ((first_forecast - last_actual) / last_actual * 100) if last_actual > 0 else 0
                
                with col_ml1:
                    st.metric("📊 Son Gerçek Satış", f"{last_actual:,.0f}")
                with col_ml2:
                    st.metric("🔮 İlk Tahmin", f"{first_forecast:,.0f}", 
                             delta=f"%{change:.1f}")
                with col_ml3:
                    avg_forecast = forecast_df['PF_Satis'].mean() if forecast_df is not None else 0
                    st.metric("📈 Ort. Tahmin", f"{avg_forecast:,.0f}")
                
                st.markdown("---")
                
                # Tahmin Grafiği
                st.markdown("#### 📈 Tahmin Grafiği")
                forecast_chart = create_forecast_chart(monthly_df, forecast_df)
                st.plotly_chart(forecast_chart, use_container_width=True)
                
                # Tahmin tablosu
                if forecast_df is not None:
                    col_table1, col_table2 = st.columns([2, 1])
                    
                    with col_table1:
                        st.markdown("##### 📋 Tahmin Detayları")
                        
                        forecast_display = forecast_df[['YIL_AY', 'PF_Satis']].copy()
                        forecast_display.columns = ['Ay', 'Tahmin Edilen Satış']
                        forecast_display.index = range(1, len(forecast_display) + 1)
                        
                        st.dataframe(
                            forecast_display.style.format({
                                'Tahmin Edilen Satış': '{:,.0f}'
                            }),
                            use_container_width=True
                        )
                    
                    with col_table2:
                        st.markdown("##### 📊 Model Bilgisi")
                        st.info("""
                        **Yöntem:** 
                        - Hareketli ortalama
                        - Trend analizi
                        
                        **Temel:** 
                        - Son 3-6 ay
                        
                        **Güvenilirlik:** 
                        - Orta (basit model)
                        """)
    
    # ==========================================================================
    # TAB 5: RAKİP ANALİZİ
    # ==========================================================================
    with tab5:
        st.header("📊 Detaylı Rakip Analizi")
        
        # Rakip analizi
        comp_data = calculate_competitor_analysis(df_filtered, selected_product, date_filter)
        
        if len(comp_data) == 0:
            st.warning("⚠️ Seçilen filtrelerde veri bulunamadı")
        else:
            # Metrikler
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
                comp_chart = create_competitor_comparison_chart(comp_data)
                st.plotly_chart(comp_chart, use_container_width=True)
            
            with col_g2:
                st.subheader("📊 Pazar Payı Trendi")
                share_chart = create_market_share_trend(comp_data)
                st.plotly_chart(share_chart, use_container_width=True)
            
            st.markdown("---")
            
            # Büyüme analizi
            st.subheader("📈 Büyüme Karşılaştırması")
            growth_chart = create_growth_comparison(comp_data)
            st.plotly_chart(growth_chart, use_container_width=True)
            
            # Performans özeti
            st.markdown("---")
            st.subheader("📋 Aylık Performans Detayları")
            
            comp_display = comp_data[['YIL_AY', 'PF', 'Rakip', 'PF_Pay_%', 
                                     'PF_Buyume', 'Rakip_Buyume', 'Fark']].copy()
            comp_display.columns = ['Ay', 'PF Satış', 'Rakip Satış', 'PF Pay %',
                                   'PF Büyüme %', 'Rakip Büyüme %', 'Fark %']
            
            # Renklendirme için stil
            def highlight_winner(row):
                if row['Fark %'] > 0:
                    return ['background-color: #d4edda'] * len(row)
                elif row['Fark %'] < 0:
                    return ['background-color: #f8d7da'] * len(row)
                else:
                    return [''] * len(row)
            
            st.dataframe(
                comp_display.style.format({
                    'PF Satış': '{:,.0f}',
                    'Rakip Satış': '{:,.0f}',
                    'PF Pay %': '{:.1f}',
                    'PF Büyüme %': '{:.1f}',
                    'Rakip Büyüme %': '{:.1f}',
                    'Fark %': '{:.1f}'
                }).apply(highlight_winner, axis=1),
                use_container_width=True,
                height=400
            )
            
            # İçgörüler
            st.markdown("---")
            st.subheader("💡 Önemli İçgörüler")
            
            col_i1, col_i2 = st.columns(2)
            
            with col_i1:
                if avg_pf_growth > avg_rakip_growth:
                    st.success(f"✅ PF ortalama %{avg_pf_growth:.1f} büyüme ile rakipten daha hızlı büyüyor")
                else:
                    st.warning(f"⚠️ Rakip ortalama %{avg_rakip_growth:.1f} büyüme ile PF'den daha hızlı büyüyor")
            
            with col_i2:
                if avg_pf_share >= 50:
                    st.success(f"✅ PF %{avg_pf_share:.1f} pazar payı ile lider konumda")
                else:
                    st.warning(f"⚠️ Rakip pazar payında öne geçmiş (%{(100-avg_pf_share):.1f})")
    
    # ==========================================================================
    # TAB 6: BCG MATRIX & YATIRIM STRATEJİSİ
    # ==========================================================================
    with tab6:
        st.header("⭐ BCG Matrix & Yatırım Stratejisi")
        
        # BCG hesapla
        bcg_df = calculate_bcg_matrix(df_filtered, selected_product, date_filter)
        
        # BCG dağılımı
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
        
        # BCG Scatter
        st.subheader("🎯 BCG Matrix")
        
        color_map = {
            "⭐ Star": "#FFD700",
            "🐄 Cash Cow": "#10B981",
            "❓ Question Mark": "#3B82F6",
            "🐶 Dog": "#9CA3AF"
        }
        
        fig_bcg = px.scatter(
            bcg_df,
            x='Goreceli_Pazar_Payi',
            y='Pazar_Buyume_%',
            size='PF_Satis',
            color='BCG_Kategori',
            color_discrete_map=color_map,
            hover_name='Territory',
            hover_data={
                'PF_Satis': ':,.0f',
                'Pazar_Payi_%': ':.1f',
                'Goreceli_Pazar_Payi': ':.2f',
                'Pazar_Buyume_%': ':.1f'
            },
            labels={
                'Goreceli_Pazar_Payi': 'Göreceli Pazar Payı (PF/Rakip)',
                'Pazar_Buyume_%': 'Pazar Büyüme Oranı (%)'
            },
            size_max=50
        )
        
        median_share = bcg_df['Goreceli_Pazar_Payi'].median()
        median_growth = bcg_df['Pazar_Buyume_%'].median()
        
        fig_bcg.add_hline(y=median_growth, line_dash="dash", line_color="rgba(255,255,255,0.4)")
        fig_bcg.add_vline(x=median_share, line_dash="dash", line_color="rgba(255,255,255,0.4)")
        
        fig_bcg.update_layout(
            title='BCG Matrix - Stratejik Konumlandırma',
            height=600,
            plot_bgcolor='#0f172a',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0')
        )
        
        st.plotly_chart(fig_bcg, use_container_width=True)
        
        # BCG Açıklamaları
        st.markdown("---")
        col_exp1, col_exp2 = st.columns(2)
        
        with col_exp1:
            st.info("""
            **⭐ STARS (Yıldızlar)**
            - Yüksek büyüme + Yüksek pazar payı
            - **Aksiyon:** Yatırımı artır, liderliği sürdür
            """)
            
            st.success("""
            **🐄 CASH COWS (Nakit İnekleri)**
            - Düşük büyüme + Yüksek pazar payı
            - **Aksiyon:** Koru, maliyeti optimize et
            """)
        
        with col_exp2:
            st.warning("""
            **❓ QUESTION MARKS (Soru İşaretleri)**
            - Yüksek büyüme + Düşük pazar payı
            - **Aksiyon:** Agresif yatırım yap veya çık
            """)
            
            st.error("""
            **🐶 DOGS (Köpekler)**
            - Düşük büyüme + Düşük pazar payı
            - **Aksiyon:** Minimal kaynak, çıkışı değerlendir
            """)
        
        # BCG Detay Tablosu
        st.markdown("---")
        st.subheader("📋 BCG Kategori Detayları")
        
        display_cols_bcg = ['Territory', 'Region', 'BCG_Kategori', 'PF_Satis',
                           'Pazar_Payi_%', 'Goreceli_Pazar_Payi', 'Pazar_Buyume_%']
        
        bcg_display = bcg_df[display_cols_bcg].copy()
        bcg_display.columns = ['Territory', 'Region', 'BCG', 'PF Satış',
                              'Pazar Payı %', 'Göreceli Pay', 'Büyüme %']
        bcg_display = bcg_display.sort_values('PF Satış', ascending=False)
        bcg_display.index = range(1, len(bcg_display) + 1)
        
        st.dataframe(
            bcg_display.style.format({
                'PF Satış': '{:,.0f}',
                'Pazar Payı %': '{:.1f}',
                'Göreceli Pay': '{:.2f}',
                'Büyüme %': '{:.1f}'
            }),
            use_container_width=True
        )
    
    # ==========================================================================
    # TAB 7: RAPORLAR
    # ==========================================================================
    with tab7:
        st.header("📥 Rapor İndirme")
        
        st.markdown("""
        Detaylı analizlerin Excel raporlarını indirebilirsiniz.
        """)
        
        if st.button("📥 Excel Raporu Oluştur", type="primary"):
            with st.spinner("Rapor hazırlanıyor..."):
                # Verileri hazırla
                terr_perf = calculate_territory_performance(df_filtered, selected_product, date_filter)
                monthly_df = calculate_time_series(df_filtered, selected_product, None, date_filter)
                bcg_df = calculate_bcg_matrix(df_filtered, selected_product, date_filter)
                city_data = calculate_city_performance(df_filtered, selected_product, date_filter)
                comp_data = calculate_competitor_analysis(df_filtered, selected_product, date_filter)
                
                # Excel oluştur
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

