"""
🎯 TİCARİ PORTFÖY ANALİZ SİSTEMİ v4.0
Tamamen Çalışır, Sade ve Güçlü
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from io import BytesIO
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ============================================================================
# SAYFA AYARLARI
# ============================================================================

st.set_page_config(
    page_title="Portföy Analizi v4.0",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CSS STİLLERİ
# ============================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
    }
    
    .main-header {
        font-size: 3rem;
        font-weight: 900;
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #ffd700 0%, #f59e0b 50%, #d97706 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 30px rgba(255, 215, 0, 0.3);
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 2.5rem;
        font-weight: 900;
        background: linear-gradient(135deg, #60a5fa 0%, #3b82f6 50%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    div[data-testid="metric-container"] {
        background: rgba(30, 41, 59, 0.8);
        padding: 2rem;
        border-radius: 16px;
        border: 1px solid rgba(59, 130, 246, 0.3);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #94a3b8;
        font-weight: 600;
        padding: 1rem 2rem;
        background: rgba(30, 41, 59, 0.5);
        border-radius: 8px 8px 0 0;
        margin: 0 0.25rem;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.5);
    }
    
    h1, h2, h3 {
        color: #f1f5f9 !important;
        font-weight: 700;
    }
    
    p, span, div, label {
        color: #cbd5e1;
    }
    
    .stDataFrame {
        background: rgba(30, 41, 59, 0.5);
        border-radius: 8px;
    }
    
    div[data-testid="stExpander"] {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# YARDIMCI FONKSİYONLAR
# ============================================================================

def safe_divide(a, b):
    """Güvenli bölme işlemi"""
    return np.where(b != 0, a / b, 0)

def get_product_cols(product):
    """Ürüne göre kolon adlarını döndür"""
    map_dict = {
        "TROCMETAM": {"pf": "TROCMETAM", "rakip": "DIGER TROCMETAM"},
        "CORTIPOL": {"pf": "CORTIPOL", "rakip": "DIGER CORTIPOL"},
        "DEKSAMETAZON": {"pf": "DEKSAMETAZON", "rakip": "DIGER DEKSAMETAZON"},
        "PF IZOTONIK": {"pf": "PF IZOTONIK", "rakip": "DIGER IZOTONIK"}
    }
    return map_dict.get(product, {"pf": product, "rakip": f"DIGER {product}"})

# ============================================================================
# VERİ YÜKLEME
# ============================================================================

@st.cache_data
def load_data(file):
    """Excel dosyasını yükle ve işle"""
    try:
        df = pd.read_excel(file)
        
        # Tarih işleme
        if 'DATE' in df.columns:
            df['DATE'] = pd.to_datetime(df['DATE'])
            df['YIL_AY'] = df['DATE'].dt.strftime('%Y-%m')
            df['AY'] = df['DATE'].dt.month
            df['YIL'] = df['DATE'].dt.year
            df['AY_ADI'] = df['DATE'].dt.strftime('%B')
        
        # Text kolonları temizle
        text_cols = ['TERRITORIES', 'REGION', 'MANAGER', 'CITY']
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.upper().str.strip()
        
        return df
    except Exception as e:
        st.error(f"❌ Veri yükleme hatası: {str(e)}")
        return None

# ============================================================================
# ANALİZ FONKSİYONLARI
# ============================================================================

def calc_territory_perf(df, product):
    """Territory bazlı performans hesaplama"""
    cols = get_product_cols(product)
    
    # Grup kolonları
    group_cols = ['TERRITORIES']
    for c in ['REGION', 'CITY', 'MANAGER']:
        if c in df.columns:
            group_cols.append(c)
    
    # Aggregation
    terr = df.groupby(group_cols).agg({
        cols['pf']: 'sum',
        cols['rakip']: 'sum'
    }).reset_index()
    
    terr.columns = list(terr.columns[:len(group_cols)]) + ['PF_Satis', 'Rakip_Satis']
    
    # Hesaplamalar
    terr['Toplam_Pazar'] = terr['PF_Satis'] + terr['Rakip_Satis']
    terr['Pazar_Payi_%'] = safe_divide(terr['PF_Satis'], terr['Toplam_Pazar']) * 100
    terr['Buyume_Pot'] = terr['Toplam_Pazar'] - terr['PF_Satis']
    terr['Goreceli_Pay'] = safe_divide(terr['PF_Satis'], terr['Rakip_Satis'])
    
    return terr.sort_values('PF_Satis', ascending=False)

def calc_bcg_full(df, product):
    """Tam BCG Matrix (4 Kategori)"""
    terr = calc_territory_perf(df, product)
    
    # Büyüme oranı hesapla (varsa)
    if 'YIL_AY' in df.columns and len(df['YIL_AY'].unique()) > 1:
        # Son 2 dönem karşılaştırması
        periods = sorted(df['YIL_AY'].unique())
        if len(periods) >= 2:
            last_period = periods[-1]
            prev_period = periods[-2]
            
            cols = get_product_cols(product)
            
            last_sales = df[df['YIL_AY'] == last_period].groupby('TERRITORIES')[cols['pf']].sum()
            prev_sales = df[df['YIL_AY'] == prev_period].groupby('TERRITORIES')[cols['pf']].sum()
            
            growth = ((last_sales - prev_sales) / prev_sales * 100).fillna(0)
            terr['Growth_%'] = terr['TERRITORIES'].map(growth).fillna(0)
        else:
            terr['Growth_%'] = 0
    else:
        terr['Growth_%'] = 0
    
    # BCG kategorileri
    median_growth = terr['Growth_%'].median()
    median_share = terr['Goreceli_Pay'].median()
    
    def assign_bcg(row):
        growth = row['Growth_%']
        share = row['Goreceli_Pay']
        
        if growth >= median_growth and share >= median_share:
            return "⭐ Star"
        elif growth < median_growth and share >= median_share:
            return "🐄 Cash Cow"
        elif growth >= median_growth and share < median_share:
            return "❓ Question Mark"
        else:
            return "🐶 Dog"
    
    terr['BCG'] = terr.apply(assign_bcg, axis=1)
    
    return terr

def calc_strategy(df):
    """Yatırım stratejisi belirleme"""
    df = df.copy()
    df = df[df["PF_Satis"] > 0]
    
    if len(df) == 0:
        return df
    
    # Pazar segmentasyonu
    try:
        df["Pazar_Seg"] = pd.qcut(df["Toplam_Pazar"], q=3, 
                                   labels=["Küçük", "Orta", "Büyük"], 
                                   duplicates='drop')
    except:
        df["Pazar_Seg"] = "Orta"
    
    # Pay segmentasyonu
    try:
        df["Pay_Seg"] = pd.qcut(df["Pazar_Payi_%"], q=3, 
                                labels=["Düşük", "Orta", "Yüksek"], 
                                duplicates='drop')
    except:
        df["Pay_Seg"] = "Orta"
    
    # Strateji belirleme
    def assign_strat(row):
        pazar = str(row["Pazar_Seg"])
        pay = str(row["Pay_Seg"])
        
        if pazar in ["Büyük", "Orta"] and pay == "Düşük":
            return "🚀 Agresif Büyüme"
        elif pazar == "Büyük" and pay == "Yüksek":
            return "🛡️ Koruma"
        elif pay == "Yüksek":
            return "💰 Hasat"
        else:
            return "👁️ İzleme"
    
    df["Strateji"] = df.apply(assign_strat, axis=1)
    
    # Öncelik puanı
    df["Oncelik"] = (
        (df["Toplam_Pazar"] / df["Toplam_Pazar"].max() * 40) +
        (df["Buyume_Pot"] / df["Buyume_Pot"].max() * 30) +
        ((100 - df["Pazar_Payi_%"]) / 100 * 30)
    )
    
    return df

def calc_time_series(df, product, freq='M'):
    """Zaman serisi analizi"""
    cols = get_product_cols(product)
    
    if freq == 'M':
        if 'YIL_AY' not in df.columns:
            return None
        df['period'] = df['YIL_AY']
    elif freq == 'W':
        if 'DATE' not in df.columns:
            return None
        df['period'] = df['DATE'].dt.strftime('%Y-W%U')
    else:
        return None
    
    ts = df.groupby('period').agg({
        cols['pf']: 'sum',
        cols['rakip']: 'sum'
    }).reset_index()
    
    ts.columns = ['Period', 'PF', 'Rakip']
    ts['Total'] = ts['PF'] + ts['Rakip']
    ts['Share_%'] = safe_divide(ts['PF'], ts['Total']) * 100
    ts['Growth_%'] = ts['PF'].pct_change() * 100
    
    return ts

def forecast_linear(ts, periods=6):
    """Doğrusal regresyon ile tahmin"""
    if len(ts) < 3:
        return None
    
    X = np.arange(len(ts)).reshape(-1, 1)
    y = ts['PF'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    future_X = np.arange(len(ts), len(ts) + periods).reshape(-1, 1)
    preds = model.predict(future_X)
    preds = np.maximum(preds, 0)  # Negatif değerleri sıfırla
    
    return pd.DataFrame({
        'Period': [f"T+{i+1}" for i in range(periods)],
        'Forecast': preds,
        'Trend': ['↗️' if model.coef_[0] > 0 else '↘️'] * periods
    })

def monte_carlo_sim(df, n_sim=1000):
    """Monte Carlo simülasyonu"""
    top10 = df.nlargest(10, 'PF_Satis')
    
    np.random.seed(42)
    results = {}
    
    for idx, row in top10.iterrows():
        terr = row['TERRITORIES']
        current = row['PF_Satis']
        
        # Parametreler
        growth_mean = 0.05
        growth_std = 0.15
        
        # Simülasyon
        sims = current * (1 + np.random.normal(growth_mean, growth_std, n_sim))
        sims = np.maximum(sims, 0)
        
        results[terr] = {
            'current': current,
            'mean': sims.mean(),
            'p10': np.percentile(sims, 10),
            'p50': np.percentile(sims, 50),
            'p90': np.percentile(sims, 90),
            'risk': sims.std() / sims.mean() * 100 if sims.mean() > 0 else 0
        }
    
    return results

def perform_clustering(df, n_clusters=4):
    """K-Means clustering"""
    features = ['PF_Satis', 'Toplam_Pazar', 'Pazar_Payi_%', 'Goreceli_Pay']
    features = [f for f in features if f in df.columns]
    
    if len(features) < 2 or len(df) < n_clusters:
        return df
    
    X = df[features].fillna(0)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    cluster_names = {
        0: "🔴 Düşük Performans",
        1: "🟡 Orta Performans",
        2: "🟢 Yüksek Performans",
        3: "🔵 Potansiyel"
    }
    
    df['Cluster_Adi'] = df['Cluster'].map(cluster_names)
    
    return df

def calc_manager_perf(df, product):
    """Manager performans analizi"""
    if 'MANAGER' not in df.columns:
        return None
    
    cols = get_product_cols(product)
    
    mgr = df.groupby('MANAGER').agg({
        cols['pf']: 'sum',
        cols['rakip']: 'sum',
        'TERRITORIES': 'nunique'
    }).reset_index()
    
    mgr.columns = ['Manager', 'PF', 'Rakip', 'Territory_Count']
    mgr['Total'] = mgr['PF'] + mgr['Rakip']
    mgr['Share_%'] = safe_divide(mgr['PF'], mgr['Total']) * 100
    mgr['Avg_Per_Territory'] = safe_divide(mgr['PF'], mgr['Territory_Count'])
    mgr['Rank'] = mgr['PF'].rank(ascending=False).astype(int)
    
    return mgr.sort_values('PF', ascending=False)

def generate_swot(df):
    """SWOT analizi"""
    swot = {
        'Güçlü Yönler': [],
        'Zayıf Yönler': [],
        'Fırsatlar': [],
        'Tehditler': []
    }
    
    # Strengths
    high_share = df[df['Pazar_Payi_%'] > 50]
    if len(high_share) > 0:
        swot['Güçlü Yönler'].append(f"✅ {len(high_share)} territoryde %50+ pazar payı")
    
    top_seller = df.nlargest(1, 'PF_Satis').iloc[0]
    swot['Güçlü Yönler'].append(f"✅ En güçlü territory: {top_seller['TERRITORIES']} ({top_seller['PF_Satis']:,.0f} kutu)")
    
    # Weaknesses
    low_share = df[df['Pazar_Payi_%'] < 10]
    if len(low_share) > 3:
        swot['Zayıf Yönler'].append(f"⚠️ {len(low_share)} territoryde %10'dan düşük pay")
    
    zero_sales = df[df['PF_Satis'] == 0]
    if len(zero_sales) > 0:
        swot['Zayıf Yönler'].append(f"⚠️ {len(zero_sales)} territoryde sıfır satış")
    
    # Opportunities
    big_opp = df[
        (df['Toplam_Pazar'] > df['Toplam_Pazar'].median()) &
        (df['Pazar_Payi_%'] < 20)
    ]
    if len(big_opp) > 0:
        total_pot = big_opp['Buyume_Pot'].sum()
        swot['Fırsatlar'].append(f"💡 {len(big_opp)} büyük potansiyelli territory ({total_pot:,.0f} kutu)")
    
    # Threats
    dominant_comp = df[df['Goreceli_Pay'] < 0.5]
    if len(dominant_comp) > 5:
        swot['Tehditler'].append(f"⚡ {len(dominant_comp)} territoryde rakip çok güçlü")
    
    return swot

# ============================================================================
# GÖRSELLEŞTİRME FONKSİYONLARI
# ============================================================================

def create_bar_chart(df, n=20, title="Top Territoryler"):
    """Grouped bar chart"""
    top = df.nlargest(n, 'PF_Satis')
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=top['TERRITORIES'],
        y=top['PF_Satis'],
        name='PF Satış',
        marker_color='#3B82F6',
        text=top['PF_Satis'].apply(lambda x: f'{x:,.0f}'),
        textposition='outside'
    ))
    
    fig.add_trace(go.Bar(
        x=top['TERRITORIES'],
        y=top['Rakip_Satis'],
        name='Rakip Satış',
        marker_color='#EF4444',
        text=top['Rakip_Satis'].apply(lambda x: f'{x:,.0f}'),
        textposition='outside'
    ))
    
    fig.update_layout(
        barmode='group',
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=12),
        title=dict(text=title, font=dict(size=18, color='white')),
        xaxis=dict(tickangle=-45),
        legend=dict(bgcolor='rgba(30, 41, 59, 0.8)')
    )
    
    return fig

def create_bcg_scatter(df):
    """BCG Matrix scatter plot"""
    color_map = {
        '⭐ Star': '#FFD700',
        '🐄 Cash Cow': '#10B981',
        '❓ Question Mark': '#F59E0B',
        '🐶 Dog': '#EF4444'
    }
    
    fig = px.scatter(
        df,
        x='Goreceli_Pay',
        y='Growth_%',
        size='Toplam_Pazar',
        color='BCG',
        hover_name='TERRITORIES',
        hover_data={
            'PF_Satis': ':,.0f',
            'Pazar_Payi_%': ':.1f',
            'Goreceli_Pay': ':.2f',
            'Growth_%': ':.1f'
        },
        color_discrete_map=color_map,
        size_max=60,
        title="BCG Matrix"
    )
    
    # Median çizgileri
    median_share = df['Goreceli_Pay'].median()
    median_growth = df['Growth_%'].median()
    
    fig.add_hline(y=median_growth, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=median_share, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        height=600,
        plot_bgcolor='#0f172a',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis_title="Göreceli Pazar Payı (PF/Rakip)",
        yaxis_title="Büyüme Oranı (%)"
    )
    
    return fig

def create_strategy_pie(df):
    """Strateji dağılımı pie chart"""
    strat_counts = df['Strateji'].value_counts()
    
    colors = {
        '🚀 Agresif Büyüme': '#3B82F6',
        '🛡️ Koruma': '#10B981',
        '💰 Hasat': '#F59E0B',
        '👁️ İzleme': '#6B7280'
    }
    
    fig = go.Figure(data=[go.Pie(
        labels=strat_counts.index,
        values=strat_counts.values,
        hole=0.4,
        marker=dict(colors=[colors.get(x, '#6B7280') for x in strat_counts.index]),
        textinfo='label+percent',
        textfont=dict(size=14)
    )])
    
    fig.update_layout(
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        title="Strateji Dağılımı",
        showlegend=True
    )
    
    return fig

def create_ts_line(ts):
    """Time series line chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=ts['Period'],
        y=ts['PF'],
        name='PF Satış',
        line=dict(color='#3B82F6', width=3),
        mode='lines+markers'
    ))
    
    fig.add_trace(go.Scatter(
        x=ts['Period'],
        y=ts['Rakip'],
        name='Rakip Satış',
        line=dict(color='#EF4444', width=3),
        mode='lines+markers'
    ))
    
    fig.update_layout(
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        title="Satış Trendi",
        xaxis_title="Periyot",
        yaxis_title="Satış (Kutu)",
        hovermode='x unified'
    )
    
    return fig

def create_growth_bar(ts):
    """Büyüme oranı bar chart"""
    ts_clean = ts.dropna(subset=['Growth_%'])
    
    colors = ['#10B981' if x >= 0 else '#EF4444' for x in ts_clean['Growth_%']]
    
    fig = go.Figure(data=[go.Bar(
        x=ts_clean['Period'],
        y=ts_clean['Growth_%'],
        marker_color=colors,
        text=ts_clean['Growth_%'].apply(lambda x: f'{x:+.1f}%'),
        textposition='outside'
    )])
    
    fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
    
    fig.update_layout(
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        title="Büyüme Oranı",
        xaxis_title="Periyot",
        yaxis_title="Büyüme (%)",
        showlegend=False
    )
    
    return fig

def create_mc_box(results):
    """Monte Carlo box plot"""
    fig = go.Figure()
    
    for terr_name, res in results.items():
        fig.add_trace(go.Box(
            y=[res['p10'], res['p50'], res['p90']],
            name=terr_name[:25],
            marker_color='#3B82F6',
            boxmean='sd'
        ))
    
    fig.update_layout(
        height=500,
        plot_bgcolor='#0f172a',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        title="Monte Carlo Simülasyon Sonuçları (P10-P50-P90)",
        yaxis_title="Tahmini Satış",
        showlegend=True
    )
    
    return fig

def create_cluster_scatter(df):
    """Clustering scatter plot"""
    fig = px.scatter(
        df,
        x='Toplam_Pazar',
        y='Pazar_Payi_%',
        color='Cluster_Adi',
        size='PF_Satis',
        hover_name='TERRITORIES',
        size_max=50,
        title="Territory Clustering"
    )
    
    fig.update_layout(
        height=600,
        plot_bgcolor='#0f172a',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis_title="Pazar Büyüklüğü",
        yaxis_title="Pazar Payı (%)"
    )
    
    return fig

def create_manager_bar(mgr_df):
    """Manager performans bar chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=mgr_df['Manager'],
        y=mgr_df['PF'],
        marker_color='#3B82F6',
        text=mgr_df['PF'].apply(lambda x: f'{x:,.0f}'),
        textposition='outside',
        name='PF Satış'
    ))
    
    fig.update_layout(
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        title="Manager Performansı",
        xaxis_title="Manager",
        yaxis_title="Satış (Kutu)",
        xaxis=dict(tickangle=-45)
    )
    
    return fig

# ============================================================================
# ANA UYGULAMA
# ============================================================================

def main():
    # Header
    st.markdown('<h1 class="main-header">💊 TİCARİ PORTFÖY ANALİZİ v4.0</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("📂 Veri Yönetimi")
    uploaded = st.sidebar.file_uploader("Excel Dosyası Yükle", type=['xlsx', 'xls'])
    
    if not uploaded:
        st.info("👈 Lütfen sol taraftan Excel dosyası yükleyin")
        st.markdown("""
        ### 📋 Beklenen Veri Formatı:
        - **TERRITORIES**: Territory adı
        - **DATE**: Tarih
        - **[ÜRÜN_ADI]**: PF satış verisi
        - **DIGER [ÜRÜN_ADI]**: Rakip satış verisi
        - **REGION**: Bölge (opsiyonel)
        - **MANAGER**: Manager (opsiyonel)
        - **CITY**: Şehir (opsiyonel)
        """)
        st.stop()
    
    # Veri yükleme
    df = load_data(uploaded)
    if df is None:
        st.stop()
    
    st.sidebar.success(f"✅ {len(df):,} satır yüklendi")
    
    # Filtreler
    st.sidebar.header("🎯 Filtreler")
    
    # Ürün seçimi
    products = ["CORTIPOL", "TROCMETAM", "DEKSAMETAZON", "PF IZOTONIK"]
    product = st.sidebar.selectbox("💊 Ürün Seçin", products)
    
    # Territory filtresi
    territories =["TÜMÜ"] + sorted(df['TERRITORIES'].unique().tolist())
territory = st.sidebar.selectbox("🏢 Territory", territories)

# Region filtresi (varsa)
if 'REGION' in df.columns:
    regions = ["TÜMÜ"] + sorted(df['REGION'].unique().tolist())
    region = st.sidebar.selectbox("🌍 Bölge", regions)
else:
    region = "TÜMÜ"

# Veriyi filtrele
df_filt = df.copy()

if territory != "TÜMÜ":
    df_filt = df_filt[df_filt['TERRITORIES'] == territory]

if region != "TÜMÜ" and 'REGION' in df_filt.columns:
    df_filt = df_filt[df_filt['REGION'] == region]

# Sekmeler
tabs = st.tabs([
    "📊 Dashboard",
    "🏢 Territory Analizi",
    "⭐ BCG & Strateji",
    "📈 Zaman Serisi",
    "🎲 Monte Carlo",
    "🎯 Clustering",
    "👔 Manager Analizi",
    "📥 Raporlar"
])

# ========================================================================
# SEKME 1: DASHBOARD
# ========================================================================

with tabs[0]:
    st.header("📊 Genel Bakış Dashboard")
    
    cols = get_product_cols(product)
    
    # Metrikler
    total_pf = df_filt[cols['pf']].sum()
    total_rakip = df_filt[cols['rakip']].sum()
    total_market = total_pf + total_rakip
    share = (total_pf / total_market * 100) if total_market > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("💊 PF Satış", f"{total_pf:,.0f}", "kutu")
    col2.metric("🏪 Toplam Pazar", f"{total_market:,.0f}", "kutu")
    col3.metric("📊 Pazar Payı", f"%{share:.1f}", "")
    col4.metric("🎯 Büyüme Potansiyeli", f"{total_rakip:,.0f}", "kutu")
    
    st.markdown("---")
    
    # Territory performans
    terr = calc_territory_perf(df_filt, product)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏆 Top 20 Territory")
        fig = create_bar_chart(terr, n=20)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Pazar Payı Dağılımı")
        
        # Histogram
        fig = px.histogram(
            terr,
            x='Pazar_Payi_%',
            nbins=20,
            title="Pazar Payı Dağılımı",
            labels={'Pazar_Payi_%': 'Pazar Payı (%)'}
        )
        fig.update_layout(
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Özet istatistikler
    st.subheader("📊 Özet İstatistikler")
    
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Toplam Territory", len(terr))
    col2.metric("Ortalama Pazar Payı", f"%{terr['Pazar_Payi_%'].mean():.1f}")
    col3.metric("Medyan Pazar Payı", f"%{terr['Pazar_Payi_%'].median():.1f}")
    col4.metric("Sıfır Satış", len(terr[terr['PF_Satis'] == 0]))

# ========================================================================
# SEKME 2: TERRITORY ANALİZİ
# ========================================================================

with tabs[1]:
    st.header("🏢 Detaylı Territory Analizi")
    
    terr = calc_territory_perf(df_filt, product)
    
    # Filtreleme seçenekleri
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_share = st.slider("Min Pazar Payı (%)", 0, 100, 0)
    
    with col2:
        min_sales = st.number_input("Min PF Satış", 0, int(terr['PF_Satis'].max()), 0)
    
    with col3:
        sort_by = st.selectbox("Sırala", 
                               ['PF_Satis', 'Pazar_Payi_%', 'Buyume_Pot', 'Goreceli_Pay'])
    
    # Filtreleme
    terr_filt = terr[
        (terr['Pazar_Payi_%'] >= min_share) &
        (terr['PF_Satis'] >= min_sales)
    ].sort_values(sort_by, ascending=False)
    
    # Tablo
    st.dataframe(
        terr_filt.style.format({
            'PF_Satis': '{:,.0f}',
            'Rakip_Satis': '{:,.0f}',
            'Toplam_Pazar': '{:,.0f}',
            'Pazar_Payi_%': '{:.1f}%',
            'Buyume_Pot': '{:,.0f}',
            'Goreceli_Pay': '{:.2f}'
        }).background_gradient(subset=['Pazar_Payi_%'], cmap='RdYlGn'),
        use_container_width=True,
        height=600
    )
    
    # İndirme
    csv = terr_filt.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Tabloyu İndir (CSV)",
        csv,
        f"territory_analiz_{datetime.now().strftime('%Y%m%d')}.csv",
        "text/csv"
    )

# ========================================================================
# SEKME 3: BCG & STRATEJİ
# ========================================================================

with tabs[2]:
    st.header("⭐ BCG Matrix & Yatırım Stratejisi")
    
    # BCG hesaplama
    bcg_df = calc_bcg_full(df_filt, product)
    strat_df = calc_strategy(bcg_df)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("BCG Matrix")
        fig_bcg = create_bcg_scatter(strat_df)
        st.plotly_chart(fig_bcg, use_container_width=True)
        
        # BCG dağılımı
        bcg_counts = strat_df['BCG'].value_counts()
        st.markdown("### 📊 BCG Dağılımı")
        for bcg_cat, count in bcg_counts.items():
            pct = count / len(strat_df) * 100
            st.markdown(f"**{bcg_cat}**: {count} territory (%{pct:.1f})")
    
    with col2:
        st.subheader("Yatırım Stratejisi")
        fig_strat = create_strategy_pie(strat_df)
        st.plotly_chart(fig_strat, use_container_width=True)
        
        # Strateji açıklamaları
        st.markdown("### 📋 Strateji Açıklamaları")
        st.markdown("""
        - **🚀 Agresif Büyüme**: Büyük pazar, düşük pay → Yoğun yatırım
        - **🛡️ Koruma**: Büyük pazar, yüksek pay → Konumu koru
        - **💰 Hasat**: Yüksek pay → Kar maksimizasyonu
        - **👁️ İzleme**: Düşük öncelik → Minimal kaynak
        """)
    
    # Öncelikli territoryler
    st.markdown("---")
    st.subheader("🎯 Öncelikli Territoryler (Top 15)")
    
    top_prior = strat_df.nlargest(15, 'Oncelik')
    
    st.dataframe(
        top_prior[['TERRITORIES', 'BCG', 'Strateji', 'PF_Satis', 
                   'Pazar_Payi_%', 'Buyume_Pot', 'Oncelik']].style.format({
            'PF_Satis': '{:,.0f}',
            'Pazar_Payi_%': '{:.1f}%',
            'Buyume_Pot': '{:,.0f}',
            'Oncelik': '{:.0f}'
        }).background_gradient(subset=['Oncelik'], cmap='YlOrRd'),
        use_container_width=True
    )
    
    # SWOT Analizi
    st.markdown("---")
    st.subheader("🎯 SWOT Analizi")
    
    swot = generate_swot(strat_df)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 💪 Güçlü Yönler")
        for item in swot['Güçlü Yönler']:
            st.markdown(f"- {item}")
        
        st.markdown("#### 🎯 Fırsatlar")
        for item in swot['Fırsatlar']:
            st.markdown(f"- {item}")
    
    with col2:
        st.markdown("#### ⚠️ Zayıf Yönler")
        for item in swot['Zayıf Yönler']:
            st.markdown(f"- {item}")
        
        st.markdown("#### ⚡ Tehditler")
        for item in swot['Tehditler']:
            st.markdown(f"- {item}")

# ========================================================================
# SEKME 4: ZAMAN SERİSİ
# ========================================================================

with tabs[3]:
    st.header("📈 Zaman Serisi Analizi")
    
    if 'YIL_AY' not in df_filt.columns:
        st.warning("⚠️ Zaman serisi analizi için DATE kolonu gerekli")
    else:
        # Periyot seçimi
        freq_opt = st.radio("Periyot", ["Aylık", "Haftalık"], horizontal=True)
        freq = 'M' if freq_opt == "Aylık" else 'W'
        
        # Time series hesaplama
        ts = calc_time_series(df_filt, product, freq)
        
        if ts is not None and len(ts) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Satış Trendi")
                fig_ts = create_ts_line(ts)
                st.plotly_chart(fig_ts, use_container_width=True)
            
            with col2:
                st.subheader("📈 Büyüme Oranı")
                fig_growth = create_growth_bar(ts)
                st.plotly_chart(fig_growth, use_container_width=True)
            
            # Pazar payı trendi
            st.subheader("📊 Pazar Payı Trendi")
            fig_share = go.Figure()
            fig_share.add_trace(go.Scatter(
                x=ts['Period'],
                y=ts['Share_%'],
                mode='lines+markers',
                line=dict(color='#10B981', width=3),
                fill='tozeroy',
                fillcolor='rgba(16, 185, 129, 0.2)'
            ))
            fig_share.update_layout(
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                yaxis_title="Pazar Payı (%)"
            )
            st.plotly_chart(fig_share, use_container_width=True)
            
            # Tahminleme
            st.markdown("---")
            st.subheader("🔮 Tahminleme")
            
            if len(ts) >= 3:
                periods = st.slider("Tahmin Periyodu", 1, 12, 6)
                
                forecast = forecast_linear(ts, periods)
                
                if forecast is not None:
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        # Tahmin grafiği
                        fig_forecast = go.Figure()
                        
                        fig_forecast.add_trace(go.Scatter(
                            x=ts['Period'],
                            y=ts['PF'],
                            name='Gerçek',
                            mode='lines+markers',
                            line=dict(color='#3B82F6', width=3)
                        ))
                        
                        fig_forecast.add_trace(go.Scatter(
                            x=forecast['Period'],
                            y=forecast['Forecast'],
                            name='Tahmin',
                            mode='lines+markers',
                            line=dict(color='#F59E0B', width=3, dash='dash')
                        ))
                        
                        fig_forecast.update_layout(
                            height=400,
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white'),
                            title="Satış Tahmini"
                        )
                        st.plotly_chart(fig_forecast, use_container_width=True)
                    
                    with col2:
                        st.markdown("### 📊 Tahmin Değerleri")
                        st.dataframe(
                            forecast.style.format({
                                'Forecast': '{:,.0f}'
                            }),
                            use_container_width=True
                        )
            else:
                st.info("ℹ️ Tahminleme için en az 3 periyot verisi gerekli")
        else:
            st.warning("⚠️ Zaman serisi verisi bulunamadı")

# ========================================================================
# SEKME 5: MONTE CARLO
# ========================================================================

with tabs[4]:
    st.header("🎲 Monte Carlo Simülasyonu")
    
    st.markdown("""
    Monte Carlo simülasyonu, gelecekteki satış değerlerinin olasılık dağılımını tahmin eder.
    - **Varsayım**: %5 ortalama büyüme, %15 standart sapma
    - **Simülasyon**: Her territory için 1000 senaryo
    - **Çıktı**: P10 (kötümser), P50 (beklenen), P90 (iyimser)
    """)
    
    terr = calc_territory_perf(df_filt, product)
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        n_sims = st.select_slider(
            "Simülasyon Sayısı",
            options=[100, 500, 1000, 2000, 5000],
            value=1000
        )
    
    with col2:
        st.info(f"ℹ️ Top 10 territory için {n_sims:,} simülasyon çalıştırılacak")
    
    if st.button("🚀 Simülasyonu Başlat", type="primary"):
        with st.spinner("Simülasyon çalışıyor..."):
            results = monte_carlo_sim(terr, n_sims)
            
            # Box plot
            st.subheader("📊 Simülasyon Sonuçları")
            fig_mc = create_mc_box(results)
            st.plotly_chart(fig_mc, use_container_width=True)
            
            # Sonuç tablosu
            st.subheader("📋 Detaylı Sonuçlar")
            
            mc_df = pd.DataFrame(results).T
            mc_df.index.name = 'Territory'
            mc_df = mc_df.reset_index()
            
            st.dataframe(
                mc_df.style.format({
                    'current': '{:,.0f}',
                    'mean': '{:,.0f}',
                    'p10': '{:,.0f}',
                    'p50': '{:,.0f}',
                    'p90': '{:,.0f}',
                    'risk': '{:.1f}%'
                }).background_gradient(subset=['risk'], cmap='RdYlGn_r'),
                use_container_width=True
            )
            
            # Risk analizi
            st.subheader("⚠️ Risk Analizi")
            
            high_risk = mc_df[mc_df['risk'] > 20]
            
            if len(high_risk) > 0:
                st.warning(f"⚠️ {len(high_risk)} territory yüksek risk taşıyor (CV > 20%)")
                st.dataframe(high_risk[['Territory', 'current', 'mean', 'risk']], 
                           use_container_width=True)
            else:
                st.success("✅ Tüm territoryler makul risk seviyesinde")

# ========================================================================
# SEKME 6: CLUSTERING
# ========================================================================

with tabs[5]:
    st.header("🎯 Territory Clustering (K-Means)")
    
    st.markdown("""
    K-Means algoritması ile territoryler benzerliklerine göre gruplandırılır.
    - **Özellikler**: PF Satış, Pazar Büyüklüğü, Pazar Payı, Göreceli Konum
    - **Amaç**: Benzer özelliklere sahip territoryleri tanımlamak
    """)
    
    terr = calc_territory_perf(df_filt, product)
    
    n_clusters = st.slider("Cluster Sayısı", 2, 6, 4)
    
    if st.button("🔍 Clustering Yap", type="primary"):
        with st.spinner("Clustering çalışıyor..."):
            clustered = perform_clustering(terr, n_clusters)
            
            if 'Cluster' in clustered.columns:
                # Scatter plot
                st.subheader("📊 Cluster Görselleştirme")
                fig_cluster = create_cluster_scatter(clustered)
                st.plotly_chart(fig_cluster, use_container_width=True)
                
                # Cluster özellikleri
                st.subheader("📋 Cluster Özellikleri")
                
                cluster_summary = clustered.groupby('Cluster_Adi').agg({
                    'TERRITORIES': 'count',
                    'PF_Satis': 'mean',
                    'Toplam_Pazar': 'mean',
                    'Pazar_Payi_%': 'mean'
                }).reset_index()
                
                cluster_summary.columns = ['Cluster', 'Territory_Count', 
                                          'Avg_PF_Sales', 'Avg_Market', 'Avg_Share']
                
                st.dataframe(
                    cluster_summary.style.format({
                        'Avg_PF_Sales': '{:,.0f}',
                        'Avg_Market': '{:,.0f}',
                        'Avg_Share': '{:.1f}%'
                    }),
                    use_container_width=True
                )
                
                # Her cluster'ın territorylerini göster
                st.subheader("🏢 Cluster Detayları")
                
                for cluster_name in sorted(clustered['Cluster_Adi'].unique()):
                    with st.expander(f"📁 {cluster_name}"):
                        cluster_data = clustered[clustered['Cluster_Adi'] == cluster_name]
                        st.dataframe(
                            cluster_data[['TERRITORIES', 'PF_Satis', 'Pazar_Payi_%', 
                                        'Toplam_Pazar']].style.format({
                                'PF_Satis': '{:,.0f}',
                                'Pazar_Payi_%': '{:.1f}%',
                                'Toplam_Pazar': '{:,.0f}'
                            }),
                            use_container_width=True
                        )
            else:
                st.error("❌ Clustering başarısız")

# ========================================================================
# SEKME 7: MANAGER ANALİZİ
# ========================================================================

with tabs[6]:
    st.header("👔 Manager Performans Analizi")
    
    if 'MANAGER' not in df_filt.columns:
        st.warning("⚠️ MANAGER kolonu veri setinde bulunamadı")
    else:
        mgr_df = calc_manager_perf(df_filt, product)
        
        if mgr_df is not None:
            # Genel metrikler
            col1, col2, col3, col4 = st.columns(4)
            
            col1.metric("👥 Toplam Manager", len(mgr_df))
            col2.metric("🏆 En İyi Satış", f"{mgr_df['PF'].max():,.0f}")
            col3.metric("📊 Ort. Pazar Payı", f"%{mgr_df['Share_%'].mean():.1f}")
            col4.metric("🎯 Toplam Territory", mgr_df['Territory_Count'].sum())
            
            st.markdown("---")
            
            # Bar chart
            st.subheader("📊 Manager Satış Performansı")
            fig_mgr = create_manager_bar(mgr_df)
            st.plotly_chart(fig_mgr, use_container_width=True)
            
            # Detay tablo
            st.subheader("📋 Detaylı Manager Analizi")
            
            st.dataframe(
                mgr_df.style.format({
                    'PF': '{:,.0f}',
                    'Rakip': '{:,.0f}',
                    'Total': '{:,.0f}',
                    'Share_%': '{:.1f}%',
                    'Avg_Per_Territory': '{:,.0f}'
                }).background_gradient(subset=['PF'], cmap='YlGnBu'),
                use_container_width=True
            )
            
            # Territory dağılımı
            st.subheader("🏢 Manager Bazlı Territory Dağılımı")
            
            fig_terr_dist = px.bar(
                mgr_df,
                x='Manager',
                y='Territory_Count',
                title="Manager Başına Territory Sayısı",
                text='Territory_Count'
            )
            fig_terr_dist.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            st.plotly_chart(fig_terr_dist, use_container_width=True)
        else:
            st.error("❌ Manager analizi yapılamadı")

# ========================================================================
# SEKME 8: RAPORLAR
# ========================================================================

with tabs[7]:
    st.header("📥 Rapor İndirme")
    
    st.markdown("""
    Bu sayfadan analiz sonuçlarınızı farklı formatlarda indirebilirsiniz.
    """)
    
    # Rapor seçenekleri
    st.subheader("📊 Hangi Raporları İndirmek İstersiniz?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        inc_territory = st.checkbox("Territory Analizi", value=True)
        inc_bcg = st.checkbox("BCG & Strateji", value=True)
        inc_ts = st.checkbox("Zaman Serisi", value=False)
    
    with col2:
        inc_manager = st.checkbox("Manager Performans", value=False)
        inc_summary = st.checkbox("Özet İstatistikler", value=True)
    
    st.markdown("---")
    
    # Excel export
    st.subheader("📥 Excel Raporu")
    
    if st.button("📊 Excel Raporu Oluştur", type="primary"):
        with st.spinner("Rapor hazırlanıyor..."):
            output = BytesIO()
            
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # Territory
                if inc_territory:
                    terr = calc_territory_perf(df_filt, product)
                    terr.to_excel(writer, sheet_name='Territory_Analizi', index=False)
                
                # BCG & Strateji
                if inc_bcg:
                    bcg_df = calc_bcg_full(df_filt, product)
                    strat_df = calc_strategy(bcg_df)
                    strat_df.to_excel(writer, sheet_name='BCG_Strateji', index=False)
                
                # Time Series
                if inc_ts and 'YIL_AY' in df_filt.columns:
                    ts = calc_time_series(df_filt, product, 'M')
                    if ts is not None:
                        ts.to_excel(writer, sheet_name='Zaman_Serisi', index=False)
                
                # Manager
                if inc_manager and 'MANAGER' in df_filt.columns:
                    mgr = calc_manager_perf(df_filt, product)
                    if mgr is not None:
                        mgr.to_excel(writer, sheet_name='Manager_Analizi', index=False)
                
                # Summary
                if inc_summary:
                    cols = get_product_cols(product)
                    summary_data = {
                        'Metrik': [
                            'Toplam PF Satış',
                            'Toplam Pazar',
                            'Pazar Payı (%)',
                            'Territory Sayısı',
                            'Ortalama Pazar Payı (%)',
                            'Sıfır Satış Territory'
                        ],
                        'Değer': [
                            df_filt[cols['pf']].sum(),
                            df_filt[cols['pf']].sum() + df_filt[cols['rakip']].sum(),
                            (df_filt[cols['pf']].sum() / (df_filt[cols['pf']].sum() + df_filt[cols['rakip']].sum()) * 100) if (df_filt[cols['pf']].sum() + df_filt[cols['rakip']].sum()) > 0 else 0,
                            len(df_filt['TERRITORIES'].unique()),
                            calc_territory_perf(df_filt, product)['Pazar_Payi_%'].mean(),
                            len(calc_territory_perf(df_filt, product)[calc_territory_perf(df_filt, product)['PF_Satis'] == 0])
                        ]
                    }
                    summary_df = pd.DataFrame(summary_data)
                    summary_df.to_excel(writer, sheet_name='Ozet', index=False)
            
            st.success("✅ Rapor hazır!")
            
            st.download_button(
                "📥 Excel Raporunu İndir",
                output.getvalue(),
                f"portfolio_analiz_{product}_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    # CSV export
    st.markdown("---")
    st.subheader("📥 CSV Raporu (Territory Analizi)")
    
    terr = calc_territory_perf(df_filt, product)
    csv = terr.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        "📥 CSV İndir",
        csv,
        f"territory_analiz_{product}_{datetime.now().strftime('%Y%m%d')}.csv",
        "text/csv"
    )
============================================================================
ÇALIŞTIR
============================================================================
if name == "main":
main()
