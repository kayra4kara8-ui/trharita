"""
🎯 TİCARİ PORTFÖY ANALİZ SİSTEMİ
Territory Bazlı Zaman Serisi ve Yatırım Stratejisi Analizi

Features:
- Territory × Ürün bazlı performans analizi
- Aylık zaman serisi ve trend analizi  
- BCG Matrix stratejik konumlandırma
- Sankey akış diyagramları
- Yatırım stratejisi önerileri
- Türkiye haritası görselleştirme
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
from io import BytesIO

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
# CUSTOM CSS
# =============================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #0EA5E9;
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
    df['REGION'] = df['REGION'].str.upper().str.strip()
    df['MANAGER'] = df['MANAGER'].str.upper().str.strip()
    
    return df

# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def calculate_territory_performance(df, product):
    """Territory bazlı performans analizi"""
    cols = get_product_columns(product)
    
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

def calculate_time_series(df, product, territory=None):
    """Aylık zaman serisi analizi"""
    cols = get_product_columns(product)
    
    # Filtreleme
    df_filtered = df.copy()
    if territory and territory != "TÜMÜ":
        df_filtered = df_filtered[df_filtered['TERRITORIES'] == territory]
    
    # Aylık toplam
    monthly = df_filtered.groupby('YIL_AY').agg({
        cols['pf']: 'sum',
        cols['rakip']: 'sum'
    }).reset_index().sort_values('YIL_AY')
    
    monthly.columns = ['YIL_AY', 'PF_Satis', 'Rakip_Satis']
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

def calculate_bcg_matrix(df, product):
    """BCG Matrix kategorileri hesapla"""
    # Territory performansı
    terr_perf = calculate_territory_performance(df, product)
    
    # Pazar büyüme oranı hesapla (ilk 6 ay vs son 6 ay)
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

def calculate_investment_strategy(bcg_df):
    """Yatırım stratejisi önerileri"""
    def assign_strategy(row):
        bcg = row['BCG_Kategori']
        growth = row['Pazar_Buyume_%']
        
        if '⭐' in bcg:
            return '🚀 Agresif Hızlandırma' if growth > 10 else '🛡️ Koruma'
        elif '🐄' in bcg:
            return '🛡️ Koruma'
        elif '❓' in bcg:
            return '🚀 Agresif Hızlandırma' if growth > 5 else '👁️ İzlem'
        else:  # Dog
            return '🚪 Çıkış/Minimize' if growth < -5 else '👁️ İzlem'
    
    bcg_df['Yatirim_Stratejisi'] = bcg_df.apply(assign_strategy, axis=1)
    
    # Aksiyon önerileri
    def suggest_action(row):
        strategy = row['Yatirim_Stratejisi']
        if '🚀' in strategy:
            return 'Yatırımı artır, pazar payını hızla yükselt'
        elif '🛡️' in strategy:
            return 'Mevcut konumu koru, maliyetleri optimize et'
        elif '👁️' in strategy:
            return 'Düşük yatırımla izle, fırsatları değerlendir'
        else:
            return 'Kaynak tahsisini azalt veya çıkışı değerlendir'
    
    bcg_df['Aksiyon'] = bcg_df.apply(suggest_action, axis=1)
    
    return bcg_df

def calculate_ytd_comparison(df, product):
    """YTD ve dönem karşılaştırmaları"""
    cols = get_product_columns(product)
    max_date = df['DATE'].max()
    
    # Tüm dönem
    all_period = df[cols['pf']].sum()
    
    # YTD
    ytd = df[df['DATE'].dt.year == max_date.year][cols['pf']].sum()
    
    # Son 3 ay
    last_3m = df[df['DATE'] >= (max_date - pd.DateOffset(months=3))][cols['pf']].sum()
    
    # Önceki 3 ay
    prev_3m = df[(df['DATE'] >= (max_date - pd.DateOffset(months=6))) & 
                 (df['DATE'] < (max_date - pd.DateOffset(months=3)))][cols['pf']].sum()
    
    # Son 6 ay
    last_6m = df[df['DATE'] >= (max_date - pd.DateOffset(months=6))][cols['pf']].sum()
    
    # Değişim hesapla
    change_3m = safe_divide(last_3m - prev_3m, prev_3m) * 100 if prev_3m > 0 else 0
    
    return {
        'Tum_Donem': all_period,
        'YTD': ytd,
        'Son_6_Ay': last_6m,
        'Son_3_Ay': last_3m,
        'Onceki_3_Ay': prev_3m,
        'Degisim_3M_%': change_3m[0] if isinstance(change_3m, np.ndarray) else change_3m
    }

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_territory_bar_chart(df, top_n=20):
    """Territory performans bar chart"""
    top_terr = df.nlargest(top_n, 'PF_Satis')
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=top_terr['Territory'],
        y=top_terr['PF_Satis'],
        name='PF Satış',
        marker_color='#3B82F6',
        text=top_terr['PF_Satis'].apply(lambda x: f'{x:,.0f}'),
        textposition='outside'
    ))
    
    fig.add_trace(go.Bar(
        x=top_terr['Territory'],
        y=top_terr['Rakip_Satis'],
        name='Rakip Satış',
        marker_color='#EF4444',
        text=top_terr['Rakip_Satis'].apply(lambda x: f'{x:,.0f}'),
        textposition='outside'
    ))
    
    fig.update_layout(
        title=f'Top {top_n} Territory - PF vs Rakip',
        xaxis_title='Territory',
        yaxis_title='Satış',
        barmode='group',
        height=500,
        xaxis=dict(tickangle=-45),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_time_series_chart(monthly_df):
    """Zaman serisi line chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=monthly_df['YIL_AY'],
        y=monthly_df['PF_Satis'],
        mode='lines+markers',
        name='PF Satış',
        line=dict(color='#3B82F6', width=3),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=monthly_df['YIL_AY'],
        y=monthly_df['Rakip_Satis'],
        mode='lines+markers',
        name='Rakip Satış',
        line=dict(color='#EF4444', width=3),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=monthly_df['YIL_AY'],
        y=monthly_df['MA_3'],
        mode='lines',
        name='3 Aylık Ort.',
        line=dict(color='#10B981', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title='Aylık Satış Trendi',
        xaxis_title='Ay',
        yaxis_title='Satış',
        height=400,
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_market_share_chart(monthly_df):
    """Pazar payı trendi"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=monthly_df['YIL_AY'],
        y=monthly_df['Pazar_Payi_%'],
        mode='lines+markers',
        fill='tozeroy',
        line=dict(color='#8B5CF6', width=2),
        marker=dict(size=8),
        name='Pazar Payı %'
    ))
    
    fig.update_layout(
        title='Pazar Payı Trendi',
        xaxis_title='Ay',
        yaxis_title='Pazar Payı (%)',
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_growth_chart(monthly_df):
    """Büyüme oranı chart"""
    fig = go.Figure()
    
    # PF Büyüme
    colors_pf = ['#10B981' if x > 0 else '#EF4444' for x in monthly_df['PF_Buyume_%']]
    fig.add_trace(go.Bar(
        x=monthly_df['YIL_AY'],
        y=monthly_df['PF_Buyume_%'],
        name='PF Büyüme %',
        marker_color=colors_pf,
        text=monthly_df['PF_Buyume_%'].apply(lambda x: f'{x:.1f}%' if not pd.isna(x) else ''),
        textposition='outside'
    ))
    
    fig.update_layout(
        title='Aylık Büyüme Oranları',
        xaxis_title='Ay',
        yaxis_title='Büyüme (%)',
        height=400,
        xaxis=dict(tickangle=-45),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_bcg_scatter(bcg_df):
    """BCG Matrix scatter plot"""
    # Renk haritası
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
    
    # Median çizgileri
    median_share = bcg_df['Goreceli_Pazar_Payi'].median()
    median_growth = bcg_df['Pazar_Buyume_%'].median()
    
    fig.add_hline(y=median_growth, line_dash="dash", line_color="rgba(255,255,255,0.4)")
    fig.add_vline(x=median_share, line_dash="dash", line_color="rgba(255,255,255,0.4)")
    
    # Kadran etiketleri
    max_x = bcg_df['Goreceli_Pazar_Payi'].max()
    max_y = bcg_df['Pazar_Buyume_%'].max()
    
    annotations = [
        dict(x=median_share + (max_x - median_share) * 0.5, y=median_growth + (max_y - median_growth) * 0.5,
             text="⭐ STARS", showarrow=False, font=dict(size=16, color="rgba(255,215,0,0.5)")),
        dict(x=median_share + (max_x - median_share) * 0.5, y=median_growth * 0.5,
             text="❓ QUESTIONS", showarrow=False, font=dict(size=16, color="rgba(59,130,246,0.5)")),
        dict(x=median_share * 0.5, y=median_growth + (max_y - median_growth) * 0.5,
             text="🐄 COWS", showarrow=False, font=dict(size=16, color="rgba(16,185,129,0.5)")),
        dict(x=median_share * 0.5, y=median_growth * 0.5,
             text="🐶 DOGS", showarrow=False, font=dict(size=16, color="rgba(156,163,175,0.5)"))
    ]
    
    fig.update_layout(
        title='BCG Matrix - Stratejik Portföy Konumlandırma',
        height=600,
        plot_bgcolor='#0f172a',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e2e8f0'),
        annotations=annotations
    )
    
    return fig

def create_sankey_diagram(df, product, top_n=15):
    """Sankey akış diyagramı: Region → Territory → BCG"""
    # BCG hesapla
    bcg_df = calculate_bcg_matrix(df, product)
    top_terr = bcg_df.nlargest(top_n, 'PF_Satis')
    
    # Node'ları oluştur
    regions = top_terr['Region'].unique().tolist()
    territories = top_terr['Territory'].tolist()
    bcg_categories = top_terr['BCG_Kategori'].unique().tolist()
    
    nodes = regions + territories + bcg_categories
    node_dict = {node: idx for idx, node in enumerate(nodes)}
    
    # Link'leri oluştur
    sources, targets, values, colors_link = [], [], [], []
    
    # Region → Territory
    for idx, row in top_terr.iterrows():
        sources.append(node_dict[row['Region']])
        targets.append(node_dict[row['Territory']])
        values.append(row['PF_Satis'])
        colors_link.append('rgba(59, 130, 246, 0.3)')
    
    # Territory → BCG
    for idx, row in top_terr.iterrows():
        sources.append(node_dict[row['Territory']])
        targets.append(node_dict[row['BCG_Kategori']])
        values.append(row['PF_Satis'])
        
        # BCG'ye göre renk
        if '⭐' in row['BCG_Kategori']:
            colors_link.append('rgba(255, 215, 0, 0.4)')
        elif '🐄' in row['BCG_Kategori']:
            colors_link.append('rgba(16, 185, 129, 0.4)')
        elif '❓' in row['BCG_Kategori']:
            colors_link.append('rgba(59, 130, 246, 0.4)')
        else:
            colors_link.append('rgba(156, 163, 175, 0.4)')
    
    # Node renkleri
    node_colors = []
    for node in nodes:
        if node in regions:
            node_colors.append('#3B82F6')
        elif node in territories:
            node_colors.append('#8B5CF6')
        else:  # BCG
            if '⭐' in node:
                node_colors.append('#FFD700')
            elif '🐄' in node:
                node_colors.append('#10B981')
            elif '❓' in node:
                node_colors.append('#3B82F6')
            else:
                node_colors.append('#9CA3AF')
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color='white', width=2),
            label=nodes,
            color=node_colors
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=colors_link
        )
    )])
    
    fig.update_layout(
        title=f'Sankey Akış: Region → Territory → BCG (Top {top_n})',
        height=600,
        font=dict(size=10, color='white'),
        plot_bgcolor='#0f172a',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

# =============================================================================
# MAIN APP
# =============================================================================

def main():
    # Header
    st.markdown('<h1 class="main-header">💊 TİCARİ PORTFÖY ANALİZ SİSTEMİ</h1>', unsafe_allow_html=True)
    st.markdown("**Territory Bazlı Performans, Zaman Serisi ve Yatırım Stratejisi Analizi**")
    
    # Sidebar
    st.sidebar.header("📂 Dosya Yükleme")
    uploaded_file = st.sidebar.file_uploader(
        "Excel Dosyası Yükleyin",
        type=['xlsx', 'xls'],
        help="Ticari Ürün 2025 verisi"
    )
    
    if not uploaded_file:
        st.info("👈 Lütfen sol taraftan Excel dosyasını yükleyin")
        st.markdown("""
        ### 📋 Gerekli Kolonlar:
        - **DATE**: Tarih (aylık)
        - **MANAGER**: Ticaret müdürü
        - **TERRITORIES**: Territory adı
        - **REGION**: Bölge
        - **CITY**: Şehir
        - **CORTIPOL, DIGER CORTIPOL**: Ürün satışları
        - **TROCMETAM, DIGER TROCMETAM**: Ürün satışları
        - **DEKSAMETAZON, DIGER DEKSAMETAZON**: Ürün satışları
        - **PF IZOTONIK, DIGER IZOTONIK**: Ürün satışları
        """)
        st.stop()
    
    # Veriyi yükle
    try:
        df = load_excel_data(uploaded_file)
        st.sidebar.success(f"✅ {len(df)} satır veri yüklendi")
        
        # Veri özeti
        with st.sidebar.expander("📊 Veri Özeti"):
            st.write(f"📅 Tarih: {df['DATE'].min().strftime('%Y-%m')} - {df['DATE'].max().strftime('%Y-%m')}")
            st.write(f"🏢 Territory: {df['TERRITORIES'].nunique()}")
            st.write(f"🗺️ Bölge: {df['REGION'].nunique()}")
            st.write(f"🏙️ Şehir: {df['CITY'].nunique()}")
            st.write(f"👤 Manager: {df['MANAGER'].nunique()}")
    except Exception as e:
        st.error(f"❌ Veri yükleme hatası: {str(e)}")
        st.stop()
    
    # Ürün seçimi
    st.sidebar.markdown("---")
    st.sidebar.header("💊 Ürün Seçimi")
    selected_product = st.sidebar.selectbox(
        "Ürün",
        ["TROCMETAM", "CORTIPOL", "DEKSAMETAZON", "PF IZOTONIK"],
        help="Analiz edilecek ürünü seçin"
    )
    
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
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Genel Bakış",
        "🏢 Territory Analizi", 
        "📈 Zaman Serisi",
        "⭐ BCG Matrix",
        "🎯 Yatırım Stratejisi",
        "📥 Raporlar"
    ])
    
    # ==========================================================================
    # TAB 1: GENEL BAKIŞ
    # ==========================================================================
    with tab1:
        st.header("📊 Genel Performans Özeti")
        
        # Temel metrikler
        cols = get_product_columns(selected_product)
        total_pf = df_filtered[cols['pf']].sum()
        total_rakip = df_filtered[cols['rakip']].sum()
        total_market = total_pf + total_rakip
        market_share = (total_pf / total_market * 100) if total_market > 0 else 0
        active_territories = df_filtered['TERRITORIES'].nunique()
        
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
        
        # YTD Karşılaştırma
        st.subheader("📅 Dönem Karşılaştırmaları")
        ytd_data = calculate_ytd_comparison(df_filtered, selected_product)
        
        col_ytd1, col_ytd2, col_ytd3, col_ytd4 = st.columns(4)
        
        with col_ytd1:
            st.metric("📆 YTD", f"{ytd_data['YTD']:,.0f}")
        
        with col_ytd2:
            st.metric("📆 Son 6 Ay", f"{ytd_data['Son_6_Ay']:,.0f}")
        
        with col_ytd3:
            st.metric("📆 Son 3 Ay", f"{ytd_data['Son_3_Ay']:,.0f}")
        
        with col_ytd4:
            change = ytd_data['Degisim_3M_%']
            st.metric(
                "📈 Değişim (3M)", 
                f"%{change:.1f}",
                delta=f"%{change:.1f}"
            )
        
        st.markdown("---")
        
        # Top 10 Territory
        st.subheader("🏆 Top 10 Territory")
        terr_perf = calculate_territory_performance(df_filtered, selected_product)
        top10 = terr_perf.head(10)
        
        # Bar chart
        fig_top10 = create_territory_bar_chart(terr_perf, top_n=10)
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
    # TAB 2: TERRITORY ANALİZİ
    # ==========================================================================
    with tab2:
        st.header("🏢 Territory Bazlı Detaylı Analiz")
        
        # Territory performansı
        terr_perf = calculate_territory_performance(df_filtered, selected_product)
        
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
            fig_bar = create_territory_bar_chart(terr_sorted, top_n=show_n)
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
        
        # Region bazlı özet
        st.markdown("---")
        st.subheader("🗺️ Region Bazlı Özet")
        
        region_summary = terr_sorted.groupby('Region').agg({
            'PF_Satis': 'sum',
            'Rakip_Satis': 'sum',
            'Toplam_Pazar': 'sum',
            'Territory': 'count'
        }).reset_index()
        
        region_summary['Pazar_Payi_%'] = (
            region_summary['PF_Satis'] / region_summary['Toplam_Pazar'] * 100
        )
        region_summary = region_summary.sort_values('PF_Satis', ascending=False)
        region_summary.columns = ['Region', 'PF Satış', 'Rakip Satış', 
                                  'Toplam Pazar', 'Territory Sayısı', 'Pazar Payı %']
        
        col_r1, col_r2 = st.columns(2)
        
        with col_r1:
            st.dataframe(
                region_summary.style.format({
                    'PF Satış': '{:,.0f}',
                    'Rakip Satış': '{:,.0f}',
                    'Toplam Pazar': '{:,.0f}',
                    'Pazar Payı %': '{:.1f}'
                }),
                use_container_width=True,
                hide_index=True
            )
        
        with col_r2:
            fig_region = px.bar(
                region_summary,
                x='Region',
                y='PF Satış',
                color='Pazar Payı %',
                color_continuous_scale='Blues',
                text='PF Satış'
            )
            fig_region.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
            fig_region.update_layout(height=400, xaxis=dict(tickangle=-45))
            st.plotly_chart(fig_region, use_container_width=True)
    
    # ==========================================================================
    # TAB 3: ZAMAN SERİSİ
    # ==========================================================================
    with tab3:
        st.header("📈 Zaman Serisi Analizi")
        
        # Territory seçimi
        territory_for_ts = st.selectbox(
            "Zaman serisi için Territory seçin",
            ["TÜMÜ"] + sorted(df_filtered['TERRITORIES'].unique()),
            key='ts_territory'
        )
        
        # Zaman serisi hesapla
        monthly_df = calculate_time_series(df_filtered, selected_product, territory_for_ts)
        
        if len(monthly_df) == 0:
            st.warning("⚠️ Seçilen filtrelerde veri bulunamadı")
        else:
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
                fig_ts = create_time_series_chart(monthly_df)
                st.plotly_chart(fig_ts, use_container_width=True)
            
            with col_chart2:
                st.markdown("#### 🎯 Pazar Payı Trendi")
                fig_share = create_market_share_chart(monthly_df)
                st.plotly_chart(fig_share, use_container_width=True)
            
            st.markdown("---")
            
            # Büyüme analizi
            st.subheader("📈 Büyüme Analizi")
            
            col_growth1, col_growth2 = st.columns(2)
            
            with col_growth1:
                fig_growth = create_growth_chart(monthly_df)
                st.plotly_chart(fig_growth, use_container_width=True)
            
            with col_growth2:
                st.markdown("#### 📊 Büyüme İstatistikleri")
                
                growth_stats = monthly_df[['PF_Buyume_%', 'Rakip_Buyume_%', 'Goreceli_Buyume_%']].describe()
                
                st.dataframe(
                    growth_stats.style.format("{:.2f}"),
                    use_container_width=True
                )
                
                # Son 3 ay analizi
                last_3_months = monthly_df.tail(3)
                st.markdown("##### 📅 Son 3 Ay Ortalamaları")
                
                avg_pf_3m = last_3_months['PF_Buyume_%'].mean()
                avg_rakip_3m = last_3_months['Rakip_Buyume_%'].mean()
                
                st.metric("PF Büyüme (3M Ort.)", f"%{avg_pf_3m:.1f}")
                st.metric("Rakip Büyüme (3M Ort.)", f"%{avg_rakip_3m:.1f}")
                st.metric("Göreceli Avantaj", f"%{(avg_pf_3m - avg_rakip_3m):.1f}")
            
            st.markdown("---")
            
            # Hareketli ortalamalar
            st.subheader("📉 Hareketli Ortalamalar")
            
            fig_ma = go.Figure()
            
            fig_ma.add_trace(go.Scatter(
                x=monthly_df['YIL_AY'],
                y=monthly_df['PF_Satis'],
                mode='lines',
                name='Gerçek',
                line=dict(color='#3B82F6', width=2)
            ))
            
            fig_ma.add_trace(go.Scatter(
                x=monthly_df['YIL_AY'],
                y=monthly_df['MA_3'],
                mode='lines',
                name='3 Aylık MA',
                line=dict(color='#10B981', width=3, dash='dash')
            ))
            
            fig_ma.add_trace(go.Scatter(
                x=monthly_df['YIL_AY'],
                y=monthly_df['MA_6'],
                mode='lines',
                name='6 Aylık MA',
                line=dict(color='#EF4444', width=3, dash='dot')
            ))
            
            fig_ma.update_layout(
                title='Hareketli Ortalamalar - Trend Analizi',
                xaxis_title='Ay',
                yaxis_title='PF Satış',
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_ma, use_container_width=True)
            
            st.markdown("---")
            
            # Detaylı veri tablosu
            st.subheader("📋 Detaylı Aylık Veri")
            
            monthly_display = monthly_df.copy()
            monthly_display.columns = [
                'Yıl-Ay', 'PF Satış', 'Rakip Satış', 'Toplam Pazar',
                'Pazar Payı %', 'PF Büyüme %', 'Rakip Büyüme %',
                'Göreceli Büyüme %', 'MA-3', 'MA-6'
            ]
            
            st.dataframe(
                monthly_display.style.format({
                    'PF Satış': '{:,.0f}',
                    'Rakip Satış': '{:,.0f}',
                    'Toplam Pazar': '{:,.0f}',
                    'Pazar Payı %': '{:.1f}',
                    'PF Büyüme %': '{:.1f}',
                    'Rakip Büyüme %': '{:.1f}',
                    'Göreceli Büyüme %': '{:.1f}',
                    'MA-3': '{:,.0f}',
                    'MA-6': '{:,.0f}'
                }).background_gradient(subset=['Göreceli Büyüme %'], cmap='RdYlGn'),
                use_container_width=True,
                height=400
            )
    
    # ==========================================================================
    # TAB 4: BCG MATRIX
    # ==========================================================================
    with tab4:
        st.header("⭐ BCG Matrix - Stratejik Portföy Konumlandırma")
        
        # BCG hesapla
        bcg_df = calculate_bcg_matrix(df_filtered, selected_product)
        
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
        st.subheader("🎯 BCG Matrix Scatter Plot")
        fig_bcg = create_bcg_scatter(bcg_df)
        st.plotly_chart(fig_bcg, use_container_width=True)
        
        st.markdown("---")
        
        # BCG Açıklamaları
        col_exp1, col_exp2 = st.columns(2)
        
        with col_exp1:
            st.info("""
            **⭐ STARS (Yıldızlar)**
            - Yüksek büyüme + Yüksek pazar payı
            - En değerli portfolio parçası
            - **Aksiyon:** Yatırımı artır, liderliği sürdür
            """)
            
            st.success("""
            **🐄 CASH COWS (Nakit İnekleri)**
            - Düşük büyüme + Yüksek pazar payı
            - Stabil gelir kaynağı
            - **Aksiyon:** Koru, maliyeti optimize et
            """)
        
        with col_exp2:
            st.warning("""
            **❓ QUESTION MARKS (Soru İşaretleri)**
            - Yüksek büyüme + Düşük pazar payı
            - En yüksek fırsat potansiyeli
            - **Aksiyon:** Agresif yatırım yap veya çık
            """)
            
            st.error("""
            **🐶 DOGS (Köpekler)**
            - Düşük büyüme + Düşük pazar payı
            - Düşük öncelik
            - **Aksiyon:** Minimal kaynak, çıkışı değerlendir
            """)
        
        st.markdown("---")
        
        # BCG Detay Tablosu
        st.subheader("📋 BCG Kategori Detayları")
        
        bcg_filter = st.multiselect(
            "BCG Kategorisi Filtrele",
            bcg_df['BCG_Kategori'].unique(),
            default=bcg_df['BCG_Kategori'].unique()
        )
        
        bcg_filtered = bcg_df[bcg_df['BCG_Kategori'].isin(bcg_filter)]
        
        display_cols_bcg = ['Territory', 'Region', 'BCG_Kategori', 'PF_Satis',
                           'Pazar_Payi_%', 'Goreceli_Pazar_Payi', 'Pazar_Buyume_%']
        
        bcg_display = bcg_filtered[display_cols_bcg].copy()
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
        
        st.markdown("---")
        
        # Sankey Diagram
        st.subheader("🌊 Sankey Akış Diyagramı")
        
        sankey_n = st.slider("Gösterilecek Territory Sayısı", 10, 30, 15)
        fig_sankey = create_sankey_diagram(df_filtered, selected_product, top_n=sankey_n)
        st.plotly_chart(fig_sankey, use_container_width=True)
    
    # ==========================================================================
    # TAB 5: YATIRIM STRATEJİSİ
    # ==========================================================================
    with tab5:
        st.header("🎯 Yatırım Stratejisi Önerileri")
        
        # BCG + Strateji hesapla
        bcg_df = calculate_bcg_matrix(df_filtered, selected_product)
        strategy_df = calculate_investment_strategy(bcg_df)
        
        # Strateji dağılımı
        st.subheader("📊 Strateji Dağılımı")
        
        strategy_counts = strategy_df['Yatirim_Stratejisi'].value_counts()
        
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        
        strategies = [
            ('🚀 Agresif Hızlandırma', '#EF4444'),
            ('🛡️ Koruma', '#10B981'),
            ('👁️ İzlem', '#6B7280'),
            ('🚪 Çıkış/Minimize', '#9CA3AF')
        ]
        
        for idx, (strategy, color) in enumerate(strategies):
            with [col_s1, col_s2, col_s3, col_s4][idx]:
                count = strategy_counts.get(strategy, 0)
                pf_sum = strategy_df[strategy_df['Yatirim_Stratejisi'] == strategy]['PF_Satis'].sum()
                st.markdown(f"""
                <div style="background: {color}; padding: 1rem; border-radius: 0.5rem; color: white; text-align: center;">
                    <h3>{strategy.split()[0]}</h3>
                    <h2>{count}</h2>
                    <p>{pf_sum:,.0f} PF</p>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Strateji bazlı analiz
        st.subheader("📈 Strateji Bazlı Performans")
        
        strategy_summary = strategy_df.groupby('Yatirim_Stratejisi').agg({
            'PF_Satis': 'sum',
            'Toplam_Pazar': 'sum',
            'Pazar_Payi_%': 'mean',
            'Pazar_Buyume_%': 'mean',
            'Territory': 'count'
        }).reset_index()
        
        strategy_summary.columns = ['Strateji', 'Toplam PF', 'Toplam Pazar',
                                    'Ort. Pazar Payı %', 'Ort. Büyüme %', 'Territory Sayısı']
        
        col_strat1, col_strat2 = st.columns(2)
        
        with col_strat1:
            st.dataframe(
                strategy_summary.style.format({
                    'Toplam PF': '{:,.0f}',
                    'Toplam Pazar': '{:,.0f}',
                    'Ort. Pazar Payı %': '{:.1f}',
                    'Ort. Büyüme %': '{:.1f}'
                }),
                use_container_width=True,
                hide_index=True
            )
        
        with col_strat2:
            fig_strat_pie = px.pie(
                strategy_summary,
                values='Toplam PF',
                names='Strateji',
                color='Strateji',
                color_discrete_map={
                    '🚀 Agresif Hızlandırma': '#EF4444',
                    '🛡️ Koruma': '#10B981',
                    '👁️ İzlem': '#6B7280',
                    '🚪 Çıkış/Minimize': '#9CA3AF'
                },
                title='PF Satış Dağılımı (Stratejiye Göre)'
            )
            fig_strat_pie.update_layout(height=400)
            st.plotly_chart(fig_strat_pie, use_container_width=True)
        
        st.markdown("---")
        
        # Öncelikli aksiyonlar
        st.subheader("🚀 Öncelikli Aksiyonlar")
        
        # Agresif hızlandırma kategorisindekiler
        aggressive = strategy_df[strategy_df['Yatirim_Stratejisi'] == '🚀 Agresif Hızlandırma'].nlargest(5, 'PF_Satis')
        
        if len(aggressive) > 0:
            st.markdown("#### 1️⃣ Agresif Yatırım Yapılacak Territory'ler")
            
            for idx, row in aggressive.iterrows():
                with st.expander(f"🎯 {row['Territory']} - {row['Region']}"):
                    col_a1, col_a2, col_a3 = st.columns(3)
                    
                    with col_a1:
                        st.metric("PF Satış", f"{row['PF_Satis']:,.0f}")
                        st.metric("Pazar Payı", f"%{row['Pazar_Payi_%']:.1f}")
                    
                    with col_a2:
                        st.metric("Büyüme Oranı", f"%{row['Pazar_Buyume_%']:.1f}")
                        st.metric("BCG", row['BCG_Kategori'])
                    
                    with col_a3:
                        potential = row['Toplam_Pazar'] - row['PF_Satis']
                        st.metric("Büyüme Potansiyeli", f"{potential:,.0f}")
                        st.metric("Manager", row['Manager'])
                    
                    st.info(f"**💡 Önerilen Aksiyon:** {row['Aksiyon']}")
        
        # Koruma kategorisindekiler
        protect = strategy_df[strategy_df['Yatirim_Stratejisi'] == '🛡️ Koruma'].nlargest(5, 'PF_Satis')
        
        if len(protect) > 0:
            st.markdown("#### 2️⃣ Koruma Stratejisi Uygulanacak Territory'ler")
            
            for idx, row in protect.iterrows():
                with st.expander(f"🛡️ {row['Territory']} - {row['Region']}"):
                    col_p1, col_p2 = st.columns(2)
                    
                    with col_p1:
                        st.metric("PF Satış", f"{row['PF_Satis']:,.0f}")
                        st.metric("Pazar Payı", f"%{row['Pazar_Payi_%']:.1f}")
                    
                    with col_p2:
                        st.metric("BCG", row['BCG_Kategori'])
                        st.metric("Manager", row['Manager'])
                    
                    st.success(f"**💡 Önerilen Aksiyon:** {row['Aksiyon']}")
        
        st.markdown("---")
        
        # Detaylı strateji tablosu
        st.subheader("📋 Tüm Territory'ler - Strateji Detayı")
        
        strategy_filter = st.multiselect(
            "Strateji Filtrele",
            strategy_df['Yatirim_Stratejisi'].unique(),
            default=strategy_df['Yatirim_Stratejisi'].unique()
        )
        
        strategy_filtered = strategy_df[strategy_df['Yatirim_Stratejisi'].isin(strategy_filter)]
        
        display_cols_strategy = ['Territory', 'Region', 'Manager', 'BCG_Kategori',
                                'Yatirim_Stratejisi', 'PF_Satis', 'Pazar_Payi_%',
                                'Pazar_Buyume_%', 'Aksiyon']
        
        strategy_display = strategy_filtered[display_cols_strategy].copy()
        strategy_display.columns = ['Territory', 'Region', 'Manager', 'BCG',
                                    'Strateji', 'PF Satış', 'Pazar Payı %',
                                    'Büyüme %', 'Aksiyon']
        strategy_display = strategy_display.sort_values('PF Satış', ascending=False)
        strategy_display.index = range(1, len(strategy_display) + 1)
        
        st.dataframe(
            strategy_display.style.format({
                'PF Satış': '{:,.0f}',
                'Pazar Payı %': '{:.1f}',
                'Büyüme %': '{:.1f}'
            }),
            use_container_width=True,
            height=500
        )
    
    # ==========================================================================
    # TAB 6: RAPORLAR
    # ==========================================================================
    with tab6:
        st.header("📥 Rapor İndirme")
        
        st.markdown("""
        Bu bölümden analizlerin Excel raporlarını indirebilirsiniz.
        Raporlar aşağıdaki sayfaları içerir:
        - Territory Performans Analizi
        - Zaman Serisi Verileri
        - BCG Matrix Kategorileri
        - Yatırım Stratejisi Önerileri
        """)
        
        col_r1, col_r2 = st.columns(2)
        
        with col_r1:
            st.subheader("📊 Kapsamlı Analiz Raporu")
            
            if st.button("📥 Excel Raporu Oluştur", type="primary"):
                with st.spinner("Rapor hazırlanıyor..."):
                    # Verileri hazırla
                    terr_perf = calculate_territory_performance(df_filtered, selected_product)
                    monthly_df = calculate_time_series(df_filtered, selected_product)
                    bcg_df = calculate_bcg_matrix(df_filtered, selected_product)
                    strategy_df = calculate_investment_strategy(bcg_df)
                    ytd_data = calculate_ytd_comparison(df_filtered, selected_product)
                    
                    # Excel oluştur
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        # Sayfa 1: Territory Performans
                        terr_perf.to_excel(writer, sheet_name='Territory Performans', index=False)
                        
                        # Sayfa 2: Zaman Serisi
                        monthly_df.to_excel(writer, sheet_name='Zaman Serisi', index=False)
                        
                        # Sayfa 3: BCG Matrix
                        bcg_df.to_excel(writer, sheet_name='BCG Matrix', index=False)
                        
                        # Sayfa 4: Yatırım Stratejisi
                        strategy_df.to_excel(writer, sheet_name='Yatırım Stratejisi', index=False)
                        
                        # Sayfa 5: YTD Özet
                        ytd_df = pd.DataFrame([ytd_data])
                        ytd_df.to_excel(writer, sheet_name='YTD Özet', index=False)
                    
                    st.success("✅ Rapor hazır!")
                    
                    st.download_button(
                        label="💾 Excel Raporunu İndir",
                        data=output.getvalue(),
                        file_name=f"ticari_portfoy_raporu_{selected_product}_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
        
        with col_r2:
            st.subheader("📄 Özet Rapor")
            st.info("Yönetici sunumu için özet rapor (yakında)")
            
            # Gelecekte PDF veya PowerPoint export eklenebilir

if __name__ == "__main__":
    main()

