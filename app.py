import streamlit as st
import geopandas as gpd
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json
from shapely.geometry import LineString, MultiLineString
from datetime import datetime
import warnings
import numpy as np

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Ticari Ürün Analizi", layout="wide")
st.title("💊 Ticari Ürün Satış Analizi - Detaylı Türkiye Haritası")

# =============================================================================
# BÖLGE RENKLERİ (COĞRAFİ & MODERN)
# =============================================================================
REGION_COLORS = {
    "MARMARA": "#0EA5E9",              # Sky Blue - Deniz ve boğazlar
    "BATI ANADOLU": "#14B8A6",         # Turkuaz-yeşil arası
    "EGE": "#FCD34D",                  # BAL SARI (Batı Anadolu ile aynı)
    "İÇ ANADOLU": "#F59E0B",           # Amber - Kuru bozkır
    "GÜNEY DOĞU ANADOLU": "#E07A5F",   # Terracotta 
    "KUZEY ANADOLU": "#059669",        # Emerald - Yemyeşil ormanlar
    "KARADENİZ": "#059669",            # Emerald (Kuzey Anadolu ile aynı)
    "AKDENİZ": "#8B5CF6",              # Violet - Akdeniz
    "DOĞU ANADOLU": "#7C3AED",         # Purple - Yüksek dağlar
    "DİĞER": "#64748B"                 # Slate Gray
}

# =============================================================================
# ŞEHİR EŞLEŞTİRME (MASTER)
# =============================================================================
FIX_CITY_MAP = {
    "AGRI": "AĞRI",
    "BARTÄ±N": "BARTIN",
    "BINGÃ¶L": "BİNGÖL",
    "DÃ¼ZCE": "DÜZCE",
    "ELAZIG": "ELAZIĞ",
    "ESKISEHIR": "ESKİŞEHİR",
    "GÃ¼MÃ¼SHANE": "GÜMÜŞHANE",
    "HAKKARI": "HAKKARİ",
    "ISTANBUL": "İSTANBUL",
    "IZMIR": "İZMİR",
    "IÄ\x9fDIR": "IĞDIR",
    "KARABÃ¼K": "KARABÜK",
    "KINKKALE": "KIRIKKALE",
    "KIRSEHIR": "KIRŞEHİR",
    "KÃ¼TAHYA": "KÜTAHYA",
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
    "K. MARAS": "KAHRAMANMARAŞ"
}

# =============================================================================
# NORMALIZATION
# =============================================================================
def normalize_city(name):
    if pd.isna(name):
        return None

    name = str(name).upper().strip()

    tr_map = {
        "İ": "I", "Ğ": "G", "Ü": "U",
        "Ş": "S", "Ö": "O",
        "Ç": "C", "Â": "A"
    }

    for k, v in tr_map.items():
        name = name.replace(k, v)

    return name

@st.cache_data
def load_excel(file):
    df = pd.read_excel(file)
    df['DATE'] = pd.to_datetime(df['DATE'])
    return df

@st.cache_data
def load_geo_from_file(file):
    gdf = gpd.read_file(file)
    gdf["raw_name"] = gdf["name"].str.upper()
    gdf["fixed_name"] = gdf["raw_name"].replace(FIX_CITY_MAP)
    gdf["CITY_KEY"] = gdf["fixed_name"].apply(normalize_city)
    return gdf

def prepare_product_data(df, gdf, product, start_date, end_date):
    df_filtered = df[(df['DATE'] >= start_date) & (df['DATE'] <= end_date)].copy()
    
    if product == "TROCMETAM":
        pf_col, other_col = "TROCMETAM", "DIGER TROCMETAM"
    elif product == "CORTIPOL":
        pf_col, other_col = "CORTIPOL", "DIGER CORTIPOL"
    elif product == "DEKSAMETAZON":
        pf_col, other_col = "DEKSAMETAZON", "DIGER DEKSAMETAZON"
    else:
        pf_col, other_col = "PF IZOTONIK", "DIGER IZOTONIK"
    
    city_df = df_filtered.groupby(['CITY', 'REGION', 'MANAGER']).agg({
        pf_col: 'sum', other_col: 'sum'
    }).reset_index()
    
    city_df.columns = ['Şehir', 'Bölge', 'Müdür', 'PF Satış', 'Rakip Satış']
    city_df['Toplam Pazar'] = city_df['PF Satış'] + city_df['Rakip Satış']
    city_df['Pazar Payı %'] = (city_df['PF Satış'] / city_df['Toplam Pazar'] * 100).round(2).fillna(0)
    
    city_df["Şehir_fix"] = city_df["Şehir"].str.upper().replace(FIX_CITY_MAP)
    city_df["CITY_KEY"] = city_df["Şehir_fix"].apply(normalize_city)
    city_df["Bölge"] = city_df["Bölge"].str.upper()
    city_df["Müdür"] = city_df["Müdür"].str.upper()
    
    merged = gdf.merge(city_df, on="CITY_KEY", how="left")
    merged["Şehir"] = merged["fixed_name"]
    merged[["PF Satış", "Rakip Satış", "Toplam Pazar", "Pazar Payı %"]] = merged[["PF Satış", "Rakip Satış", "Toplam Pazar", "Pazar Payı %"]].fillna(0)
    merged["Bölge"] = merged["Bölge"].fillna("DİĞER")
    merged["Müdür"] = merged["Müdür"].fillna("YOK")
    
    bolge_df = merged.groupby("Bölge", as_index=False).agg({
        "PF Satış": "sum", "Toplam Pazar": "sum"
    }).sort_values("PF Satış", ascending=False)
    bolge_df["Pazar Payı %"] = (bolge_df["PF Satış"] / bolge_df["Toplam Pazar"] * 100).round(2).fillna(0)
    
    return merged, bolge_df, city_df

def get_time_series(df, product, region=None):
    if product == "TROCMETAM":
        pf_col, other_col = "TROCMETAM", "DIGER TROCMETAM"
    elif product == "CORTIPOL":
        pf_col, other_col = "CORTIPOL", "DIGER CORTIPOL"
    elif product == "DEKSAMETAZON":
        pf_col, other_col = "DEKSAMETAZON", "DIGER DEKSAMETAZON"
    else:
        pf_col, other_col = "PF IZOTONIK", "DIGER IZOTONIK"
    
    df_filtered = df.copy()
    if region:
        df_filtered = df_filtered[df_filtered['REGION'] == region]
    
    monthly = df_filtered.groupby('DATE').agg({pf_col: 'sum', other_col: 'sum'}).reset_index()
    monthly.columns = ['Tarih', 'PF Satış', 'Rakip Satış']
    monthly['Toplam Pazar'] = monthly['PF Satış'] + monthly['Rakip Satış']
    monthly['Pazar Payı %'] = (monthly['PF Satış'] / monthly['Toplam Pazar'] * 100).round(2)
    return monthly

def lines_to_lonlat(geom):
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
    return gdf_region.geometry.unary_union.centroid.x, gdf_region.geometry.unary_union.centroid.y

def create_detailed_figure(gdf, filtered_pf):
    """DETAYLI HARİTA - TÜM ŞEHİRLER VE BÖLGELER GÖRÜLEBİLİR"""
    fig = go.Figure()
    
    # Her bölge ayrı renk
    for region in gdf["Bölge"].unique():
        region_gdf = gdf[gdf["Bölge"] == region]
        color = REGION_COLORS.get(region, "#78909C")
        
        fig.add_choropleth(
            geojson=json.loads(region_gdf.to_json()),
            locations=region_gdf.index,
            z=[1]*len(region_gdf),
            colorscale=[[0,color],[1,color]],
            marker_line_color="white",
            marker_line_width=1,
            showscale=False,
            customdata=list(zip(region_gdf["Şehir"], region_gdf["Bölge"], region_gdf["PF Satış"], region_gdf["Pazar Payı %"])),
            hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<br>PF: %{customdata[2]:,.0f}<br>Pay: %{customdata[3]:.1f}%<extra></extra>",
            name=region
        )
    
    # Sınırlar
    lons, lats = [], []
    for geom in gdf.geometry.boundary:
        lo, la = lines_to_lonlat(geom)
        lons += lo; lats += la
    fig.add_scattergeo(lon=lons, lat=lats, mode="lines", line=dict(color="white", width=0.5), hoverinfo="skip", showlegend=False)
    
    # BÖLGE ETİKETLERİ - Resimde olduğu gibi
    label_lons, label_lats, label_texts = [], [], []
    for region in gdf["Bölge"].unique():
        region_gdf = gdf[gdf["Bölge"] == region]
        total = region_gdf["PF Satış"].sum()
        percent = (total / filtered_pf * 100) if filtered_pf > 0 else 0
        lon, lat = get_region_center(region_gdf)
        label_lons.append(lon); label_lats.append(lat)
        label_texts.append(f"<b>{region}</b><br>{total:,.0f} ({percent:.1f}%)")
    
    fig.add_scattergeo(
        lon=label_lons, lat=label_lats, mode="text", text=label_texts,
        textfont=dict(size=9, color="black", family="Arial Black"),
        hoverinfo="skip", showlegend=False
    )
    
    fig.update_layout(
        geo=dict(
            projection=dict(type="mercator"),
            center=dict(lat=39, lon=35),
            lonaxis=dict(range=[25, 45]),
            lataxis=dict(range=[35, 43]),
            visible=False,
            bgcolor="rgba(250,250,250,1)"
        ),
        height=700,
        margin=dict(l=0, r=0, t=20, b=0),
        paper_bgcolor="white"
    )
    return fig

# SIDEBAR
st.sidebar.header("📂 Dosya Yükleme")
uploaded_excel = st.sidebar.file_uploader("📊 Excel", type=['xlsx'])
uploaded_geojson = st.sidebar.file_uploader("🗺️ GeoJSON", type=['geojson'])

if not uploaded_excel or not uploaded_geojson:
    st.warning("⚠️ Lütfen Excel ve GeoJSON dosyalarını yükleyin!")
    st.stop()

raw_df = load_excel(uploaded_excel)
geo = load_geo_from_file(uploaded_geojson)
st.sidebar.success("✅ Dosyalar yüklendi!")

st.sidebar.header("📊 Ürün Seçimi")
selected_product = st.sidebar.selectbox("💊 Ürün", ["TROCMETAM", "CORTIPOL", "DEKSAMETAZON", "PF IZOTONIK"])

# TARİH FİLTRELEME
st.sidebar.header("📅 Tarih Seçimi")
min_date, max_date = raw_df['DATE'].min(), raw_df['DATE'].max()
date_mode = st.sidebar.radio("Mod", ["Son 3 Ay", "Son 6 Ay", "Tüm Veriler", "Özel"], index=0)

if date_mode == "Son 3 Ay":
    end_date = max_date
    start_date = end_date - pd.DateOffset(months=3)
elif date_mode == "Son 6 Ay":
    end_date = max_date
    start_date = end_date - pd.DateOffset(months=6)
elif date_mode == "Tüm Veriler":
    start_date, end_date = min_date, max_date
else:
    col_d1, col_d2 = st.sidebar.columns(2)
    with col_d1:
        start_date = pd.to_datetime(st.date_input("Başlangıç", min_date, min_value=min_date, max_value=max_date))
    with col_d2:
        end_date = pd.to_datetime(st.date_input("Bitiş", max_date, min_value=min_date, max_value=max_date))

merged, bolge_df, city_df = prepare_product_data(raw_df, geo, selected_product, start_date, end_date)

st.sidebar.header("🔍 Filtreler")
selected_mudur = st.sidebar.selectbox("Müdür", ["TÜMÜ"] + sorted(merged["Müdür"].unique()))
selected_bolge = st.sidebar.selectbox("Bölge", ["TÜMÜ"] + sorted([b for b in merged["Bölge"].unique() if b != "DİĞER"]))

if selected_mudur != "TÜMÜ":
    merged = merged[merged["Müdür"] == selected_mudur]
if selected_bolge != "TÜMÜ":
    merged = merged[merged["Bölge"] == selected_bolge]

filtered_pf = merged["PF Satış"].sum()
filtered_market = merged["Toplam Pazar"].sum()

# HARİTA
st.markdown(f"### 🗺️ {selected_product} - Türkiye Dağılımı")
st.caption(f"📆 {start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')}")
fig = create_detailed_figure(merged, filtered_pf)
st.plotly_chart(fig, use_container_width=True)

# METRİKLER
col1, col2, col3, col4 = st.columns(4)
col1.metric("💊 PF Satış", f"{filtered_pf:,.0f}")
col2.metric("🏪 Pazar", f"{filtered_market:,.0f}")
col3.metric("📊 Pay %", f"%{(filtered_pf/filtered_market*100 if filtered_market>0 else 0):.1f}")
col4.metric("🏙️ Şehir", f"{(merged['PF Satış']>0).sum()}")

st.markdown("---")

# ZAMAN SERİSİ ANALİZLERİ
st.subheader("📈 Zaman Serisi Analizleri")

monthly_ts = get_time_series(raw_df, selected_product, selected_bolge if selected_bolge != "TÜMÜ" else None)

# 1. AYLIK TREND
col_ts1, col_ts2 = st.columns(2)
with col_ts1:
    st.markdown("#### 📅 Aylık Satış Trendi")
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(x=monthly_ts['Tarih'], y=monthly_ts['PF Satış'], name='PF', line=dict(color='#3B82F6', width=3), marker=dict(size=8)))
    fig_ts.add_trace(go.Scatter(x=monthly_ts['Tarih'], y=monthly_ts['Rakip Satış'], name='Rakip', line=dict(color='#EF4444', width=3), marker=dict(size=8)))
    fig_ts.update_layout(height=400, hovermode='x unified')
    st.plotly_chart(fig_ts, use_container_width=True)

with col_ts2:
    st.markdown("#### 📊 Pazar Payı Trendi")
    fig_share = go.Figure()
    fig_share.add_trace(go.Scatter(x=monthly_ts['Tarih'], y=monthly_ts['Pazar Payı %'], fill='tozeroy', line=dict(color='#10B981', width=2)))
    fig_share.update_layout(height=400, yaxis=dict(title='%'))
    st.plotly_chart(fig_share, use_container_width=True)

# 2. BÜYÜME ANALİZİ
st.markdown("#### 📊 Aylık Büyüme Analizi")
monthly_growth = monthly_ts.copy()
monthly_growth['Büyüme %'] = monthly_growth['PF Satış'].pct_change() * 100
monthly_growth['Rakip Büyüme %'] = monthly_growth['Rakip Satış'].pct_change() * 100

col_g1, col_g2 = st.columns(2)
with col_g1:
    fig_growth = go.Figure()
    fig_growth.add_trace(go.Bar(
        x=monthly_growth['Tarih'], y=monthly_growth['Büyüme %'],
        name='PF Büyüme',
        marker_color=['#10B981' if x > 0 else '#EF4444' for x in monthly_growth['Büyüme %']]
    ))
    fig_growth.update_layout(height=350, yaxis=dict(title='Büyüme %'))
    st.plotly_chart(fig_growth, use_container_width=True)

with col_g2:
    st.markdown("##### 📈 Ortalama Büyüme")
    avg_3 = monthly_growth.tail(3)['Büyüme %'].mean()
    avg_6 = monthly_growth.tail(6)['Büyüme %'].mean()
    avg_all = monthly_growth['Büyüme %'].mean()
    st.metric("Son 3 Ay", f"{avg_3:.1f}%")
    st.metric("Son 6 Ay", f"{avg_6:.1f}%")
    st.metric("Tüm Dönem", f"{avg_all:.1f}%")

# 3. DÖNEM KARŞILAŞTIRMASI
st.markdown("#### 🔄 Dönem Karşılaştırmaları")
col_c1, col_c2, col_c3 = st.columns(3)

# Son 3 vs önceki 3
latest_3 = raw_df[raw_df['DATE'] >= (max_date - pd.DateOffset(months=3))]
prev_3 = raw_df[(raw_df['DATE'] >= (max_date - pd.DateOffset(months=6))) & (raw_df['DATE'] < (max_date - pd.DateOffset(months=3)))]

if selected_product == "TROCMETAM":
    pf_col = "TROCMETAM"
elif selected_product == "CORTIPOL":
    pf_col = "CORTIPOL"
elif selected_product == "DEKSAMETAZON":
    pf_col = "DEKSAMETAZON"
else:
    pf_col = "PF IZOTONIK"

latest_3_total = latest_3[pf_col].sum()
prev_3_total = prev_3[pf_col].sum()
growth_3 = ((latest_3_total - prev_3_total) / prev_3_total * 100) if prev_3_total > 0 else 0

with col_c1:
    st.metric("📅 Son 3 Ay", f"{latest_3_total:,.0f}")
with col_c2:
    st.metric("📅 Önceki 3 Ay", f"{prev_3_total:,.0f}")
with col_c3:
    st.metric("📈 Değişim", f"{growth_3:+.1f}%", delta=f"{growth_3:+.1f}%")

# 4. YILI AYLIK KARŞILAŞTIRMA
st.markdown("#### 📊 Yıl İçi Aylık Performans")
yearly_comparison = raw_df.copy()
yearly_comparison['Ay'] = yearly_comparison['DATE'].dt.month
yearly_comparison['Ay Adı'] = yearly_comparison['DATE'].dt.strftime('%B')
monthly_perf = yearly_comparison.groupby('Ay Adı')[pf_col].sum().reset_index()

fig_yearly = px.bar(monthly_perf, x='Ay Adı', y=pf_col, color=pf_col, color_continuous_scale='Blues')
fig_yearly.update_layout(height=350, xaxis=dict(tickangle=-45))
st.plotly_chart(fig_yearly, use_container_width=True)

# 5. HAREKETLI ORTALAMALAR
st.markdown("#### 📈 Hareketli Ortalamalar (3 Ay)")
monthly_ts['MA_3'] = monthly_ts['PF Satış'].rolling(window=3).mean()
monthly_ts['MA_6'] = monthly_ts['PF Satış'].rolling(window=6).mean()

fig_ma = go.Figure()
fig_ma.add_trace(go.Scatter(x=monthly_ts['Tarih'], y=monthly_ts['PF Satış'], name='Gerçek', line=dict(color='#3B82F6')))
fig_ma.add_trace(go.Scatter(x=monthly_ts['Tarih'], y=monthly_ts['MA_3'], name='3 Ay MA', line=dict(color='#10B981', dash='dash')))
fig_ma.add_trace(go.Scatter(x=monthly_ts['Tarih'], y=monthly_ts['MA_6'], name='6 Ay MA', line=dict(color='#EF4444', dash='dot')))
fig_ma.update_layout(height=400)
st.plotly_chart(fig_ma, use_container_width=True)

# 6. YTD (YEAR-TO-DATE) ANALİZ
st.markdown("#### 📊 Year-to-Date (YTD) Performans")
ytd_data = raw_df[raw_df['DATE'].dt.year == max_date.year]
ytd_monthly = ytd_data.groupby(ytd_data['DATE'].dt.month)[pf_col].sum().reset_index()
ytd_monthly['Kümülatif'] = ytd_monthly[pf_col].cumsum()

fig_ytd = go.Figure()
fig_ytd.add_trace(go.Bar(x=ytd_monthly['DATE'], y=ytd_monthly[pf_col], name='Aylık'))
fig_ytd.add_trace(go.Scatter(x=ytd_monthly['DATE'], y=ytd_monthly['Kümülatif'], name='Kümülatif', yaxis='y2', line=dict(color='#EF4444', width=3)))
fig_ytd.update_layout(
    height=400,
    yaxis2=dict(title='Kümülatif', overlaying='y', side='right')
)
st.plotly_chart(fig_ytd, use_container_width=True)

st.markdown("---")

# TABLOLAR
st.subheader("📊 Detay Tablolar")
col_t1, col_t2 = st.columns(2)

with col_t1:
    st.markdown("##### 🗺️ Bölge Performans")
    st.dataframe(bolge_df[bolge_df["PF Satış"] > 0], use_container_width=True, hide_index=True)

with col_t2:
    st.markdown("##### 🏙️ Top 20 Şehir")
    top20 = city_df.nlargest(20, "PF Satış")[["Şehir", "Bölge", "PF Satış", "Pazar Payı %", "Müdür"]]
    st.dataframe(top20, use_container_width=True, hide_index=True)

# EXPORT
from io import BytesIO
st.markdown("---")
st.subheader("📥 Rapor İndir")
output = BytesIO()
with pd.ExcelWriter(output, engine='openpyxl') as writer:
    city_df.to_excel(writer, sheet_name='Şehir', index=False)
    bolge_df.to_excel(writer, sheet_name='Bölge', index=False)
    monthly_ts.to_excel(writer, sheet_name='Aylık Trend', index=False)
    monthly_growth.to_excel(writer, sheet_name='Büyüme', index=False)

st.download_button(
    "📥 Detaylı Excel Raporu",
    output.getvalue(),
    f"{selected_product}_{datetime.now().strftime('%Y%m%d')}.xlsx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)


