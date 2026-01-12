import streamlit as st
import geopandas as gpd
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json
from shapely.geometry import LineString, MultiLineString
from datetime import datetime
import warnings
import os

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Ticari Ürün Analizi", layout="wide")
st.title("💊 Ticari Ürün Satış Analizi - Türkiye Haritası")

REGION_COLORS = {
    "MARMARA": "#0EA5E9", "BATI ANADOLU": "#14B8A6", "EGE": "#FCD34D",
    "İÇ ANADOLU": "#F59E0B", "GÜNEY DOĞU ANADOLU": "#E07A5F",
    "KUZEY ANADOLU": "#059669", "KARADENİZ": "#059669",
    "AKDENİZ": "#8B5CF6", "DOĞU ANADOLU": "#7C3AED", "DİĞER": "#64748B"
}

FIX_CITY_MAP = {
    "AGRI": "AĞRI", "BARTÄ±N": "BARTIN", "BINGÃ¶L": "BİNGÖL",
    "DÃ¼ZCE": "DÜZCE", "ELAZIG": "ELAZIĞ", "ESKISEHIR": "ESKİŞEHİR",
    "ISTANBUL": "İSTANBUL", "IZMIR": "İZMİR", "K. MARAS": "KAHRAMANMARAŞ"
}

def normalize_city(name):
    if pd.isna(name): return None
    name = str(name).upper().strip()
    for k, v in {"İ": "I", "Ğ": "G", "Ü": "U", "Ş": "S", "Ö": "O", "Ç": "C"}.items():
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

def create_figure(gdf, view_mode, filtered_pf):
    fig = go.Figure()
    for region in gdf["Bölge"].unique():
        region_gdf = gdf[gdf["Bölge"] == region]
        color = REGION_COLORS.get(region, "#CCCCCC")
        fig.add_choropleth(geojson=json.loads(region_gdf.to_json()), locations=region_gdf.index, z=[1]*len(region_gdf), colorscale=[[0,color],[1,color]], marker_line_color="white", marker_line_width=1.5, showscale=False, customdata=list(zip(region_gdf["Şehir"], region_gdf["Bölge"], region_gdf["PF Satış"], region_gdf["Pazar Payı %"])), hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<br>%{customdata[2]:,.0f}<br>%{customdata[3]:.1f}%<extra></extra>", name=region)
    
    lons, lats = [], []
    for geom in gdf.geometry.boundary:
        lo, la = lines_to_lonlat(geom)
        lons += lo; lats += la
    fig.add_scattergeo(lon=lons, lat=lats, mode="lines", line=dict(color="rgba(255,255,255,0.8)", width=1), hoverinfo="skip", showlegend=False)
    
    if view_mode == "Bölge Görünümü":
        label_lons, label_lats, label_texts = [], [], []
        for region in gdf["Bölge"].unique():
            region_gdf = gdf[gdf["Bölge"] == region]
            total = region_gdf["PF Satış"].sum()
            if total > 0:
                percent = (total / filtered_pf * 100) if filtered_pf > 0 else 0
                lon, lat = get_region_center(region_gdf)
                label_lons.append(lon); label_lats.append(lat)
                label_texts.append(f"<b>{region}</b><br>{total:,.0f} ({percent:.1f}%)")
        fig.add_scattergeo(lon=label_lons, lat=label_lats, mode="text", text=label_texts, textfont=dict(size=10, color="black"), hoverinfo="skip", showlegend=False)
    
    fig.update_layout(geo=dict(projection=dict(type="mercator"), center=dict(lat=39, lon=35), lonaxis=dict(range=[25, 45]), lataxis=dict(range=[35, 43]), visible=False), height=750, margin=dict(l=0, r=0, t=40, b=0))
    return fig

# SIDEBAR - DOSYA YÜKLEME
st.sidebar.header("📂 Dosya Yükleme")
uploaded_excel = st.sidebar.file_uploader("📊 Excel Dosyası", type=['xlsx'], help="Ticari Ürün verileri")
uploaded_geojson = st.sidebar.file_uploader("🗺️ GeoJSON", type=['geojson'], help="turkey.geojson")

if not uploaded_excel or not uploaded_geojson:
    st.warning("⚠️ Lütfen Excel ve GeoJSON dosyalarını yükleyin!")
    st.stop()

raw_df = load_excel(uploaded_excel)
geo = load_geo_from_file(uploaded_geojson)
st.sidebar.success("✅ Dosyalar yüklendi!")

st.sidebar.markdown("---")
st.sidebar.header("📊 Ürün & Tarih")
selected_product = st.sidebar.selectbox("💊 Ürün", ["TROCMETAM", "CORTIPOL", "DEKSAMETAZON", "PF IZOTONIK"])

min_date, max_date = raw_df['DATE'].min(), raw_df['DATE'].max()
date_mode = st.sidebar.radio("Tarih Modu", ["Son 3 Ay", "Tüm Veriler", "Özel Aralık"], index=0)

if date_mode == "Son 3 Ay":
    end_date = max_date
    start_date = end_date - pd.DateOffset(months=3)
elif date_mode == "Tüm Veriler":
    start_date, end_date = min_date, max_date
else:
    col_d1, col_d2 = st.sidebar.columns(2)
    with col_d1:
        start_date = pd.to_datetime(st.date_input("Başlangıç", min_date, min_value=min_date, max_value=max_date))
    with col_d2:
        end_date = pd.to_datetime(st.date_input("Bitiş", max_date, min_value=min_date, max_value=max_date))

merged, bolge_df, city_df = prepare_product_data(raw_df, geo, selected_product, start_date, end_date)

st.sidebar.markdown("---")
st.sidebar.header("🔍 Filtreler")
view_mode = st.sidebar.radio("Görünüm", ["Bölge Görünümü", "Şehir Görünümü"])
selected_mudur = st.sidebar.selectbox("Müdür", ["TÜMÜ"] + sorted(merged["Müdür"].unique()))
selected_bolge = st.sidebar.selectbox("Bölge", ["TÜMÜ"] + sorted([b for b in merged["Bölge"].unique() if b != "DİĞER"]))

if selected_mudur != "TÜMÜ":
    merged = merged[merged["Müdür"] == selected_mudur]
if selected_bolge != "TÜMÜ":
    merged = merged[merged["Bölge"] == selected_bolge]

filtered_pf = merged["PF Satış"].sum()
filtered_market = merged["Toplam Pazar"].sum()

# HARİTA
st.markdown(f"### 🗺️ {selected_product} - Türkiye")
fig = create_figure(merged, view_mode, filtered_pf)
st.plotly_chart(fig, use_container_width=True)

# METRİKLER
col1, col2, col3, col4 = st.columns(4)
col1.metric("💊 PF Satış", f"{filtered_pf:,.0f}")
col2.metric("🏪 Pazar", f"{filtered_market:,.0f}")
col3.metric("📊 Pay %", f"%{(filtered_pf/filtered_market*100 if filtered_market>0 else 0):.1f}")
col4.metric("🏙️ Şehir", f"{(merged['PF Satış']>0).sum()}")

st.markdown("---")

# ZAMAN SERİSİ
st.subheader("📈 Zaman Serisi")
col_ts1, col_ts2 = st.columns(2)

with col_ts1:
    monthly_ts = get_time_series(raw_df, selected_product, selected_bolge if selected_bolge != "TÜMÜ" else None)
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(x=monthly_ts['Tarih'], y=monthly_ts['PF Satış'], name='PF', line=dict(color='#3B82F6', width=3), marker=dict(size=8)))
    fig_ts.add_trace(go.Scatter(x=monthly_ts['Tarih'], y=monthly_ts['Rakip Satış'], name='Rakip', line=dict(color='#EF4444', width=3), marker=dict(size=8)))
    fig_ts.update_layout(height=400)
    st.plotly_chart(fig_ts, use_container_width=True)

with col_ts2:
    fig_share = go.Figure()
    fig_share.add_trace(go.Scatter(x=monthly_ts['Tarih'], y=monthly_ts['Pazar Payı %'], fill='tozeroy', line=dict(color='#10B981', width=2)))
    fig_share.update_layout(height=400, yaxis=dict(title='Pazar Payı %'))
    st.plotly_chart(fig_share, use_container_width=True)

st.markdown("---")

# TABLOLAR
st.subheader("📊 Bölge Performans")
st.dataframe(bolge_df[bolge_df["PF Satış"] > 0], use_container_width=True, hide_index=True)

st.subheader("🏙️ Top 20 Şehir")
city_display = city_df.sort_values("PF Satış", ascending=False).head(20)
st.dataframe(city_display[["Şehir", "Bölge", "PF Satış", "Toplam Pazar", "Pazar Payı %", "Müdür"]], use_container_width=True)

# GRAFİKLER
col_v1, col_v2 = st.columns(2)
with col_v1:
    top10 = city_df.nlargest(10, "PF Satış")
    st.plotly_chart(px.bar(top10, x="PF Satış", y="Şehir", orientation='h', color="Pazar Payı %"), use_container_width=True)

with col_v2:
    st.plotly_chart(px.pie(bolge_df[bolge_df["PF Satış"]>0], values="PF Satış", names="Bölge", color="Bölge", color_discrete_map=REGION_COLORS), use_container_width=True)

# EXPORT
from io import BytesIO
output = BytesIO()
with pd.ExcelWriter(output, engine='openpyxl') as writer:
    city_df.to_excel(writer, sheet_name='Şehir', index=False)
    bolge_df.to_excel(writer, sheet_name='Bölge', index=False)
    monthly_ts.to_excel(writer, sheet_name='Trend', index=False)

st.download_button("📥 Rapor İndir", output.getvalue(), f"{selected_product}_{datetime.now().strftime('%Y%m%d')}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
