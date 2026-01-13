 (cd "$(git rev-parse --show-toplevel)" && git apply --3way <<'EOF' 
diff --git a/app.py b/app.py
index 5a576398901d2ad2297d87b481c19f5f3e3409b1..980ffc951daa0d878e6ea2f9fd4ee8d4166c8a79 100644
--- a/app.py
+++ b/app.py
@@ -1,47 +1,48 @@
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
-import plotly.graph_objects as go
-import plotly.express as px
-from datetime import datetime, timedelta
-import warnings
-from io import BytesIO
-import json
+import plotly.graph_objects as go
+import plotly.express as px
+from datetime import datetime, timedelta
+import warnings
+from io import BytesIO
+import json
+from pathlib import Path
 
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
@@ -236,81 +237,184 @@ def normalize_city_name(city_name):
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
-def load_geojson():
-    """Türkiye GeoJSON'ını yükle"""
-    try:
-        with open('/mnt/user-data/uploads/turkey.geojson', 'r', encoding='utf-8') as f:
-            return json.load(f)
-    except:
-        return None
+def load_geojson():
+    """Türkiye GeoJSON'ını yükle"""
+    candidates = [
+        Path("turkey.geojson"),
+        Path("/mnt/user-data/uploads/turkey.geojson"),
+    ]
+    for path in candidates:
+        if path.exists():
+            with path.open("r", encoding="utf-8") as f:
+                return json.load(f)
+    return None
 
 # =============================================================================
 # ANALYSIS FUNCTIONS
 # =============================================================================
 
-def calculate_city_performance(df, product, date_filter=None):
-    """Şehir bazlı performans analizi"""
-    cols = get_product_columns(product)
-    
-    # Tarih filtresi
-    if date_filter:
-        df = df[(df['DATE'] >= date_filter[0]) & (df['DATE'] <= date_filter[1])]
-    
-    # Şehir bazlı toplam
-    city_perf = df.groupby(['CITY_NORMALIZED']).agg({
-        cols['pf']: 'sum',
-        cols['rakip']: 'sum'
-    }).reset_index()
-    
-    city_perf.columns = ['City', 'PF_Satis', 'Rakip_Satis']
-    city_perf['Toplam_Pazar'] = city_perf['PF_Satis'] + city_perf['Rakip_Satis']
-    city_perf['Pazar_Payi_%'] = safe_divide(city_perf['PF_Satis'], city_perf['Toplam_Pazar']) * 100
-    
-    return city_perf
+def calculate_city_performance(df, product, date_filter=None):
+    """Şehir bazlı performans analizi"""
+    cols = get_product_columns(product)
+    
+    # Tarih filtresi
+    if date_filter:
+        df = df[(df['DATE'] >= date_filter[0]) & (df['DATE'] <= date_filter[1])]
+
+    region_map = df.groupby('CITY_NORMALIZED')['REGION'].agg(
+        lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]
+    )
+    manager_map = df.groupby('CITY_NORMALIZED')['MANAGER'].agg(
+        lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]
+    )
+    
+    # Şehir bazlı toplam
+    city_perf = df.groupby(['CITY_NORMALIZED']).agg({
+        cols['pf']: 'sum',
+        cols['rakip']: 'sum'
+    }).reset_index()
+    
+    city_perf.columns = ['City', 'PF_Satis', 'Rakip_Satis']
+    city_perf['Toplam_Pazar'] = city_perf['PF_Satis'] + city_perf['Rakip_Satis']
+    city_perf['Pazar_Payi_%'] = safe_divide(city_perf['PF_Satis'], city_perf['Toplam_Pazar']) * 100
+    city_perf['Bölge'] = city_perf['City'].map(region_map).fillna("DİĞER")
+    city_perf['Manager'] = city_perf['City'].map(manager_map).fillna("YOK")
+    
+    return city_perf
+
+def calculate_city_strategy(city_df):
+    """Şehir bazlı yatırım stratejisi hesapla"""
+    df = city_df.copy()
+    if df.empty:
+        return df
+
+    df["Büyüme_Potansiyeli"] = df["Toplam_Pazar"] - df["PF_Satis"]
+
+    def safe_qcut(series, labels):
+        try:
+            return pd.qcut(series, q=3, labels=labels, duplicates="drop")
+        except ValueError:
+            return pd.Series([labels[1]] * len(series), index=series.index)
+
+    df["Pazar_Buyuklugu"] = safe_qcut(
+        df["Toplam_Pazar"], ["Küçük", "Orta", "Büyük"]
+    )
+    df["Performans"] = safe_qcut(
+        df["PF_Satis"], ["Düşük", "Orta", "Yüksek"]
+    )
+    df["Pazar_Payi_Segment"] = safe_qcut(
+        df["Pazar_Payi_%"], ["Düşük", "Orta", "Yüksek"]
+    )
+    df["Buyume_Potansiyeli_Segment"] = safe_qcut(
+        df["Büyüme_Potansiyeli"], ["Düşük", "Orta", "Yüksek"]
+    )
+
+    def assign_strategy(row):
+        if (row["Pazar_Buyuklugu"] in ["Büyük", "Orta"] and
+                row["Pazar_Payi_Segment"] == "Düşük" and
+                row["Buyume_Potansiyeli_Segment"] in ["Yüksek", "Orta"]):
+            return "🚀 Agresif"
+        if (row["Pazar_Buyuklugu"] in ["Büyük", "Orta"] and
+                row["Pazar_Payi_Segment"] == "Orta" and
+                row["Performans"] in ["Orta", "Yüksek"]):
+            return "⚡ Hızlandırılmış"
+        if row["Pazar_Buyuklugu"] == "Büyük" and row["Pazar_Payi_Segment"] == "Yüksek":
+            return "🛡️ Koruma"
+        if (row["Pazar_Buyuklugu"] == "Küçük" and
+                row["Buyume_Potansiyeli_Segment"] == "Yüksek" and
+                row["Performans"] in ["Orta", "Yüksek"]):
+            return "💎 Potansiyel"
+        return "👁️ İzleme"
+
+    df["Yatirim_Stratejisi"] = df.apply(assign_strategy, axis=1)
+
+    df["Skor_Pazar"] = df["Toplam_Pazar"].rank(pct=True)
+    df["Skor_Buyume"] = df["Büyüme_Potansiyeli"].rank(pct=True)
+    df["Skor_Pay_Ters"] = 1 - df["Pazar_Payi_%"].rank(pct=True)
+    df["Oncelik_Skoru"] = (df["Skor_Pazar"] * 0.4 + df["Skor_Buyume"] * 0.4 + df["Skor_Pay_Ters"] * 0.2) * 100
+    df["Oncelik_Skoru"] = df["Oncelik_Skoru"].round(1)
+
+    return df
+
+def generate_city_insights(strategy_df):
+    """Şehir bazlı aksiyon önerileri üret"""
+    if strategy_df.empty:
+        return []
+
+    insights = []
+    median_market = strategy_df["Toplam_Pazar"].median()
+    buyume_alanlari = strategy_df[
+        (strategy_df["Toplam_Pazar"] >= median_market) &
+        (strategy_df["Pazar_Payi_%"] < 10)
+    ].nlargest(3, "Büyüme_Potansiyeli")
+
+    if not buyume_alanlari.empty:
+        cities = ", ".join(buyume_alanlari["City"].tolist())
+        insights.append(f"🚀 **Yüksek fırsat:** {cities} şehirlerinde pazar büyük ama pay düşük. Agresif saha yatırımı önerilir.")
+
+    koruma = strategy_df[
+        (strategy_df["Pazar_Payi_%"] >= 40) &
+        (strategy_df["Toplam_Pazar"] >= median_market)
+    ].nlargest(3, "PF_Satis")
+
+    if not koruma.empty:
+        cities = ", ".join(koruma["City"].tolist())
+        insights.append(f"🛡️ **Koruma alanları:** {cities} lider bölgeler; mevcut müşteri koruma ve sadakat programları öncelikli.")
+
+    dusuk_pay = strategy_df[
+        (strategy_df["Pazar_Payi_%"] < 5) &
+        (strategy_df["Toplam_Pazar"] > 0)
+    ].nlargest(3, "Toplam_Pazar")
+
+    if not dusuk_pay.empty:
+        cities = ", ".join(dusuk_pay["City"].tolist())
+        insights.append(f"⚡ **Giriş fırsatı:** {cities} şehirlerinde düşük pay görülüyor. Distribütör ağı veya kampanya ile giriş önerilir.")
+
+    return insights
 
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
@@ -880,60 +984,191 @@ def main():
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
         
-        st.dataframe(
-            city_display.style.format({
-                'PF_Satis': '{:,.0f}',
-                'Rakip_Satis': '{:,.0f}',
-                'Toplam_Pazar': '{:,.0f}',
-                'Pazar_Payi_%': '{:.1f}'
-            }).background_gradient(subset=['Pazar_Payi_%'], cmap='RdYlGn'),
-            use_container_width=True,
-            height=400
-        )
+        st.dataframe(
+            city_display.style.format({
+                'PF_Satis': '{:,.0f}',
+                'Rakip_Satis': '{:,.0f}',
+                'Toplam_Pazar': '{:,.0f}',
+                'Pazar_Payi_%': '{:.1f}'
+            }).background_gradient(subset=['Pazar_Payi_%'], cmap='RdYlGn'),
+            use_container_width=True,
+            height=400
+        )
+
+        st.markdown("---")
+        st.header("🎯 Şehir Stratejisi & Fırsat Analizi")
+
+        strategy_df = calculate_city_strategy(city_data)
+        if not strategy_df.empty:
+            col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns(5)
+            strategy_counts = strategy_df["Yatirim_Stratejisi"].value_counts()
+            strategy_counts_df = strategy_counts.rename_axis("Yatirim_Stratejisi").reset_index(name="count")
+
+            with col_s1:
+                st.metric("🚀 Agresif", strategy_counts.get("🚀 Agresif", 0))
+            with col_s2:
+                st.metric("⚡ Hızlandırılmış", strategy_counts.get("⚡ Hızlandırılmış", 0))
+            with col_s3:
+                st.metric("🛡️ Koruma", strategy_counts.get("🛡️ Koruma", 0))
+            with col_s4:
+                st.metric("💎 Potansiyel", strategy_counts.get("💎 Potansiyel", 0))
+            with col_s5:
+                st.metric("👁️ İzleme", strategy_counts.get("👁️ İzleme", 0))
+
+            st.markdown("---")
+
+            col_chart1, col_chart2 = st.columns(2)
+
+            with col_chart1:
+                st.subheader("🏆 Öncelikli 10 Şehir")
+                top10_priority = strategy_df.nlargest(10, "Oncelik_Skoru")
+                fig_priority = px.bar(
+                    top10_priority,
+                    x="Oncelik_Skoru",
+                    y="City",
+                    orientation="h",
+                    color="Yatirim_Stratejisi",
+                    color_discrete_map={
+                        "🚀 Agresif": "#EF4444",
+                        "⚡ Hızlandırılmış": "#F59E0B",
+                        "🛡️ Koruma": "#10B981",
+                        "💎 Potansiyel": "#8B5CF6",
+                        "👁️ İzleme": "#6B7280"
+                    },
+                    text="Oncelik_Skoru"
+                )
+                fig_priority.update_traces(texttemplate="%{text:.1f}", textposition="outside")
+                fig_priority.update_layout(height=450, yaxis={'categoryorder': 'total ascending'})
+                st.plotly_chart(fig_priority, use_container_width=True)
+
+            with col_chart2:
+                st.subheader("🎯 Strateji Dağılımı")
+                fig_pie = px.pie(
+                    strategy_counts_df,
+                    values="count",
+                    names="Yatirim_Stratejisi",
+                    color="Yatirim_Stratejisi",
+                    color_discrete_map={
+                        "🚀 Agresif": "#EF4444",
+                        "⚡ Hızlandırılmış": "#F59E0B",
+                        "🛡️ Koruma": "#10B981",
+                        "💎 Potansiyel": "#8B5CF6",
+                        "👁️ İzleme": "#6B7280"
+                    }
+                )
+                fig_pie.update_traces(textposition="inside", textinfo="percent+label")
+                fig_pie.update_layout(height=450)
+                st.plotly_chart(fig_pie, use_container_width=True)
+
+            st.markdown("---")
+            st.subheader("🗺️ Bölge → Strateji → Şehir Haritası")
+            treemap_df = strategy_df.copy()
+            treemap_df["Strateji_Kisa"] = treemap_df["Yatirim_Stratejisi"].str.replace("🚀 ", "").str.replace("⚡ ", "").str.replace("🛡️ ", "").str.replace("💎 ", "").str.replace("👁️ ", "")
+            fig_treemap = px.treemap(
+                treemap_df,
+                path=[px.Constant("TÜRKİYE"), "Bölge", "Strateji_Kisa", "City"],
+                values="PF_Satis",
+                color="Pazar_Payi_%",
+                color_continuous_scale="Blues",
+                color_continuous_midpoint=treemap_df["Pazar_Payi_%"].median(),
+                hover_data={
+                    "PF_Satis": ":,.0f",
+                    "Pazar_Payi_%": ":.1f",
+                    "Toplam_Pazar": ":,.0f"
+                }
+            )
+            fig_treemap.update_layout(height=600)
+            st.plotly_chart(fig_treemap, use_container_width=True)
+
+            st.markdown("---")
+            st.subheader("🔥 Bölge × Strateji Isı Haritası")
+            heatmap_data = strategy_df.pivot_table(
+                index="Bölge",
+                columns="Yatirim_Stratejisi",
+                values="PF_Satis",
+                aggfunc="sum",
+                fill_value=0
+            )
+            fig_heatmap = px.imshow(
+                heatmap_data,
+                labels=dict(x="Strateji", y="Bölge", color="PF Satış"),
+                color_continuous_scale="YlOrRd",
+                aspect="auto",
+                text_auto=".0f"
+            )
+            fig_heatmap.update_layout(height=500)
+            st.plotly_chart(fig_heatmap, use_container_width=True)
+
+            st.markdown("---")
+            st.subheader("💡 Otomatik Aksiyon Önerileri")
+            for insight in generate_city_insights(strategy_df):
+                st.markdown(f"- {insight}")
+
+            st.markdown("---")
+            st.subheader("📋 Strateji Detay Tablosu")
+            display_strategy = strategy_df[[
+                "City", "Bölge", "PF_Satis", "Toplam_Pazar", "Pazar_Payi_%",
+                "Yatirim_Stratejisi", "Oncelik_Skoru", "Manager"
+            ]].copy()
+            display_strategy = display_strategy.sort_values("Oncelik_Skoru", ascending=False)
+            display_strategy.columns = [
+                "Şehir", "Bölge", "PF Satış", "Toplam Pazar", "Pazar Payı %",
+                "Strateji", "Öncelik Skoru", "Manager"
+            ]
+            st.dataframe(
+                display_strategy.style.format({
+                    "PF Satış": "{:,.0f}",
+                    "Toplam Pazar": "{:,.0f}",
+                    "Pazar Payı %": "{:.1f}",
+                    "Öncelik Skoru": "{:.1f}"
+                }),
+                use_container_width=True,
+                height=400
+            )
     
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
         
 
EOF
)
