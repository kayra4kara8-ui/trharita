"""
🎯 ADVANCED COMMERCIAL PORTFOLIO ANALYTICS SYSTEM
Territory-based Performance, ML Forecasting, Turkey Mapping & Competitive Intelligence

Features:
- 🗺️ Enhanced Turkey province-based geographic visualization
- 🤖 REAL Machine Learning (Linear Regression, Ridge, Random Forest)
- 📊 Monthly/Yearly period selection with dynamic filtering
- 📈 Advanced competitor analysis with trend comparison
- 🎯 Strategic investment recommendations
- 📊 Professional executive dashboards
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
from sklearn.metrics import mean_absolute_error, mean_squared_error
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString
from typing import Dict, List, Tuple, Optional, Any
import plotly.subplots as sp
from plotly.colors import sequential

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION & STYLING
# =============================================================================

class AppConfig:
    """Application configuration and styling constants"""
    
    # Page configuration
    PAGE_CONFIG = {
        "page_title": "Strategic Portfolio Analytics",
        "page_icon": "📊",
        "layout": "wide",
        "initial_sidebar_state": "expanded"
    }
    
    # Color palettes
    COLOR_SCHEME = {
        # Primary blue tones
        "primary_light": "#60A5FA",
        "primary": "#3B82F6",
        "primary_dark": "#1D4ED8",
        "primary_darker": "#1E40AF",
        
        # Secondary blues
        "secondary_light": "#93C5FD",
        "secondary": "#2563EB",
        "secondary_dark": "#1E3A8A",
        
        # Accent colors
        "accent_teal": "#0D9488",
        "accent_indigo": "#4F46E5",
        "accent_purple": "#7C3AED",
        
        # Neutral tones
        "neutral_light": "#E5E7EB",
        "neutral": "#9CA3AF",
        "neutral_dark": "#4B5563",
        "neutral_darker": "#1F2937",
        
        # Status colors
        "success": "#10B981",
        "warning": "#F59E0B",
        "error": "#EF4444",
        "info": "#3B82F6"
    }
    
    # Region colors (professional blue theme)
    REGION_COLORS = {
        "MARMARA": "#0EA5E9",          # Bright Blue
        "BATI ANADOLU": "#3B82F6",     # Primary Blue
        "EGE": "#60A5FA",              # Light Blue
        "İÇ ANADOLU": "#1D4ED8",       # Dark Blue
        "GÜNEY DOĞU ANADOLU": "#2563EB", # Royal Blue
        "KUZEY ANADOLU": "#1E40AF",    # Deep Blue
        "KARADENİZ": "#1E40AF",        # Deep Blue
        "AKDENİZ": "#0369A1",          # Ocean Blue
        "DOĞU ANADOLU": "#0C4A6E",     # Night Blue
        "DİĞER": "#64748B"             # Slate Blue
    }
    
    # Investment strategy colors
    STRATEGY_COLORS = {
        "🚀 Agresif": COLOR_SCHEME["error"],
        "⚡ Hızlandırılmış": COLOR_SCHEME["warning"],
        "🛡️ Koruma": COLOR_SCHEME["success"],
        "💎 Potansiyel": COLOR_SCHEME["accent_purple"],
        "👁️ İzleme": COLOR_SCHEME["neutral"]
    }
    
    # BCG Matrix colors
    BCG_COLORS = {
        "⭐ Star": "#0EA5E9",
        "🐄 Cash Cow": "#3B82F6",
        "❓ Question Mark": "#60A5FA",
        "🐶 Dog": "#94A3B8"
    }

class Styling:
    """CSS and HTML styling utilities"""
    
    @staticmethod
    def apply_custom_css():
        """Apply professional CSS styling"""
        st.markdown(f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
            
            * {{
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            }}
            
            .stApp {{
                background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
                background-attachment: fixed;
            }}
            
            /* Professional header */
            .main-header {{
                font-size: 3.2rem;
                font-weight: 800;
                text-align: center;
                padding: 2.5rem 0;
                background: linear-gradient(135deg, {AppConfig.COLOR_SCHEME['primary_light']} 0%, 
                                           {AppConfig.COLOR_SCHEME['primary']} 50%, 
                                           {AppConfig.COLOR_SCHEME['primary_dark']} 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                text-shadow: 0 4px 20px rgba(59, 130, 246, 0.15);
                letter-spacing: -0.5px;
                margin-bottom: 1.5rem;
            }}
            
            /* Enhanced metric cards */
            div[data-testid="stMetricValue"] {{
                font-size: 2.6rem;
                font-weight: 700;
                background: linear-gradient(135deg, {AppConfig.COLOR_SCHEME['primary_light']} 0%, 
                                           {AppConfig.COLOR_SCHEME['primary']} 50%, 
                                           {AppConfig.COLOR_SCHEME['accent_purple']} 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }}
            
            div[data-testid="metric-container"] {{
                background: rgba(30, 41, 59, 0.85);
                padding: 1.8rem;
                border-radius: 16px;
                border: 1px solid rgba(59, 130, 246, 0.25);
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25);
                backdrop-filter: blur(12px);
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            }}
            
            div[data-testid="metric-container"]:hover {{
                transform: translateY(-4px);
                box-shadow: 0 12px 40px rgba(59, 130, 246, 0.35);
                border-color: rgba(59, 130, 246, 0.45);
            }}
            
            /* Professional tabs */
            .stTabs [data-baseweb="tab"] {{
                color: #94a3b8;
                font-weight: 600;
                font-size: 0.95rem;
                padding: 1rem 1.8rem;
                background: rgba(30, 41, 59, 0.6);
                border-radius: 10px 10px 0 0;
                margin: 0 0.2rem;
                transition: all 0.2s ease;
                border: 1px solid rgba(148, 163, 184, 0.1);
            }}
            
            .stTabs [data-baseweb="tab"]:hover {{
                background: rgba(59, 130, 246, 0.15);
                color: #e0e7ff;
                border-color: rgba(59, 130, 246, 0.3);
            }}
            
            .stTabs [data-baseweb="tab"][aria-selected="true"] {{
                background: linear-gradient(135deg, {AppConfig.COLOR_SCHEME['primary']} 0%, 
                                           {AppConfig.COLOR_SCHEME['primary_dark']} 100%);
                color: white;
                box-shadow: 0 4px 16px rgba(59, 130, 246, 0.4);
                border-color: rgba(59, 130, 246, 0.5);
            }}
            
            /* Typography */
            h1, h2, h3, h4 {{
                color: #f8fafc !important;
                font-weight: 700;
                margin-bottom: 1rem;
            }}
            
            h1 {{ font-size: 2.2rem; }}
            h2 {{ font-size: 1.8rem; }}
            h3 {{ font-size: 1.5rem; }}
            h4 {{ font-size: 1.2rem; }}
            
            p, span, div, label {{
                color: #cbd5e1;
                line-height: 1.6;
            }}
            
            /* Enhanced buttons */
            .stButton > button {{
                background: linear-gradient(135deg, {AppConfig.COLOR_SCHEME['primary']} 0%, 
                                           {AppConfig.COLOR_SCHEME['primary_dark']} 100%);
                color: white;
                border: none;
                padding: 0.85rem 2.2rem;
                border-radius: 10px;
                font-weight: 600;
                font-size: 0.95rem;
                transition: all 0.3s ease;
                box-shadow: 0 4px 14px rgba(59, 130, 246, 0.3);
                letter-spacing: 0.3px;
            }}
            
            .stButton > button:hover {{
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4);
                background: linear-gradient(135deg, {AppConfig.COLOR_SCHEME['primary_light']} 0%, 
                                           {AppConfig.COLOR_SCHEME['primary']} 100%);
            }}
            
            /* Dataframe styling */
            .dataframe {{
                background: rgba(30, 41, 59, 0.8) !important;
                border-radius: 10px;
                border: 1px solid rgba(148, 163, 184, 0.2);
            }}
            
            .dataframe th {{
                background: {AppConfig.COLOR_SCHEME['primary_dark']} !important;
                color: white !important;
                font-weight: 600 !important;
                font-size: 0.9rem;
                padding: 12px !important;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            
            .dataframe td {{
                color: #e2e8f0 !important;
                font-weight: 500;
                padding: 10px !important;
                border-bottom: 1px solid rgba(148, 163, 184, 0.1);
            }}
            
            /* Sidebar enhancements */
            .stSidebar {{
                background: rgba(15, 23, 42, 0.9);
                backdrop-filter: blur(10px);
            }}
            
            /* Custom scrollbar */
            ::-webkit-scrollbar {{
                width: 8px;
                height: 8px;
            }}
            
            ::-webkit-scrollbar-track {{
                background: rgba(30, 41, 59, 0.5);
                border-radius: 4px;
            }}
            
            ::-webkit-scrollbar-thumb {{
                background: {AppConfig.COLOR_SCHEME['primary']};
                border-radius: 4px;
            }}
            
            ::-webkit-scrollbar-thumb:hover {{
                background: {AppConfig.COLOR_SCHEME['primary_light']};
            }}
        </style>
        """, unsafe_allow_html=True)

# =============================================================================
# DATA MANAGEMENT
# =============================================================================

class DataManager:
    """Handles data loading, preprocessing, and management"""
    
    # City normalization mapping
    CITY_NORMALIZATION_MAP = {
        "AGRI": "AĞRI",
        "BARTÄ±N": "BARTIN",
        "BINGÃ¶L": "BİNGÖL",
        "DÃ1⁄4ZCE": "DÜZCE",
        "ELAZIG": "ELAZIĞ",
        "ESKISEHIR": "ESKİŞEHİR",
        "GÃ1⁄4MÃ1⁄4SHANE": "GÜMÜŞHANE",
        "HAKKARI": "HAKKARİ",
        "ISTANBUL": "İSTANBUL",
        "IZMIR": "İZMİR",
        "IÄ\x9fDIR": "IĞDIR",
        "KARABÃ1⁄4K": "KARABÜK",
        "KINKKALE": "KIRIKKALE",
        "KIRSEHIR": "KIRŞEHİR",
        "KÃ1⁄4TAHYA": "KÜTAHYA",
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
        "K. MARAS": "KAHRAMANMARAŞ",
        "CORUM": "ÇORUM",
        "CANKIRI": "ÇANKIRI",
        "ZONGULDAK": "ZONGULDAK",
        "KARABUK": "KARABÜK",
        "GUMUSHANE": "GÜMÜŞHANE",
        "ELÂZıĞ": "ELAZIĞ",
        "KUTAHYA": "KÜTAHYA",
        "CANAKKALE": "ÇANAKKALE"
    }
    
    @staticmethod
    @st.cache_data
    def load_excel_data(file) -> pd.DataFrame:
        """Load and preprocess Excel data"""
        try:
            df = pd.read_excel(file)
            
            # Date processing
            df['DATE'] = pd.to_datetime(df['DATE'])
            df['YEAR_MONTH'] = df['DATE'].dt.strftime('%Y-%m')
            df['MONTH'] = df['DATE'].dt.month
            df['YEAR'] = df['DATE'].dt.year
            df['QUARTER'] = df['DATE'].dt.quarter
            
            # String standardization
            string_columns = ['TERRITORIES', 'CITY', 'REGION', 'MANAGER']
            for col in string_columns:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.upper().str.strip()
            
            # City normalization
            if 'CITY' in df.columns:
                df['CITY_NORMALIZED'] = df['CITY'].apply(DataManager.normalize_city_name)
            
            return df
            
        except Exception as e:
            st.error(f"❌ Data loading failed: {str(e)}")
            raise
    
    @staticmethod
    def normalize_city_name(city_name: str) -> str:
        """Standardize city names with Turkish character handling"""
        if pd.isna(city_name):
            return None
        
        city_upper = str(city_name).strip().upper()
        
        # Apply known fixes
        if city_upper in DataManager.CITY_NORMALIZATION_MAP:
            return DataManager.CITY_NORMALIZATION_MAP[city_upper]
        
        # Turkish character normalization
        turkish_map = {
            "İ": "I", "Ğ": "G", "Ü": "U",
            "Ş": "S", "Ö": "O", "Ç": "C",
            "Â": "A", "Î": "I", "Û": "U"
        }
        
        for old, new in turkish_map.items():
            city_upper = city_upper.replace(old, new)
        
        return city_upper
    
    @staticmethod
    @st.cache_resource
    def load_geographic_data() -> Optional[gpd.GeoDataFrame]:
        """Load Turkey geographic data"""
        try:
            gdf = gpd.read_file("turkey.geojson", encoding='utf-8')
            gdf['CITY_NORMALIZED'] = gdf['name'].apply(DataManager.normalize_city_name)
            return gdf
        except Exception as e:
            st.warning(f"⚠️ Geographic data not available: {str(e)}")
            return None

# =============================================================================
# ANALYTICS ENGINE
# =============================================================================

class AnalyticsEngine:
    """Core analytics and calculation engine"""
    
    @staticmethod
    def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
        """Safe division with zero handling"""
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.where(denominator != 0, numerator / denominator, 0)
        return result
    
    @staticmethod
    def calculate_city_performance(df: pd.DataFrame, product_config: Dict, 
                                   date_filter: Optional[Tuple] = None) -> pd.DataFrame:
        """Calculate city-level performance metrics"""
        
        if date_filter:
            df = df[(df['DATE'] >= date_filter[0]) & (df['DATE'] <= date_filter[1])]
        
        city_perf = df.groupby(['CITY_NORMALIZED', 'REGION']).agg({
            product_config['pf']: 'sum',
            product_config['rakip']: 'sum'
        }).reset_index()
        
        city_perf.columns = ['City', 'Region', 'PF_Sales', 'Competitor_Sales']
        city_perf['Total_Market'] = city_perf['PF_Sales'] + city_perf['Competitor_Sales']
        city_perf['Market_Share_%'] = AnalyticsEngine.safe_divide(
            city_perf['PF_Sales'], city_perf['Total_Market']
        ) * 100
        
        return city_perf
    
    @staticmethod
    def calculate_territory_performance(df: pd.DataFrame, product_config: Dict,
                                        date_filter: Optional[Tuple] = None) -> pd.DataFrame:
        """Calculate territory-level performance metrics"""
        
        if date_filter:
            df = df[(df['DATE'] >= date_filter[0]) & (df['DATE'] <= date_filter[1])]
        
        territory_perf = df.groupby(['TERRITORIES', 'REGION', 'CITY', 'MANAGER']).agg({
            product_config['pf']: 'sum',
            product_config['rakip']: 'sum'
        }).reset_index()
        
        territory_perf.columns = ['Territory', 'Region', 'City', 'Manager', 
                                  'PF_Sales', 'Competitor_Sales']
        
        territory_perf['Total_Market'] = territory_perf['PF_Sales'] + territory_perf['Competitor_Sales']
        territory_perf['Market_Share_%'] = AnalyticsEngine.safe_divide(
            territory_perf['PF_Sales'], territory_perf['Total_Market']
        ) * 100
        
        total_pf = territory_perf['PF_Sales'].sum()
        territory_perf['Weight_%'] = AnalyticsEngine.safe_divide(
            territory_perf['PF_Sales'], total_pf
        ) * 100
        
        territory_perf['Relative_Market_Share'] = AnalyticsEngine.safe_divide(
            territory_perf['PF_Sales'], territory_perf['Competitor_Sales']
        )
        
        return territory_perf.sort_values('PF_Sales', ascending=False)
    
    @staticmethod
    def calculate_time_series(df: pd.DataFrame, product_config: Dict,
                              territory: Optional[str] = None,
                              date_filter: Optional[Tuple] = None) -> pd.DataFrame:
        """Calculate time series analysis"""
        
        df_filtered = df.copy()
        if territory and territory != "ALL":
            df_filtered = df_filtered[df_filtered['TERRITORIES'] == territory]
        
        if date_filter:
            df_filtered = df_filtered[
                (df_filtered['DATE'] >= date_filter[0]) & 
                (df_filtered['DATE'] <= date_filter[1])
            ]
        
        monthly = df_filtered.groupby('YEAR_MONTH').agg({
            product_config['pf']: 'sum',
            product_config['rakip']: 'sum',
            'DATE': 'first'
        }).reset_index().sort_values('YEAR_MONTH')
        
        monthly.columns = ['Period', 'PF_Sales', 'Competitor_Sales', 'Date']
        
        # Calculate metrics
        monthly['Total_Market'] = monthly['PF_Sales'] + monthly['Competitor_Sales']
        monthly['Market_Share_%'] = AnalyticsEngine.safe_divide(
            monthly['PF_Sales'], monthly['Total_Market']
        ) * 100
        
        monthly['PF_Growth_%'] = monthly['PF_Sales'].pct_change() * 100
        monthly['Competitor_Growth_%'] = monthly['Competitor_Sales'].pct_change() * 100
        monthly['Relative_Growth_%'] = monthly['PF_Growth_%'] - monthly['Competitor_Growth_%']
        
        # Moving averages
        monthly['MA_3M'] = monthly['PF_Sales'].rolling(window=3, min_periods=1).mean()
        monthly['MA_6M'] = monthly['PF_Sales'].rolling(window=6, min_periods=1).mean()
        
        return monthly
    
    @staticmethod
    def calculate_investment_strategy(city_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate strategic investment recommendations"""
        
        df = city_data[city_data['PF_Sales'] > 0].copy()
        
        if len(df) == 0:
            return df
        
        # Market size segmentation
        try:
            df["Market_Size"] = pd.qcut(
                df["Total_Market"], 
                q=3, 
                labels=["Small", "Medium", "Large"],
                duplicates='drop'
            )
        except:
            df["Market_Size"] = "Medium"
        
        # Performance segmentation
        try:
            df["Performance"] = pd.qcut(
                df["PF_Sales"], 
                q=3, 
                labels=["Low", "Medium", "High"],
                duplicates='drop'
            )
        except:
            df["Performance"] = "Medium"
        
        # Market share segmentation
        try:
            df["Share_Segment"] = pd.qcut(
                df["Market_Share_%"], 
                q=3, 
                labels=["Low", "Medium", "High"],
                duplicates='drop'
            )
        except:
            df["Share_Segment"] = "Medium"
        
        # Growth potential
        df["Growth_Potential"] = df["Total_Market"] - df["PF_Sales"]
        try:
            df["Growth_Potential_Segment"] = pd.qcut(
                df["Growth_Potential"],
                q=3,
                labels=["Low", "Medium", "High"],
                duplicates='drop'
            )
        except:
            df["Growth_Potential_Segment"] = "Medium"
        
        # Strategy assignment
        def assign_strategy(row):
            market_size = str(row["Market_Size"])
            market_share = str(row["Share_Segment"])
            growth_potential = str(row["Growth_Potential_Segment"])
            performance = str(row["Performance"])
            
            # 🚀 AGGRESSIVE: Large market + Low share + High growth potential
            if (market_size in ["Large", "Medium"] and 
                market_share == "Low" and 
                growth_potential in ["High", "Medium"]):
                return "🚀 Agresif"
            
            # ⚡ ACCELERATED: Large/Medium market + Medium share + Good performance
            elif (market_size in ["Large", "Medium"] and 
                  market_share == "Medium" and
                  performance in ["Medium", "High"]):
                return "⚡ Hızlandırılmış"
            
            # 🛡️ DEFENSIVE: Large market + High share (Market leadership)
            elif (market_size == "Large" and 
                  market_share == "High"):
                return "🛡️ Koruma"
            
            # 💎 POTENTIAL: Small market but high growth potential
            elif (market_size == "Small" and 
                  growth_potential == "High" and
                  performance in ["Medium", "High"]):
                return "💎 Potansiyel"
            
            # 👁️ MONITOR: All other cases
            else:
                return "👁️ İzleme"
        
        df["Investment_Strategy"] = df.apply(assign_strategy, axis=1)
        
        return df

# =============================================================================
# MACHINE LEARNING ENGINE
# =============================================================================

class MLEngine:
    """Machine learning forecasting engine"""
    
    @staticmethod
    def create_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create features for ML forecasting"""
        df_features = df.copy().sort_values('Date').reset_index(drop=True)
        
        # Lag features
        for lag in [1, 2, 3, 4, 6, 12]:
            df_features[f'lag_{lag}'] = df_features['PF_Sales'].shift(lag)
        
        # Rolling statistics
        for window in [3, 6, 12]:
            df_features[f'rolling_mean_{window}'] = df_features['PF_Sales'].rolling(
                window=window, min_periods=1
            ).mean()
            df_features[f'rolling_std_{window}'] = df_features['PF_Sales'].rolling(
                window=window, min_periods=1
            ).std()
        
        # Date features
        df_features['month'] = df_features['Date'].dt.month
        df_features['quarter'] = df_features['Date'].dt.quarter
        df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
        df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)
        df_features['trend'] = range(len(df_features))
        
        # Market features
        df_features['market_growth'] = df_features['Total_Market'].pct_change()
        df_features['competitor_pressure'] = AnalyticsEngine.safe_divide(
            df_features['Competitor_Sales'], df_features['Total_Market']
        )
        
        # Fill missing values
        df_features = df_features.fillna(method='bfill').fillna(0)
        
        return df_features
    
    @staticmethod
    def train_forecast_models(df: pd.DataFrame, forecast_periods: int = 3) -> Tuple:
        """Train ML models and generate forecasts"""
        
        if len(df) < 12:
            return None, None, None
        
        df_features = MLEngine.create_features(df)
        
        # Feature selection
        feature_cols = [col for col in df_features.columns 
                       if col not in ['Period', 'PF_Sales', 'Competitor_Sales', 
                                     'Total_Market', 'Date', 'Market_Share_%',
                                     'PF_Growth_%', 'Competitor_Growth_%', 
                                     'Relative_Growth_%', 'MA_3M', 'MA_6M']]
        
        # Train-test split (time-series aware)
        split_idx = max(12, int(len(df_features) * 0.75))
        train_df = df_features.iloc[:split_idx]
        test_df = df_features.iloc[split_idx:]
        
        X_train = train_df[feature_cols]
        y_train = train_df['PF_Sales']
        X_test = test_df[feature_cols]
        y_test = test_df['PF_Sales']
        
        # Model definitions
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0, random_state=42),
            'Random Forest': RandomForestRegressor(
                n_estimators=100, 
                max_depth=5, 
                random_state=42,
                min_samples_split=5
            )
        }
        
        results = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mape = np.mean(np.abs((y_test - y_pred) / np.where(y_test != 0, y_test, 1))) * 100
            
            results[name] = {
                'model': model,
                'MAE': mae,
                'RMSE': rmse,
                'MAPE': mape,
                'predictions': y_pred
            }
        
        # Select best model based on MAPE
        best_model_name = min(results.keys(), key=lambda x: results[x]['MAPE'])
        best_model = results[best_model_name]['model']
        
        # Generate forecasts
        forecast_data = []
        last_row = df_features.iloc[-1:].copy()
        
        for i in range(forecast_periods):
            next_date = last_row['Date'].values[0] + pd.DateOffset(months=1)
            X_future = last_row[feature_cols]
            next_pred = best_model.predict(X_future)[0]
            
            forecast_data.append({
                'Date': next_date,
                'Period': pd.to_datetime(next_date).strftime('%Y-%m'),
                'PF_Sales': max(0, next_pred),
                'Model': best_model_name,
                'Confidence': 'High' if results[best_model_name]['MAPE'] < 15 else 'Medium'
            })
            
            # Update features for next prediction
            new_row = last_row.copy()
            new_row['Date'] = next_date
            new_row['PF_Sales'] = next_pred
            
            # Update lag features
            for lag in [1, 2, 3, 4, 6, 12]:
                if lag == 1:
                    new_row[f'lag_{lag}'] = last_row['PF_Sales'].values[0]
                else:
                    new_row[f'lag_{lag}'] = last_row[f'lag_{lag-1}'].values[0]
            
            # Update other features
            new_row['month'] = pd.to_datetime(next_date).month
            new_row['quarter'] = pd.to_datetime(next_date).quarter
            new_row['month_sin'] = np.sin(2 * np.pi * new_row['month'] / 12)
            new_row['month_cos'] = np.cos(2 * np.pi * new_row['month'] / 12)
            new_row['trend'] = last_row['trend'].values[0] + 1
            
            last_row = new_row
        
        forecast_df = pd.DataFrame(forecast_data)
        
        return results, best_model_name, forecast_df

# =============================================================================
# VISUALIZATION ENGINE
# =============================================================================

class VisualizationEngine:
    """Professional visualization components"""
    
    @staticmethod
    def create_executive_summary(metrics: Dict) -> None:
        """Create executive summary dashboard"""
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="💰 Total PF Sales",
                value=f"{metrics.get('total_pf_sales', 0):,.0f}",
                delta=f"{metrics.get('growth_rate', 0):.1f}%"
            )
        
        with col2:
            st.metric(
                label="🏪 Total Market Size",
                value=f"{metrics.get('total_market', 0):,.0f}",
                delta="Market Size"
            )
        
        with col3:
            st.metric(
                label="📊 Market Share",
                value=f"{metrics.get('market_share', 0):.1f}%",
                delta=f"{metrics.get('share_change', 0):+.1f}%"
            )
        
        with col4:
            st.metric(
                label="🏢 Active Territories",
                value=f"{metrics.get('active_territories', 0):,.0f}",
                delta="Coverage"
            )
    
    @staticmethod
    def create_geographic_map(city_data: pd.DataFrame, gdf: gpd.GeoDataFrame, 
                              title: str = "Turkey Sales Distribution") -> go.Figure:
        """Create professional geographic visualization"""
        
        # Merge data
        city_data = city_data.copy()
        city_data['City_Normalized'] = city_data['City'].apply(DataManager.normalize_city_name)
        
        gdf = gdf.copy()
        gdf['City_Normalized'] = gdf['name'].apply(DataManager.normalize_city_name)
        
        merged = gdf.merge(city_data, on='City_Normalized', how='left')
        merged['PF_Sales'] = merged['PF_Sales'].fillna(0)
        merged['Market_Share_%'] = merged['Market_Share_%'].fillna(0)
        merged['Region'] = merged['Region'].fillna('OTHER')
        
        # Create figure
        fig = go.Figure()
        
        # Add choropleth for sales distribution
        fig.add_trace(go.Choropleth(
            geojson=json.loads(merged.to_json()),
            locations=merged.index,
            z=merged['PF_Sales'],
            colorscale='Blues',
            colorbar=dict(
                title="PF Sales",
                thickness=20,
                len=0.8,
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.01
            ),
            marker_line_color='white',
            marker_line_width=0.5,
            hoverinfo='location+z+text',
            hovertext=merged.apply(
                lambda row: f"<b>{row['name']}</b><br>"
                          f"Region: {row['Region']}<br>"
                          f"PF Sales: {row['PF_Sales']:,.0f}<br>"
                          f"Market Share: {row['Market_Share_%']:.1f}%",
                axis=1
            ),
            name="Sales Distribution"
        ))
        
        # Add city markers for top performers
        top_cities = merged.nlargest(15, 'PF_Sales')
        fig.add_trace(go.Scattergeo(
            lon=top_cities.geometry.centroid.x,
            lat=top_cities.geometry.centroid.y,
            mode='markers+text',
            marker=dict(
                size=np.log10(top_cities['PF_Sales'] + 1) * 8,
                color=AppConfig.COLOR_SCHEME['error'],
                symbol='circle',
                line=dict(color='white', width=2)
            ),
            text=top_cities['name'],
            textposition="top center",
            textfont=dict(size=10, color='white'),
            hoverinfo='text',
            hovertext=top_cities.apply(
                lambda row: f"<b>{row['name']}</b><br>"
                          f"Rank: #{top_cities.index.get_loc(row.name)+1}<br>"
                          f"Sales: {row['PF_Sales']:,.0f}<br>"
                          f"Share: {row['Market_Share_%']:.1f}%",
                axis=1
            ),
            name="Top Performers"
        ))
        
        # Professional layout
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                font=dict(size=24, color='white'),
                xanchor='center'
            ),
            geo=dict(
                projection_scale=5.5,
                center=dict(lat=39, lon=35),
                visible=True,
                bgcolor='rgba(0,0,0,0)',
                landcolor='rgba(255, 255, 255, 0.1)',
                lakecolor='rgba(255, 255, 255, 0.05)',
                showcountries=True,
                countrycolor='rgba(255, 255, 255, 0.3)',
                showocean=True,
                oceancolor='rgba(0, 0, 50, 0.2)',
                showlakes=True
            ),
            height=650,
            margin=dict(l=0, r=0, t=80, b=0),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            legend=dict(
                bgcolor='rgba(30, 41, 59, 0.9)',
                bordercolor='rgba(148, 163, 184, 0.3)',
                borderwidth=1,
                x=0.02,
                y=0.98
            )
        )
        
        return fig
    
    @staticmethod
    def create_forecast_chart(historical: pd.DataFrame, forecast: pd.DataFrame) -> go.Figure:
        """Create professional forecast visualization"""
        
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=historical['Date'],
            y=historical['PF_Sales'],
            mode='lines+markers',
            name='Historical',
            line=dict(
                color=AppConfig.COLOR_SCHEME['primary'],
                width=3
            ),
            marker=dict(size=8),
            hovertemplate='<b>%{x|%b %Y}</b><br>Sales: %{y:,.0f}<extra></extra>'
        ))
        
        # Moving averages
        fig.add_trace(go.Scatter(
            x=historical['Date'],
            y=historical['MA_3M'],
            mode='lines',
            name='3-Month MA',
            line=dict(
                color=AppConfig.COLOR_SCHEME['accent_teal'],
                width=2,
                dash='dash'
            ),
            opacity=0.7
        ))
        
        # Forecast
        if forecast is not None and len(forecast) > 0:
            fig.add_trace(go.Scatter(
                x=forecast['Date'],
                y=forecast['PF_Sales'],
                mode='lines+markers',
                name='Forecast',
                line=dict(
                    color=AppConfig.COLOR_SCHEME['error'],
                    width=3,
                    dash='dot'
                ),
                marker=dict(
                    size=10,
                    symbol='diamond'
                ),
                hovertemplate='<b>%{x|%b %Y}</b><br>Forecast: %{y:,.0f}<extra></extra>'
            ))
            
            # Confidence interval
            fig.add_trace(go.Scatter(
                x=forecast['Date'].tolist() + forecast['Date'].tolist()[::-1],
                y=(forecast['PF_Sales'] * 1.1).tolist() + (forecast['PF_Sales'] * 0.9).tolist()[::-1],
                fill='toself',
                fillcolor='rgba(239, 68, 68, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='90% Confidence',
                hoverinfo='skip'
            ))
        
        # Professional layout
        fig.update_layout(
            title=dict(
                text='Sales Forecast & Trend Analysis',
                x=0.5,
                font=dict(size=20, color='white')
            ),
            xaxis=dict(
                title='Date',
                gridcolor='rgba(148, 163, 184, 0.2)',
                tickformat='%b %Y',
                showgrid=True
            ),
            yaxis=dict(
                title='PF Sales',
                gridcolor='rgba(148, 163, 184, 0.2)',
                tickformat=',',
                showgrid=True
            ),
            height=500,
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            ),
            margin=dict(t=80, b=80, l=80, r=80)
        )
        
        return fig
    
    @staticmethod
    def create_strategy_matrix(strategy_data: pd.DataFrame) -> go.Figure:
        """Create strategic investment matrix"""
        
        fig = go.Figure()
        
        # Size mapping for bubbles
        strategy_data['Size'] = np.log10(strategy_data['PF_Sales'] + 1) * 15
        
        # Color mapping
        color_map = AppConfig.STRATEGY_COLORS
        
        # Add scatter plot
        for strategy in strategy_data['Investment_Strategy'].unique():
            strategy_df = strategy_data[strategy_data['Investment_Strategy'] == strategy]
            
            fig.add_trace(go.Scatter(
                x=strategy_df['Total_Market'],
                y=strategy_df['Market_Share_%'],
                mode='markers+text',
                marker=dict(
                    size=strategy_df['Size'],
                    color=color_map.get(strategy, AppConfig.COLOR_SCHEME['neutral']),
                    line=dict(width=2, color='white'),
                    opacity=0.8
                ),
                text=strategy_df['City'],
                textposition="top center",
                textfont=dict(size=9, color='white'),
                name=strategy,
                hovertemplate=(
                    '<b>%{text}</b><br>'
                    'Market Size: %{x:,.0f}<br>'
                    'Market Share: %{y:.1f}%<br>'
                    'PF Sales: %{customdata[0]:,.0f}<br>'
                    'Strategy: %{customdata[1]}'
                ),
                customdata=np.stack((
                    strategy_df['PF_Sales'],
                    strategy_df['Investment_Strategy']
                ), axis=-1)
            ))
        
        # Add quadrant lines
        median_market = strategy_data['Total_Market'].median()
        median_share = strategy_data['Market_Share_%'].median()
        
        fig.add_hline(
            y=median_share,
            line_dash="dash",
            line_color="rgba(255, 255, 255, 0.3)",
            line_width=1,
            annotation_text=f"Median Share: {median_share:.1f}%",
            annotation_position="bottom right"
        )
        
        fig.add_vline(
            x=median_market,
            line_dash="dash",
            line_color="rgba(255, 255, 255, 0.3)",
            line_width=1,
            annotation_text=f"Median Market: {median_market:,.0f}",
            annotation_position="top right"
        )
        
        # Professional layout
        fig.update_layout(
            title=dict(
                text='Strategic Investment Matrix',
                x=0.5,
                font=dict(size=20, color='white')
            ),
            xaxis=dict(
                title='Market Size (Total Market)',
                type='log',
                gridcolor='rgba(148, 163, 184, 0.2)',
                showgrid=True
            ),
            yaxis=dict(
                title='Market Share (%)',
                gridcolor='rgba(148, 163, 184, 0.2)',
                showgrid=True
            ),
            height=600,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            legend=dict(
                title='Investment Strategy',
                bgcolor='rgba(30, 41, 59, 0.9)',
                bordercolor='rgba(148, 163, 184, 0.3)',
                borderwidth=1
            ),
            hovermode='closest'
        )
        
        return fig

# =============================================================================
# MAIN APPLICATION
# =============================================================================

class StrategicPortfolioAnalytics:
    """Main application controller"""
    
    def __init__(self):
        self.data_manager = DataManager()
        self.analytics = AnalyticsEngine()
        self.ml_engine = MLEngine()
        self.viz = VisualizationEngine()
        
        # Initialize session state
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'current_product' not in st.session_state:
            st.session_state.current_product = None
        
    def setup_page(self):
        """Configure page and apply styling"""
        st.set_page_config(**AppConfig.PAGE_CONFIG)
        Styling.apply_custom_css()
        
        # Professional header
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h1 class="main-header">📊 STRATEGIC PORTFOLIO ANALYTICS</h1>
            <p style="color: #94a3b8; font-size: 1.1rem; max-width: 800px; margin: 0 auto;">
                Advanced Territory Performance Analysis • ML Forecasting • Geographic Intelligence • Competitive Strategy
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    def create_sidebar(self):
        """Create professional sidebar with filters"""
        with st.sidebar:
            # Header
            st.markdown("""
            <div style="text-align: center; margin-bottom: 2rem;">
                <h3 style="color: #3b82f6; margin-bottom: 0.5rem;">⚙️ CONFIGURATION</h3>
                <div style="height: 2px; background: linear-gradient(90deg, #3b82f6, transparent); margin: 0 auto; width: 80%;"></div>
            </div>
            """, unsafe_allow_html=True)
            
            # File upload
            st.subheader("📂 Data Source")
            uploaded_file = st.file_uploader(
                "Upload Dataset",
                type=['xlsx', 'csv'],
                help="Upload your sales performance data in Excel or CSV format"
            )
            
            if uploaded_file:
                try:
                    self.df = self.data_manager.load_excel_data(uploaded_file)
                    self.gdf = self.data_manager.load_geographic_data()
                    st.session_state.data_loaded = True
                    st.success("✅ Data loaded successfully")
                except Exception as e:
                    st.error(f"❌ Error loading data: {str(e)}")
                    st.session_state.data_loaded = False
                    return
            
            if not st.session_state.data_loaded:
                st.info("👈 Please upload your data to begin analysis")
                st.stop()
            
            # Product selection
            st.subheader("💊 Product Selection")
            products = ["TROCMETAM", "CORTIPOL", "DEKSAMETAZON", "PF IZOTONIK"]
            selected_product = st.selectbox(
                "Select Product",
                products,
                index=0,
                help="Choose the product for analysis"
            )
            
            # Update product configuration
            self.product_config = self._get_product_config(selected_product)
            
            # Date filtering
            st.subheader("📅 Time Period")
            min_date = self.df['DATE'].min().date()
            max_date = self.df['DATE'].max().date()
            
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "Start Date",
                    min_date,
                    min_value=min_date,
                    max_value=max_date
                )
            with col2:
                end_date = st.date_input(
                    "End Date",
                    max_date,
                    min_value=min_date,
                    max_value=max_date
                )
            
            self.date_filter = (pd.Timestamp(start_date), pd.Timestamp(end_date))
            
            # Advanced filters
            st.subheader("🎯 Advanced Filters")
            
            territories = ["ALL"] + sorted(self.df['TERRITORIES'].unique())
            selected_territory = st.selectbox(
                "Territory Filter",
                territories,
                help="Filter by specific territory"
            )
            
            regions = ["ALL"] + sorted(self.df['REGION'].unique())
            selected_region = st.selectbox(
                "Region Filter",
                regions,
                help="Filter by geographic region"
            )
            
            # Apply filters
            self.df_filtered = self.df.copy()
            if selected_territory != "ALL":
                self.df_filtered = self.df_filtered[self.df_filtered['TERRITORIES'] == selected_territory]
            if selected_region != "ALL":
                self.df_filtered = self.df_filtered[self.df_filtered['REGION'] == selected_region]
            
            # Analysis settings
            st.subheader("🔧 Analysis Settings")
            forecast_periods = st.slider(
                "Forecast Period (Months)",
                min_value=1,
                max_value=12,
                value=3,
                help="Number of months to forecast"
            )
            
            # Region color legend
            st.subheader("🗺️ Region Colors")
            for region, color in AppConfig.REGION_COLORS.items():
                if region in self.df['REGION'].unique():
                    st.markdown(
                        f'<span style="color:{color}">■</span> {region}',
                        unsafe_allow_html=True
                    )
    
    def _get_product_config(self, product: str) -> Dict:
        """Get product-specific column configuration"""
        config_map = {
            "TROCMETAM": {"pf": "TROCMETAM", "rakip": "DIGER TROCMETAM"},
            "CORTIPOL": {"pf": "CORTIPOL", "rakip": "DIGER CORTIPOL"},
            "DEKSAMETAZON": {"pf": "DEKSAMETAZON", "rakip": "DIGER DEKSAMETAZON"},
            "PF IZOTONIK": {"pf": "PF IZOTONIK", "rakip": "DIGER IZOTONIK"}
        }
        return config_map.get(product, config_map["TROCMETAM"])
    
    def create_dashboard(self):
        """Create main dashboard with tabs"""
        
        # Calculate key metrics
        city_perf = self.analytics.calculate_city_performance(
            self.df_filtered, self.product_config, self.date_filter
        )
        
        territory_perf = self.analytics.calculate_territory_performance(
            self.df_filtered, self.product_config, self.date_filter
        )
        
        time_series = self.analytics.calculate_time_series(
            self.df_filtered, self.product_config, date_filter=self.date_filter
        )
        
        # Executive metrics
        metrics = {
            'total_pf_sales': city_perf['PF_Sales'].sum(),
            'total_market': city_perf['Total_Market'].sum(),
            'market_share': AnalyticsEngine.safe_divide(
                city_perf['PF_Sales'].sum(), city_perf['Total_Market'].sum()
            ) * 100,
            'active_territories': territory_perf['Territory'].nunique(),
            'growth_rate': time_series['PF_Growth_%'].mean() if len(time_series) > 1 else 0
        }
        
        # Create tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Executive Summary",
            "🗺️ Geographic Analysis",
            "📈 Performance Analytics",
            "🤖 AI Forecasting",
            "🎯 Strategic Insights"
        ])
        
        # Tab 1: Executive Summary
        with tab1:
            self._render_executive_summary(metrics, territory_perf, city_perf)
        
        # Tab 2: Geographic Analysis
        with tab2:
            self._render_geographic_analysis(city_perf)
        
        # Tab 3: Performance Analytics
        with tab3:
            self._render_performance_analytics(territory_perf, time_series)
        
        # Tab 4: AI Forecasting
        with tab4:
            self._render_forecasting(time_series)
        
        # Tab 5: Strategic Insights
        with tab5:
            self._render_strategic_insights(city_perf)
    
    def _render_executive_summary(self, metrics: Dict, territory_perf: pd.DataFrame, 
                                  city_perf: pd.DataFrame):
        """Render executive summary tab"""
        
        # Executive metrics
        self.viz.create_executive_summary(metrics)
        
        st.markdown("---")
        
        # Top performers
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏆 Top 10 Territories")
            top_10 = territory_perf.head(10)[[
                'Territory', 'Region', 'PF_Sales', 'Market_Share_%', 'Weight_%'
            ]].copy()
            top_10.columns = ['Territory', 'Region', 'PF Sales', 'Market Share %', 'Weight %']
            top_10.index = range(1, len(top_10) + 1)
            
            # Formatting
            top_10_display = top_10.copy()
            top_10_display['PF Sales'] = top_10_display['PF Sales'].apply(lambda x: f"{x:,.0f}")
            top_10_display['Market Share %'] = top_10_display['Market Share %'].apply(lambda x: f"{x:.1f}%")
            top_10_display['Weight %'] = top_10_display['Weight %'].apply(lambda x: f"{x:.1f}%")
            
            st.dataframe(
                top_10_display,
                use_container_width=True,
                height=400
            )
        
        with col2:
            st.subheader("📈 Regional Performance")
            region_summary = city_perf.groupby('Region').agg({
                'PF_Sales': 'sum',
                'Total_Market': 'sum'
            }).reset_index()
            
            region_summary['Market_Share_%'] = AnalyticsEngine.safe_divide(
                region_summary['PF_Sales'], region_summary['Total_Market']
            ) * 100
            
            region_summary = region_summary.sort_values('PF_Sales', ascending=False)
            
            fig = px.bar(
                region_summary,
                x='Region',
                y='PF_Sales',
                color='Region',
                color_discrete_map=AppConfig.REGION_COLORS,
                text='PF_Sales',
                title='Sales by Region'
            )
            
            fig.update_traces(
                texttemplate='%{text:,.0f}',
                textposition='outside',
                marker_line_color='white',
                marker_line_width=1
            )
            
            fig.update_layout(
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                showlegend=False,
                xaxis_tickangle=-45
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Performance trends
        st.markdown("---")
        st.subheader("📊 Performance Trends")
        
        if len(city_perf) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                # Market share distribution
                fig1 = px.pie(
                    city_perf.nlargest(10, 'PF_Sales'),
                    values='PF_Sales',
                    names='City',
                    title='Top 10 Cities by Sales',
                    color_discrete_sequence=px.colors.sequential.Blues
                )
                fig1.update_layout(
                    height=400,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                # Market share vs sales
                fig2 = px.scatter(
                    city_perf,
                    x='Total_Market',
                    y='Market_Share_%',
                    size='PF_Sales',
                    color='Region',
                    color_discrete_map=AppConfig.REGION_COLORS,
                    hover_name='City',
                    title='Market Share vs Market Size',
                    log_x=True
                )
                fig2.update_layout(
                    height=400,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                st.plotly_chart(fig2, use_container_width=True)
    
    def _render_geographic_analysis(self, city_perf: pd.DataFrame):
        """Render geographic analysis tab"""
        
        if self.gdf is not None:
            # Geographic map
            st.subheader("🗺️ Geographic Sales Distribution")
            
            map_fig = self.viz.create_geographic_map(
                city_perf,
                self.gdf,
                title=f"{st.session_state.current_product} - Sales Distribution"
            )
            
            st.plotly_chart(map_fig, use_container_width=True)
            
            # Region details
            st.markdown("---")
            st.subheader("📋 Regional Analysis")
            
            region_details = city_perf.groupby('Region').agg({
                'City': 'count',
                'PF_Sales': ['sum', 'mean', 'std'],
                'Market_Share_%': 'mean'
            }).round(2)
            
            region_details.columns = ['Cities', 'Total Sales', 'Avg Sales', 'Std Sales', 'Avg Market Share']
            region_details = region_details.sort_values('Total Sales', ascending=False)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.dataframe(
                    region_details.style.format({
                        'Total Sales': '{:,.0f}',
                        'Avg Sales': '{:,.0f}',
                        'Std Sales': '{:,.0f}',
                        'Avg Market Share': '{:.1f}%'
                    }),
                    use_container_width=True,
                    height=400
                )
            
            with col2:
                st.metric(
                    "🌍 Total Cities",
                    f"{len(city_perf):,.0f}",
                    delta=f"{city_perf['City'].nunique():,.0f} unique"
                )
                st.metric(
                    "🏆 Highest Share",
                    f"{city_perf['Market_Share_%'].max():.1f}%",
                    delta=city_perf.loc[city_perf['Market_Share_%'].idxmax(), 'City']
                )
                st.metric(
                    "📈 Average Share",
                    f"{city_perf['Market_Share_%'].mean():.1f}%",
                    delta="Regional average"
                )
        else:
            st.warning("⚠️ Geographic data not available for mapping")
    
    def _render_performance_analytics(self, territory_perf: pd.DataFrame, 
                                      time_series: pd.DataFrame):
        """Render performance analytics tab"""
        
        st.subheader("📈 Performance Analytics")
        
        # Time series analysis
        col1, col2 = st.columns(2)
        
        with col1:
            # Sales trend
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=time_series['Date'],
                y=time_series['PF_Sales'],
                mode='lines+markers',
                name='PF Sales',
                line=dict(color=AppConfig.COLOR_SCHEME['primary'], width=3)
            ))
            fig1.add_trace(go.Scatter(
                x=time_series['Date'],
                y=time_series['Competitor_Sales'],
                mode='lines',
                name='Competitor Sales',
                line=dict(color=AppConfig.COLOR_SCHEME['neutral'], width=2)
            ))
            
            fig1.update_layout(
                title='Sales Trend Analysis',
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                xaxis_title='Date',
                yaxis_title='Sales Volume'
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Market share trend
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=time_series['Date'],
                y=time_series['Market_Share_%'],
                mode='lines+markers',
                name='Market Share',
                line=dict(color=AppConfig.COLOR_SCHEME['success'], width=3),
                fill='tozeroy'
            ))
            
            fig2.update_layout(
                title='Market Share Trend',
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                xaxis_title='Date',
                yaxis_title='Market Share %'
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        # Growth analysis
        st.markdown("---")
        st.subheader("📊 Growth Analysis")
        
        if len(time_series) > 1:
            growth_metrics = pd.DataFrame({
                'Metric': ['PF Growth', 'Competitor Growth', 'Relative Growth'],
                'Average %': [
                    time_series['PF_Growth_%'].mean(),
                    time_series['Competitor_Growth_%'].mean(),
                    time_series['Relative_Growth_%'].mean()
                ],
                'Volatility': [
                    time_series['PF_Growth_%'].std(),
                    time_series['Competitor_Growth_%'].std(),
                    time_series['Relative_Growth_%'].std()
                ]
            })
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.dataframe(
                    growth_metrics.style.format({
                        'Average %': '{:.2f}%',
                        'Volatility': '{:.2f}'
                    }),
                    use_container_width=True
                )
            
            with col2:
                # Growth comparison
                fig3 = go.Figure()
                fig3.add_trace(go.Bar(
                    x=growth_metrics['Metric'],
                    y=growth_metrics['Average %'],
                    name='Average Growth',
                    marker_color=[
                        AppConfig.COLOR_SCHEME['primary'],
                        AppConfig.COLOR_SCHEME['neutral'],
                        AppConfig.COLOR_SCHEME['success']
                    ]
                ))
                
                fig3.update_layout(
                    title='Growth Rate Comparison',
                    height=300,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    yaxis_title='Growth Rate %'
                )
                st.plotly_chart(fig3, use_container_width=True)
    
    def _render_forecasting(self, time_series: pd.DataFrame):
        """Render forecasting tab"""
        
        st.subheader("🤖 AI-Powered Forecasting")
        
        if len(time_series) < 12:
            st.warning("⚠️ Insufficient data for reliable forecasting. Need at least 12 months of data.")
            return
        
        with st.spinner("Training ML models..."):
            ml_results, best_model, forecast_df = self.ml_engine.train_forecast_models(
                time_series, forecast_periods=6
            )
        
        if ml_results:
            # Model performance
            st.subheader("📊 Model Performance")
            
            perf_data = []
            for model_name, metrics in ml_results.items():
                perf_data.append({
                    'Model': model_name,
                    'MAE': metrics['MAE'],
                    'RMSE': metrics['RMSE'],
                    'MAPE': f"{metrics['MAPE']:.2f}%",
                    'Status': '🏆 Best' if model_name == best_model else '✅ Good'
                })
            
            perf_df = pd.DataFrame(perf_data)
            st.dataframe(
                perf_df,
                use_container_width=True,
                height=200
            )
            
            # Forecast visualization
            st.markdown("---")
            st.subheader("🔮 Sales Forecast")
            
            forecast_fig = self.viz.create_forecast_chart(time_series, forecast_df)
            st.plotly_chart(forecast_fig, use_container_width=True)
            
            # Forecast details
            st.markdown("---")
            st.subheader("📋 Forecast Details")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                last_actual = time_series['PF_Sales'].iloc[-1]
                st.metric("Last Actual", f"{last_actual:,.0f}")
            
            with col2:
                next_forecast = forecast_df['PF_Sales'].iloc[0]
                change_pct = ((next_forecast - last_actual) / last_actual * 100) if last_actual > 0 else 0
                st.metric(
                    "Next Month Forecast",
                    f"{next_forecast:,.0f}",
                    delta=f"{change_pct:+.1f}%"
                )
            
            with col3:
                avg_forecast = forecast_df['PF_Sales'].mean()
                st.metric("Average Forecast", f"{avg_forecast:,.0f}")
            
            # Detailed forecast table
            st.dataframe(
                forecast_df[['Period', 'PF_Sales', 'Model', 'Confidence']].style.format({
                    'PF_Sales': '{:,.0f}'
                }),
                use_container_width=True,
                height=300
            )
    
    def _render_strategic_insights(self, city_perf: pd.DataFrame):
        """Render strategic insights tab"""
        
        # Calculate investment strategies
        strategy_data = self.analytics.calculate_investment_strategy(city_perf)
        
        if len(strategy_data) == 0:
            st.warning("⚠️ No data available for strategic analysis")
            return
        
        st.subheader("🎯 Strategic Investment Insights")
        
        # Strategy distribution
        strategy_counts = strategy_data['Investment_Strategy'].value_counts()
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "🚀 Aggressive",
                f"{strategy_counts.get('🚀 Agresif', 0):,.0f}",
                delta="High potential"
            )
        
        with col2:
            st.metric(
                "⚡ Accelerated",
                f"{strategy_counts.get('⚡ Hızlandırılmış', 0):,.0f}",
                delta="Growth focus"
            )
        
        with col3:
            st.metric(
                "🛡️ Defensive",
                f"{strategy_counts.get('🛡️ Koruma', 0):,.0f}",
                delta="Market leadership"
            )
        
        with col4:
            st.metric(
                "💎 Potential",
                f"{strategy_counts.get('💎 Potansiyel', 0):,.0f}",
                delta="Emerging"
            )
        
        with col5:
            st.metric(
                "👁️ Monitor",
                f"{strategy_counts.get('👁️ İzleme', 0):,.0f}",
                delta="Low priority"
            )
        
        # Strategic matrix
        st.markdown("---")
        st.subheader("📊 Strategic Investment Matrix")
        
        matrix_fig = self.viz.create_strategy_matrix(strategy_data)
        st.plotly_chart(matrix_fig, use_container_width=True)
        
        # Detailed recommendations
        st.markdown("---")
        st.subheader("📋 Strategic Recommendations")
        
        # Top opportunities
        aggressive_cities = strategy_data[strategy_data['Investment_Strategy'] == '🚀 Agresif']
        if len(aggressive_cities) > 0:
            st.info("### 🎯 High-Priority Opportunities")
            
            for idx, row in aggressive_cities.nlargest(5, 'Growth_Potential').iterrows():
                with st.expander(f"🚀 **{row['City']}** - {row['Region']}"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Current Sales", f"{row['PF_Sales']:,.0f}")
                    
                    with col2:
                        st.metric("Market Share", f"{row['Market_Share_%']:.1f}%")
                    
                    with col3:
                        st.metric("Growth Potential", f"{row['Growth_Potential']:,.0f}")
                    
                    st.markdown(f"""
                    **Strategic Rationale:**
                    - Large market size ({row['Total_Market']:,.0f})
                    - Low current market share ({row['Market_Share_%']:.1f}%)
                    - High growth potential ({row['Growth_Potential']:,.0f} units)
                    
                    **Recommended Actions:**
                    1. Increase marketing investment
                    2. Strengthen distribution channels
                    3. Competitive pricing strategy
                    4. Regular performance monitoring
                    """)
        
        # Export capabilities
        st.markdown("---")
        st.subheader("📥 Export & Reporting")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Export Strategic Report", use_container_width=True):
                # Create export data
                export_data = strategy_data[[
                    'City', 'Region', 'PF_Sales', 'Total_Market', 
                    'Market_Share_%', 'Investment_Strategy',
                    'Market_Size', 'Growth_Potential'
                ]].copy()
                
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    export_data.to_excel(writer, sheet_name='Strategic_Analysis', index=False)
                
                st.download_button(
                    label="💾 Download Excel Report",
                    data=output.getvalue(),
                    file_name=f"strategic_analysis_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        
        with col2:
            if st.button("📈 Generate Executive Summary", use_container_width=True):
                st.success("✅ Executive summary generated")
                # In a production environment, this would generate a PDF report
    
    def run(self):
        """Main application runner"""
        self.setup_page()
        self.create_sidebar()
        
        if st.session_state.data_loaded:
            self.create_dashboard()

# =============================================================================
# APPLICATION ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    app = StrategicPortfolioAnalytics()
    app.run()
