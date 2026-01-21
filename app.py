"""
🎯 STRATEJİK TİCARİ PORTFÖY ANALİZ SİSTEMİ - YÖNETİCİ KARAR DESTEK SİSTEMİ
McKinsey/BCG Tarzı, Kurumsal Seviye, Nesne Yönelimli Tasarım

Tasarım Felsefesi:
- Profesyonel renk paleti (Lacivert, Zümrüt Yeşili, Arduvaz Grisi)
- Modüler OOP Mimarisi
- Defansif Programlama
- Tam kapsamlı dokümantasyon
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod

# =============================================================================
# KONFİGÜRASYON SINIFI
# =============================================================================

@dataclass
class AppConfig:
    """Uygulama sabitlerini ve konfigürasyonlarını yönetir"""
    
    # Renk Paleti - McKinsey/BCG Tarzı
    COLOR_PALETTE = {
        "primary_dark": "#0F1729",        # Lacivert
        "primary_medium": "#1E293B",      # Orta Lacivert
        "primary_light": "#334155",       # Açık Lacivert
        "success_dark": "#065F46",        # Zümrüt Yeşili - Koyu
        "success_medium": "#059669",      # Zümrüt Yeşili
        "success_light": "#10B981",       # Zümrüt Yeşili - Açık
        "warning_dark": "#92400E",        # Amber - Koyu
        "warning_medium": "#D97706",      # Amber
        "warning_light": "#F59E0B",       # Amber - Açık
        "danger_dark": "#7F1D1D",         # Bordo
        "danger_medium": "#DC2626",       # Kırmızı
        "danger_light": "#EF4444",        # Kırmızı - Açık
        "neutral_dark": "#374151",        # Arduvaz Grisi - Koyu
        "neutral_medium": "#6B7280",      # Arduvaz Grisi
        "neutral_light": "#9CA3AF",       # Arduvaz Grisi - Açık
        "background_dark": "#0F1729",     # Arkaplan - Koyu
        "background_medium": "#1E293B",   # Arkaplan - Orta
        "background_light": "#334155",    # Arkaplan - Açık
        "text_primary": "#F8FAFC",        # Ana Metin
        "text_secondary": "#CBD5E1",      # İkincil Metin
        "text_muted": "#94A3B8",          # Soluk Metin
        "white": "#FFFFFF",               # Beyaz
        "black": "#000000"                # Siyah
    }
    
    # Bölge Renkleri
    REGION_COLORS = {
        "MARMARA": "#3B82F6",        # Lacivert Mavi
        "BATI ANADOLU": "#10B981",   # Zümrüt Yeşili
        "EGE": "#F59E0B",           # Amber
        "İÇ ANADOLU": "#8B5CF6",    # Mor
        "GÜNEY DOĞU ANADOLU": "#EF4444",  # Kırmızı
        "KUZEY ANADOLU": "#06B6D4", # Turkuaz
        "KARADENİZ": "#06B6D4",     # Turkuaz
        "AKDENİZ": "#8B5CF6",       # Mor
        "DOĞU ANADOLU": "#7C3AED",  # Koyu Mor
        "DİĞER": "#64748B"          # Gri
    }
    
    # BCG Matrix Renkleri
    BCG_COLORS = {
        "⭐ Star": "#F59E0B",        # Turuncu - Yüksek Büyüme, Yüksek Pay
        "🐄 Cash Cow": "#10B981",    # Yeşil - Düşük Büyüme, Yüksek Pay
        "❓ Question Mark": "#3B82F6", # Mavi - Yüksek Büyüme, Düşük Pay
        "🐶 Dog": "#64748B"          # Gri - Düşük Büyüme, Düşük Pay
    }
    
    # Yatırım Stratejisi Renkleri
    STRATEGY_COLORS = {
        "🚀 Agresif": "#EF4444",      # Kırmızı
        "⚡ Hızlandırılmış": "#F59E0B", # Turuncu
        "🛡️ Koruma": "#10B981",       # Yeşil
        "💎 Potansiyel": "#3B82F6",    # Mavi
        "👁️ İzleme": "#64748B"        # Gri
    }
    
    # Gradyan Skalaları
    GRADIENT_SCALES = {
        "sequential_blue": ["#DBEAFE", "#BFDBFE", "#93C5FD", "#60A5FA", "#3B82F6"],
        "sequential_green": ["#D1FAE5", "#A7F3D0", "#6EE7B7", "#34D399", "#10B981"],
        "diverging_red_blue": ["#EF4444", "#F59E0B", "#10B981", "#3B82F6", "#8B5CF6"],
        "temperature": ["#3B82F6", "#60A5FA", "#93C5FD", "#BFDBFE", "#DBEAFE"]
    }
    
    # Performans Eşikleri
    PERFORMANCE_THRESHOLDS = {
        "market_share_low": 20,      # Düşük pazar payı eşiği (%)
        "market_share_high": 50,     # Yüksek pazar payı eşiği (%)
        "growth_low": 5,             # Düşük büyüme eşiği (%)
        "growth_high": 15,           # Yüksek büyüme eşiği (%)
        "performance_score_low": 40, # Düşük performans skoru
        "performance_score_medium": 60, # Orta performans skoru
        "performance_score_high": 80  # Yüksek performans skoru
    }
    
    # Şehir Normalizasyon Haritası
    CITY_NORMALIZATION_MAP = {
        'ADANA': 'Adana',
        'ADIYAMAN': 'Adiyaman',
        'AFYONKARAHISAR': 'Afyonkarahisar',
        'AFYON': 'Afyonkarahisar',
        'AGRI': 'Agri',
        'AĞRI': 'Agri',
        'AKSARAY': 'Aksaray',
        'ANKARA': 'Ankara',
        'ANTALYA': 'Antalya',
        'AYDIN': 'Aydin',
        'BALIKESIR': 'Balikesir',
        'BARTIN': 'Bartin',
        'BATMAN': 'Batman',
        'BILECIK': 'Bilecik',
        'BINGOL': 'Bingol',
        'BITLIS': 'Bitlis',
        'BOLU': 'Bolu',
        'BURDUR': 'Burdur',
        'BURSA': 'Bursa',
        'CANAKKALE': 'Canakkale',
        'ÇANAKKALE': 'Canakkale',
        'CANKIRI': 'Cankiri',
        'ÇANKIRI': 'Cankiri',
        'CORUM': 'Corum',
        'ÇORUM': 'Corum',
        'DENIZLI': 'Denizli',
        'DIYARBAKIR': 'Diyarbakir',
        'DUZCE': 'Duzce',
        'DÜZCE': 'Duzce',
        'EDIRNE': 'Edirne',
        'ELAZIG': 'Elazig',
        'ELAZIĞ': 'Elazig',
        'ERZINCAN': 'Erzincan',
        'ERZURUM': 'Erzurum',
        'ESKISEHIR': 'Eskisehir',
        'ESKİŞEHİR': 'Eskisehir',
        'GAZIANTEP': 'Gaziantep',
        'GIRESUN': 'Giresun',
        'GİRESUN': 'Giresun',
        'GUMUSHANE': 'Gumushane',
        'GÜMÜŞHANE': 'Gumushane',
        'HAKKARI': 'Hakkari',
        'HAKKARİ': 'Hakkari',
        'HATAY': 'Hatay',
        'IGDIR': 'Igdir',
        'IĞDIR': 'Igdir',
        'ISPARTA': 'Isparta',
        'ISTANBUL': 'Istanbul',
        'İSTANBUL': 'Istanbul',
        'IZMIR': 'Izmir',
        'İZMİR': 'Izmir',
        'KAHRAMANMARAS': 'K. Maras',
        'KAHRAMANMARAŞ': 'K. Maras',
        'K.MARAS': 'K. Maras',
        'KMARAS': 'K. Maras',
        'KARABUK': 'Karabuk',
        'KARABÜK': 'Karabuk',
        'KARAMAN': 'Karaman',
        'KARS': 'Kars',
        'KASTAMONU': 'Kastamonu',
        'KAYSERI': 'Kayseri',
        'KIRIKKALE': 'Kinkkale',
        'KIRKLARELI': 'Kirklareli',
        'KIRKLARELİ': 'Kirklareli',
        'KIRSEHIR': 'Kirsehir',
        'KIRŞEHİR': 'Kirsehir',
        'KILIS': 'Kilis',
        'KİLİS': 'Kilis',
        'KOCAELI': 'Kocaeli',
        'KONYA': 'Konya',
        'KUTAHYA': 'Kutahya',
        'KÜTAHYA': 'Kutahya',
        'MALATYA': 'Malatya',
        'MANISA': 'Manisa',
        'MANİSA': 'Manisa',
        'MARDIN': 'Mardin',
        'MARDİN': 'Mardin',
        'MERSIN': 'Mersin',
        'MERSİN': 'Mersin',
        'MUGLA': 'Mugla',
        'MUĞLA': 'Mugla',
        'MUS': 'Mus',
        'MUŞ': 'Mus',
        'NEVSEHIR': 'Nevsehir',
        'NEVŞEHİR': 'Nevsehir',
        'NIGDE': 'Nigde',
        'NİĞDE': 'Nigde',
        'ORDU': 'Ordu',
        'OSMANIYE': 'Osmaniye',
        'OSMANİYE': 'Osmaniye',
        'RIZE': 'Rize',
        'RİZE': 'Rize',
        'SAKARYA': 'Sakarya',
        'SAMSUN': 'Samsun',
        'SIIRT': 'Siirt',
        'SİİRT': 'Siirt',
        'SINOP': 'Sinop',
        'SİNOP': 'Sinop',
        'SIVAS': 'Sivas',
        'SİVAS': 'Sivas',
        'SANLIURFA': 'Sanliurfa',
        'ŞANLIURFA': 'Sanliurfa',
        'SIRNAK': 'Sirnak',
        'ŞIRNAK': 'Sirnak',
        'TEKIRDAG': 'Tekirdag',
        'TEKİRDAĞ': 'Tekirdag',
        'TOKAT': 'Tokat',
        'TRABZON': 'Trabzon',
        'TUNCELI': 'Tunceli',
        'TUNCELİ': 'Tunceli',
        'USAK': 'Usak',
        'UŞAK': 'Usak',
        'VAN': 'Van',
        'YALOVA': 'Yalova',
        'YOZGAT': 'Yozgat',
        'ZONGULDAK': 'Zonguldak',
        'ARDAHAN': 'Ardahan',
        'AKSARAY': 'Aksaray',
        'KIRIKKALE': 'Kirikkale'
    }
    
    # Ürün Kolon Haritası
    PRODUCT_COLUMN_MAP = {
        "TROCMETAM": {"pf": "TROCMETAM", "rakip": "DIGER TROCMETAM"},
        "CORTIPOL": {"pf": "CORTIPOL", "rakip": "DIGER CORTIPOL"},
        "DEKSAMETAZON": {"pf": "DEKSAMETAZON", "rakip": "DIGER DEKSAMETAZON"},
        "PF IZOTONIK": {"pf": "PF IZOTONIK", "rakip": "DIGER IZOTONIK"}
    }
    
    # Tarih Seçenekleri
    DATE_OPTIONS = [
        "Tüm Veriler",
        "Son 3 Ay",
        "Son 6 Ay",
        "Son 1 Yıl",
        "2025",
        "2024",
        "Özel Aralık"
    ]
    
    # ML Model Parametreleri
    ML_PARAMS = {
        "forecast_periods": 3,
        "test_size": 0.2,
        "random_state": 42,
        "n_estimators": 100,
        "max_depth": 5,
        "ridge_alpha": 1.0
    }


# =============================================================================
# SOYUT TEMEL SINIFLAR
# =============================================================================

class BaseDataProcessor(ABC):
    """Veri işleme için soyut temel sınıf"""
    
    @abstractmethod
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        pass


class BaseVisualizer(ABC):
    """Görselleştirme için soyut temel sınıf"""
    
    @abstractmethod
    def create_visualization(self, data: pd.DataFrame, **kwargs) -> go.Figure:
        pass


class BaseAnalyzer(ABC):
    """Analiz için soyut temel sınıf"""
    
    @abstractmethod
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        pass


# =============================================================================
# ŞEHİR NORMALİZASYON SINIFI
# =============================================================================

class CityNormalizer:
    """
    Şehir isimlerini normalleştirir ve standardize eder.
    Türkiye'nin 81 ili için tüm yazım varyasyonlarını işler.
    """
    
    def __init__(self):
        self.config = AppConfig()
        self._setup_normalization_maps()
    
    def _setup_normalization_maps(self) -> None:
        """Normalizasyon haritalarını kur"""
        self.city_map = self.config.CITY_NORMALIZATION_MAP
        
        # Türkçe karakter dönüşümü
        self.turkish_char_map = {
            "İ": "I", "Ğ": "G", "Ü": "U", "Ş": "S", 
            "Ö": "O", "Ç": "C", "Â": "A", "Î": "I", "Û": "U"
        }
    
    def normalize(self, city_name: str) -> str:
        """
        Şehir ismini normalize eder
        
        Args:
            city_name (str): Normalize edilecek şehir ismi
            
        Returns:
            str: Normalize edilmiş şehir ismi
        """
        if pd.isna(city_name):
            return "Bilinmeyen"
        
        try:
            # String'e çevir ve temizle
            city_str = str(city_name).strip().upper()
            
            # Doğrudan eşleşme
            if city_str in self.city_map:
                return self.city_map[city_str]
            
            # Türkçe karakterleri dönüştür
            for turkish_char, latin_char in self.turkish_char_map.items():
                city_str = city_str.replace(turkish_char, latin_char)
            
            # Normalize edilmiş eşleşme
            if city_str in self.city_map:
                return self.city_map[city_str]
            
            # Kısmi eşleşme kontrolü
            for key, value in self.city_map.items():
                if city_str in key or key in city_str:
                    return value
            
            return city_str
            
        except Exception as e:
            logging.warning(f"Şehir normalizasyon hatası: {e}, Şehir: {city_name}")
            return city_name
    
    def normalize_dataframe(self, df: pd.DataFrame, column_name: str = "CITY") -> pd.DataFrame:
        """
        DataFrame'deki şehir kolonunu normalize eder
        
        Args:
            df (pd.DataFrame): İşlenecek DataFrame
            column_name (str): Şehir kolonu adı
            
        Returns:
            pd.DataFrame: Normalize edilmiş DataFrame
        """
        df = df.copy()
        
        if column_name in df.columns:
            df[f"{column_name}_NORMALIZED"] = df[column_name].apply(self.normalize)
        
        return df


# =============================================================================
# VERİ İŞLEYİCİ SINIFI
# =============================================================================

class DataProcessor(BaseDataProcessor):
    """
    Veri temizleme, dönüştürme ve hazırlama işlemlerini yönetir
    """
    
    def __init__(self):
        self.config = AppConfig()
        self.city_normalizer = CityNormalizer()
    
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Ham veriyi analize hazır hale getirir
        
        Args:
            data (pd.DataFrame): Ham veri
            
        Returns:
            pd.DataFrame: İşlenmiş veri
        """
        try:
            df = data.copy()
            
            # Temel temizlik
            df = self._clean_basic(df)
            
            # Tarih işlemleri
            df = self._process_dates(df)
            
            # Şehir normalizasyonu
            df = self.city_normalizer.normalize_dataframe(df)
            
            # Metin kolonlarını standartlaştır
            df = self._standardize_text_columns(df)
            
            # Hareketli ortalamaları hesapla
            df = self._calculate_moving_averages(df)
            
            # Yıllık büyümeyi hesapla
            df = self._calculate_yoy_growth(df)
            
            return df
            
        except Exception as e:
            logging.error(f"Veri işleme hatası: {e}")
            raise
    
    def _clean_basic(self, df: pd.DataFrame) -> pd.DataFrame:
        """Temel veri temizliği"""
        # Boş değerleri temizle
        df = df.dropna(subset=['DATE', 'TERRITORIES', 'CITY'])
        
        # Sayısal kolonları doldur
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        return df
    
    def _process_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tarih işlemleri"""
        df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
        df['YIL_AY'] = df['DATE'].dt.strftime('%Y-%m')
        df['AY'] = df['DATE'].dt.month
        df['YIL'] = df['DATE'].dt.year
        df['QUARTER'] = df['DATE'].dt.quarter
        df['HAFTA'] = df['DATE'].dt.isocalendar().week
        
        return df
    
    def _standardize_text_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Metin kolonlarını standartlaştır"""
        text_columns = ['TERRITORIES', 'REGION', 'MANAGER']
        
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.upper().str.strip()
        
        return df
    
    def _calculate_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Hareketli ortalamaları hesapla"""
        # Ürün kolonlarını bul
        product_cols = []
        for product in self.config.PRODUCT_COLUMN_MAP:
            cols = self.config.PRODUCT_COLUMN_MAP[product]
            product_cols.extend([cols['pf'], cols['rakip']])
        
        # Benzersiz kolonları al
        product_cols = list(set(product_cols))
        
        # Her territory için hareketli ortalama hesapla
        for col in product_cols:
            if col in df.columns:
                df[f'{col}_MA3'] = df.groupby('TERRITORIES')[col].transform(
                    lambda x: x.rolling(window=3, min_periods=1).mean()
                )
                df[f'{col}_MA6'] = df.groupby('TERRITORIES')[col].transform(
                    lambda x: x.rolling(window=6, min_periods=1).mean()
                )
        
        return df
    
    def _calculate_yoy_growth(self, df: pd.DataFrame) -> pd.DataFrame:
        """Yıllık büyümeyi hesapla"""
        # Her territory ve ürün için yıllık büyüme
        for product in self.config.PRODUCT_COLUMN_MAP:
            cols = self.config.PRODUCT_COLUMN_MAP[product]
            pf_col = cols['pf']
            
            if pf_col in df.columns:
                # Yıllık toplamları hesapla
                yearly_sales = df.groupby(['TERRITORIES', 'YIL'])[pf_col].sum().reset_index()
                
                # Yıllık büyümeyi hesapla
                yearly_sales[f'{pf_col}_YOY'] = yearly_sales.groupby('TERRITORIES')[pf_col].pct_change() * 100
                
                # DataFrame'e birleştir
                df = df.merge(
                    yearly_sales[['TERRITORIES', 'YIL', f'{pf_col}_YOY']],
                    on=['TERRITORIES', 'YIL'],
                    how='left'
                )
        
        return df


# =============================================================================
# HARİTA MOTORU SINIFI
# =============================================================================

class MapEngine(BaseVisualizer):
    """
    Hiyerarşik harita görselleştirmeleri oluşturur
    """
    
    def __init__(self, geojson_path: str = "turkey.geojson"):
        """
        Args:
            geojson_path (str): GeoJSON dosya yolu
        """
        self.config = AppConfig()
        self.geojson_path = geojson_path
        self.geojson_data = self._load_geojson()
    
    def _load_geojson(self) -> Optional[Dict]:
        """GeoJSON verisini yükle"""
        try:
            with open(self.geojson_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"GeoJSON yükleme hatası: {e}")
            return None
    
    def create_visualization(self, 
                           city_data: pd.DataFrame, 
                           view_mode: str = "Bölge Görünümü",
                           title: str = "Türkiye Satış Haritası") -> Optional[go.Figure]:
        """
        Harita görselleştirmesi oluşturur
        
        Args:
            city_data (pd.DataFrame): Şehir bazlı veri
            view_mode (str): Görünüm modu (Bölge/Şehir)
            title (str): Harita başlığı
            
        Returns:
            Optional[go.Figure]: Oluşturulan harita
        """
        if self.geojson_data is None:
            logging.error("GeoJSON verisi yüklenemedi")
            return None
        
        try:
            # Veriyi hazırla
            prepared_data = self._prepare_map_data(city_data)
            
            # Harita türüne göre oluştur
            if view_mode == "Bölge Görünümü":
                fig = self._create_region_map(prepared_data, title)
            else:
                fig = self._create_city_map(prepared_data, title)
            
            # Layout'u güncelle
            fig = self._update_map_layout(fig, title)
            
            return fig
            
        except Exception as e:
            logging.error(f"Harita oluşturma hatası: {e}")
            return None
    
    def _prepare_map_data(self, city_data: pd.DataFrame) -> pd.DataFrame:
        """Harita verisini hazırla"""
        city_data = city_data.copy()
        
        # Şehir isimlerini normalleştir
        normalizer = CityNormalizer()
        city_data['CITY_NORMALIZED'] = city_data['City'].apply(normalizer.normalize)
        
        # GeoJSON'daki tüm şehirleri al
        gdf = gpd.read_file(self.geojson_path)
        gdf['name_upper'] = gdf['name'].str.upper()
        
        # Şehir isimlerini düzelt
        for idx, row in gdf.iterrows():
            normalized = normalizer.normalize(row['name'])
            gdf.at[idx, 'name_normalized'] = normalized
        
        # Birleştir
        merged = gdf.merge(
            city_data,
            left_on='name_normalized',
            right_on='CITY_NORMALIZED',
            how='left'
        )
        
        # Eksik değerleri doldur
        merged['PF_Satis'] = merged['PF_Satis'].fillna(0)
        merged['Pazar_Payi_%'] = merged['Pazar_Payi_%'].fillna(0)
        merged['Region'] = merged['Region'].fillna('DİĞER')
        
        # Performans skorunu hesapla
        merged['Performance_Score'] = self._calculate_performance_score(merged)
        
        return merged
    
    def _calculate_performance_score(self, data: pd.DataFrame) -> pd.Series:
        """Performans skorunu hesapla"""
        # Pazar payı skoru (0-50)
        market_share_score = np.clip(data['Pazar_Payi_%'] * 0.5, 0, 50)
        
        # Satış büyüklüğü skoru (0-30)
        sales_score = np.clip(
            np.log1p(data['PF_Satis']) / np.log1p(data['PF_Satis'].max() + 1) * 30,
            0, 30
        )
        
        # Büyüme potansiyeli skoru (0-20)
        growth_potential = (data['Toplam_Pazar'] - data['PF_Satis']) / data['Toplam_Pazar'].clip(lower=1)
        growth_score = np.clip(growth_potential * 20, 0, 20)
        
        return market_share_score + sales_score + growth_score
    
    def _create_region_map(self, data: pd.DataFrame, title: str) -> go.Figure:
        """Bölge haritası oluştur"""
        fig = go.Figure()
        
        # Her bölge için ayrı trace
        for region in data['Region'].unique():
            region_data = data[data['Region'] == region]
            color = self.config.REGION_COLORS.get(region, self.config.COLOR_PALETTE['neutral_medium'])
            
            # Bölge verisini GeoJSON formatına çevir
            region_json = json.loads(region_data.to_json())
            
            fig.add_trace(go.Choroplethmapbox(
                geojson=region_json,
                locations=region_data.index,
                z=[1] * len(region_data),  # Sabit değer, renk için
                colorscale=[[0, color], [1, color]],
                marker_opacity=0.7,
                marker_line_width=1.5,
                marker_line_color='rgba(255, 255, 255, 0.9)',
                showscale=False,
                customdata=list(zip(
                    region_data['name'],
                    region_data['Region'],
                    region_data['PF_Satis'],
                    region_data['Pazar_Payi_%'],
                    region_data['Performance_Score']
                )),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Bölge: %{customdata[1]}<br>"
                    "PF Satış: %{customdata[2]:,.0f}<br>"
                    "Pazar Payı: %{customdata[3]:.1f}%<br>"
                    "Performans Skoru: %{customdata[4]:.0f}/100"
                    "<extra></extra>"
                ),
                name=region
            ))
        
        # Bölge merkezlerine etiket ekle
        label_data = self._calculate_region_labels(data)
        
        if len(label_data) > 0:
            fig.add_trace(go.Scattermapbox(
                lon=label_data['lon'],
                lat=label_data['lat'],
                mode='text',
                text=label_data['text'],
                textfont=dict(
                    size=12,
                    color='white',
                    family='Inter, sans-serif'
                ),
                hoverinfo='skip',
                showlegend=False
            ))
        
        return fig
    
    def _calculate_region_labels(self, data: pd.DataFrame) -> pd.DataFrame:
        """Bölge etiketlerini hesapla"""
        labels = []
        
        for region in data['Region'].unique():
            region_data = data[data['Region'] == region]
            total_sales = region_data['PF_Satis'].sum()
            
            if total_sales > 0:
                # Bölge merkezini hesapla
                centroid = region_data.geometry.unary_union.centroid
                
                labels.append({
                    'region': region,
                    'lon': centroid.x,
                    'lat': centroid.y,
                    'text': f"<b>{region}</b><br>{total_sales:,.0f}",
                    'sales': total_sales
                })
        
        return pd.DataFrame(labels)
    
    def _create_city_map(self, data: pd.DataFrame, title: str) -> go.Figure:
        """Şehir haritası oluştur"""
        fig = go.Figure()
        
        # Performans skoruna göre renk skalası
        max_score = data['Performance_Score'].max() if len(data) > 0 else 1
        
        fig.add_trace(go.Choroplethmapbox(
            geojson=self.geojson_data,
            locations=data.index,
            z=data['Performance_Score'],
            colorscale=[
                [0, self.config.COLOR_PALETTE['danger_light']],
                [0.5, self.config.COLOR_PALETTE['warning_medium']],
                [1, self.config.COLOR_PALETTE['success_medium']]
            ],
            zmin=0,
            zmax=max_score,
            marker_opacity=0.8,
            marker_line_width=1,
            marker_line_color='rgba(255, 255, 255, 0.8)',
            colorbar=dict(
                title="Performans<br>Skoru",
                titleside="right",
                thickness=15,
                len=0.8,
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.02,
                tickformat=".0f"
            ),
            customdata=list(zip(
                data['name'],
                data['Region'],
                data['PF_Satis'],
                data['Pazar_Payi_%'],
                data['Performance_Score']
            )),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Bölge: %{customdata[1]}<br>"
                "PF Satış: %{customdata[2]:,.0f}<br>"
                "Pazar Payı: %{customdata[3]:.1f}%<br>"
                "Performans Skoru: %{customdata[4]:.0f}/100"
                "<extra></extra>"
            )
        ))
        
        # Büyük şehirlere etiket ekle
        large_cities = data[data['PF_Satis'] > data['PF_Satis'].quantile(0.75)]
        
        if len(large_cities) > 0:
            fig.add_trace(go.Scattermapbox(
                lon=large_cities.geometry.centroid.x,
                lat=large_cities.geometry.centroid.y,
                mode='text',
                text=large_cities['name'],
                textfont=dict(
                    size=10,
                    color='white',
                    family='Inter, sans-serif'
                ),
                hoverinfo='skip',
                showlegend=False
            ))
        
        return fig
    
    def _update_map_layout(self, fig: go.Figure, title: str) -> go.Figure:
        """Harita layout'unu güncelle"""
        fig.update_layout(
            mapbox_style="carto-darkmatter",
            mapbox=dict(
                center=dict(lat=39.0, lon=35.0),
                zoom=5,
                bearing=0,
                pitch=0
            ),
            height=700,
            margin=dict(l=0, r=0, t=80, b=0),
            title=dict(
                text=f"<b>{title}</b>",
                x=0.5,
                font=dict(
                    size=24,
                    color=self.config.COLOR_PALETTE['text_primary'],
                    family='Inter, sans-serif'
                ),
                y=0.95
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            hoverlabel=dict(
                bgcolor=self.config.COLOR_PALETTE['background_dark'],
                font_size=12,
                font_family="Inter, sans-serif",
                font_color=self.config.COLOR_PALETTE['text_primary']
            )
        )
        
        return fig


# =============================================================================
# İÇGÖRÜ ÜRETİCİ SINIFI
# =============================================================================

class InsightGenerator(BaseAnalyzer):
    """
    Otomatik yönetici içgörüleri ve özetler oluşturur
    """
    
    def __init__(self):
        self.config = AppConfig()
        self.thresholds = self.config.PERFORMANCE_THRESHOLDS
    
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Veriyi analiz eder ve içgörüler üretir
        
        Args:
            data (pd.DataFrame): Analiz edilecek veri
            
        Returns:
            Dict[str, Any]: İçgörüler ve özetler
        """
        insights = {
            "executive_summary": [],
            "key_opportunities": [],
            "key_risks": [],
            "strategic_recommendations": [],
            "performance_metrics": {}
        }
        
        try:
            # Temel metrikleri hesapla
            insights["performance_metrics"] = self._calculate_basic_metrics(data)
            
            # İçgörüleri oluştur
            insights["executive_summary"] = self._generate_executive_summary(data)
            insights["key_opportunities"] = self._identify_opportunities(data)
            insights["key_risks"] = self._identify_risks(data)
            insights["strategic_recommendations"] = self._generate_recommendations(data)
            
            return insights
            
        except Exception as e:
            logging.error(f"İçgörü üretme hatası: {e}")
            return insights
    
    def _calculate_basic_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Temel performans metriklerini hesapla"""
        if len(data) == 0:
            return {}
        
        metrics = {}
        
        # Toplam satışlar
        metrics["total_pf_sales"] = data['PF_Satis'].sum()
        metrics["total_competitor_sales"] = data['Rakip_Satis'].sum()
        metrics["total_market"] = metrics["total_pf_sales"] + metrics["total_competitor_sales"]
        
        # Pazar payı
        if metrics["total_market"] > 0:
            metrics["market_share"] = (metrics["total_pf_sales"] / metrics["total_market"]) * 100
        else:
            metrics["market_share"] = 0
        
        # Büyüme metrikleri
        if 'PF_Buyume_%' in data.columns:
            metrics["avg_growth_rate"] = data['PF_Buyume_%'].mean()
            metrics["positive_growth_months"] = len(data[data['PF_Buyume_%'] > 0])
        
        # Territory metrikleri
        if 'TERRITORIES' in data.columns:
            metrics["active_territories"] = data['TERRITORIES'].nunique()
        
        # Şehir metrikleri
        if 'CITY_NORMALIZED' in data.columns:
            metrics["active_cities"] = data['CITY_NORMALIZED'].nunique()
        
        return metrics
    
    def _generate_executive_summary(self, data: pd.DataFrame) -> List[str]:
        """Yönetici özeti oluştur"""
        summary = []
        
        if len(data) == 0:
            return ["⚠️ Analiz için yeterli veri bulunamadı"]
        
        # Toplam satış özeti
        total_pf = data['PF_Satis'].sum()
        total_market = data['Toplam_Pazar'].sum() if 'Toplam_Pazar' in data.columns else total_pf + data['Rakip_Satis'].sum()
        market_share = (total_pf / total_market * 100) if total_market > 0 else 0
        
        summary.append(f"📊 **Toplam PF Satış:** {self._format_number(total_pf)}")
        summary.append(f"🏪 **Toplam Pazar Büyüklüğü:** {self._format_number(total_market)}")
        summary.append(f"🎯 **Pazar Payı:** {market_share:.1f}%")
        
        # Büyüme özeti
        if 'PF_Buyume_%' in data.columns:
            avg_growth = data['PF_Buyume_%'].mean()
            growth_trend = "📈" if avg_growth > 0 else "📉" if avg_growth < 0 else "➡️"
            summary.append(f"{growth_trend} **Ortalama Aylık Büyüme:** {avg_growth:.1f}%")
        
        # Territory özeti
        if 'TERRITORIES' in data.columns:
            territory_count = data['TERRITORIES'].nunique()
            summary.append(f"🏢 **Aktif Territory Sayısı:** {territory_count}")
        
        return summary
    
    def _identify_opportunities(self, data: pd.DataFrame) -> List[str]:
        """Fırsatları belirle"""
        opportunities = []
        
        if len(data) == 0:
            return opportunities
        
        # Yüksek büyüme, düşük pazar payı olan şehirler
        if 'Pazar_Payi_%' in data.columns and 'PF_Buyume_%' in data.columns:
            high_growth_low_share = data[
                (data['PF_Buyume_%'] > self.thresholds['growth_high']) &
                (data['Pazar_Payi_%'] < self.thresholds['market_share_low'])
            ]
            
            if len(high_growth_low_share) > 0:
                top_opportunities = high_growth_low_share.nlargest(3, 'PF_Buyume_%')
                for idx, row in top_opportunities.iterrows():
                    opportunities.append(
                        f"💎 **{row.get('City', 'Bilinmeyen')}**: Düşük pazar payı ({row['Pazar_Payi_%']:.1f}%) "
                        f"ancak yüksek büyüme ({row['PF_Buyume_%']:.1f}%). Potansiyel 'Soru İşareti'."
                    )
        
        # Büyük pazar, düşük penetrasyon
        if 'Toplam_Pazar' in data.columns and 'Pazar_Payi_%' in data.columns:
            large_market_low_penetration = data[
                (data['Toplam_Pazar'] > data['Toplam_Pazar'].quantile(0.75)) &
                (data['Pazar_Payi_%'] < self.thresholds['market_share_low'])
            ]
            
            if len(large_market_low_penetration) > 0:
                opportunities.append(
                    f"🏙️ **{len(large_market_low_penetration)} büyük pazarda** "
                    f"düşük penetrasyon (<{self.thresholds['market_share_low']}%) tespit edildi. "
                    f"Agresif pazarlama potansiyeli."
                )
        
        return opportunities
    
    def _identify_risks(self, data: pd.DataFrame) -> List[str]:
        """Riskleri belirle"""
        risks = []
        
        if len(data) == 0:
            return risks
        
        # Düşen pazar payı
        if 'Pazar_Payi_%' in data.columns and 'PF_Buyume_%' in data.columns:
            declining_markets = data[
                (data['Pazar_Payi_%'] > self.thresholds['market_share_high']) &
                (data['PF_Buyume_%'] < 0)
            ]
            
            if len(declining_markets) > 0:
                risks.append(
                    f"⚠️ **{len(declining_markets)} yüksek pazar paylı bölgede** "
                    f"düşüş trendi tespit edildi. 'Cash Cow'ları koruma stratejisi gerekli."
                )
        
        # Yüksek rakip büyümesi
        if 'Rakip_Buyume_%' in data.columns:
            high_competitor_growth = data[data['Rakip_Buyume_%'] > self.thresholds['growth_high']]
            
            if len(high_competitor_growth) > 0:
                top_competition = high_competitor_growth.nlargest(3, 'Rakip_Buyume_%')
                for idx, row in top_competition.iterrows():
                    risks.append(
                        f"🎯 **{row.get('City', 'Bilinmeyen')}**: Rakip büyümesi ({row['Rakip_Buyume_%']:.1f}%) "
                        f"PF büyümesinden ({row.get('PF_Buyume_%', 0):.1f}%) yüksek. "
                        f"Rakip aktivitesi izlenmeli."
                    )
        
        return risks
    
    def _generate_recommendations(self, data: pd.DataFrame) -> List[str]:
        """Stratejik öneriler oluştur"""
        recommendations = []
        
        if len(data) == 0:
            return recommendations
        
        # BCG kategorilerine göre öneriler
        if 'BCG_Kategori' in data.columns:
            bcg_counts = data['BCG_Kategori'].value_counts()
            
            if "❓ Question Mark" in bcg_counts:
                recommendations.append(
                    f"🚀 **{bcg_counts['❓ Question Mark']} 'Soru İşareti' territory** "
                    f"tespit edildi. Yatırım önceliği verilmeli."
                )
            
            if "🐄 Cash Cow" in bcg_counts:
                recommendations.append(
                    f"🛡️ **{bcg_counts['🐄 Cash Cow']} 'Cash Cow' territory** "
                    f"mevcut. Koruma ve nakit akışı optimizasyonu önerilir."
                )
            
            if "🐶 Dog" in bcg_counts:
                recommendations.append(
                    f"📉 **{bcg_counts['🐶 Dog']} 'Dog' territory** tespit edildi. "
                    f"Kaynakların yeniden tahsisi değerlendirilmeli."
                )
        
        # Pazar payı bazlı öneriler
        if 'Pazar_Payi_%' in data.columns:
            low_share_count = len(data[data['Pazar_Payi_%'] < self.thresholds['market_share_low']])
            high_share_count = len(data[data['Pazar_Payi_%'] > self.thresholds['market_share_high']])
            
            if low_share_count > 0:
                recommendations.append(
                    f"🎯 **{low_share_count} bölgede** pazar payı <%{self.thresholds['market_share_low']}. "
                    f"Penetrasyon artırma stratejileri uygulanmalı."
                )
            
            if high_share_count > 0:
                recommendations.append(
                    f"🛡️ **{high_share_count} bölgede** pazar payı >%{self.thresholds['market_share_high']}. "
                    f"Rakiplerin girişini engelleme stratejileri önerilir."
                )
        
        return recommendations
    
    def _format_number(self, num: float) -> str:
        """Sayıları formatla"""
        if pd.isna(num):
            return "0"
        
        try:
            num = float(num)
            if num == 0:
                return "0"
            elif abs(num) >= 1_000_000_000:
                return f"{num/1_000_000_000:,.1f}B"
            elif abs(num) >= 1_000_000:
                return f"{num/1_000_000:,.1f}M"
            elif abs(num) >= 1_000:
                return f"{num/1_000:,.1f}K"
            else:
                return f"{num:,.0f}"
        except:
            return str(num)


# =============================================================================
# SATIŞ TAHMİNCİ SINIFI
# =============================================================================

class SalesForecaster:
    """
    Makine öğrenmesi ile satış tahminleri yapar
    """
    
    def __init__(self):
        self.config = AppConfig()
        self.ml_params = self.config.ML_PARAMS
        self.models = {}
        self.results = {}
    
    def forecast(self, 
                data: pd.DataFrame, 
                target_column: str = "PF_Satis",
                forecast_periods: int = None) -> Dict[str, Any]:
        """
        Satış tahmini yapar
        
        Args:
            data (pd.DataFrame): Tarihsel veri
            target_column (str): Tahmin edilecek kolon
            forecast_periods (int): Tahmin periyodu sayısı
            
        Returns:
            Dict[str, Any]: Tahmin sonuçları ve model metrikleri
        """
        if forecast_periods is None:
            forecast_periods = self.ml_params["forecast_periods"]
        
        results = {
            "forecast": None,
            "model_performance": {},
            "best_model": None,
            "feature_importance": {}
        }
        
        try:
            # Veriyi hazırla
            prepared_data = self._prepare_forecast_data(data, target_column)
            
            if len(prepared_data) < 10:
                results["error"] = "Tahmin için yeterli veri yok (en az 10 gözlem gerekli)"
                return results
            
            # Feature engineering
            features_df = self._create_features(prepared_data, target_column)
            
            # Model eğitimi
            model_results = self._train_models(features_df, target_column)
            
            # En iyi modeli seç
            best_model_name = self._select_best_model(model_results)
            results["best_model"] = best_model_name
            results["model_performance"] = model_results
            
            # Tahmin yap
            forecast = self._generate_forecast(
                features_df, 
                model_results[best_model_name]["model"],
                forecast_periods
            )
            
            results["forecast"] = forecast
            
            # Feature importance
            if best_model_name == "Random Forest":
                results["feature_importance"] = self._get_feature_importance(
                    model_results[best_model_name]["model"],
                    features_df.columns.tolist()
                )
            
            return results
            
        except Exception as e:
            logging.error(f"Tahmin hatası: {e}")
            results["error"] = str(e)
            return results
    
    def _prepare_forecast_data(self, data: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """Tahmin verisini hazırla"""
        df = data.copy()
        
        # Tarih sıralaması
        if 'DATE' in df.columns:
            df = df.sort_values('DATE').reset_index(drop=True)
        
        # Target kolonu kontrolü
        if target_column not in df.columns:
            raise ValueError(f"Target kolonu '{target_column}' veride bulunamadı")
        
        return df
    
    def _create_features(self, data: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """Feature engineering"""
        df = data.copy()
        
        # Lag features
        for lag in [1, 2, 3, 6, 12]:
            df[f'lag_{lag}'] = df[target_column].shift(lag)
        
        # Rolling statistics
        df['rolling_mean_3'] = df[target_column].rolling(window=3, min_periods=1).mean()
        df['rolling_mean_6'] = df[target_column].rolling(window=6, min_periods=1).mean()
        df['rolling_mean_12'] = df[target_column].rolling(window=12, min_periods=1).mean()
        df['rolling_std_3'] = df[target_column].rolling(window=3, min_periods=1).std()
        
        # Seasonality features
        if 'DATE' in df.columns:
            df['month'] = df['DATE'].dt.month
            df['quarter'] = df['DATE'].dt.quarter
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Trend
        df['trend'] = np.arange(len(df))
        
        # YoY growth if available
        yoy_col = f'{target_column}_YOY'
        if yoy_col in df.columns:
            df[yoy_col] = df[yoy_col].fillna(0)
        
        # Fill NaN values
        df = df.fillna(method='bfill').fillna(method='ffill').fillna(0)
        
        return df
    
    def _train_models(self, data: pd.DataFrame, target_column: str) -> Dict[str, Dict]:
        """ML modellerini eğit"""
        # Feature ve target'ları ayır
        feature_cols = [col for col in data.columns if col not in ['DATE', target_column, 'YIL_AY']]
        X = data[feature_cols]
        y = data[target_column]
        
        # Train/test split
        split_idx = int(len(X) * (1 - self.ml_params["test_size"]))
        
        X_train = X.iloc[:split_idx]
        y_train = y.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_test = y.iloc[split_idx:]
        
        # Model tanımları
        models = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(alpha=self.ml_params["ridge_alpha"]),
            "Random Forest": RandomForestRegressor(
                n_estimators=self.ml_params["n_estimators"],
                max_depth=self.ml_params["max_depth"],
                random_state=self.ml_params["random_state"]
            )
        }
        
        results = {}
        
        for name, model in models.items():
            try:
                # Model eğitimi
                model.fit(X_train, y_train)
                
                # Tahminler
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                
                # Metrikler
                train_mae = mean_absolute_error(y_train, y_pred_train)
                test_mae = mean_absolute_error(y_test, y_pred_test)
                train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
                test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                
                # MAPE (Mean Absolute Percentage Error)
                train_mape = self._calculate_mape(y_train, y_pred_train)
                test_mape = self._calculate_mape(y_test, y_pred_test)
                
                # R² Score
                train_r2 = r2_score(y_train, y_pred_train)
                test_r2 = r2_score(y_test, y_pred_test)
                
                results[name] = {
                    "model": model,
                    "metrics": {
                        "train_mae": train_mae,
                        "test_mae": test_mae,
                        "train_rmse": train_rmse,
                        "test_rmse": test_rmse,
                        "train_mape": train_mape,
                        "test_mape": test_mape,
                        "train_r2": train_r2,
                        "test_r2": test_r2
                    },
                    "feature_columns": feature_cols
                }
                
            except Exception as e:
                logging.error(f"Model {name} eğitim hatası: {e}")
                continue
        
        return results
    
    def _calculate_mape(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        """MAPE hesapla"""
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        mask = y_true != 0
        if np.sum(mask) == 0:
            return 0.0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    def _select_best_model(self, model_results: Dict[str, Dict]) -> str:
        """En iyi modeli seç (test MAPE'e göre)"""
        if not model_results:
            return None
        
        best_model = None
        best_mape = float('inf')
        
        for name, result in model_results.items():
            test_mape = result["metrics"]["test_mape"]
            if test_mape < best_mape:
                best_mape = test_mape
                best_model = name
        
        return best_model
    
    def _generate_forecast(self, 
                          data: pd.DataFrame, 
                          model: Any,
                          periods: int) -> pd.DataFrame:
        """Gelecek tahmini yap"""
        # Son satırı al
        last_row = data.iloc[-1:].copy()
        feature_cols = [col for col in data.columns if col not in ['DATE', 'YIL_AY']]
        
        forecast_data = []
        
        for i in range(periods):
            # Son tarihi al
            if 'DATE' in last_row.columns:
                last_date = last_row['DATE'].iloc[0]
                next_date = last_date + pd.DateOffset(months=1)
            else:
                next_date = None
            
            # Feature'ları hazırla
            X_next = last_row[feature_cols].copy()
            
            # Tahmin yap
            next_pred = max(0, model.predict(X_next)[0])
            
            forecast_data.append({
                'DATE': next_date,
                'YIL_AY': next_date.strftime('%Y-%m') if next_date else f"T+{i+1}",
                'PF_Satis': next_pred,
                'Forecast_Type': 'Tahmin'
            })
            
            # Next row için feature'ları güncelle
            new_row = last_row.copy()
            
            # Lag'leri güncelle
            for lag in range(5, 0, -1):
                if f'lag_{lag}' in new_row.columns:
                    if lag == 1:
                        new_row[f'lag_{lag}'] = next_pred
                    else:
                        new_row[f'lag_{lag}'] = last_row[f'lag_{lag-1}'].values[0]
            
            # Rolling statistics güncelle
            if 'rolling_mean_3' in new_row.columns:
                new_row['rolling_mean_3'] = (
                    new_row['lag_1'] + new_row['lag_2'] + new_row['lag_3']
                ) / 3
            
            if 'rolling_mean_6' in new_row.columns:
                new_row['rolling_mean_6'] = (
                    new_row['lag_1'] + new_row['lag_2'] + new_row['lag_3'] +
                    new_row['lag_4'] + new_row['lag_5'] + new_row['lag_6']
                ) / 6
            
            # Tarih feature'larını güncelle
            if next_date:
                new_row['DATE'] = next_date
                new_row['month'] = next_date.month
                new_row['quarter'] = next_date.quarter
                new_row['month_sin'] = np.sin(2 * np.pi * new_row['month'] / 12)
                new_row['month_cos'] = np.cos(2 * np.pi * new_row['month'] / 12)
            
            # Trend'i artır
            if 'trend' in new_row.columns:
                new_row['trend'] = last_row['trend'].values[0] + 1
            
            last_row = new_row
        
        return pd.DataFrame(forecast_data)
    
    def _get_feature_importance(self, model: Any, feature_names: List[str]) -> Dict[str, float]:
        """Feature importance değerlerini al"""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            return dict(zip(feature_names, importances))
        elif hasattr(model, 'coef_'):
            coef = model.coef_
            return dict(zip(feature_names, coef))
        else:
            return {}


# =============================================================================
# BCG ANALİZ SINIFI
# =============================================================================

class BCGAnalyzer(BaseAnalyzer):
    """
    BCG Matrix analizleri yapar
    """
    
    def __init__(self):
        self.config = AppConfig()
    
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        BCG Matrix analizi yapar
        
        Args:
            data (pd.DataFrame): Territory bazlı veri
            
        Returns:
            Dict[str, Any]: BCG analiz sonuçları
        """
        results = {
            "bcg_matrix": None,
            "category_summary": {},
            "strategic_implications": []
        }
        
        try:
            # BCG kategorilerini hesapla
            bcg_df = self._calculate_bcg_categories(data)
            results["bcg_matrix"] = bcg_df
            
            # Kategori özeti
            results["category_summary"] = self._summarize_categories(bcg_df)
            
            # Stratejik çıkarımlar
            results["strategic_implications"] = self._derive_strategic_implications(bcg_df)
            
            return results
            
        except Exception as e:
            logging.error(f"BCG analiz hatası: {e}")
            return results
    
    def _calculate_bcg_categories(self, data: pd.DataFrame) -> pd.DataFrame:
        """BCG kategorilerini hesapla"""
        df = data.copy()
        
        # Göreceli pazar payı
        if 'Goreceli_Pazar_Payi' not in df.columns:
            df['Goreceli_Pazar_Payi'] = df['PF_Satis'] / df['Rakip_Satis'].replace(0, 1)
        
        # Pazar büyüme oranı
        if 'Pazar_Buyume_%' not in df.columns:
            # Basit büyüme hesaplaması
            df['Pazar_Buyume_%'] = df['PF_Satis'].pct_change() * 100
            df['Pazar_Buyume_%'] = df['Pazar_Buyume_%'].fillna(0)
        
        # Medyan değerler
        median_share = df['Goreceli_Pazar_Payi'].median()
        median_growth = df['Pazar_Buyume_%'].median()
        
        # BCG kategorilerini ata
        def assign_bcg_category(row):
            if pd.isna(row['Goreceli_Pazar_Payi']) or pd.isna(row['Pazar_Buyume_%']):
                return "🐶 Dog"
            
            if row['Goreceli_Pazar_Payi'] >= median_share and row['Pazar_Buyume_%'] >= median_growth:
                return "⭐ Star"
            elif row['Goreceli_Pazar_Payi'] >= median_share and row['Pazar_Buyume_%'] < median_growth:
                return "🐄 Cash Cow"
            elif row['Goreceli_Pazar_Payi'] < median_share and row['Pazar_Buyume_%'] >= median_growth:
                return "❓ Question Mark"
            else:
                return "🐶 Dog"
        
        df['BCG_Kategori'] = df.apply(assign_bcg_category, axis=1)
        
        return df
    
    def _summarize_categories(self, bcg_df: pd.DataFrame) -> Dict[str, Any]:
        """BCG kategorilerini özetle"""
        if len(bcg_df) == 0:
            return {}
        
        summary = {}
        
        for category in self.config.BCG_COLORS.keys():
            cat_data = bcg_df[bcg_df['BCG_Kategori'] == category]
            
            summary[category] = {
                "count": len(cat_data),
                "total_sales": cat_data['PF_Satis'].sum() if len(cat_data) > 0 else 0,
                "avg_market_share": cat_data['Pazar_Payi_%'].mean() if len(cat_data) > 0 else 0,
                "avg_growth": cat_data['Pazar_Buyume_%'].mean() if len(cat_data) > 0 else 0,
                "top_territories": cat_data.nlargest(3, 'PF_Satis')[['Territory', 'PF_Satis', 'Pazar_Payi_%']].to_dict('records')
            }
        
        return summary
    
    def _derive_strategic_implications(self, bcg_df: pd.DataFrame) -> List[str]:
        """Stratejik çıkarımlar oluştur"""
        implications = []
        
        if len(bcg_df) == 0:
            return implications
        
        category_counts = bcg_df['BCG_Kategori'].value_counts()
        
        # Stars için
        star_count = category_counts.get("⭐ Star", 0)
        if star_count > 0:
            star_sales = bcg_df[bcg_df['BCG_Kategori'] == "⭐ Star"]['PF_Satis'].sum()
            implications.append(
                f"🚀 **{star_count} 'Star' territory** tespit edildi (Toplam: {self._format_number(star_sales)}). "
                f"Bu territory'lere yatırım devam etmeli, büyümeleri desteklenmeli."
            )
        
        # Cash Cows için
        cow_count = category_counts.get("🐄 Cash Cow", 0)
        if cow_count > 0:
            cow_sales = bcg_df[bcg_df['BCG_Kategori'] == "🐄 Cash Cow"]['PF_Satis'].sum()
            implications.append(
                f"💰 **{cow_count} 'Cash Cow' territory** tespit edildi (Toplam: {self._format_number(cow_sales)}). "
                f"Nakit akışı üretimi maksimize edilmeli, koruma stratejisi uygulanmalı."
            )
        
        # Question Marks için
        question_count = category_counts.get("❓ Question Mark", 0)
        if question_count > 0:
            implications.append(
                f"🎯 **{question_count} 'Soru İşareti' territory** tespit edildi. "
                f"Detaylı analiz yapılıp, ya yatırım artırılmalı ya da çıkış stratejisi uygulanmalı."
            )
        
        # Dogs için
        dog_count = category_counts.get("🐶 Dog", 0)
        if dog_count > 0:
            implications.append(
                f"📉 **{dog_count} 'Dog' territory** tespit edildi. "
                f"Kaynakların verimli kullanımı için minimal yatırım veya çıkış değerlendirilmeli."
            )
        
        # Portfolio dengelenmesi
        total = len(bcg_df)
        if total > 0:
            star_ratio = (star_count / total) * 100
            cow_ratio = (cow_count / total) * 100
            
            if star_ratio < 20:
                implications.append(
                    f"⚠️ **Portföy dengesi**: Star oranı (%{star_ratio:.1f}) düşük. "
                    f"Yeni 'Star' adayları geliştirilmeli."
                )
            
            if cow_ratio < 30:
                implications.append(
                    f"⚠️ **Nakit akışı riski**: Cash Cow oranı (%{cow_ratio:.1f}) düşük. "
                    f"Star'ların Cash Cow'a dönüşümü hızlandırılmalı."
                )
        
        return implications
    
    def _format_number(self, num: float) -> str:
        """Sayıları formatla"""
        if pd.isna(num):
            return "0"
        
        try:
            num = float(num)
            if num == 0:
                return "0"
            elif abs(num) >= 1_000_000_000:
                return f"{num/1_000_000_000:,.1f}B"
            elif abs(num) >= 1_000_000:
                return f"{num/1_000_000:,.1f}M"
            elif abs(num) >= 1_000:
                return f"{num/1_000:,.1f}K"
            else:
                return f"{num:,.0f}"
        except:
            return str(num)


# =============================================================================
# YATIRIM STRATEJİSİ SINIFI
# =============================================================================

class InvestmentStrategyAnalyzer(BaseAnalyzer):
    """
    Yatırım stratejisi analizleri yapar
    """
    
    def __init__(self):
        self.config = AppConfig()
    
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Yatırım stratejisi analizi yapar
        
        Args:
            data (pd.DataFrame): Şehir bazlı veri
            
        Returns:
            Dict[str, Any]: Yatırım stratejisi sonuçları
        """
        results = {
            "strategy_matrix": None,
            "strategy_distribution": {},
            "investment_recommendations": []
        }
        
        try:
            # Yatırım stratejilerini hesapla
            strategy_df = self._calculate_investment_strategies(data)
            results["strategy_matrix"] = strategy_df
            
            # Strateji dağılımı
            results["strategy_distribution"] = self._analyze_strategy_distribution(strategy_df)
            
            # Yatırım önerileri
            results["investment_recommendations"] = self._generate_investment_recommendations(strategy_df)
            
            return results
            
        except Exception as e:
            logging.error(f"Yatırım stratejisi analiz hatası: {e}")
            return results
    
    def _calculate_investment_strategies(self, data: pd.DataFrame) -> pd.DataFrame:
        """Yatırım stratejilerini hesapla"""
        df = data.copy()
        
        if len(df) == 0:
            return df
        
        # Segmentleri hesapla
        df = self._calculate_segments(df)
        
        # Stratejileri ata
        df['Yatırım_Stratejisi'] = df.apply(self._assign_strategy, axis=1)
        
        return df
    
    def _calculate_segments(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pazar segmentlerini hesapla"""
        # Pazar büyüklüğü segmenti
        try:
            df["Pazar_Büyüklüğü_Segment"] = pd.qcut(
                df["Toplam_Pazar"], 
                q=3, 
                labels=["Küçük", "Orta", "Büyük"],
                duplicates='drop'
            )
        except:
            df["Pazar_Büyüklüğü_Segment"] = "Orta"
        
        # Performans segmenti
        try:
            df["Performans_Segment"] = pd.qcut(
                df["PF_Satis"], 
                q=3, 
                labels=["Düşük", "Orta", "Yüksek"],
                duplicates='drop'
            )
        except:
            df["Performans_Segment"] = "Orta"
        
        # Pazar payı segmenti
        try:
            df["Pazar_Payı_Segment"] = pd.qcut(
                df["Pazar_Payi_%"], 
                q=3, 
                labels=["Düşük", "Orta", "Yüksek"],
                duplicates='drop'
            )
        except:
            df["Pazar_Payı_Segment"] = "Orta"
        
        # Büyüme potansiyeli
        df["Büyüme_Potansiyeli"] = df["Toplam_Pazar"] - df["PF_Satis"]
        try:
            df["Büyüme_Potansiyeli_Segment"] = pd.qcut(
                df["Büyüme_Potansiyeli"],
                q=3,
                labels=["Düşük", "Orta", "Yüksek"],
                duplicates='drop'
            )
        except:
            df["Büyüme_Potansiyeli_Segment"] = "Orta"
        
        return df
    
    def _assign_strategy(self, row: pd.Series) -> str:
        """Satır bazında strateji ata"""
        pazar_buyuklugu = str(row.get("Pazar_Büyüklüğü_Segment", "Orta"))
        pazar_payi = str(row.get("Pazar_Payı_Segment", "Orta"))
        buyume_potansiyeli = str(row.get("Büyüme_Potansiyeli_Segment", "Orta"))
        performans = str(row.get("Performans_Segment", "Orta"))
        
        # 1. Agresif Strateji: Büyük pazar, düşük pay, yüksek potansiyel
        if (pazar_buyuklugu in ["Büyük", "Orta"] and 
            pazar_payi == "Düşük" and 
            buyume_potansiyeli in ["Yüksek", "Orta"]):
            return "🚀 Agresif"
        
        # 2. Hızlandırılmış Strateji: Orta/Büyük pazar, orta pay, orta/yüksek performans
        elif (pazar_buyuklugu in ["Büyük", "Orta"] and 
              pazar_payi == "Orta" and
              performans in ["Orta", "Yüksek"]):
            return "⚡ Hızlandırılmış"
        
        # 3. Koruma Stratejisi: Büyük pazar, yüksek pay
        elif (pazar_buyuklugu == "Büyük" and 
              pazar_payi == "Yüksek"):
            return "🛡️ Koruma"
        
        # 4. Potansiyel Stratejisi: Küçük pazar, yüksek potansiyel, orta/yüksek performans
        elif (pazar_buyuklugu == "Küçük" and 
              buyume_potansiyeli == "Yüksek" and
              performans in ["Orta", "Yüksek"]):
            return "💎 Potansiyel"
        
        # 5. İzleme Stratejisi: Diğer durumlar
        else:
            return "👁️ İzleme"
    
    def _analyze_strategy_distribution(self, strategy_df: pd.DataFrame) -> Dict[str, Any]:
        """Strateji dağılımını analiz et"""
        if len(strategy_df) == 0:
            return {}
        
        distribution = {}
        strategy_counts = strategy_df['Yatırım_Stratejisi'].value_counts()
        
        for strategy in self.config.STRATEGY_COLORS.keys():
            count = strategy_counts.get(strategy, 0)
            strategy_data = strategy_df[strategy_df['Yatırım_Stratejisi'] == strategy]
            
            distribution[strategy] = {
                "count": count,
                "percentage": (count / len(strategy_df)) * 100 if len(strategy_df) > 0 else 0,
                "total_sales": strategy_data['PF_Satis'].sum() if len(strategy_data) > 0 else 0,
                "avg_market_share": strategy_data['Pazar_Payi_%'].mean() if len(strategy_data) > 0 else 0,
                "top_cities": strategy_data.nlargest(3, 'PF_Satis')[['City', 'PF_Satis', 'Pazar_Payi_%']].to_dict('records')
            }
        
        return distribution
    
    def _generate_investment_recommendations(self, strategy_df: pd.DataFrame) -> List[str]:
        """Yatırım önerileri oluştur"""
        recommendations = []
        
        if len(strategy_df) == 0:
            return recommendations
        
        distribution = self._analyze_strategy_distribution(strategy_df)
        
        # Agresif strateji önerileri
        aggressive = distribution.get("🚀 Agresif", {})
        if aggressive.get("count", 0) > 0:
            recommendations.append(
                f"🎯 **{aggressive['count']} şehirde 'Agresif' strateji** öneriliyor. "
                f"Toplam {self._format_number(aggressive['total_sales'])} PF satış potansiyeli. "
                f"Saha gücü ve pazarlama bütçesi artırılmalı."
            )
        
        # Potansiyel strateji önerileri
        potential = distribution.get("💎 Potansiyel", {})
        if potential.get("count", 0) > 0:
            recommendations.append(
                f"💎 **{potential['count']} 'Potansiyel' şehir** tespit edildi. "
                f"Küçük ama hızlı büyüyen pazarlar. Pilot programlar başlatılmalı."
            )
        
        # Koruma stratejisi önerileri
        protection = distribution.get("🛡️ Koruma", {})
        if protection.get("count", 0) > 0:
            recommendations.append(
                f"🛡️ **{protection['count']} şehirde 'Koruma' stratejisi** gerekli. "
                f"Yüksek pazar payı korunmalı, rakip girişleri engellenmeli."
            )
        
        # Kaynak tahsisi önerisi
        total_investment_needed = sum([
            aggressive.get("count", 0) * 1.5,  # Agresif: Yüksek yatırım
            distribution.get("⚡ Hızlandırılmış", {}).get("count", 0) * 1.0,  # Orta yatırım
            potential.get("count", 0) * 0.7,  # Düşük yatırım
            protection.get("count", 0) * 0.5,  # Minimal yatırım
            distribution.get("👁️ İzleme", {}).get("count", 0) * 0.2  # Çok düşük yatırım
        ])
        
        recommendations.append(
            f"💰 **Kaynak Tahsisi**: Toplam {total_investment_needed:.1f} birim yatırım gerekli. "
            f"Öncelik sırası: Agresif → Hızlandırılmış → Potansiyel → Koruma → İzleme"
        )
        
        return recommendations
    
    def _format_number(self, num: float) -> str:
        """Sayıları formatla"""
        if pd.isna(num):
            return "0"
        
        try:
            num = float(num)
            if num == 0:
                return "0"
            elif abs(num) >= 1_000_000_000:
                return f"{num/1_000_000_000:,.1f}B"
            elif abs(num) >= 1_000_000:
                return f"{num/1_000_000:,.1f}M"
            elif abs(num) >= 1_000:
                return f"{num/1_000:,.1f}K"
            else:
                return f"{num:,.0f}"
        except:
            return str(num)


# =============================================================================
# KULLANICI ARAYÜZ YÖNETİCİSİ
# =============================================================================

class UIManager:
    """
    Streamlit UI bileşenlerini ve stilini yönetir
    """
    
    def __init__(self):
        self.config = AppConfig()
        self._setup_page_config()
        self._inject_custom_css()
    
    def _setup_page_config(self) -> None:
        """Sayfa konfigürasyonunu ayarla"""
        st.set_page_config(
            page_title="Stratejik Ticari Portföy Analiz Sistemi",
            page_icon="🎯",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def _inject_custom_css(self) -> None:
        """Özel CSS enjekte et"""
        custom_css = f"""
        <style>
            /* Temel Stil */
            * {{
                font-family: 'Inter', 'Segoe UI', sans-serif;
            }}
            
            .stApp {{
                background: linear-gradient(135deg, 
                    {self.config.COLOR_PALETTE['background_dark']} 0%, 
                    {self.config.COLOR_PALETTE['background_medium']} 50%, 
                    {self.config.COLOR_PALETTE['background_light']} 100%);
            }}
            
            /* Başlık */
            .main-header {{
                font-size: 2.8rem;
                font-weight: 800;
                text-align: center;
                padding: 1.5rem 0;
                background: linear-gradient(135deg, 
                    {self.config.COLOR_PALETTE['primary_medium']} 0%, 
                    {self.config.COLOR_PALETTE['success_medium']} 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 1rem;
            }}
            
            /* Metrik Kartları */
            div[data-testid="metric-container"] {{
                background: rgba(30, 41, 59, 0.85);
                padding: 1.2rem;
                border-radius: 12px;
                border: 1px solid rgba(59, 130, 246, 0.2);
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
                backdrop-filter: blur(10px);
                transition: all 0.3s ease;
            }}
            
            div[data-testid="metric-container"]:hover {{
                transform: translateY(-4px);
                box-shadow: 0 8px 30px rgba(59, 130, 246, 0.3);
                border-color: rgba(59, 130, 246, 0.4);
            }}
            
            div[data-testid="stMetricValue"] {{
                font-size: 2.2rem;
                font-weight: 700;
                color: {self.config.COLOR_PALETTE['text_primary']};
            }}
            
            div[data-testid="stMetricLabel"] {{
                font-size: 0.9rem;
                font-weight: 600;
                color: {self.config.COLOR_PALETTE['text_secondary']};
            }}
            
            /* Tab'ler */
            .stTabs [data-baseweb="tab-list"] {{
                gap: 0.5rem;
                background: rgba(30, 41, 59, 0.7);
                border-radius: 10px;
                padding: 0.5rem;
            }}
            
            .stTabs [data-baseweb="tab"] {{
                color: {self.config.COLOR_PALETTE['text_secondary']};
                font-weight: 600;
                padding: 0.8rem 1.5rem;
                background: rgba(30, 41, 59, 0.5);
                border-radius: 8px;
                border: 1px solid transparent;
                transition: all 0.3s ease;
            }}
            
            .stTabs [data-baseweb="tab"]:hover {{
                background: rgba(59, 130, 246, 0.15);
                color: {self.config.COLOR_PALETTE['text_primary']};
                border-color: rgba(59, 130, 246, 0.3);
            }}
            
            .stTabs [data-baseweb="tab"][aria-selected="true"] {{
                background: linear-gradient(135deg, 
                    {self.config.COLOR_PALETTE['primary_medium']} 0%, 
                    {self.config.COLOR_PALETTE['success_medium']} 100%);
                color: white;
                box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
                border: 1px solid rgba(255, 255, 255, 0.1);
            }}
            
            /* Butonlar */
            .stButton > button {{
                background: linear-gradient(135deg, 
                    {self.config.COLOR_PALETTE['primary_medium']} 0%, 
                    {self.config.COLOR_PALETTE['success_medium']} 100%);
                color: white;
                border: none;
                padding: 0.7rem 1.8rem;
                border-radius: 10px;
                font-weight: 600;
                transition: all 0.3s ease;
                box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
            }}
            
            .stButton > button:hover {{
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4);
            }}
            
            /* Sidebar */
            [data-testid="stSidebar"] {{
                background: rgba(15, 23, 41, 0.95);
                backdrop-filter: blur(15px);
                border-right: 1px solid rgba(59, 130, 246, 0.1);
            }}
            
            /* Input Alanları */
            .stSelectbox, .stSlider, .stRadio {{
                background: rgba(30, 41, 59, 0.7);
                padding: 0.5rem;
                border-radius: 8px;
                border: 1px solid rgba(59, 130, 246, 0.2);
            }}
            
            /* Dataframe */
            .dataframe {{
                border-radius: 10px;
                overflow: hidden;
            }}
            
            /* Scrollbar */
            ::-webkit-scrollbar {{
                width: 8px;
                height: 8px;
            }}
            
            ::-webkit-scrollbar-track {{
                background: rgba(30, 41, 59, 0.5);
                border-radius: 4px;
            }}
            
            ::-webkit-scrollbar-thumb {{
                background: linear-gradient(135deg, 
                    {self.config.COLOR_PALETTE['primary_medium']} 0%, 
                    {self.config.COLOR_PALETTE['success_medium']} 100%);
                border-radius: 4px;
            }}
            
            ::-webkit-scrollbar-thumb:hover {{
                background: linear-gradient(135deg, 
                    {self.config.COLOR_PALETTE['success_medium']} 0%, 
                    {self.config.COLOR_PALETTE['warning_medium']} 100%);
            }}
            
            /* Kart Stilleri */
            .custom-card {{
                background: rgba(30, 41, 59, 0.8);
                padding: 1.5rem;
                border-radius: 12px;
                border: 1px solid rgba(59, 130, 246, 0.2);
                margin-bottom: 1rem;
            }}
            
            .insight-card {{
                background: rgba(30, 41, 59, 0.9);
                padding: 1.2rem;
                border-radius: 10px;
                border-left: 4px solid {self.config.COLOR_PALETTE['success_medium']};
                margin-bottom: 0.8rem;
            }}
            
            .warning-card {{
                background: rgba(30, 41, 59, 0.9);
                padding: 1.2rem;
                border-radius: 10px;
                border-left: 4px solid {self.config.COLOR_PALETTE['warning_medium']};
                margin-bottom: 0.8rem;
            }}
            
            .danger-card {{
                background: rgba(30, 41, 59, 0.9);
                padding: 1.2rem;
                border-radius: 10px;
                border-left: 4px solid {self.config.COLOR_PALETTE['danger_medium']};
                margin-bottom: 0.8rem;
            }}
        </style>
        """
        
        st.markdown(custom_css, unsafe_allow_html=True)
    
    def create_metric_card(self, label: str, value: Any, delta: str = None) -> None:
        """
        Metrik kartı oluştur
        
        Args:
            label (str): Metrik etiketi
            value (Any): Metrik değeri
            delta (str): Delta değeri
        """
        st.metric(label=label, value=value, delta=delta)
    
    def create_insight_card(self, title: str, content: str, type: str = "info") -> None:
        """
        İçgörü kartı oluştur
        
        Args:
            title (str): Kart başlığı
            content (str): İçerik
            type (str): Kart türü (info/warning/danger)
        """
        if type == "warning":
            st.markdown(f'<div class="warning-card"><h4>{title}</h4><p>{content}</p></div>', 
                       unsafe_allow_html=True)
        elif type == "danger":
            st.markdown(f'<div class="danger-card"><h4>{title}</h4><p>{content}</p></div>', 
                       unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="insight-card"><h4>{title}</h4><p>{content}</p></div>', 
                       unsafe_allow_html=True)
    
    def create_header(self, title: str, subtitle: str = None) -> None:
        """
        Sayfa başlığı oluştur
        
        Args:
            title (str): Ana başlık
            subtitle (str): Alt başlık
        """
        st.markdown(f'<h1 class="main-header">{title}</h1>', unsafe_allow_html=True)
        
        if subtitle:
            st.markdown(f'<div style="text-align: center; color: {self.config.COLOR_PALETTE["text_secondary"]}; '
                       f'margin-bottom: 2rem;">{subtitle}</div>', unsafe_allow_html=True)


# =============================================================================
# ANA UYGULAMA SINIFI
# =============================================================================

class StrategicPortfolioAnalyzer:
    """
    Ana uygulama sınıfı - Tüm bileşenleri koordine eder
    """
    
    def __init__(self):
        self.config = AppConfig()
        self.ui = UIManager()
        self.data_processor = DataProcessor()
        self.city_normalizer = CityNormalizer()
        self.map_engine = MapEngine()
        self.insight_generator = InsightGenerator()
        self.sales_forecaster = SalesForecaster()
        self.bcg_analyzer = BCGAnalyzer()
        self.investment_strategy_analyzer = InvestmentStrategyAnalyzer()
        
        # Session state initialization
        self._init_session_state()
    
    def _init_session_state(self) -> None:
        """Session state'i başlat"""
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'processed_data' not in st.session_state:
            st.session_state.processed_data = None
        if 'current_product' not in st.session_state:
            st.session_state.current_product = "TROCMETAM"
        if 'date_filter' not in st.session_state:
            st.session_state.date_filter = None
        if 'selected_territory' not in st.session_state:
            st.session_state.selected_territory = "TÜMÜ"
    
    def run(self) -> None:
        """Ana uygulamayı çalıştır"""
        # Başlık
        self.ui.create_header(
            "🎯 Stratejik Ticari Portföy Analiz Sistemi",
            "Yönetici Karar Destek Sistemi • McKinsey/BCG Tarzı • ML Tahminleme"
        )
        
        # Sidebar
        self._create_sidebar()
        
        # Ana içerik
        self._create_main_content()
    
    def _create_sidebar(self) -> None:
        """Sidebar bileşenlerini oluştur"""
        with st.sidebar:
            # Veri Yükleme Bölümü
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.markdown('### 📂 Veri Yükleme')
            
            uploaded_file = st.file_uploader(
                "Excel Dosyası Seçin",
                type=['xlsx', 'xls'],
                key="file_uploader"
            )
            
            if uploaded_file and not st.session_state.data_loaded:
                try:
                    with st.spinner("Veri yükleniyor ve işleniyor..."):
                        # Veriyi yükle
                        raw_data = pd.read_excel(uploaded_file)
                        
                        # Veriyi işle
                        processed_data = self.data_processor.process(raw_data)
                        
                        # Session state'e kaydet
                        st.session_state.processed_data = processed_data
                        st.session_state.data_loaded = True
                        
                        st.success(f"✅ **{len(processed_data):,}** satır veri yüklendi")
                        
                except Exception as e:
                    st.error(f"❌ Veri yükleme hatası: {str(e)}")
                    st.session_state.data_loaded = False
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            if st.session_state.data_loaded:
                # Filtre Bölümü
                st.markdown('<div class="custom-card">', unsafe_allow_html=True)
                st.markdown('### 🔍 Filtreler')
                
                # Ürün Seçimi
                products = list(self.config.PRODUCT_COLUMN_MAP.keys())
                selected_product = st.selectbox(
                    "💊 Ürün",
                    products,
                    index=products.index(st.session_state.current_product)
                )
                st.session_state.current_product = selected_product
                
                # Tarih Filtresi
                date_option = st.selectbox(
                    "📅 Dönem",
                    self.config.DATE_OPTIONS,
                    key="date_option"
                )
                
                # Tarih aralığı hesapla
                if st.session_state.processed_data is not None:
                    df = st.session_state.processed_data
                    min_date = df['DATE'].min()
                    max_date = df['DATE'].max()
                    
                    if date_option == "Tüm Veriler":
                        st.session_state.date_filter = None
                    elif date_option == "Son 3 Ay":
                        start_date = max_date - pd.DateOffset(months=3)
                        st.session_state.date_filter = (start_date, max_date)
                    elif date_option == "Son 6 Ay":
                        start_date = max_date - pd.DateOffset(months=6)
                        st.session_state.date_filter = (start_date, max_date)
                    elif date_option == "Son 1 Yıl":
                        start_date = max_date - pd.DateOffset(years=1)
                        st.session_state.date_filter = (start_date, max_date)
                    elif date_option == "2025":
                        st.session_state.date_filter = (
                            pd.to_datetime('2025-01-01'), 
                            pd.to_datetime('2025-12-31')
                        )
                    elif date_option == "2024":
                        st.session_state.date_filter = (
                            pd.to_datetime('2024-01-01'), 
                            pd.to_datetime('2024-12-31')
                        )
                    else:  # Özel Aralık
                        col1, col2 = st.columns(2)
                        with col1:
                            start_date = st.date_input(
                                "Başlangıç", 
                                min_date, 
                                min_value=min_date, 
                                max_value=max_date
                            )
                        with col2:
                            end_date = st.date_input(
                                "Bitiş", 
                                max_date, 
                                min_value=min_date, 
                                max_value=max_date
                            )
                        st.session_state.date_filter = (
                            pd.to_datetime(start_date), 
                            pd.to_datetime(end_date)
                        )
                
                # Territory Filtresi
                if st.session_state.processed_data is not None:
                    territories = ["TÜMÜ"] + sorted(
                        st.session_state.processed_data['TERRITORIES'].unique()
                    )
                    selected_territory = st.selectbox(
                        "🏢 Territory",
                        territories,
                        index=territories.index(st.session_state.selected_territory)
                    )
                    st.session_state.selected_territory = selected_territory
                
                # Bölge Filtresi
                if st.session_state.processed_data is not None:
                    regions = ["TÜMÜ"] + sorted(
                        st.session_state.processed_data['REGION'].unique()
                    )
                    selected_region = st.selectbox("🗺️ Bölge", regions)
                
                # Manager Filtresi
                if st.session_state.processed_data is not None:
                    managers = ["TÜMÜ"] + sorted(
                        st.session_state.processed_data['MANAGER'].unique()
                    )
                    selected_manager = st.selectbox("👨‍💼 Manager", managers)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Harita Ayarları
                st.markdown('<div class="custom-card">', unsafe_allow_html=True)
                st.markdown('### 🗺️ Harita Ayarları')
                
                view_mode = st.radio(
                    "Görünüm Modu",
                    ["Bölge Görünümü", "Şehir Görünümü"],
                    horizontal=True
                )
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Bölge Renkleri
                st.markdown('<div class="custom-card">', unsafe_allow_html=True)
                st.markdown('### 🎨 Bölge Renkleri')
                
                cols = st.columns(2)
                region_colors = list(self.config.REGION_COLORS.items())
                
                for idx, (region, color) in enumerate(region_colors):
                    col_idx = idx % 2
                    with cols[col_idx]:
                        st.markdown(
                            f'<div style="display: flex; align-items: center; margin: 0.2rem 0;">'
                            f'<div style="width: 12px; height: 12px; background-color: {color}; '
                            f'border-radius: 2px; margin-right: 6px;"></div>'
                            f'<span style="color: {self.config.COLOR_PALETTE["text_secondary"]}; '
                            f'font-size: 0.85rem;">{region}</span>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                
                st.markdown('</div>', unsafe_allow_html=True)
    
    def _create_main_content(self) -> None:
        """Ana içerik bileşenlerini oluştur"""
        if not st.session_state.data_loaded:
            st.info("👈 Lütfen sol taraftan Excel dosyasını yükleyin")
            return
        
        # Tab'ler
        tab_titles = [
            "📊 Genel Bakış",
            "🗺️ Coğrafi Analiz",
            "🏢 Performans Detay",
            "📈 Trend & Tahmin",
            "🎯 Stratejik Analiz",
            "🤖 ML İleri Analiz",
            "📥 Raporlar"
        ]
        
        tabs = st.tabs(tab_titles)
        
        with tabs[0]:  # Genel Bakış
            self._create_overview_tab()
        
        with tabs[1]:  # Coğrafi Analiz
            self._create_geographic_tab()
        
        with tabs[2]:  # Performans Detay
            self._create_performance_tab()
        
        with tabs[3]:  # Trend & Tahmin
            self._create_trend_tab()
        
        with tabs[4]:  # Stratejik Analiz
            self._create_strategic_tab()
        
        with tabs[5]:  # ML İleri Analiz
            self._create_ml_tab()
        
        with tabs[6]:  # Raporlar
            self._create_reports_tab()
    
    def _get_filtered_data(self) -> pd.DataFrame:
        """Filtrelenmiş veriyi al"""
        if st.session_state.processed_data is None:
            return pd.DataFrame()
        
        df = st.session_state.processed_data.copy()
        
        # Tarih filtresi
        if st.session_state.date_filter:
            start_date, end_date = st.session_state.date_filter
            df = df[(df['DATE'] >= start_date) & (df['DATE'] <= end_date)]
        
        # Ürün kolonlarını al
        product_cols = self.config.PRODUCT_COLUMN_MAP[st.session_state.current_product]
        pf_col = product_cols['pf']
        rakip_col = product_cols['rakip']
        
        # Eğer kolonlar yoksa, oluştur
        if pf_col not in df.columns:
            df[pf_col] = 0
        if rakip_col not in df.columns:
            df[rakip_col] = 0
        
        # Territory filtresi
        if st.session_state.selected_territory != "TÜMÜ":
            df = df[df['TERRITORIES'] == st.session_state.selected_territory]
        
        return df
    
    def _create_overview_tab(self) -> None:
        """Genel Bakış tab'ını oluştur"""
        st.header("📊 Genel Performans Özeti")
        
        df = self._get_filtered_data()
        
        if len(df) == 0:
            st.warning("Seçilen filtrelerde veri bulunamadı")
            return
        
        # Ürün kolonlarını al
        product_cols = self.config.PRODUCT_COLUMN_MAP[st.session_state.current_product]
        pf_col = product_cols['pf']
        rakip_col = product_cols['rakip']
        
        # Temel metrikleri hesapla
        total_pf = df[pf_col].sum()
        total_rakip = df[rakip_col].sum()
        total_market = total_pf + total_rakip
        market_share = (total_pf / total_market * 100) if total_market > 0 else 0
        active_territories = df['TERRITORIES'].nunique()
        active_cities = df['CITY_NORMALIZED'].nunique()
        avg_monthly_pf = total_pf / df['YIL_AY'].nunique() if df['YIL_AY'].nunique() > 0 else 0
        
        # Metrik kartları
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            self.ui.create_metric_card(
                "💊 PF Satış",
                self._format_number(total_pf),
                f"{self._format_number(avg_monthly_pf)}/ay"
            )
        
        with col2:
            self.ui.create_metric_card(
                "🏪 Toplam Pazar",
                self._format_number(total_market),
                f"{self._format_number(total_rakip)} rakip"
            )
        
        with col3:
            self.ui.create_metric_card(
                "📊 Pazar Payı",
                f"{market_share:.1f}%",
                f"{(100-market_share):.1f}% rakip"
            )
        
        with col4:
            self.ui.create_metric_card(
                "🏢 Aktif Birimler",
                f"{active_territories} Territory",
                f"{active_cities} Şehir"
            )
        
        st.markdown("---")
        
        # İçgörüler
        st.subheader("💡 Yönetici İçgörüleri")
        
        # İçgörü oluştur
        insight_data = pd.DataFrame({
            'PF_Satis': df[pf_col],
            'Rakip_Satis': df[rakip_col],
            'Toplam_Pazar': df[pf_col] + df[rakip_col],
            'Pazar_Payi_%': (df[pf_col] / (df[pf_col] + df[rakip_col].replace(0, 1))) * 100
        })
        
        insights = self.insight_generator.analyze(insight_data)
        
        # İçgörüleri göster
        col_insight1, col_insight2 = st.columns(2)
        
        with col_insight1:
            if insights["key_opportunities"]:
                st.markdown("##### 🚀 Ana Fırsatlar")
                for opportunity in insights["key_opportunities"][:3]:  # İlk 3'ü göster
                    self.ui.create_insight_card("Fırsat", opportunity, "info")
        
        with col_insight2:
            if insights["key_risks"]:
                st.markdown("##### ⚠️ Ana Riskler")
                for risk in insights["key_risks"][:3]:  # İlk 3'ü göster
                    self.ui.create_insight_card("Risk", risk, "warning")
        
        # Özet metrikler
        st.markdown("---")
        st.subheader("📈 Performans Özeti")
        
        if insights["performance_metrics"]:
            metrics = insights["performance_metrics"]
            
            col_sum1, col_sum2, col_sum3 = st.columns(3)
            
            with col_sum1:
                st.metric("💰 Toplam PF Değeri", self._format_number(metrics.get("total_pf_sales", 0)))
            
            with col_sum2:
                st.metric("📊 Pazar Payı", f"{metrics.get('market_share', 0):.1f}%")
            
            with col_sum3:
                growth = metrics.get("avg_growth_rate", 0)
                st.metric("📈 Ort. Büyüme", f"{growth:.1f}%")
    
    def _create_geographic_tab(self) -> None:
        """Coğrafi Analiz tab'ını oluştur"""
        st.header("🗺️ Coğrafi Dağılım Analizi")
        
        df = self._get_filtered_data()
        
        if len(df) == 0:
            st.warning("Seçilen filtrelerde veri bulunamadı")
            return
        
        # Ürün kolonlarını al
        product_cols = self.config.PRODUCT_COLUMN_MAP[st.session_state.current_product]
        pf_col = product_cols['pf']
        rakip_col = product_cols['rakip']
        
        # Şehir bazlı veriyi hazırla
        city_data = df.groupby(['CITY_NORMALIZED', 'REGION']).agg({
            pf_col: 'sum',
            rakip_col: 'sum'
        }).reset_index()
        
        city_data.columns = ['City', 'Region', 'PF_Satis', 'Rakip_Satis']
        city_data['Toplam_Pazar'] = city_data['PF_Satis'] + city_data['Rakip_Satis']
        city_data['Pazar_Payi_%'] = (city_data['PF_Satis'] / city_data['Toplam_Pazar'].replace(0, 1)) * 100
        
        # Metrikler
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_pf = city_data['PF_Satis'].sum()
            st.metric("🌍 PF Satış", self._format_number(total_pf))
        
        with col2:
            active_cities = len(city_data[city_data['PF_Satis'] > 0])
            st.metric("🏙️ Aktif Şehir", str(active_cities))
        
        with col3:
            avg_share = city_data['Pazar_Payi_%'].mean()
            st.metric("🎯 Ort. Pazar Payı", f"{avg_share:.1f}%")
        
        with col4:
            top_city = city_data.loc[city_data['PF_Satis'].idxmax(), 'City'] if len(city_data) > 0 else "Yok"
            st.metric("🏆 Lider Şehir", top_city)
        
        st.markdown("---")
        
        # Harita
        st.subheader("📍 Türkiye Haritası")
        
        # Harita görünümü seçimi
        view_mode = st.radio(
            "Harita Görünümü",
            ["Bölge Görünümü", "Şehir Görünümü"],
            horizontal=True,
            key="map_view"
        )
        
        # Harita oluştur
        map_fig = self.map_engine.create_visualization(
            city_data,
            view_mode=view_mode,
            title=f"{st.session_state.current_product} - Coğrafi Dağılım"
        )
        
        if map_fig:
            st.plotly_chart(map_fig, use_container_width=True)
        else:
            st.error("Harita oluşturulamadı. GeoJSON dosyasını kontrol edin.")
        
        st.markdown("---")
        
        # Şehir Performans Tablosu
        st.subheader("📋 Şehir Performans Detayları")
        
        # Sıralama seçeneği
        sort_option = st.selectbox(
            "Sıralama Kriteri",
            ["PF Satış", "Pazar Payı", "Toplam Pazar"],
            key="city_sort"
        )
        
        sort_column_map = {
            "PF Satış": "PF_Satis",
            "Pazar Payı": "Pazar_Payi_%",
            "Toplam Pazar": "Toplam_Pazar"
        }
        
        city_sorted = city_data.sort_values(
            sort_column_map[sort_option], 
            ascending=False
        ).head(20)
        
        # Tabloyu göster
        display_cols = ['City', 'Region', 'PF_Satis', 'Toplam_Pazar', 'Pazar_Payi_%']
        city_display = city_sorted[display_cols].copy()
        city_display.columns = ['Şehir', 'Bölge', 'PF Satış', 'Toplam Pazar', 'Pazar Payı %']
        city_display.index = range(1, len(city_display) + 1)
        
        # Stil uygula
        styled_df = self._style_dataframe(
            city_display,
            color_column='Pazar Payı %'
        )
        
        st.dataframe(styled_df, use_container_width=True, height=400)
    
    def _create_performance_tab(self) -> None:
        """Performans Detay tab'ını oluştur"""
        st.header("🏢 Territory Performans Analizi")
        
        df = self._get_filtered_data()
        
        if len(df) == 0:
            st.warning("Seçilen filtrelerde veri bulunamadı")
            return
        
        # Ürün kolonlarını al
        product_cols = self.config.PRODUCT_COLUMN_MAP[st.session_state.current_product]
        pf_col = product_cols['pf']
        rakip_col = product_cols['rakip']
        
        # Territory bazlı veriyi hazırla
        territory_data = df.groupby(['TERRITORIES', 'REGION', 'CITY', 'MANAGER']).agg({
            pf_col: 'sum',
            rakip_col: 'sum'
        }).reset_index()
        
        territory_data.columns = ['Territory', 'Region', 'City', 'Manager', 'PF_Satis', 'Rakip_Satis']
        territory_data['Toplam_Pazar'] = territory_data['PF_Satis'] + territory_data['Rakip_Satis']
        territory_data['Pazar_Payi_%'] = (territory_data['PF_Satis'] / territory_data['Toplam_Pazar'].replace(0, 1)) * 100
        
        total_pf = territory_data['PF_Satis'].sum()
        territory_data['Agirlik_%'] = (territory_data['PF_Satis'] / total_pf * 100) if total_pf > 0 else 0
        
        # Metrikler
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            top_territory = territory_data.loc[territory_data['PF_Satis'].idxmax(), 'Territory'] if len(territory_data) > 0 else "Yok"
            st.metric("🥇 En İyi Territory", top_territory)
        
        with col2:
            avg_share = territory_data['Pazar_Payi_%'].mean()
            st.metric("📊 Ort. Pazar Payı", f"{avg_share:.1f}%")
        
        with col3:
            high_performance = len(territory_data[territory_data['Pazar_Payi_%'] > 50])
            st.metric("🎯 >%50 Pay", str(high_performance))
        
        with col4:
            total_territories = len(territory_data)
            st.metric("🏢 Toplam Territory", str(total_territories))
        
        st.markdown("---")
        
        # Görselleştirmeler
        col_viz1, col_viz2 = st.columns(2)
        
        with col_viz1:
            st.subheader("📊 Top 10 Territory")
            
            top_10 = territory_data.nlargest(10, 'PF_Satis')
            
            fig = px.bar(
                top_10,
                x='Territory',
                y='PF_Satis',
                color='Region',
                color_discrete_map=self.config.REGION_COLORS,
                title='En Yüksek Satış Yapan Territory\'ler',
                text_auto='.2s'
            )
            
            fig.update_layout(
                height=500,
                xaxis_tickangle=-45,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color=self.config.COLOR_PALETTE['text_primary'],
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col_viz2:
            st.subheader("🎯 Pazar Payı Dağılımı")
            
            fig = px.scatter(
                territory_data,
                x='PF_Satis',
                y='Pazar_Payi_%',
                size='Toplam_Pazar',
                color='Region',
                color_discrete_map=self.config.REGION_COLORS,
                hover_name='Territory',
                hover_data=['Manager', 'PF_Satis', 'Pazar_Payi_%'],
                title='Territory Performans Haritası'
            )
            
            fig.update_layout(
                height=500,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color=self.config.COLOR_PALETTE['text_primary'],
                xaxis_title='PF Satış',
                yaxis_title='Pazar Payı (%)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Detaylı Tablo
        st.subheader("📋 Detaylı Territory Listesi")
        
        # Filtreleme seçenekleri
        col_filter1, col_filter2 = st.columns(2)
        
        with col_filter1:
            show_count = st.slider("Gösterilecek Territory Sayısı", 10, 100, 25, 5)
        
        with col_filter2:
            sort_by = st.selectbox(
                "Sıralama Kriteri",
                ["PF Satış", "Pazar Payı", "Toplam Pazar", "Ağırlık %"],
                key="territory_sort"
            )
        
        sort_map = {
            "PF Satış": "PF_Satis",
            "Pazar Payı": "Pazar_Payi_%",
            "Toplam Pazar": "Toplam_Pazar",
            "Ağırlık %": "Agirlik_%"
        }
        
        territory_sorted = territory_data.sort_values(
            sort_map[sort_by],
            ascending=False
        ).head(show_count)
        
        # Tabloyu göster
        display_cols = ['Territory', 'Region', 'City', 'Manager', 'PF_Satis', 'Toplam_Pazar', 'Pazar_Payi_%', 'Agirlik_%']
        territory_display = territory_sorted[display_cols].copy()
        territory_display.columns = ['Territory', 'Bölge', 'Şehir', 'Manager', 'PF Satış', 'Toplam Pazar', 'Pazar Payı %', 'Ağırlık %']
        territory_display.index = range(1, len(territory_display) + 1)
        
        # Stil uygula
        styled_df = self._style_dataframe(
            territory_display,
            color_column='Pazar Payı %',
            gradient_columns=['Ağırlık %', 'PF Satış']
        )
        
        st.dataframe(styled_df, use_container_width=True, height=500)
    
    def _create_trend_tab(self) -> None:
        """Trend & Tahmin tab'ını oluştur"""
        st.header("📈 Zaman Serisi Analizi ve Trendler")
        
        df = self._get_filtered_data()
        
        if len(df) == 0:
            st.warning("Seçilen filtrelerde veri bulunamadı")
            return
        
        # Ürün kolonlarını al
        product_cols = self.config.PRODUCT_COLUMN_MAP[st.session_state.current_product]
        pf_col = product_cols['pf']
        rakip_col = product_cols['rakip']
        
        # Aylık veriyi hazırla
        monthly_data = df.groupby('YIL_AY').agg({
            pf_col: 'sum',
            rakip_col: 'sum',
            'DATE': 'first'
        }).reset_index().sort_values('YIL_AY')
        
        monthly_data.columns = ['YIL_AY', 'PF_Satis', 'Rakip_Satis', 'DATE']
        monthly_data['Toplam_Pazar'] = monthly_data['PF_Satis'] + monthly_data['Rakip_Satis']
        monthly_data['Pazar_Payi_%'] = (monthly_data['PF_Satis'] / monthly_data['Toplam_Pazar'].replace(0, 1)) * 100
        monthly_data['PF_Buyume_%'] = monthly_data['PF_Satis'].pct_change() * 100
        monthly_data['Rakip_Buyume_%'] = monthly_data['Rakip_Satis'].pct_change() * 100
        monthly_data['Goreceli_Buyume_%'] = monthly_data['PF_Buyume_%'] - monthly_data['Rakip_Buyume_%']
        
        # Metrikler
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_pf = monthly_data['PF_Satis'].mean()
            st.metric("📊 Ort. Aylık PF", self._format_number(avg_pf))
        
        with col2:
            avg_growth = monthly_data['PF_Buyume_%'].mean()
            st.metric("📈 Ort. Büyüme", f"{avg_growth:.1f}%")
        
        with col3:
            avg_share = monthly_data['Pazar_Payi_%'].mean()
            st.metric("🎯 Ort. Pazar Payı", f"{avg_share:.1f}%")
        
        with col4:
            win_months = len(monthly_data[monthly_data['Goreceli_Buyume_%'] > 0])
            total_months = len(monthly_data)
            st.metric("🏆 Kazanılan Aylar", f"{win_months}/{total_months}")
        
        st.markdown("---")
        
        # Görselleştirmeler
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.subheader("💰 Satış Trendi")
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=monthly_data['DATE'],
                y=monthly_data['PF_Satis'],
                mode='lines+markers',
                name='PF Satış',
                line=dict(color=self.config.COLOR_PALETTE['success_medium'], width=3),
                marker=dict(size=8, color='white', line=dict(width=2, color=self.config.COLOR_PALETTE['success_medium']))
            ))
            
            fig.add_trace(go.Scatter(
                x=monthly_data['DATE'],
                y=monthly_data['Rakip_Satis'],
                mode='lines+markers',
                name='Rakip Satış',
                line=dict(color=self.config.COLOR_PALETTE['danger_medium'], width=3),
                marker=dict(size=8, color='white', line=dict(width=2, color=self.config.COLOR_PALETTE['danger_medium']))
            ))
            
            fig.update_layout(
                height=500,
                xaxis_title='Tarih',
                yaxis_title='Satış',
                hovermode='x unified',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color=self.config.COLOR_PALETTE['text_primary'],
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col_chart2:
            st.subheader("📈 Büyüme Oranları")
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=monthly_data['DATE'],
                y=monthly_data['PF_Buyume_%'],
                mode='lines+markers',
                name='PF Büyüme',
                line=dict(color=self.config.COLOR_PALETTE['success_medium'], width=3),
                marker=dict(size=8, color='white', line=dict(width=2, color=self.config.COLOR_PALETTE['success_medium'])),
                fill='tozeroy',
                fillcolor='rgba(16, 185, 129, 0.1)'
            ))
            
            fig.add_trace(go.Scatter(
                x=monthly_data['DATE'],
                y=monthly_data['Rakip_Buyume_%'],
                mode='lines+markers',
                name='Rakip Büyüme',
                line=dict(color=self.config.COLOR_PALETTE['danger_medium'], width=3),
                marker=dict(size=8, color='white', line=dict(width=2, color=self.config.COLOR_PALETTE['danger_medium'])),
                fill='tozeroy',
                fillcolor='rgba(239, 68, 68, 0.1)'
            ))
            
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            
            fig.update_layout(
                height=500,
                xaxis_title='Tarih',
                yaxis_title='Büyüme (%)',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color=self.config.COLOR_PALETTE['text_primary'],
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Detaylı Tablo
        st.subheader("📋 Aylık Performans Detayları")
        
        display_cols = ['YIL_AY', 'PF_Satis', 'Rakip_Satis', 'Toplam_Pazar', 'Pazar_Payi_%', 'PF_Buyume_%', 'Rakip_Buyume_%', 'Goreceli_Buyume_%']
        monthly_display = monthly_data[display_cols].copy()
        monthly_display.columns = ['Ay', 'PF Satış', 'Rakip Satış', 'Toplam Pazar', 'Pazar Payı %', 'PF Büyüme %', 'Rakip Büyüme %', 'Göreceli Büyüme %']
        monthly_display.index = range(1, len(monthly_display) + 1)
        
        # Stil uygula
        styled_df = self._style_dataframe(
            monthly_display,
            color_column='Göreceli Büyüme %',
            gradient_columns=['Pazar Payı %', 'PF Büyüme %']
        )
        
        st.dataframe(styled_df, use_container_width=True, height=400)
    
    def _create_strategic_tab(self) -> None:
        """Stratejik Analiz tab'ını oluştur"""
        st.header("🎯 Stratejik Analiz ve Planlama")
        
        df = self._get_filtered_data()
        
        if len(df) == 0:
            st.warning("Seçilen filtrelerde veri bulunamadı")
            return
        
        # Ürün kolonlarını al
        product_cols = self.config.PRODUCT_COLUMN_MAP[st.session_state.current_product]
        pf_col = product_cols['pf']
        rakip_col = product_cols['rakip']
        
        # BCG Analizi
        st.subheader("⭐ BCG Matrix Analizi")
        
        # Territory bazlı veriyi hazırla
        territory_data = df.groupby(['TERRITORIES', 'REGION', 'CITY']).agg({
            pf_col: 'sum',
            rakip_col: 'sum'
        }).reset_index()
        
        territory_data.columns = ['Territory', 'Region', 'City', 'PF_Satis', 'Rakip_Satis']
        territory_data['Toplam_Pazar'] = territory_data['PF_Satis'] + territory_data['Rakip_Satis']
        territory_data['Pazar_Payi_%'] = (territory_data['PF_Satis'] / territory_data['Toplam_Pazar'].replace(0, 1)) * 100
        territory_data['Goreceli_Pazar_Payi'] = territory_data['PF_Satis'] / territory_data['Rakip_Satis'].replace(0, 1)
        
        # Büyüme hesapla (basit versiyon)
        territory_data['Pazar_Buyume_%'] = 0  # Gerçek uygulamada tarihsel büyüme hesaplanmalı
        
        # BCG analizi yap
        bcg_results = self.bcg_analyzer.analyze(territory_data)
        
        if bcg_results["bcg_matrix"] is not None:
            bcg_df = bcg_results["bcg_matrix"]
            
            # BCG Görselleştirme
            fig = px.scatter(
                bcg_df,
                x='Goreceli_Pazar_Payi',
                y='Pazar_Buyume_%',
                size='PF_Satis',
                color='BCG_Kategori',
                color_discrete_map=self.config.BCG_COLORS,
                hover_name='Territory',
                hover_data=['Region', 'PF_Satis', 'Pazar_Payi_%'],
                title='BCG Matrix - Stratejik Konumlandırma',
                labels={
                    'Goreceli_Pazar_Payi': 'Göreceli Pazar Payı',
                    'Pazar_Buyume_%': 'Pazar Büyüme Oranı (%)'
                }
            )
            
            # Medyan çizgileri
            median_share = bcg_df['Goreceli_Pazar_Payi'].median()
            median_growth = bcg_df['Pazar_Buyume_%'].median()
            
            fig.add_hline(y=median_growth, line_dash="dash", line_color="gray", opacity=0.5)
            fig.add_vline(x=median_share, line_dash="dash", line_color="gray", opacity=0.5)
            
            fig.update_layout(
                height=600,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color=self.config.COLOR_PALETTE['text_primary']
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # BCG Özeti
            st.markdown("---")
            st.subheader("📊 BCG Portföy Dağılımı")
            
            if bcg_results["category_summary"]:
                summary = bcg_results["category_summary"]
                
                cols = st.columns(4)
                categories = list(self.config.BCG_COLORS.keys())
                
                for idx, category in enumerate(categories):
                    with cols[idx]:
                        cat_data = summary.get(category, {})
                        count = cat_data.get("count", 0)
                        sales = cat_data.get("total_sales", 0)
                        
                        st.metric(
                            category,
                            f"{count} Territory",
                            delta=f"{self._format_number(sales)} PF"
                        )
            
            # Stratejik Çıkarımlar
            st.markdown("---")
            st.subheader("💡 Stratejik Çıkarımlar")
            
            if bcg_results["strategic_implications"]:
                for implication in bcg_results["strategic_implications"]:
                    self.ui.create_insight_card("Stratejik Öneri", implication, "info")
        
        # Yatırım Stratejisi Analizi
        st.markdown("---")
        st.subheader("💰 Yatırım Stratejisi Analizi")
        
        # Şehir bazlı veriyi hazırla
        city_data = df.groupby(['CITY_NORMALIZED', 'REGION']).agg({
            pf_col: 'sum',
            rakip_col: 'sum'
        }).reset_index()
        
        city_data.columns = ['City', 'Region', 'PF_Satis', 'Rakip_Satis']
        city_data['Toplam_Pazar'] = city_data['PF_Satis'] + city_data['Rakip_Satis']
        city_data['Pazar_Payi_%'] = (city_data['PF_Satis'] / city_data['Toplam_Pazar'].replace(0, 1)) * 100
        
        # Yatırım stratejisi analizi yap
        strategy_results = self.investment_strategy_analyzer.analyze(city_data)
        
        if strategy_results["strategy_matrix"] is not None:
            strategy_df = strategy_results["strategy_matrix"]
            
            # Strateji Dağılımı
            if strategy_results["strategy_distribution"]:
                distribution = strategy_results["strategy_distribution"]
                
                cols = st.columns(5)
                strategies = list(self.config.STRATEGY_COLORS.keys())
                
                for idx, strategy in enumerate(strategies):
                    with cols[idx]:
                        strat_data = distribution.get(strategy, {})
                        count = strat_data.get("count", 0)
                        sales = strat_data.get("total_sales", 0)
                        
                        st.metric(
                            strategy,
                            f"{count} Şehir",
                            delta=f"{self._format_number(sales)} PF"
                        )
            
            # Yatırım Önerileri
            st.markdown("---")
            st.subheader("🎯 Yatırım Önerileri")
            
            if strategy_results["investment_recommendations"]:
                for recommendation in strategy_results["investment_recommendations"]:
                    self.ui.create_insight_card("Yatırım Önerisi", recommendation, "info")
    
    def _create_ml_tab(self) -> None:
        """ML İleri Analiz tab'ını oluştur"""
        st.header("🤖 Makine Öğrenmesi İle İleri Analiz")
        
        df = self._get_filtered_data()
        
        if len(df) == 0:
            st.warning("Seçilen filtrelerde veri bulunamadı")
            return
        
        # Ürün kolonlarını al
        product_cols = self.config.PRODUCT_COLUMN_MAP[st.session_state.current_product]
        pf_col = product_cols['pf']
        
        # Aylık veriyi hazırla
        monthly_data = df.groupby('YIL_AY').agg({
            pf_col: 'sum',
            'DATE': 'first'
        }).reset_index().sort_values('YIL_AY')
        
        monthly_data.columns = ['YIL_AY', 'PF_Satis', 'DATE']
        
        # Tahmin periyodu seçimi
        st.subheader("📅 Tahmin Ayarları")
        
        col_set1, col_set2 = st.columns(2)
        
        with col_set1:
            forecast_periods = st.slider(
                "Tahmin Periyodu (Ay)",
                min_value=1,
                max_value=12,
                value=3,
                step=1
            )
        
        with col_set2:
            test_size = st.slider(
                "Test Seti Oranı (%)",
                min_value=10,
                max_value=40,
                value=20,
                step=5
            ) / 100
        
        st.markdown("---")
        
        # ML Tahminleme
        st.subheader("🔮 Satış Tahmini")
        
        if len(monthly_data) < 10:
            st.warning("Tahmin için en az 10 ay veri gereklidir")
            return
        
        with st.spinner("ML modelleri eğitiliyor..."):
            # ML parametrelerini güncelle
            self.sales_forecaster.ml_params["test_size"] = test_size
            self.sales_forecaster.ml_params["forecast_periods"] = forecast_periods
            
            # Tahmin yap
            forecast_results = self.sales_forecaster.forecast(
                monthly_data,
                target_column="PF_Satis",
                forecast_periods=forecast_periods
            )
        
        if "error" in forecast_results:
            st.error(f"Tahmin hatası: {forecast_results['error']}")
            return
        
        # Model Performansı
        st.subheader("📊 Model Performans Karşılaştırması")
        
        if forecast_results["model_performance"]:
            perf_data = []
            
            for model_name, result in forecast_results["model_performance"].items():
                metrics = result["metrics"]
                perf_data.append({
                    'Model': model_name,
                    'Test MAE': metrics["test_mae"],
                    'Test RMSE': metrics["test_rmse"],
                    'Test MAPE (%)': metrics["test_mape"],
                    'Test R²': metrics["test_r2"]
                })
            
            perf_df = pd.DataFrame(perf_data)
            perf_df = perf_df.sort_values('Test MAPE (%)')
            
            # Stil uygula
            styled_perf = self._style_dataframe(
                perf_df,
                color_column='Test MAPE (%)',
                gradient_columns=['Test R²']
            )
            
            st.dataframe(styled_perf, use_container_width=True)
            
            # En iyi model
            best_model = forecast_results["best_model"]
            best_metrics = forecast_results["model_performance"][best_model]["metrics"]
            
            st.markdown(f"**🏆 En İyi Model:** {best_model}")
            st.markdown(f"**📈 Test MAPE:** {best_metrics['test_mape']:.2f}%")
            st.markdown(f"**🎯 Test R²:** {best_metrics['test_r2']:.3f}")
        
        # Tahmin Grafiği
        st.markdown("---")
        st.subheader("📈 Gerçek vs Tahmin Edilen Satışlar")
        
        if forecast_results["forecast"] is not None:
            forecast_df = forecast_results["forecast"]
            
            # Geçmiş ve geleceği birleştir
            historical_dates = monthly_data['DATE'].tolist()
            historical_values = monthly_data['PF_Satis'].tolist()
            
            forecast_dates = forecast_df['DATE'].tolist()
            forecast_values = forecast_df['PF_Satis'].tolist()
            
            all_dates = historical_dates + forecast_dates
            all_values = historical_values + forecast_values
            all_types = ['Gerçek'] * len(historical_dates) + ['Tahmin'] * len(forecast_dates)
            
            combined_df = pd.DataFrame({
                'DATE': all_dates,
                'PF_Satis': all_values,
                'Type': all_types
            })
            
            # Grafik oluştur
            fig = go.Figure()
            
            # Gerçek veri
            real_data = combined_df[combined_df['Type'] == 'Gerçek']
            fig.add_trace(go.Scatter(
                x=real_data['DATE'],
                y=real_data['PF_Satis'],
                mode='lines+markers',
                name='Gerçek Satış',
                line=dict(color=self.config.COLOR_PALETTE['success_medium'], width=3),
                marker=dict(size=8, color='white', line=dict(width=2, color=self.config.COLOR_PALETTE['success_medium']))
            ))
            
            # Tahmin verisi
            forecast_data = combined_df[combined_df['Type'] == 'Tahmin']
            fig.add_trace(go.Scatter(
                x=forecast_data['DATE'],
                y=forecast_data['PF_Satis'],
                mode='lines+markers',
                name='Tahmin',
                line=dict(color=self.config.COLOR_PALETTE['warning_medium'], width=3, dash='dash'),
                marker=dict(size=10, symbol='diamond', color='white', 
                          line=dict(width=2, color=self.config.COLOR_PALETTE['warning_medium']))
            ))
            
            fig.update_layout(
                height=500,
                xaxis_title='Tarih',
                yaxis_title='PF Satış',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color=self.config.COLOR_PALETTE['text_primary'],
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Tahmin Detayları
            st.markdown("---")
            st.subheader("📋 Tahmin Detayları")
            
            forecast_display = forecast_df[['YIL_AY', 'PF_Satis']].copy()
            forecast_display.columns = ['Ay', 'Tahmin Edilen Satış']
            forecast_display.index = range(1, len(forecast_display) + 1)
            
            styled_forecast = self._style_dataframe(
                forecast_display,
                gradient_columns=['Tahmin Edilen Satış']
            )
            
            st.dataframe(styled_forecast, use_container_width=True)
    
    def _create_reports_tab(self) -> None:
        """Raporlar tab'ını oluştur"""
        st.header("📥 Rapor İndirme ve Dışa Aktarma")
        
        if not st.session_state.data_loaded:
            st.warning("Lütfen önce veri yükleyin")
            return
        
        st.markdown("""
        <div class="custom-card">
            <h3>📊 Kapsamlı Excel Raporu</h3>
            <p>Tüm analiz sonuçlarını içeren detaylı bir Excel raporu oluşturun.</p>
            <p>Rapor aşağıdaki sayfaları içerecektir:</p>
            <ul>
                <li>Genel Performans Özeti</li>
                <li>Territory Bazlı Analiz</li>
                <li>Şehir Bazlı Analiz</li>
                <li>Zaman Serisi Analizi</li>
                <li>BCG Matrix Sonuçları</li>
                <li>Yatırım Stratejisi Önerileri</li>
                <li>ML Tahmin Sonuçları</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("📈 Excel Raporu Oluştur", type="primary", use_container_width=True):
            with st.spinner("Rapor hazırlanıyor..."):
                try:
                    df = self._get_filtered_data()
                    
                    if len(df) == 0:
                        st.error("Rapor için yeterli veri yok")
                        return
                    
                    # Ürün kolonlarını al
                    product_cols = self.config.PRODUCT_COLUMN_MAP[st.session_state.current_product]
                    pf_col = product_cols['pf']
                    rakip_col = product_cols['rakip']
                    
                    # Tüm analizleri yap
                    
                    # 1. Territory Bazlı Analiz
                    territory_data = df.groupby(['TERRITORIES', 'REGION', 'CITY', 'MANAGER']).agg({
                        pf_col: 'sum',
                        rakip_col: 'sum'
                    }).reset_index()
                    
                    territory_data['Toplam_Pazar'] = territory_data[pf_col] + territory_data[rakip_col]
                    territory_data['Pazar_Payi_%'] = (territory_data[pf_col] / territory_data['Toplam_Pazar'].replace(0, 1)) * 100
                    
                    # 2. Şehir Bazlı Analiz
                    city_data = df.groupby(['CITY_NORMALIZED', 'REGION']).agg({
                        pf_col: 'sum',
                        rakip_col: 'sum'
                    }).reset_index()
                    
                    city_data['Toplam_Pazar'] = city_data[pf_col] + city_data[rakip_col]
                    city_data['Pazar_Payi_%'] = (city_data[pf_col] / city_data['Toplam_Pazar'].replace(0, 1)) * 100
                    
                    # 3. Zaman Serisi Analizi
                    monthly_data = df.groupby('YIL_AY').agg({
                        pf_col: 'sum',
                        rakip_col: 'sum',
                        'DATE': 'first'
                    }).reset_index().sort_values('YIL_AY')
                    
                    monthly_data['Toplam_Pazar'] = monthly_data[pf_col] + monthly_data[rakip_col]
                    monthly_data['Pazar_Payi_%'] = (monthly_data[pf_col] / monthly_data['Toplam_Pazar'].replace(0, 1)) * 100
                    monthly_data['PF_Buyume_%'] = monthly_data[pf_col].pct_change() * 100
                    
                    # 4. BCG Analizi
                    territory_for_bcg = territory_data.copy()
                    territory_for_bcg.columns = ['Territory', 'Region', 'City', 'PF_Satis', 'Rakip_Satis', 'Toplam_Pazar', 'Pazar_Payi_%']
                    territory_for_bcg['Goreceli_Pazar_Payi'] = territory_for_bcg['PF_Satis'] / territory_for_bcg['Rakip_Satis'].replace(0, 1)
                    territory_for_bcg['Pazar_Buyume_%'] = 0  # Basit versiyon
                    
                    bcg_results = self.bcg_analyzer.analyze(territory_for_bcg)
                    bcg_df = bcg_results.get("bcg_matrix", pd.DataFrame())
                    
                    # 5. Yatırım Stratejisi
                    city_for_strategy = city_data.copy()
                    city_for_strategy.columns = ['City', 'Region', 'PF_Satis', 'Rakip_Satis', 'Toplam_Pazar', 'Pazar_Payi_%']
                    
                    strategy_results = self.investment_strategy_analyzer.analyze(city_for_strategy)
                    strategy_df = strategy_results.get("strategy_matrix", pd.DataFrame())
                    
                    # 6. ML Tahminleri
                    forecast_data = pd.DataFrame()  # Basit versiyon
                    
                    # Excel dosyası oluştur
                    output = BytesIO()
                    
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        # Özet sayfası
                        summary_data = {
                            'Parametre': ['Ürün', 'Dönem', 'Territory Sayısı', 'Şehir Sayısı', 'Başlangıç Tarihi', 'Bitiş Tarihi'],
                            'Değer': [
                                st.session_state.current_product,
                                st.session_state.date_filter[0].strftime('%Y-%m-%d') + ' - ' + st.session_state.date_filter[1].strftime('%Y-%m-%d') if st.session_state.date_filter else 'Tüm Veriler',
                                territory_data['TERRITORIES'].nunique(),
                                city_data['CITY_NORMALIZED'].nunique(),
                                df['DATE'].min().strftime('%Y-%m-%d'),
                                df['DATE'].max().strftime('%Y-%m-%d')
                            ]
                        }
                        summary_df = pd.DataFrame(summary_data)
                        summary_df.to_excel(writer, sheet_name='Özet', index=False)
                        
                        # Territory Analizi
                        territory_data.to_excel(writer, sheet_name='Territory_Analizi', index=False)
                        
                        # Şehir Analizi
                        city_data.to_excel(writer, sheet_name='Şehir_Analizi', index=False)
                        
                        # Zaman Serisi
                        monthly_data.to_excel(writer, sheet_name='Zaman_Serisi', index=False)
                        
                        # BCG Matrix
                        if len(bcg_df) > 0:
                            bcg_df.to_excel(writer, sheet_name='BCG_Matrix', index=False)
                        
                        # Yatırım Stratejisi
                        if len(strategy_df) > 0:
                            strategy_df.to_excel(writer, sheet_name='Yatırım_Stratejisi', index=False)
                        
                        # ML Tahminleri
                        if len(forecast_data) > 0:
                            forecast_data.to_excel(writer, sheet_name='ML_Tahminleri', index=False)
                    
                    st.success("✅ Rapor başarıyla oluşturuldu!")
                    
                    # İndirme butonu
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f"ticari_portfoy_raporu_{st.session_state.current_product}_{timestamp}.xlsx"
                    
                    st.download_button(
                        label="💾 Raporu İndir",
                        data=output.getvalue(),
                        file_name=filename,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                    
                except Exception as e:
                    st.error(f"Rapor oluşturma hatası: {str(e)}")
    
    def _format_number(self, num: float) -> str:
        """Sayıları okunabilir formatta göster"""
        if pd.isna(num):
            return "0"
        
        try:
            num = float(num)
            if num == 0:
                return "0"
            elif abs(num) >= 1_000_000_000:
                return f"{num/1_000_000_000:,.1f}B"
            elif abs(num) >= 1_000_000:
                return f"{num/1_000_000:,.1f}M"
            elif abs(num) >= 1_000:
                return f"{num/1_000:,.1f}K"
            else:
                return f"{num:,.0f}"
        except:
            return str(num)
    
    def _style_dataframe(self, 
                         df: pd.DataFrame, 
                         color_column: str = None,
                         gradient_columns: List[str] = None) -> pd.DataFrame:
        """
        DataFrame'e stil uygular
        
        Args:
            df (pd.DataFrame): Stil uygulanacak DataFrame
            color_column (str): Renklendirilecek kolon
            gradient_columns (List[str]): Gradyan uygulanacak kolonlar
            
        Returns:
            pd.DataFrame: Stil uygulanmış DataFrame
        """
        if gradient_columns is None:
            gradient_columns = []
        
        # Sayısal kolonları formatla
        styled_df = df.copy()
        
        for col in styled_df.columns:
            if col in df.columns and df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                if any(keyword in col.lower() for keyword in ['%', 'yüzde', 'pay', 'oran', 'büyüme']):
                    # Yüzdelik format
                    styled_df[col] = df[col].apply(lambda x: f"{x:,.1f}%" if pd.notnull(x) else "")
                else:
                    # Sayısal format
                    try:
                        styled_df[col] = df[col].apply(lambda x: self._format_number(x) if pd.notnull(x) else "")
                    except:
                        styled_df[col] = df[col].astype(str)
        
        # Pandas Styler oluştur
        styler = styled_df.style
        
        # Temel stil
        styler = styler.set_properties(**{
            'background-color': 'rgba(30, 41, 59, 0.7)',
            'color': self.config.COLOR_PALETTE['text_primary'],
            'border': f'1px solid {self.config.COLOR_PALETTE["primary_light"]}',
            'text-align': 'center'
        })
        
        # Başlık satırı
        styler = styler.set_table_styles([{
            'selector': 'thead th',
            'props': [
                ('background-color', self.config.COLOR_PALETTE['primary_medium']),
                ('color', 'white'),
                ('font-weight', '700'),
                ('border', f'1px solid {self.config.COLOR_PALETTE["primary_light"]}'),
                ('padding', '10px 8px')
            ]
        }])
        
        # Hücreler
        styler = styler.set_table_styles([{
            'selector': 'td',
            'props': [
                ('padding', '8px 6px')
            ]
        }])
        
        # Gradyan uygula
        for col in gradient_columns:
            if col in df.columns and df[col].dtype in ['int64', 'float64']:
                try:
                    col_data = df[col].astype(float)
                    min_val = col_data.min()
                    max_val = col_data.max()
                    
                    if min_val != max_val:
                        styler = styler.background_gradient(
                            subset=[col],
                            cmap='RdYlGn',
                            vmin=min_val,
                            vmax=max_val,
                            gmap=col_data
                        )
                except:
                    pass
        
        # Renk sütunu
        if color_column and color_column in df.columns:
            def color_cells(val):
                try:
                    num_val = float(val)
                    if num_val >= 70:
                        return f'background-color: rgba(16, 185, 129, 0.3); color: {self.config.COLOR_PALETTE["success_medium"]}; font-weight: 600'
                    elif num_val >= 40:
                        return f'background-color: rgba(245, 158, 11, 0.3); color: {self.config.COLOR_PALETTE["warning_medium"]}; font-weight: 600'
                    else:
                        return f'background-color: rgba(239, 68, 68, 0.3); color: {self.config.COLOR_PALETTE["danger_medium"]}; font-weight: 600'
                except:
                    return ''
            
            styler = styler.map(color_cells, subset=[color_column])
        
        # Alternatif satır renkleri
        styler = styler.set_table_styles([{
            'selector': 'tbody tr:nth-child(even)',
            'props': [('background-color', 'rgba(30, 41, 59, 0.5)')]
        }, {
            'selector': 'tbody tr:nth-child(odd)',
            'props': [('background-color', 'rgba(30, 41, 59, 0.3)')]
        }])
        
        return styler


# =============================================================================
# ANA UYGULAMA GİRİŞ NOKTASI
# =============================================================================

def main():
    """
    Ana uygulama giriş noktası
    """
    # Uyarıları gizle
    warnings.filterwarnings('ignore')
    
    # Logging konfigürasyonu
    logging.basicConfig(level=logging.ERROR)
    
    try:
        # Uygulamayı başlat
        app = StrategicPortfolioAnalyzer()
        app.run()
        
    except Exception as e:
        st.error(f"Uygulama hatası: {str(e)}")
        logging.error(f"Uygulama hatası: {e}", exc_info=True)


if __name__ == "__main__":
    main()
