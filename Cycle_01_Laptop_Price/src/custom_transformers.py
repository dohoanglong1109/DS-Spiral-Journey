import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class LaptopFeatureExtractor(BaseEstimator, TransformerMixin):
    """Gộp các bước clean cơ bản không cần học tham số (stateless)"""
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        
        # 1. Clean Numeric
        X_copy['Ram'] = X_copy['Ram'].str.replace('GB', '').astype(int)
        X_copy['Weight'] = X_copy['Weight'].str.replace('kg', '').astype(float)
        
        # 2. Clean Screen Resolution (Vectorized)
        xy_resolution = X_copy['ScreenResolution'].str.extract(r'(\d+)x(\d+)')
        X_copy['PPI'] = np.sqrt(xy_resolution[0].astype(int)**2 + xy_resolution[1].astype(int)**2) / X_copy['Inches'].astype(float)
        X_copy['TouchScreen'] = X_copy['ScreenResolution'].str.contains('Touchscreen', case=False).astype(int)
        X_copy['IPS'] = X_copy['ScreenResolution'].str.contains('IPS', case=False).astype(int)
        
        # 3. Clean Memory
        drive_types = ['SSD', 'HDD', 'Flash Storage', 'Hybrid']
        for drive in drive_types:
            extracted = X_copy['Memory'].str.extract(rf'(\d+)(GB|TB)\s+{drive}')
            size = pd.to_numeric(extracted[0])
            multiplier = extracted[1].map({'GB': 1, 'TB': 1024})
            X_copy[f'{drive}'] = (size * multiplier).fillna(0).astype(int)
            
        # 4. Clean GPU, CPU, OS
        X_copy['Gpu_brand'] = X_copy['Gpu'].apply(lambda x: x.split()[0])
        
        X_copy.loc[X_copy['Gpu_brand'] == 'ARM', 'Gpu_brand'] = 'Other'
        
        X_copy['Cpu_brand'] = X_copy['Cpu'].apply(self._fetch_cpu)
        X_copy['OpSys'] = X_copy['OpSys'].apply(self._cat_os)
        
        # Xóa các cột gốc
        cols_to_drop = ['ScreenResolution', 'Inches', 'Memory', 'Cpu', 'Gpu'] 
        X_copy = X_copy.drop(columns=cols_to_drop)
        
        return X_copy

    def _fetch_cpu(self, text):
        words = text.split()
        header = " ".join(words[:3])
        if header in ['Intel Core i7', 'Intel Core i5', 'Intel Core i3']:
            return header
        elif words[0] == 'Intel':
            return 'Other Intel Processor'
        else:
            return 'AMD Processor'

    def _cat_os(self, text):
        if "Windows" in text: 
            return 'Windows'
        elif "Mac" in text or "mac" in text: 
            return "MacOS"
        else: 
            return 'Others/No OS/Linux'

class CompanyRareLabelEncoder(BaseEstimator, TransformerMixin):
    """Đây là một Stateful Transformer: Cần học xem hãng nào là hiếm từ tập Train"""
    def __init__(self, tol=10):
        self.tol = tol
        self.common_companies_ = [] # Lưu trữ trạng thái học được
        
    def fit(self, X, y=None):
        # Học từ tập Train: Tìm các hãng xuất hiện > tol lần
        brand_counts = X['Company'].value_counts()
        self.common_companies_ = brand_counts[brand_counts > self.tol].index.tolist()
        return self
        
    def transform(self, X):
        X_copy = X.copy()
        # Áp dụng cho tập Transform: Hãng nào không nằm trong danh sách đã học -> gán là 'Other'
        X_copy['Company'] = X_copy['Company'].where(X_copy['Company'].isin(self.common_companies_), 'Other')
        return X_copy
