# src/train.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Import các module từ source code của bạn
from src.config import CATEGORICAL_COLS
from src.data_ingestion import ingest_and_query_data
from src.custom_transformers import LaptopFeatureExtractor, CompanyRareLabelEncoder

def build_preprocessing_pipeline():
    """Hàm này lắp ráp toàn bộ băng chuyền làm sạch và mã hóa dữ liệu"""
    
    # 1. Người gác cổng: Chỉ áp dụng One-Hot Encoding cho các cột Category
    categorical_transformer = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat_encoder', categorical_transformer, CATEGORICAL_COLS)
        ],
        remainder='passthrough' # Các cột số (như Ram, Weight, PPI) được giữ nguyên đi tiếp
    )

    # 2. Siêu băng chuyền: Nối các Custom Transformers với Preprocessor
    full_pipeline = Pipeline(steps=[
        ('feature_extractor', LaptopFeatureExtractor()),           # Bước 1: Rút trích đặc trưng
        ('rare_company_encoder', CompanyRareLabelEncoder(tol=10)), # Bước 2: Xử lý hãng hiếm
        ('encoder_and_router', preprocessor)                       # Bước 3: One-Hot Encoding
    ])
    
    return full_pipeline

def main():
    print("🚀 BƯỚC 1: Đang tải và truy vấn dữ liệu từ SQLite...")
    df = ingest_and_query_data()
    print(f"   -> Dữ liệu ban đầu có kích thước: {df.shape}")

    print("\n✂️ BƯỚC 2: Tách Features (X) và Target (y), chia tập Train/Test...")
    X = df.drop(columns=['Price'])
    y = np.log1p(df['Price']) # Biến đổi logarit cho Price
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    print(f"   -> Tập Train: {X_train.shape[0]} dòng | Tập Test: {X_test.shape[0]} dòng")

    print("\n⚙️ BƯỚC 3: Đưa dữ liệu qua Pipeline chuẩn hóa...")
    pipeline = build_preprocessing_pipeline()
    
    # Cho tập Train HỌC (fit) và BIẾN ĐỔI (transform)
    X_train_processed = pipeline.fit_transform(X_train)
    
    # Tập Test CHỈ ĐƯỢC BIẾN ĐỔI (transform) dựa trên những gì đã học từ tập Train
    X_test_processed = pipeline.transform(X_test)

    print("\n✅ HOÀN TẤT! Dữ liệu đã sẵn sàng 100% để đưa vào Model.")
    print(f"   -> Kích thước X_train sau mã hóa (toàn số): {X_train_processed.shape}")
    print(f"   -> Kích thước X_test sau mã hóa (toàn số): {X_test_processed.shape}")
    
    # Trả về các biến này để bước tiếp theo huấn luyện mô hình
    return X_train_processed, X_test_processed, y_train, y_test

if __name__ == "__main__":
    main()