# src/train.py
import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor # Mô hình AI
from sklearn.metrics import r2_score, mean_absolute_error # Thước đo đánh giá

from src.config import CATEGORICAL_COLS, BASE_DIR
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

    # 2. Nối các Custom Transformers với Preprocessor
    full_pipeline = Pipeline(steps=[
        ('feature_extractor', LaptopFeatureExtractor()),           # 1. Rút trích đặc trưng
        ('rare_company_encoder', CompanyRareLabelEncoder(tol=10)), # 2. Xử lý hãng hiếm
        ('encoder_and_router', preprocessor),                      # 3. One-Hot Encoding        
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))                       
    ])
    
    return full_pipeline

def main():
    print("🚀 BƯỚC 1: Đang tải và truy vấn dữ liệu từ SQLite...")
    df = ingest_and_query_data()
    print(f"   -> Dữ liệu ban đầu có kích thước: {df.shape}")

    print("\n✂️ BƯỚC 2: Tách Features (X) và Target (y), chia tập Train/Test...")
    X = df.drop(columns=['Price'])
    y = np.log1p(df['Price']*300) # Biến đổi logarit cho Price
    # *300 ở đây vì giá gốc là Rupee, theo tỉ giá thì 1 Rupee = 300 VNĐ

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    print(f"   -> Tập Train: {X_train.shape[0]} dòng | Tập Test: {X_test.shape[0]} dòng")

    print("\n⚙️ BƯỚC 3: Đưa dữ liệu qua Pipeline chuẩn hóa...")
    pipeline = build_preprocessing_pipeline()
    
    # Ở bước này, Pipeline sẽ tự làm sạch X_train rồi đẩy thẳng vào RandomForest để học y_train
    pipeline.fit(X_train, y_train)

    print("\n📊 BƯỚC 4: Đánh giá mô hình trên tập Test...")
    y_pred_log = pipeline.predict(X_test)
    
    # Tính R2 Score (Độ r-bình phương: càng gần 1.0 càng tốt)
    r2 = r2_score(y_test, y_pred_log)
    
    # Tính MAE (Sai số tuyệt đối trung bình). Đảo ngược log để ra giá trị tiền thật
    y_test_real = np.expm1(y_test)
    y_pred_real = np.expm1(y_pred_log)
    mae = mean_absolute_error(y_test_real, y_pred_real)

    print(f"   -> Điểm R2 Score: {r2:.4f}")
    print(f"   -> Sai số trung bình (MAE): lệch khoảng {mae:.2f} (VND)")

    print("\n💾 BƯỚC 5: Xuất xưởng mô hình...")
    # Tạo thư mục models/ nếu chưa có
    models_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Lưu toàn bộ Pipeline (bao gồm cả các bước làm sạch và model) thành file .joblib
    model_path = os.path.join(models_dir, "laptop_price_pipeline.joblib")
    joblib.dump(pipeline, model_path)
    
    print(f"✅ HOÀN TẤT! Mô hình đã được lưu tại: {model_path}")

if __name__ == "__main__":
    main()