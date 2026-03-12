# src/config.py

# Cấu hình dữ liệu
DATA_URL = "https://raw.githubusercontent.com/campusx-official/laptop-price-predictor-regression-project/main/laptop_data.csv"
DB_PATH = "../data/02_interim/laptops.db" # Đã sửa đường dẫn theo cấu trúc chuẩn

# Cấu hình Features
CATEGORICAL_COLS = ['Company', 'TypeName', 'Cpu_brand', 'Gpu_brand', 'OpSys']