import os

# __file__ là file hiện tại (config.py). dirname 2 lần sẽ lùi từ src/ ra thư mục gốc.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Cấu hình dữ liệu
DATA_URL = "https://raw.githubusercontent.com/campusx-official/laptop-price-predictor-regression-project/main/laptop_data.csv"
DB_PATH = os.path.join(BASE_DIR, "data", "02_interim", "laptops.db")

# Cấu hình Features
CATEGORICAL_COLS = ['Company', 'TypeName', 'Cpu_brand', 'Gpu_brand', 'OpSys']