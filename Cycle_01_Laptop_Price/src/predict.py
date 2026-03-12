# src/predict.py
import pandas as pd
import numpy as np
import joblib
import os
from src.config import BASE_DIR

def predict_laptop_price(laptop_features):
    """
    Hàm nhận vào một dictionary chứa cấu hình laptop và trả về giá dự đoán.
    """
    # 1. Đường dẫn tới bộ não AI
    model_path = os.path.join(BASE_DIR, "models", "laptop_price_pipeline.joblib")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError("🚨 Không tìm thấy file Model! Hãy chạy 'python -m src.train' trước.")

    # 2. Load toàn bộ Băng chuyền + Model AI vào bộ nhớ
    pipeline = joblib.load(model_path)

    # 3. Chuyển dictionary của người dùng thành Pandas DataFrame (1 dòng)
    df_input = pd.DataFrame([laptop_features])

    # 4. Đưa qua Băng chuyền để dự đoán
    # Pipeline sẽ TỰ ĐỘNG làm sạch: cắt chữ 'GB', tính PPI, gom hãng hiếm, One-Hot encoding...
    # rồi nhét thẳng vào Random Forest.
    predicted_log_price = pipeline.predict(df_input)

    # 5. Đảo ngược phép biến đổi Logarit để ra tiền VNĐ thực tế
    real_price_vnd = np.expm1(predicted_log_price[0])

    return real_price_vnd

if __name__ == "__main__":
    # GIẢ LẬP: Khách hàng nhập thông tin trên Website
    # Lưu ý: Cấu trúc key phải khớp với các cột ban đầu trong tập train
    my_dream_laptop = {
        'Company': 'HP',
        'TypeName': 'Gaming',
        'ScreenResolution': 'FullHD 1920x1080',
        'Inches': 15.5,
        'Cpu': 'Intel Core i7 2.3GHz',
        'Ram': '16GB',          # Cố tình để nguyên chữ 'GB' xem máy có tự cắt không
        'Memory': '512GB SSD',      # Cố tình để text thô sơ
        'Gpu': 'NVIDIA Gearforce ',
        'Weight': '1.37kg',         # Cố tình để chữ 'kg'
        'OpSys': 'Windows'
    }

    print("🔍 Khách hàng vừa nhập cấu hình:")
    for key, value in my_dream_laptop.items():
        print(f"   - {key}: {value}")
        
    print("\n🧠 AI đang tính toán...")
    
    try:
        price = predict_laptop_price(my_dream_laptop)
        print(f"🎉 TING TING! Giá dự đoán cho chiếc laptop này là: {price:,.0f} VNĐ")
    except Exception as e:
        print(f"❌ Lỗi rồi: {e}")