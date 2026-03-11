#%%
import pandas as pd
import sqlite3
#%%
# 1. Dataset
url = "https://raw.githubusercontent.com/campusx-official/laptop-price-predictor-regression-project/main/laptop_data.csv"
full_df = pd.read_csv(url)
#%%
# 2. "Phẫu thuật" dataset thành 2 phần để luyện SQL JOIN
# Bảng 1: Thông tin thương hiệu và dòng máy
df_laptops = full_df[['Company', 'TypeName']].copy()
df_laptops.index.name = 'laptop_id'
df_laptops = df_laptops.reset_index()
#%%
# Bảng 2: Thông số kỹ thuật chi tiết và Giá
# Lưu ý: Cột giá trong bộ này là 'Price', RAM là 'Ram', Trọng lượng là 'Weight'
df_specs = full_df[['ScreenResolution', 'Inches', 'Cpu', 'Ram', 'Memory', 'Gpu', 'OpSys', 'Weight', 'Price']].copy()
df_specs['laptop_id'] = df_laptops['laptop_id']
#%%
# 3. Đẩy vào SQLite ảo để luyện SQL
conn = sqlite3.connect('../laptops.db')
df_laptops.to_sql('Laptops', conn, index=False, if_exists='replace')
df_specs.to_sql('Specs', conn, index=False, if_exists='replace')
conn.close() # Đóng kết nối sau khi tạo xong

# %%
