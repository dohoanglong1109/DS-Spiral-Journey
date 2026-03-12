#%%
import pandas as pd
import sqlite3
#%% Dataset
url = "https://raw.githubusercontent.com/campusx-official/laptop-price-predictor-regression-project/main/laptop_data.csv"
full_df = pd.read_csv(url)
#%% 
df_laptops = full_df[['Company', 'TypeName']].copy()
df_laptops.index.name = 'laptop_id'
df_laptops = df_laptops.reset_index()
#%%
df_specs = full_df[['ScreenResolution', 'Inches', 'Cpu', 'Ram', 'Memory', 'Gpu', 'OpSys', 'Weight', 'Price']].copy()
df_specs['laptop_id'] = df_laptops['laptop_id']
#%%
conn = sqlite3.connect('../laptops.db')
df_laptops.to_sql('Laptops', conn, index=False, if_exists='replace')
df_specs.to_sql('Specs', conn, index=False, if_exists='replace')
conn.close() # Đóng kết nối sau khi tạo xong
