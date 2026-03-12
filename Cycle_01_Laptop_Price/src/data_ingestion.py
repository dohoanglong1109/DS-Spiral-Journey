#%%
import pandas as pd
import sqlite3
import os
from src.config import DATA_URL, DB_PATH

def ingest_and_query_data():
    full_df = pd.read_csv(DATA_URL)
    
    df_laptops = full_df[['Company', 'TypeName']].copy()
    df_laptops.index.name = 'laptop_id'
    df_laptops = df_laptops.reset_index()
    
    df_specs = full_df[['ScreenResolution', 'Inches', 'Cpu', 'Ram', 'Memory', 'Gpu', 'OpSys', 'Weight', 'Price']].copy()
    df_specs['laptop_id'] = df_laptops['laptop_id']
    
    # Đảm bảo thư mục tồn tại trước khi tạo DB
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    # Đẩy vào SQLite
    with sqlite3.connect(DB_PATH) as conn:
        df_laptops.to_sql('Laptops', conn, index=False, if_exists='replace')
        df_specs.to_sql('Specs', conn, index=False, if_exists='replace')
        
        # Truy vấn JOIN ngay trong cùng một luồng
        sql_query = """
            SELECT L.Company, L.TypeName, S.ScreenResolution, S.Inches, 
                   S.Cpu, S.Ram, S.Memory, S.Gpu, S.Weight, S.OpSys, S.Price
            FROM Laptops L
            JOIN Specs S ON L.laptop_id = S.laptop_id
        """
        df_joined = pd.read_sql_query(sql_query, conn)
        
    return df_joined
