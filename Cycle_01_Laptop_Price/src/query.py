#%%
import sqlite3
import pandas as pd
# %%
# Kết nối tới đúng file đó
conn = sqlite3.connect('../laptops.db')

sql_query = """
    select L.Company, S.ScreenResolution, S.Inches, S.Cpu, S.Ram, S.Memory, S.Gpu, S.Weight, S.OpSys, S.Price
    from Laptops L
    join Specs S on L.laptop_id = S.laptop_id
"""
df = pd.read_sql_query(sql_query, conn)
# %%
df.head()
# %%
