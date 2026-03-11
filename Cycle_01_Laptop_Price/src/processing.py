import pandas as pd
import numpy as np

def clean_numeric_cols(df):
    df['Ram'] = df['Ram'].str.replace('GB', '').astype(int)
    df['Weight'] = df['Weight'].str.replace('kg', '').astype(int)
    return df    

def clean_screen_resolution(df):
    df = df.copy()
    # PPI
    xy_resolution = df['ScreenResolution'].str.extract(r'(\d+)x(\d+)')
    x_res = xy_resolution[0].astype(int)
    y_res = xy_resolution[1].astype(int)
    df['PPI'] = np.sqrt((x_res)**2 + (y_res)**2) / df['Inches'].astype(float)

    # screen resolution 
    df['TouchScreen'] = df['ScreenResolution'].str.contains('Touchscreen', case=False).astype(int)
    df['IPS'] = df['ScreenResolution'].str.contains('IPS', case=False).astype(int)

    df = df.drop(columns=['ScreenResolution', 'Inches'])
    return df

def clean_memory(df):
    ## memory
    # Tạo bản sao để tránh cảnh báo SettingWithCopyWarning
    df = df.copy() 
    
    drive_types = ['SSD', 'HDD', 'Flash Storage', 'Hybrid']
    
    for drive in drive_types:
        extracted = df['Memory'].str.extract(rf'(\d+)(GB|TB)\s+{drive}')  #-> [number, 'GB/TB']
        size = pd.to_numeric(extracted[0])
        multiplier = extracted[1].map({'GB': 1, 'TB': 1024})
        df[f'{drive}'] = (size * multiplier).fillna(0).astype(int)
    
    df.drop(columns=['Memory'], inplace=True)    
    return df

# ===Clean CPU, GPU, OS===
def fetch_cpu(text):
    words = text.split()
    header = " ".join(words[:3])
    if header == 'Intel Core i7' or header == 'Intel Core i5' or header == 'Intel Core i3':
        return header
    else:
        if words[0] == 'Intel':
            return 'Other Intel Processor'
        else:
            return 'AMD Processor'

def catOS(text):
    if "Windows" in text: 
        return 'Windows'
    if "Mac" in text or "mac" in text: 
        return "MacOS"
    else: 
        return 'Others/No OS/Linux'

def clean_cpu_gpu_os(df):
    df = df.copy()
    # gpu
    df['Gpu_brand'] = df['Gpu'].apply(lambda x: x.split()[0])
    df = df[df['Gpu_brand'] != 'ARM']

    # cpu
    df['Cpu_brand'] = df['Cpu'].apply(fetch_cpu)

    # os
    df['OpSys'] = df['OpSys'].apply(catOS)

    df.drop(columns=['Cpu', 'Gpu', 'OpSys'], inplace=True)
    return df
