import pandas as pd
import re 

def RAM_Standardization(ram):
    ram = ram.str.replace('GB', '')
    ram = ram.astype(int)

def extract_memory_gb(memory_str, drive_type):
    memory_str = str(memory_str)
    
    pattern = rf'(\d+)(GB|TB)\s+{drive_type}'
    match = re.search(pattern, memory_str, re.IGNORECASE)
    
    if match:
        size = int(match.group(1))
        unit = match.group(2).upper()

        return size * 1024 if unit == 'TB' else size
    return 0

