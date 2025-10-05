import re

def clean_nro(nro_value):
    try:
        cleaned_value = re.sub(r'[^\d]', '', nro_value)
        return int(cleaned_value)
    except (ValueError, TypeError):
        return float('inf')
