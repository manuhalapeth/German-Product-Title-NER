import pandas as pd
import re
from unidecode import unidecode

# --- File paths ---
input_file = "ALLOUT_highF1_optimized.tsv"
output_file = "ALLOUT_highF1_postprocessed.tsv"

# --- Load TSV safely ---
try:
    df = pd.read_csv(input_file, sep='\t', dtype=str, on_bad_lines='skip')
    print(f"Loaded {len(df)} rows from {input_file}")
except Exception as e:
    print(f"Error reading {input_file}: {e}")
    raise

# --- Normalize column names ---
df.columns = [col.strip() for col in df.columns]

# --- Helper functions ---
def normalize_text(text):
    """Uppercase, strip extra spaces, normalize diacritics."""
    if pd.isna(text):
        return ''
    text = unidecode(text)  # remove accents
    text = text.upper()
    text = text.strip()
    return text

def split_multiple_entries(text):
    """Split entries like 'FIAT OPEL' into separate items."""
    if pd.isna(text):
        return []
    # Split by spaces, commas, slashes, or &
    items = re.split(r'[\s,/&]+', text)
    return [i for i in items if i]

def clean_model(text):
    """Remove noise from model strings."""
    if pd.isna(text):
        return ''
    text = text.replace('(', '').replace(')', '')
    text = text.replace('.', '').strip()
    return text

# --- Normalize specified text columns ---
text_cols = ['Kompatible_Fahrzeug_Marke', 'Kompatibles_Fahrzeug_Modell', 
             'Hersteller', 'Produktart', 'Im_Lieferumfang_Enthalten']

for col in text_cols:
    if col in df.columns:
        df[col] = df[col].apply(normalize_text)

# --- Expand multiple brands/models into separate rows ---
expanded_rows = []
for _, row in df.iterrows():
    brands = split_multiple_entries(row.get('Kompatible_Fahrzeug_Marke', ''))
    models = split_multiple_entries(row.get('Kompatibles_Fahrzeug_Modell', ''))
    
    if not brands:
        brands = ['']
    if not models:
        models = ['']
    
    for brand in brands:
        for model in models:
            new_row = row.copy()
            new_row['Kompatible_Fahrzeug_Marke'] = brand
            new_row['Kompatibles_Fahrzeug_Modell'] = clean_model(model)
            expanded_rows.append(new_row)

df_clean = pd.DataFrame(expanded_rows)
print(f"Expanded to {len(df_clean)} rows")

# --- Deduplicate rows safely ---
dedup_cols = ['Record Number', 'Category Id', 
              'Kompatible_Fahrzeug_Marke', 
              'Kompatibles_Fahrzeug_Modell', 
              'Produktart']

# Keep only columns that exist
dedup_cols_existing = [c for c in dedup_cols if c in df_clean.columns]
print(f"Deduplicating using columns: {dedup_cols_existing}")

df_clean.drop_duplicates(subset=dedup_cols_existing, inplace=True)

# --- Optional: clean extra spaces in all string columns ---
df_clean = df_clean.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# --- Save cleaned TSV ---
df_clean.to_csv(output_file, sep='\t', index=False)
print(f"Post-processed file saved as {output_file}")
