import pandas as pd
import re

# Load the raw data
df = pd.read_csv('uniprot_raw_data.csv')

def extract_ec_class(ec_string):
    """
    Extract EC class (first digit) from EC number
    Example: '3.4.21.4' -> 'EC3'
    Example: '1.1.1.1; 2.3.1.5' -> 'EC1' (take first if multiple)
    """
    if pd.isna(ec_string):
        return None
    
    # Take first EC number if multiple
    first_ec = str(ec_string).split(';')[0].strip()
    
    # Extract first digit
    match = re.match(r'^(\d)', first_ec)
    if match:
        return f"EC{match.group(1)}"
    return None

# Apply extraction
df['label'] = df['ec_number'].apply(extract_ec_class)

# Remove sequences without valid EC class
df_clean = df.dropna(subset=['label']).copy()

print("="*50)
print("LABEL DISTRIBUTION")
print("="*50)
print(df_clean['label'].value_counts())

print(f"\n✅ Clean dataset: {len(df_clean)} sequences")
print(f"Removed {len(df) - len(df_clean)} sequences without valid EC class")

# Save final clean dataset
final_df = df_clean[['id', 'sequence', 'label']].copy()
final_df.to_csv('protein_dataset_clean.csv', index=False)

print("\n✅ Final dataset saved: protein_dataset_clean.csv")
print("\nFirst 5 rows:")
print(final_df.head())