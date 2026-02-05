import requests
import pandas as pd
import time
from io import StringIO

def download_uniprot_with_ec(num_sequences=10000):
    """
    Download reviewed proteins with EC numbers from UniProt
    """
    
    # UniProt REST API query
    # Filters: reviewed:true (Swiss-Prot only) + has EC number
    base_url = "https://rest.uniprot.org/uniprotkb/stream"
    
    params = {
        'query': 'reviewed:true AND ec:*',  # Reviewed proteins with EC numbers
        'format': 'tsv',
        'fields': 'accession,sequence,ec,protein_name,length,organism_name',
        'size': num_sequences
    }
    
    print(f"Downloading {num_sequences} protein sequences from UniProt...")
    print("This may take 2-3 minutes...")
    
    response = requests.get(base_url, params=params)
    
    if response.status_code == 200:
        # Parse TSV response
        df = pd.read_csv(StringIO(response.text), sep='\t')
        
        # Rename columns for clarity
        df.columns = ['id', 'sequence', 'ec_number', 'protein_name', 'length', 'organism']
        
        print(f"\n✅ Downloaded {len(df)} sequences!")
        return df
    else:
        print(f"❌ Error: {response.status_code}")
        return None

# Run the download
df = download_uniprot_with_ec(num_sequences=10000)

if df is not None:
    # Quick exploration
    print("\n" + "="*50)
    print("DATA PREVIEW")
    print("="*50)
    print(df.head())
    
    print("\n" + "="*50)
    print("DATASET STATS")
    print("="*50)
    print(f"Total sequences: {len(df)}")
    print(f"Average sequence length: {df['length'].mean():.1f}")
    print(f"Unique EC numbers: {df['ec_number'].nunique()}")
    
    # Save to CSV
    df.to_csv('uniprot_raw_data.csv', index=False)
    print("\n✅ Saved to: uniprot_raw_data.csv")