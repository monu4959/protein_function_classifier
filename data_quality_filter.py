import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

print("="*60)
print("PROTEIN SEQUENCE DATA QUALITY PIPELINE")
print("="*60)

# Load raw data
df = pd.read_csv('protein_dataset_clean.csv')
print(f"\n📊 Initial dataset: {len(df)} sequences")

# ============================================
# 1. AMINO ACID VALIDITY CHECK
# ============================================
print("\n" + "="*60)
print("STEP 1: Amino Acid Validity Analysis")
print("="*60)

VALID_AA = set('ACDEFGHIKLMNPQRSTVWY')

def check_sequence_validity(seq):
    """Check if sequence contains only valid amino acids"""
    return all(aa in VALID_AA for aa in seq)

def get_invalid_chars(seq):
    """Get set of invalid characters in sequence"""
    return set(aa for aa in seq if aa not in VALID_AA)

# Analyze sequences
df['is_valid'] = df['sequence'].apply(check_sequence_validity)
df['invalid_chars'] = df['sequence'].apply(get_invalid_chars)

# Statistics
valid_count = df['is_valid'].sum()
invalid_count = len(df) - valid_count

print(f"\n✅ Valid sequences: {valid_count} ({valid_count/len(df)*100:.2f}%)")
print(f"❌ Invalid sequences: {invalid_count} ({invalid_count/len(df)*100:.2f}%)")

# Show what invalid characters exist
if invalid_count > 0:
    all_invalid_chars = set()
    for chars in df[~df['is_valid']]['invalid_chars']:
        all_invalid_chars.update(chars)
    
    print(f"\n🔍 Found invalid amino acids: {sorted(all_invalid_chars)}")
    
    # Count frequency of each invalid character
    char_counts = Counter()
    for chars in df[~df['is_valid']]['invalid_chars']:
        char_counts.update(chars)
    
    print("\nInvalid character frequencies:")
    for char, count in char_counts.most_common():
        print(f"  '{char}': {count} sequences")

# Filter to valid sequences only
df_valid = df[df['is_valid']].copy()
print(f"\n✂️ Removed {invalid_count} sequences with invalid amino acids")

# ============================================
# 2. SEQUENCE LENGTH ANALYSIS & FILTERING
# ============================================
print("\n" + "="*60)
print("STEP 2: Sequence Length Analysis")
print("="*60)

df_valid['seq_length'] = df_valid['sequence'].str.len()

# Statistics
print(f"\nLength statistics:")
print(f"  Mean: {df_valid['seq_length'].mean():.1f}")
print(f"  Median: {df_valid['seq_length'].median():.1f}")
print(f"  Min: {df_valid['seq_length'].min()}")
print(f"  Max: {df_valid['seq_length'].max()}")
print(f"  Std: {df_valid['seq_length'].std():.1f}")

# Percentiles
percentiles = [1, 5, 25, 50, 75, 95, 99]
print(f"\nLength percentiles:")
for p in percentiles:
    val = df_valid['seq_length'].quantile(p/100)
    print(f"  {p}th: {val:.0f}")

# Length distribution by class
print(f"\nLength by EC class:")
print(df_valid.groupby('label')['seq_length'].agg(['mean', 'median', 'min', 'max']))

# ============================================
# 3. APPLY LENGTH FILTERS
# ============================================
print("\n" + "="*60)
print("STEP 3: Applying Length Filters")
print("="*60)

# Strategy: Remove extreme outliers
# Keep sequences between 50 and 1000 amino acids
# This covers 95%+ of proteins while removing edge cases

MIN_LENGTH = 50
MAX_LENGTH = 1000

before_filter = len(df_valid)
df_filtered = df_valid[
    (df_valid['seq_length'] >= MIN_LENGTH) & 
    (df_valid['seq_length'] <= MAX_LENGTH)
].copy()
after_filter = len(df_filtered)

removed = before_filter - after_filter
print(f"\nLength filter: {MIN_LENGTH}-{MAX_LENGTH} amino acids")
print(f"  Kept: {after_filter} sequences ({after_filter/before_filter*100:.2f}%)")
print(f"  Removed: {removed} sequences ({removed/before_filter*100:.2f}%)")

# ============================================
# 4. CLASS BALANCE CHECK
# ============================================
print("\n" + "="*60)
print("STEP 4: Class Distribution After Filtering")
print("="*60)

class_dist = df_filtered['label'].value_counts().sort_index()
print("\nFinal class distribution:")
print(class_dist)
print(f"\nTotal sequences: {len(df_filtered)}")

# Check for class imbalance
max_class = class_dist.max()
min_class = class_dist.min()
imbalance_ratio = max_class / min_class
print(f"\nClass imbalance ratio: {imbalance_ratio:.2f}:1")
if imbalance_ratio > 5:
    print("⚠️  Warning: Significant class imbalance detected")
    print("   Consider using class weights or stratified sampling")
else:
    print("✅ Class distribution is reasonably balanced")

# ============================================
# 5. FINAL DATASET SUMMARY
# ============================================
print("\n" + "="*60)
print("FINAL DATASET SUMMARY")
print("="*60)

original_count = len(df)
final_count = len(df_filtered)
total_removed = original_count - final_count
removal_pct = (total_removed / original_count) * 100

print(f"\n📊 Original dataset: {original_count:,} sequences")
print(f"✅ Final dataset: {final_count:,} sequences")
print(f"✂️ Total removed: {total_removed:,} ({removal_pct:.2f}%)")
print(f"\n📏 Length range: {df_filtered['seq_length'].min()}-{df_filtered['seq_length'].max()}")
print(f"📏 Average length: {df_filtered['seq_length'].mean():.1f}")

# ============================================
# 6. SAVE CLEANED DATASET
# ============================================
print("\n" + "="*60)
print("SAVING CLEANED DATASET")
print("="*60)

# Keep only necessary columns
final_df = df_filtered[['id', 'sequence', 'label']].copy()

# Save
output_file = 'protein_dataset_final.csv'
final_df.to_csv(output_file, index=False)

print(f"\n✅ Saved cleaned dataset to: {output_file}")
print(f"   Columns: {list(final_df.columns)}")
print(f"   Shape: {final_df.shape}")

# Show sample
print("\n📋 Sample of final dataset:")
print(final_df.head())

print("\n" + "="*60)
print("✅ DATA QUALITY PIPELINE COMPLETE!")
print("="*60)
print(f"\nNext step: Run training with '{output_file}'")