import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

print("="*60)
print("EXPLORATORY DATA ANALYSIS - PROTEIN DATASET")
print("="*60)

# Load the filtered dataset
df = pd.read_csv('protein_dataset_final.csv')
print(f"\n📊 Dataset loaded: {len(df)} sequences")
print(f"Columns: {list(df.columns)}")

# Calculate sequence length
df['seq_length'] = df['sequence'].str.len()

# ============================================
# 1. CLASS DISTRIBUTION ANALYSIS
# ============================================
print("\n" + "="*60)
print("1. CLASS DISTRIBUTION")
print("="*60)

class_counts = df['label'].value_counts().sort_index()
print("\nSequences per class:")
for label, count in class_counts.items():
    pct = (count / len(df)) * 100
    print(f"  {label}: {count:,} ({pct:.2f}%)")

# Calculate imbalance metrics
max_class = class_counts.max()
min_class = class_counts.min()
print(f"\nImbalance ratio: {max_class/min_class:.2f}:1")
print(f"Most common: {class_counts.idxmax()} ({max_class:,} samples)")
print(f"Least common: {class_counts.idxmin()} ({min_class:,} samples)")

# ============================================
# 2. SEQUENCE LENGTH ANALYSIS
# ============================================
print("\n" + "="*60)
print("2. SEQUENCE LENGTH STATISTICS")
print("="*60)

print("\nOverall length statistics:")
print(df['seq_length'].describe())

print("\nLength by EC class:")
length_by_class = df.groupby('label')['seq_length'].agg([
    ('count', 'count'),
    ('mean', 'mean'),
    ('median', 'median'),
    ('std', 'std'),
    ('min', 'min'),
    ('max', 'max')
]).round(1)
print(length_by_class)

# ============================================
# 3. AMINO ACID COMPOSITION ANALYSIS
# ============================================
print("\n" + "="*60)
print("3. AMINO ACID COMPOSITION")
print("="*60)

def get_aa_composition(sequence):
    """Calculate amino acid frequencies in a sequence"""
    aa_counts = Counter(sequence)
    total = len(sequence)
    return {aa: count/total for aa, count in aa_counts.items()}

# Calculate average composition across all sequences
all_aa_freqs = Counter()
total_length = 0

for seq in df['sequence']:
    all_aa_freqs.update(seq)
    total_length += len(seq)

# Convert to percentages
aa_composition = {aa: (count/total_length)*100 for aa, count in all_aa_freqs.most_common()}

print("\nTop 10 most common amino acids:")
for aa, freq in list(aa_composition.items())[:10]:
    print(f"  {aa}: {freq:.2f}%")

print("\nTop 10 least common amino acids:")
for aa, freq in list(aa_composition.items())[-10:]:
    print(f"  {aa}: {freq:.2f}%")

# ============================================
# 4. SAMPLE SEQUENCES INSPECTION
# ============================================
print("\n" + "="*60)
print("4. SAMPLE SEQUENCES")
print("="*60)

print("\nRandom samples from each class:")
for label in sorted(df['label'].unique()):
    sample = df[df['label'] == label].sample(1).iloc[0]
    print(f"\n{label}:")
    print(f"  ID: {sample['id']}")
    print(f"  Length: {sample['seq_length']}")
    print(f"  First 60 AA: {sample['sequence'][:60]}...")

# ============================================
# 5. VISUALIZATIONS
# ============================================
print("\n" + "="*60)
print("5. GENERATING VISUALIZATIONS")
print("="*60)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Protein Dataset EDA', fontsize=16, fontweight='bold')

# Plot 1: Class Distribution (Bar Chart)
ax1 = axes[0, 0]
class_counts_sorted = class_counts.sort_values(ascending=False)
bars = ax1.bar(range(len(class_counts_sorted)), class_counts_sorted.values, 
               color=sns.color_palette("husl", len(class_counts_sorted)))
ax1.set_xticks(range(len(class_counts_sorted)))
ax1.set_xticklabels(class_counts_sorted.index, rotation=0)
ax1.set_xlabel('EC Class', fontweight='bold')
ax1.set_ylabel('Number of Sequences', fontweight='bold')
ax1.set_title('Class Distribution', fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, (bar, value) in enumerate(zip(bars, class_counts_sorted.values)):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
             f'{value:,}', ha='center', va='bottom', fontsize=9)

# Plot 2: Sequence Length Distribution (Histogram)
ax2 = axes[0, 1]
ax2.hist(df['seq_length'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
ax2.axvline(df['seq_length'].mean(), color='red', linestyle='--', 
            linewidth=2, label=f'Mean: {df["seq_length"].mean():.0f}')
ax2.axvline(df['seq_length'].median(), color='orange', linestyle='--', 
            linewidth=2, label=f'Median: {df["seq_length"].median():.0f}')
ax2.set_xlabel('Sequence Length (amino acids)', fontweight='bold')
ax2.set_ylabel('Frequency', fontweight='bold')
ax2.set_title('Sequence Length Distribution', fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3)

# Plot 3: Length Distribution by Class (Box Plot)
ax3 = axes[1, 0]
class_order = sorted(df['label'].unique())
box_data = [df[df['label'] == label]['seq_length'].values for label in class_order]
bp = ax3.boxplot(box_data, labels=class_order, patch_artist=True,
                 medianprops=dict(color='red', linewidth=2),
                 boxprops=dict(facecolor='lightblue', alpha=0.7))
ax3.set_xlabel('EC Class', fontweight='bold')
ax3.set_ylabel('Sequence Length (amino acids)', fontweight='bold')
ax3.set_title('Sequence Length by EC Class', fontweight='bold')
ax3.grid(axis='y', alpha=0.3)

# Plot 4: Amino Acid Composition (Bar Chart)
ax4 = axes[1, 1]
top_aa = dict(list(aa_composition.items())[:20])
aa_sorted = sorted(top_aa.items(), key=lambda x: x[1], reverse=True)
aa_names = [x[0] for x in aa_sorted]
aa_freqs = [x[1] for x in aa_sorted]

bars = ax4.bar(range(len(aa_names)), aa_freqs, color='coral', edgecolor='black', alpha=0.7)
ax4.set_xticks(range(len(aa_names)))
ax4.set_xticklabels(aa_names, rotation=0, fontsize=10)
ax4.set_xlabel('Amino Acid', fontweight='bold')
ax4.set_ylabel('Frequency (%)', fontweight='bold')
ax4.set_title('Top 20 Amino Acid Composition', fontweight='bold')
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('eda_visualization.png', dpi=300, bbox_inches='tight')
print("\n✅ Saved visualization: eda_visualization.png")

# ============================================
# 6. ADDITIONAL INSIGHTS
# ============================================
print("\n" + "="*60)
print("6. KEY INSIGHTS & RECOMMENDATIONS")
print("="*60)

# Check for potential issues
insights = []

# Class imbalance check
imbalance_ratio = class_counts.max() / class_counts.min()
if imbalance_ratio > 3:
    insights.append(f"⚠️  Class imbalance detected ({imbalance_ratio:.1f}:1). Consider using class weights in training.")
else:
    insights.append(f"✅ Class distribution is relatively balanced ({imbalance_ratio:.1f}:1).")

# Length variation check
length_std = df['seq_length'].std()
length_mean = df['seq_length'].mean()
cv = (length_std / length_mean) * 100  # Coefficient of variation
if cv > 50:
    insights.append(f"⚠️  High length variability (CV={cv:.1f}%). Consider length-based bucketing for efficient batching.")
else:
    insights.append(f"✅ Sequence lengths are relatively consistent (CV={cv:.1f}%).")

# Very short/long sequences
very_short = (df['seq_length'] < 100).sum()
very_long = (df['seq_length'] > 800).sum()
if very_short > len(df) * 0.05:
    insights.append(f"⚠️  {very_short} sequences (<100 AA) may need special handling.")
if very_long > len(df) * 0.05:
    insights.append(f"⚠️  {very_long} sequences (>800 AA) may cause memory issues.")

# Dataset size check
if len(df) > 200000:
    insights.append(f"✅ Large dataset ({len(df):,} sequences) - good for deep learning!")
elif len(df) > 50000:
    insights.append(f"✅ Medium dataset ({len(df):,} sequences) - sufficient for CNN/RNN models.")
else:
    insights.append(f"⚠️  Small dataset ({len(df):,} sequences) - consider augmentation or transfer learning.")

print("\n" + "\n".join(insights))

# ============================================
# 7. TRAINING RECOMMENDATIONS
# ============================================
print("\n" + "="*60)
print("7. MODEL TRAINING RECOMMENDATIONS")
print("="*60)

print("\nBased on this EDA, here are recommendations:")

print("\n📋 Data Handling:")
print(f"  • Batch size: 32-64 (adjust based on GPU memory)")
print(f"  • Max sequence length: {int(df['seq_length'].quantile(0.95))} (95th percentile)")
print(f"  • Use padding for variable length sequences")

if imbalance_ratio > 2:
    print("\n⚖️  Class Imbalance:")
    print(f"  • Use class weights: {dict(1/class_counts * len(df) / len(class_counts))}")
    print(f"  • Or use weighted sampling in DataLoader")

print("\n🏗️  Architecture Suggestions:")
if df['seq_length'].mean() > 500:
    print(f"  • CNN with adaptive pooling (handles variable length)")
    print(f"  • Or Transformer with positional encoding")
else:
    print(f"  • Simple CNN works well for this length range")
    print(f"  • LSTM/GRU also suitable")

print("\n📊 Evaluation Strategy:")
print(f"  • Use stratified k-fold (5 folds recommended)")
print(f"  • Track per-class metrics (precision, recall, F1)")
print(f"  • Watch for overfitting on minority classes")

print("\n" + "="*60)
print("✅ EDA COMPLETE!")
print("="*60)
print("\nNext steps:")
print("1. Review eda_visualization.png")
print("2. Decide on class weight strategy")
print("3. Choose max_length for model (recommended: {})".format(int(df['seq_length'].quantile(0.95))))
print("4. Proceed to model training")
