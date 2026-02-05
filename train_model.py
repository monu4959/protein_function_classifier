import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import torch.optim as optim
from tqdm import tqdm
import json
import time

# ============================================
# CONFIGURATION
# ============================================
CONFIG = {
    'max_length': 750,           # Based on 95th percentile from EDA
    'batch_size': 64,            # Adjust if memory issues
    'embed_dim': 128,            # Embedding dimension
    'num_epochs': 15,            # Train for 15 epochs
    'learning_rate': 0.001,
    'dropout': 0.3,
    'sample_size': 50000,        # Start with 50k for faster iteration (set to None for full dataset)
    'random_seed': 42,
    'use_class_weights': True    # Critical for imbalanced dataset
}

print("="*60)
print("PROTEIN FUNCTION CLASSIFICATION - TRAINING PIPELINE")
print("="*60)
print(f"\nConfiguration:")
for key, value in CONFIG.items():
    print(f"  {key}: {value}")

# Set random seeds for reproducibility
torch.manual_seed(CONFIG['random_seed'])
np.random.seed(CONFIG['random_seed'])

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n🖥️  Using device: {device}")
if device.type == 'cuda':
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ============================================
# 1. LOAD AND PREPARE DATA
# ============================================
print("\n" + "="*60)
print("1. LOADING DATA")
print("="*60)

df = pd.read_csv('protein_dataset_final.csv')
print(f"✅ Loaded {len(df):,} sequences")

# Sample subset if specified (for faster training)
if CONFIG['sample_size'] is not None and CONFIG['sample_size'] < len(df):
    df = df.groupby('label', group_keys=False).apply(
        lambda x: x.sample(min(len(x), CONFIG['sample_size'] // 7), random_state=CONFIG['random_seed'])
    ).reset_index(drop=True)
    print(f"📊 Using {len(df):,} sequences for training (stratified sample)")
else:
    print(f"📊 Using full dataset: {len(df):,} sequences")

print(f"\nClass distribution:")
print(df['label'].value_counts().sort_index())

# ============================================
# 2. ENCODE LABELS AND SEQUENCES
# ============================================
print("\n" + "="*60)
print("2. ENCODING DATA")
print("="*60)

# Encode labels
label_encoder = LabelEncoder()
df['label_encoded'] = label_encoder.fit_transform(df['label'])
num_classes = len(label_encoder.classes_)
print(f"✅ Classes: {list(label_encoder.classes_)}")

# Save label encoder for later use
label_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}
with open('label_mapping.json', 'w') as f:
    json.dump(label_mapping, f, indent=2)
print(f"✅ Saved label mapping to: label_mapping.json")

# Amino acid vocabulary
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
aa_to_int = {aa: i+1 for i, aa in enumerate(AMINO_ACIDS)}  # 0 reserved for padding

def encode_sequence(seq, max_len):
    """Convert amino acid sequence to integers with padding"""
    encoded = [aa_to_int.get(aa, 0) for aa in seq[:max_len]]
    # Pad to max_len
    encoded += [0] * (max_len - len(encoded))
    return encoded

print(f"🧬 Encoding sequences (max_length={CONFIG['max_length']})...")
df['encoded_seq'] = df['sequence'].apply(lambda x: encode_sequence(x, CONFIG['max_length']))

# ============================================
# 3. CALCULATE CLASS WEIGHTS
# ============================================
print("\n" + "="*60)
print("3. CALCULATING CLASS WEIGHTS")
print("="*60)

if CONFIG['use_class_weights']:
    class_counts = df['label'].value_counts().sort_index()
    total_samples = len(df)
    class_weights = {}
    
    print("\nClass weights (for handling imbalance):")
    for label in label_encoder.classes_:
        count = class_counts[label]
        weight = total_samples / (num_classes * count)
        class_weights[label] = weight
        print(f"  {label}: {weight:.4f} (n={count:,})")
    
    # Convert to tensor
    weight_list = [class_weights[label] for label in label_encoder.classes_]
    class_weight_tensor = torch.FloatTensor(weight_list).to(device)
else:
    class_weight_tensor = None
    print("⚠️  Class weights disabled")

# ============================================
# 4. TRAIN/TEST SPLIT
# ============================================
print("\n" + "="*60)
print("4. SPLITTING DATA")
print("="*60)

X = np.array(df['encoded_seq'].tolist())
y = df['label_encoded'].values

# Stratified split to preserve class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=CONFIG['random_seed'], stratify=y
)

print(f"✅ Training samples: {len(X_train):,}")
print(f"✅ Test samples: {len(X_test):,}")
print(f"   Split ratio: {len(X_train)/len(df)*100:.1f}% / {len(X_test)/len(df)*100:.1f}%")

# ============================================
# 5. CREATE DATASETS AND DATALOADERS
# ============================================
print("\n" + "="*60)
print("5. CREATING DATALOADERS")
print("="*60)

class ProteinDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.LongTensor(sequences)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

train_dataset = ProteinDataset(X_train, y_train)
test_dataset = ProteinDataset(X_test, y_test)

train_loader = DataLoader(
    train_dataset, 
    batch_size=CONFIG['batch_size'], 
    shuffle=True,
    num_workers=0  # Set to 0 for Windows compatibility
)
test_loader = DataLoader(
    test_dataset, 
    batch_size=CONFIG['batch_size'], 
    shuffle=False,
    num_workers=0
)

print(f"✅ Train batches: {len(train_loader)}")
print(f"✅ Test batches: {len(test_loader)}")

# ============================================
# 6. DEFINE MODEL ARCHITECTURE
# ============================================
print("\n" + "="*60)
print("6. BUILDING MODEL")
print("="*60)

class ProteinCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, dropout=0.3):
        super(ProteinCNN, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Multi-scale convolutional layers
        self.conv1 = nn.Conv1d(embed_dim, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(256, 256, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(256, 256, kernel_size=7, padding=3)
        
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(256)
        
        # Global pooling
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 2, 512)  # *2 because we concat max and avg pool
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x shape: (batch, seq_len)
        x = self.embedding(x)  # (batch, seq_len, embed_dim)
        x = x.permute(0, 2, 1)  # (batch, embed_dim, seq_len)
        
        # Convolutional layers with residual connections
        x1 = self.relu(self.bn1(self.conv1(x)))
        x2 = self.relu(self.bn2(self.conv2(x1)))
        x3 = self.relu(self.bn3(self.conv3(x2)))
        
        # Add residual connection
        x = x3 + x1  # Skip connection
        
        # Global pooling (both max and average)
        x_max = self.global_max_pool(x).squeeze(-1)
        x_avg = self.global_avg_pool(x).squeeze(-1)
        x = torch.cat([x_max, x_avg], dim=1)  # (batch, 512)
        
        # Fully connected layers
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

# Initialize model
vocab_size = len(aa_to_int) + 1  # +1 for padding
model = ProteinCNN(
    vocab_size=vocab_size,
    embed_dim=CONFIG['embed_dim'],
    num_classes=num_classes,
    dropout=CONFIG['dropout']
)
model = model.to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n📊 Model Architecture:")
print(f"   Total parameters: {total_params:,}")
print(f"   Trainable parameters: {trainable_params:,}")
print(f"   Model size: ~{total_params * 4 / 1e6:.2f} MB")

# ============================================
# 7. SETUP TRAINING
# ============================================
print("\n" + "="*60)
print("7. SETUP TRAINING")
print("="*60)

criterion = nn.CrossEntropyLoss(weight=class_weight_tensor)
optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=2
)

print(f"✅ Loss function: CrossEntropyLoss (with class weights)")
print(f"✅ Optimizer: Adam (lr={CONFIG['learning_rate']})")
print(f"✅ Scheduler: ReduceLROnPlateau (patience=2)")

# ============================================
# 8. TRAINING FUNCTIONS
# ============================================

def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for sequences, labels in pbar:
        sequences, labels = sequences.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return total_loss / len(loader), 100. * correct / total

def evaluate(model, loader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(loader, desc="Evaluating", leave=False)
        for sequences, labels in pbar:
            sequences, labels = sequences.to(device), labels.to(device)
            
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    return total_loss / len(loader), 100. * correct / total, all_preds, all_labels

# ============================================
# 9. TRAIN THE MODEL
# ============================================
print("\n" + "="*60)
print("🚀 STARTING TRAINING")
print("="*60)

best_acc = 0
best_epoch = 0
training_history = {
    'train_loss': [],
    'train_acc': [],
    'test_loss': [],
    'test_acc': []
}

start_time = time.time()

for epoch in range(CONFIG['num_epochs']):
    print(f"\n{'='*60}")
    print(f"Epoch {epoch+1}/{CONFIG['num_epochs']}")
    print(f"{'='*60}")
    
    # Train
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    
    # Evaluate
    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
    
    # Update scheduler
    scheduler.step(test_acc)
    
    # Save history
    training_history['train_loss'].append(train_loss)
    training_history['train_acc'].append(train_acc)
    training_history['test_loss'].append(test_loss)
    training_history['test_acc'].append(test_acc)
    
    # Print results
    print(f"\n📊 Results:")
    print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"   Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.2f}%")
    
    # Save best model
    if test_acc > best_acc:
        best_acc = test_acc
        best_epoch = epoch + 1
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'test_acc': test_acc,
            'config': CONFIG
        }, 'best_model.pth')
        print(f"   ✅ New best model saved! Accuracy: {best_acc:.2f}%")

training_time = time.time() - start_time

print("\n" + "="*60)
print("✅ TRAINING COMPLETE!")
print("="*60)
print(f"⏱️  Total training time: {training_time/60:.1f} minutes")
print(f"🏆 Best test accuracy: {best_acc:.2f}% (epoch {best_epoch})")

# Save training history
with open('training_history.json', 'w') as f:
    json.dump(training_history, f, indent=2)
print(f"✅ Saved training history to: training_history.json")

# ============================================
# 10. FINAL EVALUATION
# ============================================
print("\n" + "="*60)
print("10. FINAL EVALUATION")
print("="*60)

# Load best model
checkpoint = torch.load('best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
print(f"✅ Loaded best model from epoch {checkpoint['epoch']+1}")

# Evaluate
_, final_acc, final_preds, final_labels = evaluate(model, test_loader, criterion, device)

# Generate classification report
print("\n📊 Classification Report:")
print("="*60)
target_names = [label_encoder.classes_[i] for i in range(num_classes)]
report = classification_report(final_labels, final_preds, target_names=target_names, digits=4)
print(report)

# Save classification report
with open('classification_report.txt', 'w') as f:
    f.write(report)
print(f"✅ Saved classification report to: classification_report.txt")

# Confusion matrix
print("\n📊 Confusion Matrix:")
print("="*60)
cm = confusion_matrix(final_labels, final_preds)
print("Rows = True labels, Columns = Predicted labels")
print(f"Classes: {target_names}")
print(cm)

print("\n" + "="*60)
print("🎉 PROJECT COMPLETE!")
print("="*60)
print("\n📁 Generated files:")
print("   • best_model.pth - Trained model weights")
print("   • label_mapping.json - Label encoder mapping")
print("   • training_history.json - Training metrics")
print("   • classification_report.txt - Performance metrics")
print("\n🚀 Next steps:")
print("   1. Review classification_report.txt for per-class performance")
print("   2. Build FastAPI deployment endpoint")
print("   3. Add to resume with key metrics!")