from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional
import torch
import torch.nn as nn
import json
import time
from datetime import datetime

# ============================================
# CONFIGURATION
# ============================================
MODEL_PATH = 'best_model_full.pth'
LABEL_MAPPING_PATH = 'label_mapping_full_2.json'
MAX_SEQUENCE_LENGTH = 750
MIN_SEQUENCE_LENGTH = 50
VALID_AMINO_ACIDS = set('ACDEFGHIKLMNPQRSTVWY')

# ============================================
# MODEL ARCHITECTURE (same as training)
# ============================================
class ProteinCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, dropout=0.3):
        super(ProteinCNN, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        self.conv1 = nn.Conv1d(embed_dim, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(256, 256, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(256, 256, kernel_size=7, padding=3)
        
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(256)
        
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc1 = nn.Linear(256 * 2, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        
        x1 = self.relu(self.bn1(self.conv1(x)))
        x2 = self.relu(self.bn2(self.conv2(x1)))
        x3 = self.relu(self.bn3(self.conv3(x2)))
        
        x = x3 + x1
        
        x_max = self.global_max_pool(x).squeeze(-1)
        x_avg = self.global_avg_pool(x).squeeze(-1)
        x = torch.cat([x_max, x_avg], dim=1)
        
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

# ============================================
# LOAD MODEL AT STARTUP
# ============================================
print("="*60)
print("LOADING PROTEIN FUNCTION CLASSIFIER")
print("="*60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Load label mapping
with open(LABEL_MAPPING_PATH, 'r') as f:
    label_mapping = json.load(f)
num_classes = len(label_mapping)
print(f"Classes: {list(label_mapping.values())}")

# Initialize model
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
aa_to_int = {aa: i+1 for i, aa in enumerate(AMINO_ACIDS)}
vocab_size = len(aa_to_int) + 1

model = ProteinCNN(vocab_size=vocab_size, embed_dim=128, num_classes=num_classes)

# Load trained weights
checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

print(f"✅ Model loaded successfully!")
print(f"   Accuracy: {checkpoint.get('test_acc', 'N/A'):.2f}%")
print("="*60)

# ============================================
# HELPER FUNCTIONS
# ============================================
def encode_sequence(seq: str, max_len: int = MAX_SEQUENCE_LENGTH) -> List[int]:
    """Encode amino acid sequence to integers"""
    encoded = [aa_to_int.get(aa, 0) for aa in seq[:max_len]]
    encoded += [0] * (max_len - len(encoded))
    return encoded

def validate_sequence(seq: str) -> tuple[bool, Optional[str]]:
    """Validate protein sequence"""
    # Check length
    if len(seq) < MIN_SEQUENCE_LENGTH:
        return False, f"Sequence too short (minimum {MIN_SEQUENCE_LENGTH} amino acids, got {len(seq)})"
    
    if len(seq) > MAX_SEQUENCE_LENGTH:
        return False, f"Sequence too long (maximum {MAX_SEQUENCE_LENGTH} amino acids, got {len(seq)})"
    
    # Check for invalid amino acids
    invalid_chars = set(seq) - VALID_AMINO_ACIDS
    if invalid_chars:
        return False, f"Invalid amino acids found: {sorted(invalid_chars)}"
    
    return True, None

# ============================================
# PYDANTIC MODELS
# ============================================
class ProteinInput(BaseModel):
    sequence: str = Field(..., description="Protein amino acid sequence (50-750 AA)")
    
    @validator('sequence')
    def validate_sequence_format(cls, v):
        v = v.upper().strip()
        is_valid, error_msg = validate_sequence(v)
        if not is_valid:
            raise ValueError(error_msg)
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "sequence": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL"
            }
        }

class BatchProteinInput(BaseModel):
    sequences: List[str] = Field(..., description="List of protein sequences")
    
    @validator('sequences')
    def validate_batch(cls, v):
        if len(v) == 0:
            raise ValueError("Batch cannot be empty")
        if len(v) > 100:
            raise ValueError("Maximum batch size is 100 sequences")
        
        # Validate each sequence
        validated = []
        for i, seq in enumerate(v):
            seq = seq.upper().strip()
            is_valid, error_msg = validate_sequence(seq)
            if not is_valid:
                raise ValueError(f"Sequence {i+1}: {error_msg}")
            validated.append(seq)
        return validated

class PredictionOutput(BaseModel):
    predicted_class: str = Field(..., description="Predicted enzyme class (EC1-EC7)")
    confidence: float = Field(..., description="Prediction confidence (0-1)")
    all_probabilities: Dict[str, float] = Field(..., description="Probabilities for all classes")
    sequence_length: int = Field(..., description="Length of input sequence")
    inference_time_ms: float = Field(..., description="Inference time in milliseconds")

class BatchPredictionOutput(BaseModel):
    predictions: List[PredictionOutput]
    total_sequences: int
    total_inference_time_ms: float

class HealthCheck(BaseModel):
    status: str
    model_loaded: bool
    device: str
    model_accuracy: float
    timestamp: str

# ============================================
# INITIALIZE FASTAPI
# ============================================
app = FastAPI(
    title="Protein Function Classifier API",
    description="Deep learning API for classifying protein sequences into enzyme classes (EC1-EC7). Trained on 268K UniProt sequences with 96% accuracy.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# API ENDPOINTS
# ============================================

@app.get("/", tags=["General"])
def root():
    """Root endpoint with API information"""
    return {
        "message": "Protein Function Classifier API",
        "version": "1.0.0",
        "model_accuracy": f"{checkpoint.get('test_acc', 0):.2f}%",
        "endpoints": {
            "/predict": "POST - Predict single protein sequence",
            "/predict/batch": "POST - Predict multiple sequences",
            "/health": "GET - Health check",
            "/docs": "GET - Interactive API documentation",
            "/stats": "GET - Model statistics"
        }
    }

@app.get("/health", response_model=HealthCheck, tags=["General"])
def health_check():
    """Check API health and model status"""
    return HealthCheck(
        status="healthy",
        model_loaded=True,
        device=str(device),
        model_accuracy=checkpoint.get('test_acc', 0),
        timestamp=datetime.now().isoformat()
    )

@app.get("/stats", tags=["General"])
def model_stats():
    """Get detailed model statistics"""
    return {
        "model_info": {
            "architecture": "Multi-scale CNN with residual connections",
            "parameters": "1.3M trainable parameters",
            "training_dataset": "268,260 UniProt sequences",
            "accuracy": f"{checkpoint.get('test_acc', 0):.2f}%",
            "classes": list(label_mapping.values())
        },
        "input_requirements": {
            "min_length": MIN_SEQUENCE_LENGTH,
            "max_length": MAX_SEQUENCE_LENGTH,
            "valid_amino_acids": "".join(sorted(VALID_AMINO_ACIDS))
        },
        "performance": {
            "typical_inference_time": "50-100ms per sequence",
            "batch_processing": "Up to 100 sequences per request",
            "device": str(device)
        }
    }

@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
def predict_protein_function(input_data: ProteinInput):
    """
    Predict the enzyme class (EC1-EC7) for a protein sequence.
    
    **Input:**
    - sequence: Protein amino acid sequence (50-750 characters, valid amino acids only)
    
    **Output:**
    - predicted_class: Most likely enzyme class
    - confidence: Prediction confidence score (0-1)
    - all_probabilities: Probability distribution across all classes
    - sequence_length: Length of input sequence
    - inference_time_ms: Time taken for prediction
    
    **Example:**
```json
    {
        "sequence": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL"
    }
```
    """
    
    start_time = time.time()
    
    seq = input_data.sequence
    
    # Encode sequence
    encoded = encode_sequence(seq)
    input_tensor = torch.LongTensor([encoded]).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)[0]
        pred_class_idx = torch.argmax(probs).item()
    
    # Format response
    predicted_label = label_mapping[str(pred_class_idx)]
    confidence = float(probs[pred_class_idx])
    
    all_probs = {
        label_mapping[str(i)]: float(probs[i]) 
        for i in range(len(label_mapping))
    }
    
    inference_time = (time.time() - start_time) * 1000  # Convert to ms
    
    return PredictionOutput(
        predicted_class=predicted_label,
        confidence=confidence,
        all_probabilities=all_probs,
        sequence_length=len(seq),
        inference_time_ms=round(inference_time, 2)
    )

@app.post("/predict/batch", response_model=BatchPredictionOutput, tags=["Prediction"])
def predict_batch(input_data: BatchProteinInput):
    """
    Predict enzyme classes for multiple protein sequences in batch.
    
    **Input:**
    - sequences: List of protein sequences (max 100 sequences per batch)
    
    **Output:**
    - predictions: List of predictions for each sequence
    - total_sequences: Number of sequences processed
    - total_inference_time_ms: Total time for batch processing
    
    **Example:**
```json
    {
        "sequences": [
            "MKTAYIAKQRQISFVKSHFSRQL...",
            "MVKVYAPASSANMSVGFDVLGAD..."
        ]
    }
```
    """
    
    start_time = time.time()
    predictions = []
    
    for seq in input_data.sequences:
        # Encode
        encoded = encode_sequence(seq)
        input_tensor = torch.LongTensor([encoded]).to(device)
        
        # Predict
        seq_start = time.time()
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)[0]
            pred_class_idx = torch.argmax(probs).item()
        
        # Format
        predicted_label = label_mapping[str(pred_class_idx)]
        confidence = float(probs[pred_class_idx])
        all_probs = {label_mapping[str(i)]: float(probs[i]) for i in range(len(label_mapping))}
        seq_time = (time.time() - seq_start) * 1000
        
        predictions.append(PredictionOutput(
            predicted_class=predicted_label,
            confidence=confidence,
            all_probabilities=all_probs,
            sequence_length=len(seq),
            inference_time_ms=round(seq_time, 2)
        ))
    
    total_time = (time.time() - start_time) * 1000
    
    return BatchPredictionOutput(
        predictions=predictions,
        total_sequences=len(predictions),
        total_inference_time_ms=round(total_time, 2)
    )

@app.get("/validate", tags=["Utility"])
def validate_sequence_endpoint(sequence: str = Query(..., description="Protein sequence to validate")):
    """
    Validate a protein sequence without making a prediction.
    
    **Parameters:**
    - sequence: Protein amino acid sequence
    
    **Returns:**
    - valid: Boolean indicating if sequence is valid
    - message: Validation message or error details
    - sequence_info: Information about the sequence if valid
    """
    
    seq = sequence.upper().strip()
    is_valid, error_msg = validate_sequence(seq)
    
    if is_valid:
        return {
            "valid": True,
            "message": "Sequence is valid",
            "sequence_info": {
                "length": len(seq),
                "amino_acid_composition": {aa: seq.count(aa) for aa in sorted(set(seq))}
            }
        }
    else:
        return {
            "valid": False,
            "message": error_msg,
            "sequence_info": None
        }

# ============================================
# ERROR HANDLERS
# ============================================
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return {
        "error": "Internal server error",
        "detail": str(exc),
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Starting Protein Function Classifier API...")
    print("📖 API documentation: http://localhost:8000/docs")
    print("🔄 Alternative docs: http://localhost:8000/redoc")
    uvicorn.run(app, host="0.0.0.0", port=8000)