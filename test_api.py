import requests
import json

BASE_URL = "http://localhost:8000"

# Test sequences
TEST_SEQUENCES = {
    "short_sequence": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLS",
    "medium_sequence": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL",
    "invalid_sequence": "MKTAYIAKQXYZ",  # Contains invalid amino acids
    "too_short": "MKTA"  # Too short
}

print("="*60)
print("TESTING PROTEIN FUNCTION CLASSIFIER API")
print("="*60)

# Test 1: Health Check
print("\n1. Testing Health Check...")
response = requests.get(f"{BASE_URL}/health")
print(f"Status: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}")

# Test 2: Model Stats
print("\n2. Testing Model Stats...")
response = requests.get(f"{BASE_URL}/stats")
print(f"Status: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}")

# Test 3: Valid Prediction
print("\n3. Testing Valid Prediction...")
response = requests.post(
    f"{BASE_URL}/predict",
    json={"sequence": TEST_SEQUENCES["medium_sequence"]}
)
print(f"Status: {response.status_code}")
result = response.json()
print(f"Predicted Class: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.4f}")
print(f"Inference Time: {result['inference_time_ms']:.2f}ms")

# Test 4: Invalid Sequence
print("\n4. Testing Invalid Sequence (should fail)...")
response = requests.post(
    f"{BASE_URL}/predict",
    json={"sequence": TEST_SEQUENCES["invalid_sequence"]}
)
print(f"Status: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}")

# Test 5: Batch Prediction
print("\n5. Testing Batch Prediction...")
response = requests.post(
    f"{BASE_URL}/predict/batch",
    json={"sequences": [TEST_SEQUENCES["short_sequence"], TEST_SEQUENCES["medium_sequence"]]}
)
print(f"Status: {response.status_code}")
result = response.json()
print(f"Total Sequences: {result['total_sequences']}")
print(f"Total Time: {result['total_inference_time_ms']:.2f}ms")
for i, pred in enumerate(result['predictions']):
    print(f"  Sequence {i+1}: {pred['predicted_class']} (confidence: {pred['confidence']:.4f})")

# Test 6: Sequence Validation
print("\n6. Testing Sequence Validation...")
response = requests.get(
    f"{BASE_URL}/validate",
    params={"sequence": TEST_SEQUENCES["medium_sequence"]}
)
print(f"Status: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}")

print("\n" + "="*60)
print("✅ API TESTING COMPLETE!")
print("="*60)