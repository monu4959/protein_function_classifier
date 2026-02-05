import requests
import json

import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "sequence": "MKLAVAALAVATVQATAVAEQLRRELQLVQKNSRFLKHLGVVSQHQMWQRVNGELATYRFRDLEAFDAQIKRLRTVAVQSPEPQIQVLKNQVAILPQQGRALKIDATQNQYIGDHQITHQ"
    }
)

print(response.json())

result = response.json()

print("="*60)
print("PROBABILITY VERIFICATION")
print("="*60)

print(f"\nPredicted Class: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.6f}")

print("\nAll Probabilities:")
total = 0
for cls, prob in sorted(result['all_probabilities'].items()):
    print(f"  {cls}: {prob:.10f}")
    total += prob

print(f"\n✅ Sum of probabilities: {total:.10f}")
print(f"Expected: 1.0")

if abs(total - 1.0) < 0.0001:
    print("\n✅ Probabilities are properly normalized!")
else:
    print(f"\n❌ ERROR: Probabilities don't sum to 1! (sum={total})")