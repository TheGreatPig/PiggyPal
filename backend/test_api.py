import requests
import json
import time

# API base URL
BASE_URL = "http://localhost:8000"

def test_health():
    """Test the health endpoint"""
    start = time.time()
    response = requests.get(f"{BASE_URL}/health")
    elapsed = time.time() - start
    print("Health Check:", response.json())
    print(f"Elapsed time: {elapsed:.4f} seconds\n")

def test_single_classification():
    """Test single transaction classification"""
    data = {
        "transaction_text": "Nintendo CA1412771920",
    }
    
    start = time.time()
    response = requests.post(f"{BASE_URL}/classify", json=data)
    elapsed = time.time() - start
    print("Single Classification:")
    print(json.dumps(response.json(), indent=2))
    print(f"Elapsed time: {elapsed:.4f} seconds\n")

def test_batch_classification():
    """Test batch transaction classification"""
    data = [
        {
            "transaction_text": "Nintendo CA1412771920",
        },
        {
            "transaction_text": "Uber *TRIP",
        },
        {
            "transaction_text": "WALMART GROCERY",
        }
    ]
    
    start = time.time()
    response = requests.post(f"{BASE_URL}/classify/batch", json=data)
    elapsed = time.time() - start
    print("Batch Classification:")
    print(json.dumps(response.json(), indent=2))
    print(f"Elapsed time: {elapsed:.4f} seconds\n")

def test_batch_large(n=20):
    """Test batch classification with a larger batch size for benchmarking"""
    data = [{"transaction_text": f"Transaction {i}"} for i in range(n)]
    start = time.time()
    response = requests.post(f"{BASE_URL}/classify/batch", json=data)
    elapsed = time.time() - start
    print(f"Large Batch Classification (n={n}):")
    print(f"Elapsed time: {elapsed:.4f} seconds\n")
    # Optionally print the result: print(json.dumps(response.json(), indent=2))

if __name__ == "__main__":
    print("Testing PiggyPal Transaction Classifier API")
    print("=" * 50)
    
    try:
        test_health()
        test_single_classification()
        test_batch_classification()
        test_batch_large(20)
        test_batch_large(50)
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API. Make sure the server is running on http://localhost:8000")
    except Exception as e:
        print(f"Error: {e}") 