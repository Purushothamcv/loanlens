#!/usr/bin/env python3
"""
Test script for the Loan Prediction API
"""

import requests
import json

# API base URL
BASE_URL = "http://localhost:8000"

def test_api():
    """Test the loan prediction API"""
    
    # Test data - example loan application
    test_data = {
        "Gender": "Male",
        "Married": "Yes",
        "Dependents": 1,
        "Education": "Graduate",
        "Self_Employed": "No",
        "ApplicantIncome": 5849.0,
        "CoapplicantIncome": 0.0,
        "LoanAmount": 146.0,
        "Loan_Amount_Term": 360.0,
        "Credit_History": "Yes",
        "Property_Area": "Semiurban"
    }
    
    try:
        # Test root endpoint
        print("Testing root endpoint...")
        response = requests.get(f"{BASE_URL}/")
        print(f"Root response: {response.json()}")
        print()
        
        # Test health check
        print("Testing health check...")
        response = requests.get(f"{BASE_URL}/health")
        print(f"Health check: {response.json()}")
        print()
        
        # Test prediction endpoint
        print("Testing prediction endpoint...")
        response = requests.post(
            f"{BASE_URL}/predict",
            headers={"Content-Type": "application/json"},
            json=test_data
        )
        
        if response.status_code == 200:
            result = response.json()
            print("Prediction successful!")
            print(f"Loan Status: {result['loan_status_text']}")
            print(f"Probability: {result['probability']:.4f}")
            print(f"Raw prediction: {result['loan_status']}")
        else:
            print(f"Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API server.")
        print("Please make sure the server is running on http://localhost:8000")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_api()
