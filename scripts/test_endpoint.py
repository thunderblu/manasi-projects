from google.cloud import aiplatform

def test_endpoint():
    endpoint = aiplatform.Endpoint(
        endpoint_name="4602032306335514624"  # Your endpoint ID
    )

    # Test instance
    test_instance = {
        "Date": "9/9/24",
        "Open": "151.19000244140625",
        "High": "153.25",
        "Low": "149.91999816894531",
        "Volume": "1000000"
    }

    prediction = endpoint.predict([test_instance])
    print(f"Prediction: {prediction}")

if __name__ == "__main__":
    test_endpoint() 