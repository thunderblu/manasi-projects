from typing import Dict
from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value

def predict_tabular_classification_sample():
    project = "322603165747"
    endpoint_id = "4602032306335514624"
    location = "us-east1"
    api_endpoint = "us-east1-aiplatform.googleapis.com"  # Changed to match your region

    # Initialize client with correct API endpoint
    client_options = {"api_endpoint": api_endpoint}
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)

    # Prepare the instance data
    instance_dict = {
        "Date": "12/9/24",
        "Open": "151.19000244140625",
        "High": "153.25",
        "Low": "149.91999816894531",
        "Volume": "1000000"
    }

    # Convert the instance dict to the required format
    instance = json_format.ParseDict(instance_dict, Value())
    instances = [instance]
    
    # Parameters (empty in this case)
    parameters_dict = {}
    parameters = json_format.ParseDict(parameters_dict, Value())

    # Get the endpoint path
    endpoint = client.endpoint_path(
        project=project, location=location, endpoint=endpoint_id
    )

    # Make prediction
    response = client.predict(
        endpoint=endpoint, instances=instances, parameters=parameters
    )

    print("Response:")
    print(" deployed_model_id:", response.deployed_model_id)
    
    # Print predictions
    predictions = response.predictions
    for prediction in predictions:
        print(" prediction:", dict(prediction))

if __name__ == "__main__":
    try:
        predict_tabular_classification_sample()
    except Exception as e:
        print("Error making prediction:", str(e))
