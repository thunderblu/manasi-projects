from google.cloud import aiplatform

def deploy_model():
    # Initialize Vertex AI
    aiplatform.init(
        project='322603165747',
        location='us-east1'
    )

    # Get the model
    model = aiplatform.Model('YOUR_MODEL_ID')  # Get this from Vertex AI console

    # Deploy to endpoint
    endpoint = model.deploy(
        machine_type='n1-standard-4',
        min_replica_count=1,
        max_replica_count=1,
        sync=True
    )

    print(f"Model deployed successfully to endpoint: {endpoint.resource_name}")

if __name__ == "__main__":
    deploy_model() 