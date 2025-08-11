# Stock Prediction Model Setup Guide

This guide helps you set up and run the stock prediction model on a new laptop. Follow these steps carefully to ensure proper setup and execution.

## Prerequisites

- Python 3.9
- Git
- Anaconda/Miniconda (recommended)
- Google Cloud Platform account
- Basic understanding of terminal/command line

## Installation Steps

### 1. Basic Setup 

bash
Clone the repository
git clone <your-repository-url>
cd testedinall
Create and activate virtual environment
conda create -n myenv python=3.9
conda activate myenv
Install requirements
pip install -r requirements.txt

### 2. GCP Setup

# Install Google Cloud SDK from: https://cloud.google.com/sdk/docs/install

# Login to GCP
gcloud auth login

# Set project
gcloud config set project <your-project-id>

# Configure application credentials
gcloud auth application-default login

### 3. Environment Configuration

Create a `.env` file in the project root:
plaintext
GCP_PROJECT_ID=<your-project-id>
GCP_REGION=<your-region>
BUCKET_NAME=<your-bucket-name>
MODEL_REGISTRY_URI=<your-registry-uri>

### 4. Project Structure

Ensure your project has this structure:

```
testedinall/
├── data/
├── models/
│   ├── __init__.py
│   ├── train.py
│   ├── bias_checker.py
│   ├── experiment_tracker.py
│   ├── hyperparameter_tuner.py
│   ├── model_registry.py
│   ├── model_selector.py
│   └── model_trainer.py
├── mlruns/
├── requirements.txt
└── .env
```

## Running the Project

### 1. Training Models

```bash
# Activate environment
conda activate myenv

# Run training
python -m models.train
```

### 2. Viewing Results

```bash
# Start MLflow UI
mlflow ui --host localhost --port 5001

# Access in browser
http://localhost:5001
```

## Troubleshooting

### Common Issues

1. GCP Credentials Error
```bash
gcloud auth application-default login
```

2. Port Already in Use
```bash
pkill -f mlflow
mlflow ui --port 5001
```

3. Missing Packages
```bash
pip install <package-name>
```

### Required Packages

Main dependencies:
- mlflow
- pandas
- numpy
- scikit-learn
- xgboost
- python-dotenv
- google-cloud-storage
- google-cloud-aiplatform

## Monitoring

### Logs
```bash
# Check training logs
tail -f logs/training.log
```

### MLflow
- Access MLflow UI at http://localhost:5001
- View experiments, runs, and metrics
- Compare model performances

## Best Practices

1. **Security**
   - Keep GCP credentials secure
   - Don't commit .env file
   - Use secure API keys

2. **Development**
   - Test with small data first
   - Update requirements.txt when adding packages
   - Monitor resource usage

3. **Data Management**
   - Keep data organized in data/
   - Backup important results
   - Version control your data

## Support

For issues or questions:
1. Check logs for error messages
2. Review GCP console for cloud issues
3. Check MLflow UI for training details

## License

[Your License Information]

## Contributors

[Your Name/Team Information]

---
Last Updated: [Current Date]
```

This README provides a comprehensive guide for setting up and running the project on a new laptop. Would you like me to add or modify any section?
