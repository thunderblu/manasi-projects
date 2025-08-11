FROM apache/airflow:2.7.3-python3.9

USER root

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    libpq-dev \
    gcc \
    g++ \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Ensure the logs directory exists with correct permissions
RUN mkdir -p /opt/airflow/logs && \
    chown -R airflow /opt/airflow/logs && \
    chmod -R 755 /opt/airflow/logs    

USER airflow

# Set the working directory to Airflow's home
WORKDIR /opt/airflow

# Copy requirements file
COPY requirements.txt /opt/airflow/requirements.txt

# Install Python packages
RUN pip install --no-cache-dir -r /opt/airflow/requirements.txt

# Copy source code
COPY src/ /opt/airflow/src/ 