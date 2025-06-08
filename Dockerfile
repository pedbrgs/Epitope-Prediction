# Start from Python 3.12 image
FROM python:3.12-slim
# Install git
RUN apt-get update && apt-get install -y git
# Create working directory
WORKDIR /epitope-prediction/
# Copy requirements
COPY requirements.txt .
# Install requirements
RUN pip install --no-cache-dir -r requirements.txt
# Copy source-code
COPY . .