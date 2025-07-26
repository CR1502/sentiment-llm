# Base image with Python and CUDA CPU-only support
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install OS dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirement spec
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code and model
COPY app ./app
COPY models ./models

# Expose port for FastAPI
EXPOSE 8000

# Start server
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]