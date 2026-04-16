# Use official Python image
FROM python:3.11-slim

# Prevent Python from writing .pyc files & enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir chromadb streamlit sentence-transformers

# Copy project files
COPY . .

# Expose Render's expected port
EXPOSE 10000

# Run Streamlit app (dynamic PORT support for Render)
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT:-10000} --server.address=0.0.0.0"]