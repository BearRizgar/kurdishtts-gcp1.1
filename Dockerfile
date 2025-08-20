# Use the official Python image optimized for Cloud Run
FROM python:3.10-slim

# Set environment variables for production
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Optimize math libraries for Cloud Run performance
ENV OPENBLAS_NUM_THREADS=2
ENV MKL_NUM_THREADS=2
ENV OMP_NUM_THREADS=2
ENV VECLIB_MAXIMUM_THREADS=2

# Set work directory
WORKDIR /app

# Install system dependencies optimized for Google Cloud Run
# Include audio libraries and math optimizations
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    libsndfile1-dev \
    libasound2-dev \
    libportaudio2 \
    libportaudiocpp0 \
    portaudio19-dev \
    libopenblas-dev \
    liblapack-dev \
    libblas-dev \
    libfftw3-dev \
    git \
    gcc \
    g++ \
    build-essential \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create models directory for persistent storage
RUN mkdir -p /app/models/sorani /app/models/kurmanji /app/outputs

# Install Python dependencies with optimizations
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Note: Running as root for Cloud Run with service account authentication
# Google Cloud Run manages security and access control at the platform level

# Health check disabled for Cloud Run - Cloud Run handles health checks internally
# HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=3 \
#     CMD curl -f http://localhost:$PORT/health || exit 1

# Expose port
EXPOSE 8080

# Run with optimized gunicorn settings for Google Cloud Run
# Use PORT environment variable that Cloud Run provides
# Single worker to avoid model loading conflicts and memory issues
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-8080} --workers 1 --worker-class sync --timeout 300 --worker-connections 1000 --max-requests 50 --max-requests-jitter 5 --preload app:app"] 