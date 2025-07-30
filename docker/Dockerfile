# Movie Genre Prediction - Docker Image
# Multi-stage build for optimized production image

FROM python:3.9-slim as builder

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.9-slim as production

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy Python packages from builder stage
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application files
COPY src/ ./src/
COPY models/ ./models/
COPY templates/ ./templates/
COPY ["data/Dataset/movie_posters/movie_genres.csv", "./data/Dataset/movie_posters/movie_genres.csv"]

# Create necessary directories
RUN mkdir -p "./data/Dataset/movie_posters/sample_posters/images/"

# Copy a few sample images for testing (optional)
# COPY data/Dataset/movie_posters/sample_posters/images/*.jpg ./data/Dataset/movie_posters/sample_posters/images/

# Set proper permissions
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8080

# Set environment variables
ENV PYTHONPATH=/app
ENV FLASK_APP=src/web_app.py
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/health', timeout=5)" || exit 1

# Default command
CMD ["python", "src/web_app.py"]