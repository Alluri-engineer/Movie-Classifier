# 🐳 Docker Deployment Guide

Complete Docker setup for the Movie Genre Prediction application with production-ready configuration.

## 🚀 Quick Start

### Option 1: Simple Docker Run
```bash
# Build the image
./docker-build.sh

# Run the container  
./docker-run.sh

# Access the app
open http://localhost:8080
```

### Option 2: Docker Compose (Recommended)
```bash
# Start all services
docker-compose up -d

# Access the app
open http://localhost:8080
```

## 📋 Prerequisites

- **Docker**: Version 20.0+ 
- **Docker Compose**: Version 2.0+
- **System Requirements**: 4GB RAM, 2GB free disk space

### Install Docker
```bash
# macOS (using Homebrew)
brew install --cask docker

# Ubuntu/Debian
sudo apt-get update
sudo apt-get install docker.io docker-compose

# CentOS/RHEL
sudo yum install docker docker-compose
```

## 🏗️ Docker Configuration

### Dockerfile Features
- **Multi-stage build** for optimized image size
- **Non-root user** for security
- **Health checks** for monitoring
- **Production optimizations**

### Image Details
```dockerfile
Base Image: python:3.9-slim
Final Size: ~800MB (optimized)
Architecture: x86_64, ARM64 support
Security: Non-root user, minimal packages
```

## 🛠️ Build Options

### Basic Build
```bash
# Build with default tag
./docker-build.sh

# Build with custom tag
./docker-build.sh v1.0.0
```

### Manual Build
```bash
# Build the image
docker build -t movie-genre-predictor:latest .

# View built images
docker images movie-genre-predictor
```

## 🚀 Deployment Options

### 1. Development Mode
```bash
# Run with auto-reload (development)
docker run -p 8080:8080 \
  -v $(pwd):/app \
  -e FLASK_ENV=development \
  movie-genre-predictor:latest
```

### 2. Production Mode
```bash
# Run optimized for production
docker run -d \
  --name movie-genre-app \
  -p 8080:8080 \
  --restart unless-stopped \
  movie-genre-predictor:latest
```

### 3. Docker Compose (Full Stack)
```bash
# Development setup
docker-compose up -d

# Production setup with Redis and Nginx
docker-compose --profile production up -d
```

## 🔧 Configuration

### Environment Variables
```bash
# Core settings
FLASK_ENV=production          # production/development
PYTHONUNBUFFERED=1           # For proper logging
MODEL_PATH=/app/models/movie_genre_classifier.h5
LABELS_PATH=/app/models/genre_labels.pkl

# Optional settings
REDIS_URL=redis://redis:6379  # For caching
LOG_LEVEL=INFO               # DEBUG/INFO/WARNING/ERROR
```

### Volume Mounts
```bash
# Persistent uploads
-v ./uploads:/app/uploads

# Log persistence  
-v ./logs:/app/logs

# Custom model (optional)
-v ./custom-model.h5:/app/models/movie_genre_classifier.h5
```

## 📊 Monitoring & Health

### Health Checks
```bash
# Check container health
docker ps

# View health status
docker inspect movie-genre-app --format='{{.State.Health.Status}}'

# Manual health check
curl http://localhost:8080/health
```

### Logs
```bash
# View live logs
docker logs -f movie-genre-app

# View specific number of lines
docker logs --tail 100 movie-genre-app

# Export logs
docker logs movie-genre-app > app.log 2>&1
```

## 🔍 Troubleshooting

### Common Issues

**Port Already in Use**
```bash
# Find process using port
lsof -i :8080

# Use different port
docker run -p 8081:8080 movie-genre-predictor:latest
```

**Model File Not Found**
```bash
# Check model files exist
ls -la models/

# Rebuild with fresh files
docker-compose build --no-cache
```

**Memory Issues**
```bash
# Check Docker memory limits
docker stats movie-genre-app

# Increase memory limit
docker run --memory=2g movie-genre-predictor:latest
```

**Permission Errors**
```bash
# Fix file permissions
chmod -R 755 models/ templates/

# Run as root (not recommended for production)
docker run --user root movie-genre-predictor:latest
```

### Debug Mode
```bash
# Run with shell access
docker run -it --entrypoint /bin/bash movie-genre-predictor:latest

# Exec into running container
docker exec -it movie-genre-app /bin/bash

# Check Python environment
docker exec movie-genre-app python -c "import tensorflow; print(tensorflow.__version__)"
```

## 📈 Production Deployment

### Docker Swarm
```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml movie-genre-stack
```

### Kubernetes
```yaml
# Example Kubernetes deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: movie-genre-predictor
spec:
  replicas: 3
  selector:
    matchLabels:
      app: movie-genre-predictor
  template:
    metadata:
      labels:
        app: movie-genre-predictor
    spec:
      containers:
      - name: app
        image: movie-genre-predictor:latest
        ports:
        - containerPort: 8080
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
```

### AWS ECS/Fargate
```bash
# Build for AWS
docker build --platform linux/amd64 -t movie-genre-predictor:aws .

# Tag for ECR
docker tag movie-genre-predictor:aws 123456789.dkr.ecr.us-east-1.amazonaws.com/movie-genre-predictor:latest

# Push to ECR
docker push 123456789.dkr.ecr.us-east-1.amazonaws.com/movie-genre-predictor:latest
```

## 🔐 Security Best Practices

### Image Security
- ✅ **Non-root user** (appuser)
- ✅ **Minimal base image** (python:3.9-slim)
- ✅ **No secrets in image**
- ✅ **Vulnerability scanning**

### Runtime Security
```bash
# Run with security options
docker run \
  --read-only \
  --tmpfs /tmp \
  --tmpfs /var/run \
  --no-new-privileges \
  --cap-drop ALL \
  movie-genre-predictor:latest
```

### Secrets Management
```bash
# Use Docker secrets (Swarm)
echo "secret-key" | docker secret create model-key -

# Use environment file
docker run --env-file .env movie-genre-predictor:latest
```

## 📝 Management Commands

### Container Lifecycle
```bash
# Start container
docker start movie-genre-app

# Stop container
docker stop movie-genre-app

# Restart container
docker restart movie-genre-app

# Remove container
docker rm movie-genre-app

# Remove image
docker rmi movie-genre-predictor:latest
```

### Cleanup
```bash
# Remove all stopped containers
docker container prune

# Remove unused images
docker image prune

# Remove all unused resources
docker system prune -a
```

## 🎯 Performance Optimization

### Image Size Reduction
- ✅ Multi-stage builds
- ✅ .dockerignore file
- ✅ Minimal dependencies
- ✅ Layer caching

### Runtime Performance
```bash
# Allocate more memory
docker run --memory=4g movie-genre-predictor:latest

# Use multiple CPU cores
docker run --cpus=2 movie-genre-predictor:latest

# Enable production WSGI server
docker run -e FLASK_ENV=production movie-genre-predictor:latest
```

## 📚 Additional Resources

- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Docker Security](https://docs.docker.com/engine/security/)
- [Docker Compose](https://docs.docker.com/compose/)

---

**Your Movie Genre Prediction app is now fully containerized! 🐳✨**