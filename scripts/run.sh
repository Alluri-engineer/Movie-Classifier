#!/bin/bash

# Movie Genre Prediction - Docker Run Script
# This script runs the Docker container for the movie genre prediction application

set -e  # Exit immediately if a command exits with a non-zero status

echo "🎬 Movie Genre Prediction - Docker Run"
echo "====================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="movie-genre-predictor"
IMAGE_TAG="${1:-latest}"
CONTAINER_NAME="movie-genre-app"
HOST_PORT="${2:-8080}"
CONTAINER_PORT="8080"

FULL_IMAGE_NAME="${IMAGE_NAME}:${IMAGE_TAG}"

echo -e "${BLUE}📋 Run Configuration:${NC}"
echo "  • Image: ${FULL_IMAGE_NAME}"
echo "  • Container Name: ${CONTAINER_NAME}"
echo "  • Port Mapping: ${HOST_PORT}:${CONTAINER_PORT}"
echo "  • Access URL: http://localhost:${HOST_PORT}"
echo ""

# Check if Docker is running
echo -e "${YELLOW}🔧 Checking Docker...${NC}"
if ! docker info &> /dev/null; then
    echo -e "${RED}❌ Docker daemon is not running. Please start Docker.${NC}"
    exit 1
fi

# Check if image exists
if ! docker image inspect "${FULL_IMAGE_NAME}" &> /dev/null; then
    echo -e "${RED}❌ Docker image '${FULL_IMAGE_NAME}' not found.${NC}"
    echo "Please build the image first:"
    echo "  ./scripts/build.sh"
    exit 1
fi

echo -e "${GREEN}✅ Docker and image ready${NC}"
echo ""

# Stop and remove existing container if it exists
echo -e "${YELLOW}🧹 Cleaning up existing container...${NC}"
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Stopping existing container..."
    docker stop "${CONTAINER_NAME}" &> /dev/null || true
    echo "Removing existing container..."
    docker rm "${CONTAINER_NAME}" &> /dev/null || true
    echo -e "${GREEN}✅ Cleanup completed${NC}"
else
    echo "No existing container found"
fi
echo ""

# Create necessary directories for volumes
echo -e "${YELLOW}📁 Creating directories...${NC}"
mkdir -p ./data/uploads ./data/logs
echo -e "${GREEN}✅ Directories created${NC}"
echo ""

# Run the container
echo -e "${BLUE}🚀 Starting container...${NC}"
echo "Container will be accessible at: http://localhost:${HOST_PORT}"
echo ""

# Run with proper error handling
if docker run \
    --name "${CONTAINER_NAME}" \
    --detach \
    --publish "${HOST_PORT}:${CONTAINER_PORT}" \
    --volume "$(pwd)/data/uploads:/app/uploads" \
    --volume "$(pwd)/data/logs:/app/logs" \
    --env FLASK_ENV=production \
    --env PYTHONUNBUFFERED=1 \
    --restart unless-stopped \
    "${FULL_IMAGE_NAME}"; then
    
    echo -e "${GREEN}✅ Container started successfully!${NC}"
    echo ""
    
    # Wait a moment for the container to start
    echo -e "${YELLOW}⏳ Waiting for application to start...${NC}"
    sleep 5
    
    # Check container status
    if docker ps --filter "name=${CONTAINER_NAME}" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -q "${CONTAINER_NAME}"; then
        echo -e "${GREEN}🎉 Application is running!${NC}"
        echo ""
        
        # Display container information
        echo -e "${BLUE}📊 Container Information:${NC}"
        docker ps --filter "name=${CONTAINER_NAME}" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
        echo ""
        
        # Test the health endpoint
        echo -e "${YELLOW}🏥 Testing health endpoint...${NC}"
        sleep 2
        if curl -s "http://localhost:${HOST_PORT}/health" > /dev/null; then
            echo -e "${GREEN}✅ Health check passed${NC}"
        else
            echo -e "${YELLOW}⚠️  Health check pending (application may still be starting)${NC}"
        fi
        echo ""
        
        # Display usage information
        echo -e "${BLUE}🌐 Access Information:${NC}"
        echo "  • Web Interface: http://localhost:${HOST_PORT}"
        echo "  • API Endpoint: http://localhost:${HOST_PORT}/api/predict"
        echo "  • Health Check: http://localhost:${HOST_PORT}/health"
        echo ""
        
        echo -e "${BLUE}🛠️  Management Commands:${NC}"
        echo "  • View logs: docker logs ${CONTAINER_NAME}"
        echo "  • Stop container: docker stop ${CONTAINER_NAME}"
        echo "  • Remove container: docker rm ${CONTAINER_NAME}"
        echo "  • Shell access: docker exec -it ${CONTAINER_NAME} /bin/bash"
        echo ""
        
    else
        echo -e "${RED}❌ Container failed to start properly${NC}"
        echo "Check logs with: docker logs ${CONTAINER_NAME}"
        exit 1
    fi
    
else
    echo -e "${RED}❌ Failed to start container!${NC}"
    echo "Please check the error messages above."
    exit 1
fi

echo -e "${GREEN}🏁 Application is ready to use!${NC}"