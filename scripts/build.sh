#!/bin/bash

# Movie Genre Prediction - Docker Build Script
# This script builds the Docker image for the movie genre prediction application

set -e  # Exit immediately if a command exits with a non-zero status

echo "🎬 Movie Genre Prediction - Docker Build"
echo "========================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="movie-genre-predictor"
IMAGE_TAG="${1:-latest}"
FULL_IMAGE_NAME="${IMAGE_NAME}:${IMAGE_TAG}"

echo -e "${BLUE}📋 Build Configuration:${NC}"
echo "  • Image Name: ${FULL_IMAGE_NAME}"
echo "  • Build Context: $(pwd)"
echo "  • Dockerfile: ./docker/Dockerfile"
echo ""

# Check if Docker is installed and running
echo -e "${YELLOW}🔧 Checking Docker installation...${NC}"
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed. Please install Docker first.${NC}"
    exit 1
fi

if ! docker info &> /dev/null; then
    echo -e "${RED}❌ Docker daemon is not running. Please start Docker.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Docker is ready${NC}"
echo ""

# Check if required files exist
echo -e "${YELLOW}📁 Checking required files...${NC}"
REQUIRED_FILES=(
    "docker/Dockerfile"
    "requirements.txt"
    "src/web_app.py"
    "models/movie_genre_classifier.h5"
    "models/genre_labels.pkl"
    "templates/index.html"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [[ ! -f "$file" ]]; then
        echo -e "${RED}❌ Required file missing: $file${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ Found: $file${NC}"
done
echo ""

# Build the Docker image
echo -e "${BLUE}🏗️  Building Docker image...${NC}"
echo "This may take a few minutes for the first build..."
echo ""

if docker build -f docker/Dockerfile -t "${FULL_IMAGE_NAME}" .; then
    echo ""
    echo -e "${GREEN}🎉 Docker image built successfully!${NC}"
    echo ""
    
    # Display image information
    echo -e "${BLUE}📊 Image Information:${NC}"
    docker images "${IMAGE_NAME}" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
    echo ""
    
    # Display next steps
    echo -e "${YELLOW}🚀 Next Steps:${NC}"
    echo "1. Run the container:"
    echo "   docker run -p 8080:8080 ${FULL_IMAGE_NAME}"
    echo ""
    echo "2. Or use docker-compose:"
    echo "   cd docker && docker-compose up"
    echo ""
    echo "3. Access the application:"
    echo "   http://localhost:8080"
    echo ""
    
else
    echo ""
    echo -e "${RED}❌ Docker build failed!${NC}"
    echo "Please check the error messages above and try again."
    exit 1
fi

echo -e "${GREEN}🏁 Build process completed successfully!${NC}"