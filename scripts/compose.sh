#!/bin/bash

# Movie Genre Prediction - Docker Compose Script
# This script manages the Docker Compose stack for the movie genre prediction application

set -e  # Exit immediately if a command exits with a non-zero status

echo "🎬 Movie Genre Prediction - Docker Compose"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
COMPOSE_FILE="docker/docker-compose.yml"
ACTION="${1:-up}"

echo -e "${BLUE}📋 Compose Configuration:${NC}"
echo "  • Compose File: ${COMPOSE_FILE}"
echo "  • Action: ${ACTION}"
echo ""

# Check if Docker and Docker Compose are available
echo -e "${YELLOW}🔧 Checking Docker setup...${NC}"
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed. Please install Docker first.${NC}"
    exit 1
fi

if ! docker info &> /dev/null; then
    echo -e "${RED}❌ Docker daemon is not running. Please start Docker.${NC}"
    exit 1
fi

if ! docker compose version &> /dev/null && ! docker-compose version &> /dev/null; then
    echo -e "${RED}❌ Docker Compose is not available. Please install Docker Compose.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Docker setup is ready${NC}"
echo ""

# Check if compose file exists
if [[ ! -f "${COMPOSE_FILE}" ]]; then
    echo -e "${RED}❌ Docker Compose file not found: ${COMPOSE_FILE}${NC}"
    exit 1
fi

# Execute the requested action
case "${ACTION}" in
    "up")
        echo -e "${BLUE}🚀 Starting services...${NC}"
        cd docker
        if command -v "docker compose" &> /dev/null; then
            docker compose up --build -d
        else
            docker-compose up --build -d
        fi
        echo -e "${GREEN}✅ Services started successfully!${NC}"
        echo ""
        echo -e "${BLUE}🌐 Access Information:${NC}"
        echo "  • Web Interface: http://localhost:8080"
        echo "  • API Endpoint: http://localhost:8080/api/predict"
        echo "  • Health Check: http://localhost:8080/health"
        ;;
    "down")
        echo -e "${BLUE}🛑 Stopping services...${NC}"
        cd docker
        if command -v "docker compose" &> /dev/null; then
            docker compose down
        else
            docker-compose down
        fi
        echo -e "${GREEN}✅ Services stopped successfully!${NC}"
        ;;
    "logs")
        echo -e "${BLUE}📋 Showing logs...${NC}"
        cd docker
        if command -v "docker compose" &> /dev/null; then
            docker compose logs -f
        else
            docker-compose logs -f
        fi
        ;;
    "status")
        echo -e "${BLUE}📊 Service status:${NC}"
        cd docker
        if command -v "docker compose" &> /dev/null; then
            docker compose ps
        else
            docker-compose ps
        fi
        ;;
    *)
        echo -e "${YELLOW}📖 Usage:${NC}"
        echo "  $0 [ACTION]"
        echo ""
        echo -e "${BLUE}Available actions:${NC}"
        echo "  • up      - Start services (default)"
        echo "  • down    - Stop services"
        echo "  • logs    - Show logs"
        echo "  • status  - Show service status"
        echo ""
        echo -e "${BLUE}Examples:${NC}"
        echo "  $0 up      # Start all services"
        echo "  $0 down    # Stop all services"
        echo "  $0 logs    # View logs"
        echo "  $0 status  # Check status"
        ;;
esac

echo -e "${GREEN}🏁 Operation completed!${NC}"