# 🎬 Movie Genre Prediction from Posters

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13.0-orange.svg)](https://tensorflow.org)
[![Flask](https://img.shields.io/badge/Flask-3.0.0-green.svg)](https://flask.palletsprojects.com)
[![Docker](https://img.shields.io/badge/Docker-Supported-blue.svg)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Predict movie genres from poster images using deep learning!** This project uses a trained EfficientNet-B0 model to classify movie posters into multiple genres with confidence scores.

## 🌟 Features

- **🤖 AI-Powered Predictions**: EfficientNet-B0 model with 24 genre classifications
- **🎯 Multi-Label Classification**: Predicts multiple genres per movie poster
- **📊 Confidence Scoring**: Visual confidence bars for each prediction
- **🌐 Web Interface**: Modern, responsive UI for easy interaction
- **🔌 RESTful API**: Programmatic access for developers
- **🐳 Docker Ready**: Containerized deployment with Docker Compose
- **⚡ Fast Inference**: Sub-second predictions
- **🧪 Comprehensive Testing**: Full test suite included

## 🚀 Quick Start

### Option 1: Docker (Recommended)

```bash
# Clone the repository
git clone <your-repo-url>
cd Image-Prediction

# Start with Docker Compose
./scripts/compose.sh up
```

**Access the app at:** http://localhost:8080

### Option 2: Local Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Start the web application
python src/web_app.py
```

**Access the app at:** http://localhost:8080

## 📁 Project Structure

```
Image Prediction/
├── 📄 README.md                    # Project documentation
├── 📄 requirements.txt             # Python dependencies
├── 🐳 docker/                      # Docker configuration
│   ├── Dockerfile                  # Multi-stage Docker build
│   ├── docker-compose.yml          # Service orchestration
│   └── docker.md                   # Docker documentation
├── 📁 scripts/                     # Automation scripts
│   ├── build.sh                    # Docker build script
│   ├── run.sh                      # Container run script
│   └── compose.sh                  # Docker Compose helper
├── 📁 src/                         # Source code
│   ├── web_app.py                  # Flask web application
│   └── model_trainer.py            # Model training pipeline
├── 📁 templates/                   # Web templates
│   └── index.html                  # Main UI template
├── 📁 models/                      # Trained models
│   ├── movie_genre_classifier.h5   # EfficientNet model
│   ├── movie_genre_classifier_backup.h5
│   └── genre_labels.pkl            # Label encoder
├── 📁 data/                        # Dataset and storage
│   ├── Dataset/movie_posters/      # Training data
│   ├── uploads/                    # User uploads
│   └── logs/                       # Application logs
└── 📁 tests/                       # Test suite
    └── test_predictions.py         # Comprehensive tests
```

## 🎯 Supported Genres

The model can predict the following 24 movie genres:

- **Action** • **Adventure** • **Animation** • **Biography** • **Comedy**
- **Crime** • **Documentary** • **Drama** • **Family** • **Fantasy**
- **History** • **Horror** • **Music** • **Mystery** • **Romance**
- **Sci-Fi** • **Sport** • **Thriller** • **War** • **Western**
- **Musical** • **Film-Noir** • **News** • **Short**

## 🛠️ Technical Details

### Model Architecture
- **Base Model**: EfficientNet-B0 (pre-trained on ImageNet)
- **Input Size**: 224×224×3 RGB images
- **Architecture**: Transfer learning with custom classification head
- **Parameters**: ~4.8M trainable parameters
- **Training**: Two-phase approach (frozen → fine-tuned)

### Performance Metrics
- **Model Type**: Multi-label classification
- **Confidence Threshold**: 0.3 (configurable)
- **Inference Time**: <1 second per image
- **Accuracy**: Varies by genre (detailed metrics in training logs)

### Web Application
- **Framework**: Flask 3.0.0
- **Frontend**: Responsive HTML/CSS with confidence visualization
- **API**: RESTful endpoints for programmatic access
- **Health Checks**: Built-in monitoring endpoints

## 🔌 API Usage

### Web Interface
```
GET  /           # Main web interface
POST /           # Upload image via web form
```

### REST API
```
POST /api/predict    # Predict genres from uploaded image
GET  /health         # Health check endpoint
```

### Example API Usage

```python
import requests

# Predict genres from image
with open('movie_poster.jpg', 'rb') as image_file:
    files = {'file': image_file}
    response = requests.post('http://localhost:8080/api/predict', files=files)
    
    if response.status_code == 200:
        data = response.json()
        print(f"Predicted genres: {data['predicted_genres']}")
        print(f"Confidence scores: {data['confidence_scores']}")
```

### Example Response
```json
{
  "predicted_genres": ["Action", "Adventure", "Sci-Fi"],
  "confidence_scores": {
    "Action": 0.89,
    "Adventure": 0.76,
    "Sci-Fi": 0.62,
    "Drama": 0.23
  },
  "model_version": "EfficientNet-B0 (Improved)",
  "threshold_used": 0.3
}
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Start the web application first
python src/web_app.py

# Run tests in another terminal
python tests/test_predictions.py
```

The test suite includes:
- **Flask Connection Test**: Verify web app is running
- **Single Image Test**: Test prediction on one image
- **Multiple Images Test**: Batch testing with accuracy metrics
- **API Endpoint Test**: Validate REST API functionality

## 🏋️ Training Your Own Model

To train a new model with your own dataset:

```bash
# Prepare your dataset (CSV + images)
# Format: columns should include 'imdbId', 'Title', 'Genre'
# Images should be named as {imdbId}.jpg

# Update paths in model_trainer.py
# Then run training
python src/model_trainer.py
```

### Training Features
- **Data Augmentation**: Automatic image transformations
- **Early Stopping**: Prevents overfitting
- **Learning Rate Scheduling**: Adaptive learning rates
- **Model Checkpointing**: Saves best performing models
- **Comprehensive Evaluation**: Detailed performance metrics

## 🐳 Docker Deployment

### Development
```bash
# Build and run locally
./scripts/build.sh
./scripts/run.sh
```

### Production with Docker Compose
```bash
# Full stack with Redis and Nginx (optional)
./scripts/compose.sh up
```

### Docker Features
- **Multi-stage builds** for optimized image size
- **Non-root user** for security
- **Health checks** for monitoring
- **Volume mounts** for data persistence
- **Production-ready** configuration

## 📊 Performance Monitoring

Monitor your deployment:

```bash
# Check container health
docker ps

# View application logs
./scripts/compose.sh logs

# Health check endpoint
curl http://localhost:8080/health
```

## 🔧 Configuration

### Environment Variables
```bash
MODEL_PATH=/app/models/movie_genre_classifier.h5
LABELS_PATH=/app/models/genre_labels.pkl
FLASK_ENV=production
PYTHONUNBUFFERED=1
```

### Model Parameters
- **Confidence Threshold**: Adjustable in `web_app.py` (default: 0.3)
- **Input Size**: 224×224 pixels (configurable in model training)
- **Batch Size**: Configurable for training and inference

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset**: Movie poster dataset with genre labels
- **EfficientNet**: Google's efficient CNN architecture
- **TensorFlow**: Deep learning framework
- **Flask**: Lightweight web framework
- **Contributors**: All contributors to this project

## 📞 Support

- **Issues**: Report bugs or request features via GitHub Issues
- **Documentation**: Check the `docker/docker.md` and `setup.md` files
- **API Docs**: Available at `/health` endpoint

---

**⭐ Star this repository if you found it helpful!**

Made with ❤️ using Python, TensorFlow, and Docker.