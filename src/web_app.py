"""
Movie Genre Prediction Web Application

A Flask web application that predicts movie genres from poster images
using a trained EfficientNet deep learning model.
"""

from flask import Flask, request, render_template, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import tempfile
import os

app = Flask(__name__, template_folder='../templates')

# Load the trained model and label encoder
MODEL_PATH = os.getenv('MODEL_PATH', '/app/models/movie_genre_classifier_backup.h5')
LABELS_PATH = os.getenv('LABELS_PATH', '/app/models/genre_labels.pkl')

print("Loading movie genre prediction model...")
try:
    model = load_model(MODEL_PATH)
    with open(LABELS_PATH, 'rb') as f:
        genre_encoder = pickle.load(f)
    MODEL_LOADED = True
except Exception as e:
    print(f"⚠️  Warning: Could not load model - {str(e)}")
    print("🚀 Running in demo mode without model")
    model = None
    genre_encoder = None
    MODEL_LOADED = False

MODEL_INFO = "EfficientNet-B0 (Improved)"
INPUT_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.3

if MODEL_LOADED:
    print(f"✅ Model loaded: {MODEL_INFO}")
    print(f"✅ Available genres: {len(genre_encoder.classes_)}")
else:
    print("⚠️  Running without ML model - demo mode only")

@app.route('/', methods=['GET', 'POST'])
def predict_genres():
    """Main route for genre prediction"""
    if request.method == 'POST':
        # Validate file upload
        if 'file' not in request.files:
            return 'No file uploaded'
        
        file = request.files['file']
        if file.filename == '':
            return 'No file selected'
        
        if file:
            if not MODEL_LOADED:
                return render_template(
                    'index.html', 
                    genres=['Demo', 'Sample', 'Genres'],
                    confidences={'Demo': 0.8, 'Sample': 0.7, 'Genres': 0.6},
                    model_version=f"{MODEL_INFO} (Demo Mode - No Model Loaded)",
                    demo_mode=True
                )
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                file.save(temp_file.name)
                temp_path = temp_file.name
            
            try:
                # Process image
                image = tf.keras.preprocessing.image.load_img(temp_path, target_size=INPUT_SIZE)
                image_array = tf.keras.preprocessing.image.img_to_array(image)
                image_array = image_array / 255.0  # Normalize to [0, 1]
                image_array = np.expand_dims(image_array, axis=0)
                
                # Make prediction
                predictions = model.predict(image_array, verbose=0)
                
                # Apply threshold and get genres
                predicted_binary = (predictions > CONFIDENCE_THRESHOLD).astype(int)
                predicted_genres = genre_encoder.inverse_transform(predicted_binary)
                
                # Get confidence scores for all genres
                confidence_scores = predictions[0]
                all_confidences = {
                    genre: float(confidence_scores[i]) 
                    for i, genre in enumerate(genre_encoder.classes_)
                }
                
                # Clean up temporary file
                os.unlink(temp_path)
                
                return render_template(
                    'index.html', 
                    genres=predicted_genres[0] if len(predicted_genres) > 0 else [],
                    confidences=all_confidences,
                    model_version=MODEL_INFO
                )
            
            except Exception as e:
                # Clean up temporary file on error
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                return f'Error processing image: {str(e)}'
    
    return render_template('index.html', model_version=MODEL_INFO)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for programmatic predictions"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            file.save(temp_file.name)
            temp_path = temp_file.name
        
        # Process image
        image = tf.keras.preprocessing.image.load_img(temp_path, target_size=INPUT_SIZE)
        image_array = tf.keras.preprocessing.image.img_to_array(image)
        image_array = image_array / 255.0  # Normalize to [0, 1]
        image_array = np.expand_dims(image_array, axis=0)
        
        # Make prediction
        predictions = model.predict(image_array, verbose=0)
        
        # Apply threshold and get genres
        predicted_binary = (predictions > CONFIDENCE_THRESHOLD).astype(int)
        predicted_genres = genre_encoder.inverse_transform(predicted_binary)
        
        # Get confidence scores
        confidence_scores = predictions[0]
        all_confidences = {
            genre: float(confidence_scores[i]) 
            for i, genre in enumerate(genre_encoder.classes_)
        }
        
        # Clean up temporary file
        os.unlink(temp_path)
        
        return jsonify({
            'predicted_genres': predicted_genres[0].tolist() if len(predicted_genres) > 0 else [],
            'confidence_scores': all_confidences,
            'model_version': MODEL_INFO,
            'threshold_used': CONFIDENCE_THRESHOLD
        })
        
    except Exception as e:
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model': MODEL_INFO,
        'model_loaded': MODEL_LOADED,
        'genres_available': len(genre_encoder.classes_) if MODEL_LOADED else 0
    })

if __name__ == '__main__':
    print("🎬 Starting Movie Genre Prediction Web App...")
    print(f"🌐 Access the app at: http://0.0.0.0:8080")
    app.run(debug=False, host='0.0.0.0', port=8080, threaded=True)