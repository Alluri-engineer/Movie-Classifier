"""
Test Script for Movie Genre Prediction

Tests the trained model with sample images and validates predictions.
"""

import requests
import pandas as pd
import os
import random
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle

class MovieGenreTester:
    """Test the movie genre prediction system"""
    
    def __init__(self, flask_url='http://127.0.0.1:8080'):
        self.flask_url = flask_url
        self.csv_path = '../data/Dataset/Movie Genre from its poster/MovieGenre.csv'
        self.images_dir = '../data/Dataset/Movie Genre from its poster/SampleMoviePosters/SampleMoviePosters'
    
    def get_movie_info(self, imdb_id):
        """Get movie information from CSV based on IMDB ID"""
        try:
            df = pd.read_csv(self.csv_path, encoding='latin-1')
            movie_info = df[df['imdbId'] == int(imdb_id)]
            if not movie_info.empty:
                return {
                    'title': movie_info.iloc[0]['Title'],
                    'actual_genres': movie_info.iloc[0]['Genre'].split('|') if pd.notna(movie_info.iloc[0]['Genre']) else [],
                    'year': movie_info.iloc[0]['Year'] if 'Year' in movie_info.columns else 'Unknown'
                }
        except:
            pass
        return {'title': 'Unknown', 'actual_genres': [], 'year': 'Unknown'}
    
    def test_flask_connection(self):
        """Test if Flask app is running"""
        try:
            response = requests.get(self.flask_url, timeout=5)
            if response.status_code == 200:
                print("✅ Flask app is running")
                return True
            else:
                print(f"❌ Flask app returned status {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            print(f"❌ Cannot connect to Flask app at {self.flask_url}")
            print("💡 Make sure to run: python src/web_app.py")
            return False
        except Exception as e:
            print(f"❌ Error connecting to Flask app: {e}")
            return False
    
    def test_single_image(self, image_filename=None):
        """Test prediction on a single image"""
        if not image_filename:
            # Pick a random image
            try:
                image_files = [f for f in os.listdir(self.images_dir) if f.endswith('.jpg')]
                image_filename = random.choice(image_files)
            except:
                print(f"❌ Cannot access images directory: {self.images_dir}")
                return
        
        image_path = os.path.join(self.images_dir, image_filename)
        
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return
        
        imdb_id = image_filename.replace('.jpg', '')
        movie_info = self.get_movie_info(imdb_id)
        
        print(f"\n🎯 Testing Image: {image_filename}")
        print(f"🎭 Movie: {movie_info['title']} ({movie_info['year']})")
        print(f"🏷️  Actual Genres: {', '.join(movie_info['actual_genres'])}")
        
        try:
            with open(image_path, 'rb') as image_file:
                files = {'file': image_file}
                response = requests.post(self.flask_url, files=files, timeout=10)
                
                if response.status_code == 200:
                    # Parse predictions from HTML
                    genres = []
                    confidence_scores = []
                    
                    lines = response.text.split('\n')
                    in_results = False
                    
                    for line in lines:
                        if '<h2>🎯 Predicted Genres:</h2>' in line:
                            in_results = True
                        elif in_results and '<strong>' in line and '</strong>' in line:
                            # Extract genre name
                            start = line.find('<strong>') + 8
                            end = line.find('</strong>')
                            if start > 7 and end > start:
                                genre = line[start:end]
                                genres.append(genre)
                        elif in_results and 'Confidence:' in line:
                            # Extract confidence score
                            parts = line.split('Confidence:')
                            if len(parts) > 1:
                                conf_part = parts[1].strip()
                                if '%' in conf_part:
                                    conf_str = conf_part.split('%')[0].strip()
                                    try:
                                        confidence_scores.append(float(conf_str))
                                    except:
                                        pass
                        elif '</ul>' in line and in_results:
                            break
                    
                    print(f"🤖 Predicted Genres:")
                    if genres:
                        for i, genre in enumerate(genres):
                            conf = confidence_scores[i] if i < len(confidence_scores) else "N/A"
                            print(f"   • {genre}: {conf}%")
                        
                        # Calculate accuracy
                        if movie_info['actual_genres']:
                            common_genres = set(movie_info['actual_genres']) & set(genres)
                            if common_genres:
                                print(f"✅ Matching Genres: {', '.join(common_genres)}")
                            else:
                                print("❌ No matching genres")
                    else:
                        print("   • No genres predicted")
                    
                else:
                    print(f"❌ Prediction failed: HTTP {response.status_code}")
                    
        except Exception as e:
            print(f"❌ Error during prediction: {e}")
    
    def test_multiple_images(self, num_tests=5):
        """Test multiple random images"""
        try:
            image_files = [f for f in os.listdir(self.images_dir) if f.endswith('.jpg')]
            test_images = random.sample(image_files, min(num_tests, len(image_files)))
        except:
            print(f"❌ Cannot access images directory: {self.images_dir}")
            return
        
        print(f"\n🎬 Testing {len(test_images)} random movie posters...")
        print("=" * 60)
        
        successful_predictions = 0
        matching_predictions = 0
        
        for i, image_file in enumerate(test_images, 1):
            print(f"\n📸 Test {i}/{len(test_images)}: {image_file}")
            
            image_path = os.path.join(self.images_dir, image_file)
            imdb_id = image_file.replace('.jpg', '')
            movie_info = self.get_movie_info(imdb_id)
            
            print(f"🎭 {movie_info['title']} - Actual: {', '.join(movie_info['actual_genres'])}")
            
            try:
                with open(image_path, 'rb') as img_file:
                    files = {'file': img_file}
                    response = requests.post(self.flask_url, files=files, timeout=10)
                    
                    if response.status_code == 200:
                        successful_predictions += 1
                        
                        # Quick check for predictions
                        if '<strong>' in response.text and 'Predicted Genres:' in response.text:
                            genre_count = response.text.count('<strong>') - 1  # Subtract 1 for model version
                            print(f"✅ Predicted {genre_count} genres")
                            
                            # Check for matches (simplified)
                            has_match = False
                            for actual_genre in movie_info['actual_genres']:
                                if actual_genre in response.text:
                                    has_match = True
                                    break
                            
                            if has_match:
                                matching_predictions += 1
                                print("🎯 Has matching genre!")
                        else:
                            print("⚠️  No genres predicted")
                    else:
                        print(f"❌ Error: HTTP {response.status_code}")
                        
            except Exception as e:
                print(f"❌ Error: {e}")
        
        # Summary
        print(f"\n📊 TEST SUMMARY:")
        print(f"✅ Successful predictions: {successful_predictions}/{len(test_images)} ({successful_predictions/len(test_images)*100:.1f}%)")
        if successful_predictions > 0:
            print(f"🎯 Predictions with matches: {matching_predictions}/{successful_predictions} ({matching_predictions/successful_predictions*100:.1f}%)")
    
    def test_api_endpoint(self):
        """Test the API endpoint"""
        print(f"\n🔌 Testing API endpoint...")
        
        # Get a random test image
        try:
            image_files = [f for f in os.listdir(self.images_dir) if f.endswith('.jpg')]
            test_image = random.choice(image_files)
            image_path = os.path.join(self.images_dir, test_image)
        except:
            print(f"❌ Cannot access test images")
            return
        
        try:
            with open(image_path, 'rb') as image_file:
                files = {'file': image_file}
                response = requests.post(f"{self.flask_url}/api/predict", files=files, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    print("✅ API endpoint working")
                    print(f"🤖 Model: {data.get('model_version', 'Unknown')}")
                    print(f"🎯 Predicted Genres: {data.get('predicted_genres', [])}")
                    print(f"📊 Threshold Used: {data.get('threshold_used', 'Unknown')}")
                else:
                    print(f"❌ API error: HTTP {response.status_code}")
                    print(response.text[:200])
                    
        except Exception as e:
            print(f"❌ API test error: {e}")

def main():
    """Main testing function"""
    print("🎬 Movie Genre Prediction - Test Suite")
    print("=" * 50)
    
    tester = MovieGenreTester()
    
    # Test Flask connection
    if not tester.test_flask_connection():
        return
    
    # Test single image
    print(f"\n1️⃣ SINGLE IMAGE TEST")
    print("-" * 30)
    tester.test_single_image()
    
    # Test multiple images
    print(f"\n2️⃣ MULTIPLE IMAGES TEST")
    print("-" * 30)
    tester.test_multiple_images(5)
    
    # Test API endpoint
    print(f"\n3️⃣ API ENDPOINT TEST")
    print("-" * 30)
    tester.test_api_endpoint()
    
    print(f"\n🌐 Web Interface: {tester.flask_url}")
    print("✨ Testing completed!")

if __name__ == "__main__":
    main()