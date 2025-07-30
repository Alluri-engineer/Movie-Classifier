"""
Movie Genre Classification Model Training

This script trains a movie genre classifier using EfficientNet-B0 architecture
with transfer learning for multi-label classification of movie posters.
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0, DenseNet169
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, hamming_loss
import pickle
import os
from collections import Counter

class MovieGenreTrainer:
    """Movie genre classification model trainer"""
    
    def __init__(self, model_type='efficientnet', input_size=(224, 224), batch_size=32):
        self.model_type = model_type
        self.input_size = input_size
        self.batch_size = batch_size
        self.model = None
        self.genre_encoder = None
        
    def load_dataset(self, csv_path, images_dir):
        """Load and preprocess the movie dataset"""
        print("📊 Loading movie dataset...")
        df = pd.read_csv(csv_path, encoding='latin-1')
        df.dropna(inplace=True)
        
        # Process genres
        df['Genre'] = df['Genre'].apply(lambda x: x.split('|'))
        
        # Remove movies with rare genres (less than 50 occurrences)
        genre_counts = Counter([genre for genres in df['Genre'] for genre in genres])
        common_genres = [genre for genre, count in genre_counts.items() if count >= 50]
        df['Genre'] = df['Genre'].apply(lambda x: [g for g in x if g in common_genres])
        df = df[df['Genre'].map(len) > 0]  # Remove movies with no common genres
        
        print(f"✅ Dataset loaded: {len(df)} movies with {len(common_genres)} genres")
        print(f"📈 Most common genres: {genre_counts.most_common(10)}")
        
        # Setup multi-label encoder
        self.genre_encoder = MultiLabelBinarizer()
        y = self.genre_encoder.fit_transform(df['Genre'])
        X = df['imdbId'].values
        
        print(f"🏷️  Label matrix shape: {y.shape}")
        print(f"🎭 Genre classes: {self.genre_encoder.classes_}")
        
        return train_test_split(X, y, test_size=0.2, random_state=42)
    
    def create_data_generator(self, X, y, images_dir, augment=True):
        """Create data generator with preprocessing and augmentation"""
        def generator():
            indices = np.arange(len(X))
            while True:
                np.random.shuffle(indices)
                for i in range(0, len(indices), self.batch_size):
                    batch_indices = indices[i:i + self.batch_size]
                    batch_X, batch_y = [], []
                    
                    for idx in batch_indices:
                        image_path = f'{images_dir}/{X[idx]}.jpg'
                        try:
                            # Load and preprocess image
                            img = tf.keras.preprocessing.image.load_img(
                                image_path, target_size=self.input_size
                            )
                            img = tf.keras.preprocessing.image.img_to_array(img)
                            
                            # Normalize to [0, 1]
                            img = img / 255.0
                            
                            # Simple augmentation for training
                            if augment and np.random.random() > 0.5:
                                img = tf.image.flip_left_right(img)
                            if augment and np.random.random() > 0.5:
                                img = tf.image.adjust_brightness(img, delta=0.1)
                            
                            batch_X.append(img)
                            batch_y.append(y[idx])
                            
                        except (FileNotFoundError, Exception):
                            continue
                    
                    if batch_X:
                        yield (np.array(batch_X), np.array(batch_y))
        
        return generator
    
    def build_model(self):
        """Build the model architecture"""
        print(f"🏗️  Building {self.model_type} model...")
        
        # Choose base model
        if self.model_type == 'efficientnet':
            base_model = EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.input_size, 3)
            )
        elif self.model_type == 'densenet':
            base_model = DenseNet169(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.input_size, 3)
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Freeze base model initially
        base_model.trainable = False
        
        # Add custom classification head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        predictions = Dense(len(self.genre_encoder.classes_), activation='sigmoid')(x)
        
        self.model = Model(inputs=base_model.input, outputs=predictions)
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        print(f"✅ Model built with {self.model.count_params():,} parameters")
        return self.model
    
    def train_model(self, X_train, y_train, X_val, y_val, images_dir, epochs=30):
        """Train the model using two-phase approach"""
        print("🚀 Starting model training...")
        
        # Create data generators
        train_gen = self.create_data_generator(X_train, y_train, images_dir, augment=True)
        val_gen = self.create_data_generator(X_val, y_val, images_dir, augment=False)
        
        # Calculate steps
        train_steps = max(1, len(X_train) // self.batch_size)
        val_steps = max(1, len(X_val) // self.batch_size)
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss'),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-7, monitor='val_loss'),
            ModelCheckpoint(
                '../models/movie_genre_classifier.h5',
                save_best_only=True,
                monitor='val_loss'
            )
        ]
        
        # Phase 1: Train head only
        print("📚 Phase 1: Training classification head...")
        history1 = self.model.fit(
            train_gen(),
            steps_per_epoch=train_steps,
            epochs=min(15, epochs//2),
            validation_data=val_gen(),
            validation_steps=val_steps,
            callbacks=callbacks,
            verbose=1
        )
        
        # Phase 2: Fine-tune with lower learning rate
        if epochs > 15:
            print("🔬 Phase 2: Fine-tuning entire model...")
            self.model.layers[0].trainable = True  # Unfreeze base model
            
            # Lower learning rate for fine-tuning
            self.model.compile(
                optimizer=Adam(learning_rate=0.0001),
                loss='binary_crossentropy',
                metrics=['accuracy', 'precision', 'recall']
            )
            
            history2 = self.model.fit(
                train_gen(),
                steps_per_epoch=train_steps,
                epochs=epochs - 15,
                validation_data=val_gen(),
                validation_steps=val_steps,
                callbacks=callbacks,
                verbose=1
            )
        
        print("✅ Training completed!")
        return self.model
    
    def evaluate_model(self, X_test, y_test, images_dir):
        """Evaluate the trained model"""
        print("📊 Evaluating model performance...")
        test_gen = self.create_data_generator(X_test, y_test, images_dir, augment=False)
        test_steps = max(1, len(X_test) // self.batch_size)
        
        # Get predictions
        predictions = self.model.predict(test_gen(), steps=test_steps)
        pred_binary = (predictions > 0.3).astype(int)
        
        # Calculate metrics
        hamming = hamming_loss(y_test[:len(pred_binary)], pred_binary)
        print(f"📈 Hamming Loss: {hamming:.4f}")
        
        # Per-genre classification report
        target_names = self.genre_encoder.classes_
        print("\n📋 Per-Genre Classification Report:")
        print(classification_report(
            y_test[:len(pred_binary)], 
            pred_binary, 
            target_names=target_names,
            zero_division=0
        ))
    
    def save_model_and_encoder(self):
        """Save the trained model and label encoder"""
        model_path = '../models/movie_genre_classifier.h5'
        encoder_path = '../models/genre_labels.pkl'
        
        self.model.save(model_path)
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.genre_encoder, f)
        
        print(f"💾 Model saved to {model_path}")
        print(f"💾 Label encoder saved to {encoder_path}")

def main():
    """Main training function"""
    # Configuration
    MODEL_TYPE = 'efficientnet'  # 'efficientnet' or 'densenet'
    CSV_PATH = '../data/Dataset/Movie Genre from its poster/MovieGenre.csv'
    IMAGES_DIR = '../data/Dataset/Movie Genre from its poster/SampleMoviePosters/SampleMoviePosters'
    EPOCHS = 30
    
    print("🎬 Movie Genre Classification - Model Training")
    print("=" * 60)
    
    # Initialize trainer
    trainer = MovieGenreTrainer(model_type=MODEL_TYPE, input_size=(224, 224))
    
    # Load and prepare data
    X_train, X_val, y_train, y_val = trainer.load_dataset(CSV_PATH, IMAGES_DIR)
    
    # Build model
    trainer.build_model()
    
    # Train model
    trainer.train_model(X_train, y_train, X_val, y_val, IMAGES_DIR, epochs=EPOCHS)
    
    # Evaluate model
    trainer.evaluate_model(X_val, y_val, IMAGES_DIR)
    
    # Save model and encoder
    trainer.save_model_and_encoder()
    
    print("\n🎉 Training completed successfully!")
    print("🌐 You can now run the web app with: python src/web_app.py")

if __name__ == "__main__":
    main()