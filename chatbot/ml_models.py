"""
Enhanced ML Model Architecture
Implements sophisticated neural network models for the chatbot:
- LSTM-based sequence models for intent classification
- Attention mechanisms for better context understanding
- Response generation models
- Model evaluation and metrics
"""

import os
import numpy as np
from typing import List, Tuple, Dict, Optional
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from .logger import get_logger

logger = get_logger(__name__)


class LSTMIntentClassifier:
    """
    LSTM-based intent classifier with attention mechanism.
    Provides superior performance for intent classification.
    """
    
    def __init__(self, max_sequence_length: int = 50, 
                 max_vocab_size: int = 10000,
                 embedding_dim: int = 128,
                 lstm_units: int = 64):
        """
        Initialize LSTM intent classifier.
        
        Args:
            max_sequence_length: Maximum length of input sequences
            max_vocab_size: Maximum vocabulary size
            embedding_dim: Dimension of word embeddings
            lstm_units: Number of LSTM units
        """
        logger.info(f"Initializing LSTMIntentClassifier "
                   f"(seq_len={max_sequence_length}, vocab={max_vocab_size}, "
                   f"embed_dim={embedding_dim}, lstm={lstm_units})")
        
        self.max_sequence_length = max_sequence_length
        self.max_vocab_size = max_vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        
        # Tokenizer for text processing
        self.tokenizer = Tokenizer(num_words=max_vocab_size, oov_token='<UNK>')
        
        # Model components
        self.model = None
        self.intent_labels = []
        self.label_to_idx = {}
        self.idx_to_label = {}
        
        logger.info("LSTMIntentClassifier initialized")
    
    def build_model(self, num_classes: int, use_attention: bool = True) -> models.Model:
        """
        Build LSTM model architecture with optional attention.
        
        Args:
            num_classes: Number of intent classes
            use_attention: Whether to use attention mechanism
            
        Returns:
            Compiled Keras model
        """
        logger.info(f"Building LSTM model for {num_classes} classes "
                   f"(attention={'enabled' if use_attention else 'disabled'})")
        
        # Input layer
        input_layer = layers.Input(shape=(self.max_sequence_length,), name='input')
        
        # Embedding layer
        embedding = layers.Embedding(
            input_dim=self.max_vocab_size,
            output_dim=self.embedding_dim,
            input_length=self.max_sequence_length,
            mask_zero=True,
            name='embedding'
        )(input_layer)
        
        # Dropout for regularization
        embedding = layers.Dropout(0.3)(embedding)
        
        # Bidirectional LSTM layers
        lstm1 = layers.Bidirectional(
            layers.LSTM(self.lstm_units, return_sequences=True, dropout=0.2),
            name='bi_lstm_1'
        )(embedding)
        
        if use_attention:
            # Attention mechanism
            attention = layers.Attention(name='attention')([lstm1, lstm1])
            lstm_output = layers.Concatenate()([lstm1, attention])
        else:
            lstm_output = lstm1
        
        # Second LSTM layer
        lstm2 = layers.Bidirectional(
            layers.LSTM(self.lstm_units // 2, return_sequences=False, dropout=0.2),
            name='bi_lstm_2'
        )(lstm_output)
        
        # Dense layers
        dense = layers.Dense(64, activation='relu', name='dense_1')(lstm2)
        dense = layers.Dropout(0.3)(dense)
        dense = layers.Dense(32, activation='relu', name='dense_2')(dense)
        
        # Output layer
        output = layers.Dense(num_classes, activation='softmax', name='output')(dense)
        
        # Create model
        model = models.Model(inputs=input_layer, outputs=output, name='lstm_intent_classifier')
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.SparseCategoricalAccuracy()]
        )
        
        logger.info("LSTM model built successfully")
        return model
    
    def prepare_data(self, texts: List[str], labels: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data by tokenizing and padding.
        
        Args:
            texts: List of text samples
            labels: List of intent labels
            
        Returns:
            Tuple of (X, y) numpy arrays
        """
        logger.info(f"Preparing data: {len(texts)} samples")
        
        # Build label mappings
        unique_labels = sorted(list(set(labels)))
        self.intent_labels = unique_labels
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
        # Fit tokenizer on texts
        self.tokenizer.fit_on_texts(texts)
        
        # Convert texts to sequences
        sequences = self.tokenizer.texts_to_sequences(texts)
        
        # Pad sequences
        X = pad_sequences(sequences, maxlen=self.max_sequence_length, padding='post')
        
        # Convert labels to indices
        y = np.array([self.label_to_idx[label] for label in labels])
        
        logger.info(f"Data prepared: X.shape={X.shape}, y.shape={y.shape}, "
                   f"vocab_size={len(self.tokenizer.word_index)}")
        
        return X, y
    
    def train(self, texts: List[str], labels: List[str],
              validation_split: float = 0.2,
              epochs: int = 20,
              batch_size: int = 32,
              use_attention: bool = True) -> Dict[str, any]:
        """
        Train the LSTM model.
        
        Args:
            texts: Training texts
            labels: Training labels
            validation_split: Fraction of data to use for validation
            epochs: Number of training epochs
            batch_size: Batch size
            use_attention: Whether to use attention mechanism
            
        Returns:
            Training history dictionary
        """
        logger.info(f"Training LSTM model: {len(texts)} samples, "
                   f"{epochs} epochs, batch_size={batch_size}")
        
        # Prepare data
        X, y = self.prepare_data(texts, labels)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        logger.info(f"Train set: {len(X_train)}, Validation set: {len(X_val)}")
        
        # Build model
        num_classes = len(self.intent_labels)
        self.model = self.build_model(num_classes, use_attention=use_attention)
        
        # Callbacks
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            verbose=1,
            min_lr=1e-6
        )
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        logger.info("Model training completed")
        
        return history.history
    
    def predict(self, text: str) -> Tuple[str, float]:
        """
        Predict intent for a single text.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (predicted_intent, confidence)
        """
        if self.model is None:
            logger.error("Model not trained yet")
            raise ValueError("Model must be trained before prediction")
        
        # Preprocess text
        sequence = self.tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequence, maxlen=self.max_sequence_length, padding='post')
        
        # Predict
        prediction = self.model.predict(padded, verbose=0)
        predicted_idx = np.argmax(prediction[0])
        confidence = float(prediction[0][predicted_idx])
        
        predicted_label = self.idx_to_label[predicted_idx]
        
        logger.debug(f"Predicted: {predicted_label} (confidence: {confidence:.3f})")
        
        return predicted_label, confidence
    
    def predict_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        """
        Predict intents for multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of (predicted_intent, confidence) tuples
        """
        if self.model is None:
            raise ValueError("Model must be trained before prediction")
        
        # Preprocess texts
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=self.max_sequence_length, padding='post')
        
        # Predict
        predictions = self.model.predict(padded, verbose=0)
        
        results = []
        for pred in predictions:
            predicted_idx = np.argmax(pred)
            confidence = float(pred[predicted_idx])
            predicted_label = self.idx_to_label[predicted_idx]
            results.append((predicted_label, confidence))
        
        return results
    
    def evaluate(self, texts: List[str], labels: List[str]) -> Dict[str, any]:
        """
        Evaluate model performance.
        
        Args:
            texts: Test texts
            labels: True labels
            
        Returns:
            Evaluation metrics dictionary
        """
        logger.info(f"Evaluating model on {len(texts)} samples")
        
        # Prepare data
        sequences = self.tokenizer.texts_to_sequences(texts)
        X = pad_sequences(sequences, maxlen=self.max_sequence_length, padding='post')
        y_true = np.array([self.label_to_idx[label] for label in labels])
        
        # Predict
        predictions = self.model.predict(X, verbose=0)
        y_pred = np.argmax(predictions, axis=1)
        
        # Calculate metrics
        accuracy = np.mean(y_true == y_pred)
        
        # Classification report
        report = classification_report(
            y_true, y_pred,
            target_names=self.intent_labels,
            output_dict=True
        )
        
        # Confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        results = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': conf_matrix.tolist(),
            'num_samples': len(texts)
        }
        
        logger.info(f"Evaluation completed: accuracy={accuracy:.3f}")
        
        return results
    
    def save(self, model_dir: str):
        """
        Save model and tokenizer.
        
        Args:
            model_dir: Directory to save model files
        """
        try:
            os.makedirs(model_dir, exist_ok=True)
            
            # Save model
            model_path = os.path.join(model_dir, 'lstm_intent_model.keras')
            self.model.save(model_path)
            
            # Save tokenizer and labels
            import pickle
            metadata_path = os.path.join(model_dir, 'model_metadata.pkl')
            metadata = {
                'tokenizer': self.tokenizer,
                'intent_labels': self.intent_labels,
                'label_to_idx': self.label_to_idx,
                'idx_to_label': self.idx_to_label,
                'max_sequence_length': self.max_sequence_length
            }
            
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            logger.info(f"Model saved to {model_dir}")
        except Exception as e:
            logger.exception(f"Failed to save model: {e}")
            raise
    
    def load(self, model_dir: str):
        """
        Load model and tokenizer.
        
        Args:
            model_dir: Directory containing model files
        """
        try:
            # Load model
            model_path = os.path.join(model_dir, 'lstm_intent_model.keras')
            self.model = tf.keras.models.load_model(model_path)
            
            # Load metadata
            import pickle
            metadata_path = os.path.join(model_dir, 'model_metadata.pkl')
            
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            
            self.tokenizer = metadata['tokenizer']
            self.intent_labels = metadata['intent_labels']
            self.label_to_idx = metadata['label_to_idx']
            self.idx_to_label = metadata['idx_to_label']
            self.max_sequence_length = metadata['max_sequence_length']
            
            logger.info(f"Model loaded from {model_dir}")
        except Exception as e:
            logger.exception(f"Failed to load model: {e}")
            raise
