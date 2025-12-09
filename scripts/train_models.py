"""
Training Pipeline for Chatbot ML Models
Handles complete training workflow:
- Data loading and preprocessing
- Data splitting (train/validation/test)
- Model training with proper validation
- Model evaluation and metrics
- Model persistence
"""

import os
import sys
from typing import Dict, List, Tuple
import numpy as np
from sklearn.model_selection import train_test_split

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chatbot.dataset import ChatbotDataset
from chatbot.preprocessing import TextPreprocessor
from chatbot.embeddings import WordEmbeddings
from chatbot.ml_models import LSTMIntentClassifier
from chatbot.logger import get_logger

logger = get_logger(__name__)


class TrainingPipeline:
    """
    Complete training pipeline for chatbot models.
    Manages data preprocessing, splitting, training, and evaluation.
    """
    
    def __init__(self, model_dir: str = 'models'):
        """
        Initialize training pipeline.
        
        Args:
            model_dir: Directory to save trained models
        """
        logger.info(f"Initializing TrainingPipeline (model_dir={model_dir})")
        
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize components
        self.dataset = ChatbotDataset()
        self.preprocessor = TextPreprocessor(download_nltk_data=True)
        self.embeddings = None
        self.intent_model = None
        
        logger.info("TrainingPipeline initialized")
    
    def prepare_data(self, test_size: float = 0.2, 
                    val_size: float = 0.1) -> Dict[str, any]:
        """
        Prepare and split data for training.
        
        Args:
            test_size: Fraction of data for testing
            val_size: Fraction of training data for validation
            
        Returns:
            Dictionary containing train/val/test splits
        """
        logger.info("Preparing data for training")
        
        # Get intent training data
        texts, labels = self.dataset.get_intent_training_data()
        
        logger.info(f"Total samples: {len(texts)}")
        logger.info(f"Unique intents: {len(set(labels))}")
        
        # Clean and preprocess texts
        logger.info("Preprocessing texts...")
        cleaned_texts = []
        for text in texts:
            cleaned = self.preprocessor.clean_text(text, lowercase=True, 
                                                   preserve_currency=True)
            cleaned_texts.append(cleaned)
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            cleaned_texts, labels, 
            test_size=test_size, 
            random_state=42,
            stratify=labels
        )
        
        # Second split: separate validation from training
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_ratio,
            random_state=42,
            stratify=y_temp
        )
        
        logger.info(f"Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        
        return {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test),
            'all_texts': cleaned_texts,
            'all_labels': labels
        }
    
    def train_word_embeddings(self, texts: List[str], 
                             embedding_dim: int = 100,
                             epochs: int = 10) -> WordEmbeddings:
        """
        Train word embeddings on the dataset.
        
        Args:
            texts: Training texts
            embedding_dim: Dimension of embeddings
            epochs: Number of training epochs
            
        Returns:
            Trained WordEmbeddings object
        """
        logger.info(f"Training word embeddings (dim={embedding_dim}, epochs={epochs})")
        
        # Initialize embeddings
        self.embeddings = WordEmbeddings(
            embedding_dim=embedding_dim,
            window_size=2
        )
        
        # Train embeddings
        self.embeddings.train(texts, epochs=epochs, batch_size=128)
        
        # Save embeddings
        embeddings_path = os.path.join(self.model_dir, 'word_embeddings.pkl')
        self.embeddings.save(embeddings_path)
        
        logger.info(f"Word embeddings trained and saved to {embeddings_path}")
        
        return self.embeddings
    
    def train_intent_classifier(self, data_splits: Dict[str, any],
                                use_lstm: bool = True,
                                epochs: int = 20,
                                batch_size: int = 32) -> Dict[str, any]:
        """
        Train intent classification model.
        
        Args:
            data_splits: Dictionary with train/val/test splits
            use_lstm: Whether to use LSTM model (True) or simple model (False)
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Training results and metrics
        """
        logger.info(f"Training intent classifier (LSTM={use_lstm})")
        
        X_train, y_train = data_splits['train']
        X_val, y_val = data_splits['val']
        X_test, y_test = data_splits['test']
        
        if use_lstm:
            # Train LSTM model
            self.intent_model = LSTMIntentClassifier(
                max_sequence_length=50,
                max_vocab_size=10000,
                embedding_dim=128,
                lstm_units=64
            )
            
            # Train model
            history = self.intent_model.train(
                texts=X_train,
                labels=y_train,
                validation_split=0.0,  # We already have validation set
                epochs=epochs,
                batch_size=batch_size,
                use_attention=True
            )
            
            # Evaluate on test set
            eval_results = self.intent_model.evaluate(X_test, y_test)
            
            # Save model
            lstm_model_dir = os.path.join(self.model_dir, 'lstm_intent_classifier')
            self.intent_model.save(lstm_model_dir)
            
            logger.info(f"LSTM model trained and saved to {lstm_model_dir}")
            logger.info(f"Test accuracy: {eval_results['accuracy']:.4f}")
            
            return {
                'history': history,
                'evaluation': eval_results,
                'model_type': 'LSTM'
            }
        else:
            # Use existing simple TensorFlow model from nlp.py
            logger.info("Using simple TensorFlow model (from nlp.py)")
            return {'model_type': 'Simple'}
    
    def evaluate_models(self, data_splits: Dict[str, any]) -> Dict[str, any]:
        """
        Evaluate all trained models.
        
        Args:
            data_splits: Dictionary with test data
            
        Returns:
            Evaluation results
        """
        logger.info("Evaluating models...")
        
        X_test, y_test = data_splits['test']
        
        results = {}
        
        # Evaluate LSTM model if trained
        if self.intent_model is not None:
            lstm_results = self.intent_model.evaluate(X_test, y_test)
            results['lstm'] = lstm_results
            
            logger.info(f"LSTM Model Accuracy: {lstm_results['accuracy']:.4f}")
        
        # Test word embeddings if trained
        if self.embeddings is not None:
            # Test similarity
            test_words = ['spent', 'earned', 'groceries', 'rent']
            embedding_results = {}
            
            for word in test_words:
                similar = self.embeddings.get_similar_words(word, top_k=3)
                if similar:
                    embedding_results[word] = similar
                    logger.info(f"Similar to '{word}': {similar}")
            
            results['embeddings'] = embedding_results
        
        return results
    
    def run_full_training(self, train_embeddings: bool = True,
                         train_lstm: bool = True,
                         embedding_epochs: int = 10,
                         lstm_epochs: int = 20) -> Dict[str, any]:
        """
        Run complete training pipeline.
        
        Args:
            train_embeddings: Whether to train word embeddings
            train_lstm: Whether to train LSTM model
            embedding_epochs: Epochs for embedding training
            lstm_epochs: Epochs for LSTM training
            
        Returns:
            Complete training results
        """
        logger.info("=" * 60)
        logger.info("Starting Full Training Pipeline")
        logger.info("=" * 60)
        
        results = {}
        
        # Step 1: Prepare data
        logger.info("\n[Step 1/4] Preparing data...")
        data_splits = self.prepare_data(test_size=0.2, val_size=0.1)
        results['data_info'] = {
            'train_size': len(data_splits['train'][0]),
            'val_size': len(data_splits['val'][0]),
            'test_size': len(data_splits['test'][0]),
            'total_size': len(data_splits['all_texts'])
        }
        
        # Step 2: Train word embeddings
        if train_embeddings:
            logger.info("\n[Step 2/4] Training word embeddings...")
            self.train_word_embeddings(
                texts=data_splits['all_texts'],
                embedding_dim=100,
                epochs=embedding_epochs
            )
            results['embeddings_trained'] = True
        else:
            logger.info("\n[Step 2/4] Skipping word embeddings training")
            results['embeddings_trained'] = False
        
        # Step 3: Train LSTM intent classifier
        if train_lstm:
            logger.info("\n[Step 3/4] Training LSTM intent classifier...")
            training_results = self.train_intent_classifier(
                data_splits=data_splits,
                use_lstm=True,
                epochs=lstm_epochs,
                batch_size=32
            )
            results['lstm_training'] = training_results
        else:
            logger.info("\n[Step 3/4] Skipping LSTM training")
            results['lstm_training'] = None
        
        # Step 4: Evaluate models
        logger.info("\n[Step 4/4] Evaluating models...")
        eval_results = self.evaluate_models(data_splits)
        results['evaluation'] = eval_results
        
        logger.info("\n" + "=" * 60)
        logger.info("Training Pipeline Completed Successfully!")
        logger.info("=" * 60)
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def _print_summary(self, results: Dict[str, any]):
        """
        Print training summary.
        
        Args:
            results: Training results dictionary
        """
        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)
        
        # Data info
        if 'data_info' in results:
            info = results['data_info']
            print(f"\nDataset Information:")
            print(f"  Total samples: {info['total_size']}")
            print(f"  Training set: {info['train_size']}")
            print(f"  Validation set: {info['val_size']}")
            print(f"  Test set: {info['test_size']}")
        
        # LSTM results
        if results.get('lstm_training') and 'evaluation' in results['lstm_training']:
            eval_res = results['lstm_training']['evaluation']
            print(f"\nLSTM Intent Classifier:")
            print(f"  Test Accuracy: {eval_res['accuracy']:.4f}")
            print(f"  Number of intents: {len(eval_res['classification_report']) - 3}")
        
        # Embeddings
        if results.get('embeddings_trained'):
            print(f"\nWord Embeddings:")
            print(f"  Status: Trained successfully")
            if 'embeddings' in results.get('evaluation', {}):
                print(f"  Sample similarities available")
        
        print("\n" + "=" * 60)


def main():
    """
    Main function to run training pipeline.
    Can be run as standalone script.
    """
    # Configure logging
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize pipeline
    pipeline = TrainingPipeline(model_dir='models')
    
    # Run full training
    results = pipeline.run_full_training(
        train_embeddings=True,
        train_lstm=True,
        embedding_epochs=10,
        lstm_epochs=20
    )
    
    print("\nTraining complete! Models saved in 'models/' directory.")
    

if __name__ == '__main__':
    main()
