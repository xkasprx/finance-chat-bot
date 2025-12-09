"""
Word Embeddings Module
Implements word embeddings to represent words as vectors in continuous space.
Supports:
- Training custom word embeddings on dataset
- Using pre-trained embeddings
- Word similarity and analogies
- Vector operations
"""

import os
import numpy as np
import pickle
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import tensorflow as tf
from tensorflow.keras import layers, models
from .logger import get_logger
from .preprocessing import TextPreprocessor

logger = get_logger(__name__)


class WordEmbeddings:
    """
    Word embeddings for representing words as dense vectors.
    Trains embeddings using Skip-gram or CBOW approach.
    """
    
    def __init__(self, embedding_dim: int = 100, window_size: int = 2):
        """
        Initialize word embeddings.
        
        Args:
            embedding_dim: Dimension of embedding vectors
            window_size: Context window size for training
        """
        logger.info(f"Initializing WordEmbeddings (dim={embedding_dim}, window={window_size})")
        
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.preprocessor = TextPreprocessor(download_nltk_data=False)
        
        # Vocabulary and embeddings
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.embeddings_matrix = None
        self.model = None
        
        # Special tokens
        self.PAD_TOKEN = '<PAD>'
        self.UNK_TOKEN = '<UNK>'
        
        logger.info("WordEmbeddings initialized")
    
    def build_vocabulary(self, texts: List[str], min_frequency: int = 1):
        """
        Build vocabulary from texts.
        
        Args:
            texts: List of text documents
            min_frequency: Minimum word frequency to include in vocabulary
        """
        logger.info(f"Building vocabulary from {len(texts)} texts")
        
        # Count word frequencies
        word_freq = defaultdict(int)
        
        for text in texts:
            tokens = self.preprocessor.preprocess(text, lowercase=True, 
                                                 remove_stopwords=False)
            for token in tokens:
                word_freq[token] += 1
        
        # Filter by minimum frequency and create mappings
        self.word_to_idx = {self.PAD_TOKEN: 0, self.UNK_TOKEN: 1}
        idx = 2
        
        for word, freq in sorted(word_freq.items(), key=lambda x: x[1], reverse=True):
            if freq >= min_frequency:
                self.word_to_idx[word] = idx
                idx += 1
        
        # Create reverse mapping
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        
        logger.info(f"Vocabulary built with {len(self.word_to_idx)} words")
    
    def generate_training_data(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate training data for Skip-gram model.
        Creates (target, context) pairs from texts.
        
        Args:
            texts: List of text documents
            
        Returns:
            Tuple of (target_words, context_words) as numpy arrays
        """
        logger.info("Generating training data for embeddings")
        
        target_words = []
        context_words = []
        
        for text in texts:
            # Tokenize and convert to indices
            tokens = self.preprocessor.preprocess(text, lowercase=True, 
                                                 remove_stopwords=False)
            indices = [self.word_to_idx.get(token, self.word_to_idx[self.UNK_TOKEN]) 
                      for token in tokens]
            
            # Generate context pairs
            for i, target_idx in enumerate(indices):
                # Get context window
                start = max(0, i - self.window_size)
                end = min(len(indices), i + self.window_size + 1)
                
                for j in range(start, end):
                    if j != i:  # Skip the target word itself
                        target_words.append(target_idx)
                        context_words.append(indices[j])
        
        logger.info(f"Generated {len(target_words)} training pairs")
        return np.array(target_words), np.array(context_words)
    
    def train(self, texts: List[str], epochs: int = 10, batch_size: int = 128,
              learning_rate: float = 0.01):
        """
        Train word embeddings using Skip-gram model.
        
        Args:
            texts: List of text documents
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
        """
        logger.info(f"Training word embeddings for {epochs} epochs")
        
        # Build vocabulary if not already built
        if not self.word_to_idx:
            self.build_vocabulary(texts)
        
        # Generate training data
        target_words, context_words = self.generate_training_data(texts)
        
        vocab_size = len(self.word_to_idx)
        
        # Build Skip-gram model
        # Target word input
        target_input = layers.Input(shape=(1,), name='target_word')
        
        # Embedding layer for target words
        target_embedding = layers.Embedding(
            input_dim=vocab_size,
            output_dim=self.embedding_dim,
            input_length=1,
            name='target_embedding'
        )(target_input)
        target_embedding = layers.Reshape((self.embedding_dim,))(target_embedding)
        
        # Context word input
        context_input = layers.Input(shape=(1,), name='context_word')
        
        # Embedding layer for context words
        context_embedding = layers.Embedding(
            input_dim=vocab_size,
            output_dim=self.embedding_dim,
            input_length=1,
            name='context_embedding'
        )(context_input)
        context_embedding = layers.Reshape((self.embedding_dim,))(context_embedding)
        
        # Dot product to get similarity
        dot_product = layers.Dot(axes=1)([target_embedding, context_embedding])
        
        # Output layer
        output = layers.Dense(1, activation='sigmoid')(dot_product)
        
        # Create model
        self.model = models.Model(inputs=[target_input, context_input], outputs=output)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Create labels (all pairs are positive examples)
        labels = np.ones(len(target_words))
        
        # Train the model
        self.model.fit(
            [target_words, context_words],
            labels,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1
        )
        
        # Extract embeddings matrix from the trained model
        embedding_layer = self.model.get_layer('target_embedding')
        self.embeddings_matrix = embedding_layer.get_weights()[0]
        
        logger.info("Word embeddings training completed")
    
    def get_embedding(self, word: str) -> Optional[np.ndarray]:
        """
        Get embedding vector for a word.
        
        Args:
            word: Input word
            
        Returns:
            Embedding vector or None if word not in vocabulary
        """
        word = word.lower()
        idx = self.word_to_idx.get(word)
        
        if idx is None:
            logger.debug(f"Word '{word}' not in vocabulary")
            return None
        
        if self.embeddings_matrix is None:
            logger.warning("Embeddings not trained yet")
            return None
        
        return self.embeddings_matrix[idx]
    
    def get_similar_words(self, word: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find most similar words using cosine similarity.
        
        Args:
            word: Input word
            top_k: Number of similar words to return
            
        Returns:
            List of (word, similarity_score) tuples
        """
        embedding = self.get_embedding(word)
        
        if embedding is None or self.embeddings_matrix is None:
            return []
        
        # Calculate cosine similarities with all words
        # Normalize embeddings
        norm_embeddings = self.embeddings_matrix / (
            np.linalg.norm(self.embeddings_matrix, axis=1, keepdims=True) + 1e-10
        )
        norm_embedding = embedding / (np.linalg.norm(embedding) + 1e-10)
        
        # Compute similarities
        similarities = np.dot(norm_embeddings, norm_embedding)
        
        # Get top-k similar words (excluding the word itself)
        word_idx = self.word_to_idx[word.lower()]
        similarities[word_idx] = -1  # Exclude the word itself
        
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        similar_words = [
            (self.idx_to_word[idx], float(similarities[idx]))
            for idx in top_indices
        ]
        
        return similar_words
    
    def word_analogy(self, word_a: str, word_b: str, word_c: str, top_k: int = 1) -> List[Tuple[str, float]]:
        """
        Solve word analogies: word_a is to word_b as word_c is to ?
        Example: "king" is to "queen" as "man" is to "woman"
        
        Args:
            word_a: First word in analogy
            word_b: Second word in analogy
            word_c: Third word in analogy
            top_k: Number of results to return
            
        Returns:
            List of (word, score) tuples
        """
        emb_a = self.get_embedding(word_a)
        emb_b = self.get_embedding(word_b)
        emb_c = self.get_embedding(word_c)
        
        if None in [emb_a, emb_b, emb_c] or self.embeddings_matrix is None:
            logger.warning("Cannot perform analogy: some words not in vocabulary")
            return []
        
        # Vector arithmetic: b - a + c
        target_vector = emb_b - emb_a + emb_c
        
        # Normalize
        norm_embeddings = self.embeddings_matrix / (
            np.linalg.norm(self.embeddings_matrix, axis=1, keepdims=True) + 1e-10
        )
        norm_target = target_vector / (np.linalg.norm(target_vector) + 1e-10)
        
        # Find most similar
        similarities = np.dot(norm_embeddings, norm_target)
        
        # Exclude input words
        for word in [word_a, word_b, word_c]:
            idx = self.word_to_idx.get(word.lower())
            if idx is not None:
                similarities[idx] = -1
        
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = [
            (self.idx_to_word[idx], float(similarities[idx]))
            for idx in top_indices
        ]
        
        return results
    
    def save(self, filepath: str):
        """
        Save embeddings to file.
        
        Args:
            filepath: Path to save embeddings
        """
        try:
            data = {
                'word_to_idx': self.word_to_idx,
                'idx_to_word': self.idx_to_word,
                'embeddings_matrix': self.embeddings_matrix,
                'embedding_dim': self.embedding_dim,
                'window_size': self.window_size
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            
            logger.info(f"Embeddings saved to {filepath}")
        except Exception as e:
            logger.exception(f"Failed to save embeddings: {e}")
            raise
    
    def load(self, filepath: str):
        """
        Load embeddings from file.
        
        Args:
            filepath: Path to load embeddings from
        """
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.word_to_idx = data['word_to_idx']
            self.idx_to_word = data['idx_to_word']
            self.embeddings_matrix = data['embeddings_matrix']
            self.embedding_dim = data['embedding_dim']
            self.window_size = data['window_size']
            
            logger.info(f"Embeddings loaded from {filepath}")
        except Exception as e:
            logger.exception(f"Failed to load embeddings: {e}")
            raise
    
    def get_sentence_embedding(self, text: str, method: str = 'mean') -> Optional[np.ndarray]:
        """
        Get embedding for a sentence by aggregating word embeddings.
        
        Args:
            text: Input text
            method: Aggregation method ('mean', 'sum', 'max')
            
        Returns:
            Sentence embedding vector
        """
        tokens = self.preprocessor.preprocess(text, lowercase=True, 
                                             remove_stopwords=False)
        
        embeddings = []
        for token in tokens:
            emb = self.get_embedding(token)
            if emb is not None:
                embeddings.append(emb)
        
        if not embeddings:
            return None
        
        embeddings = np.array(embeddings)
        
        if method == 'mean':
            return np.mean(embeddings, axis=0)
        elif method == 'sum':
            return np.sum(embeddings, axis=0)
        elif method == 'max':
            return np.max(embeddings, axis=0)
        else:
            logger.warning(f"Unknown aggregation method: {method}. Using mean.")
            return np.mean(embeddings, axis=0)
