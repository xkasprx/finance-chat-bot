"""
Advanced Text Preprocessing Module
Handles text cleaning, tokenization, and preprocessing for NLP tasks.
Implements various preprocessing techniques including:
- Text normalization and cleaning
- Advanced tokenization (word, subword, character)
- Stopword removal and lemmatization
- Special character handling
"""

import re
import string
from typing import List, Dict, Set, Tuple
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from .logger import get_logger

logger = get_logger(__name__)


class TextPreprocessor:
    """
    Advanced text preprocessing for chatbot NLP tasks.
    Provides comprehensive text cleaning and tokenization capabilities.
    """
    
    def __init__(self, download_nltk_data: bool = True):
        """
        Initialize the text preprocessor.
        
        Args:
            download_nltk_data: Whether to download required NLTK data
        """
        logger.info("Initializing TextPreprocessor")
        
        # Download required NLTK data
        if download_nltk_data:
            self._download_nltk_resources()
        
        # Initialize NLTK tools
        try:
            self.lemmatizer = WordNetLemmatizer()
            self.stemmer = PorterStemmer()
            self.stop_words = set(stopwords.words('english'))
            logger.info("NLTK tools initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize NLTK tools: {e}. Some features may not work.")
            self.lemmatizer = None
            self.stemmer = None
            self.stop_words = set()
        
        # Custom stopwords to keep (financial terms we want to preserve)
        self.preserve_words = {'spent', 'paid', 'earned', 'income', 'expense', 'most', 'least'}
        
        # Remove preserved words from stopwords
        self.stop_words = self.stop_words - self.preserve_words
    
    def _download_nltk_resources(self):
        """Download required NLTK resources."""
        resources = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 'punkt_tab']
        
        for resource in resources:
            try:
                nltk.download(resource, quiet=True)
                logger.debug(f"Downloaded NLTK resource: {resource}")
            except Exception as e:
                logger.warning(f"Failed to download NLTK resource {resource}: {e}")
    
    def clean_text(self, text: str, lowercase: bool = True, 
                   remove_punctuation: bool = False,
                   remove_numbers: bool = False,
                   preserve_currency: bool = True) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Input text to clean
            lowercase: Convert text to lowercase
            remove_punctuation: Remove punctuation marks
            remove_numbers: Remove numeric characters
            preserve_currency: Preserve currency symbols and amounts
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Preserve currency amounts before cleaning
        currency_pattern = r'\$\d+(?:\.\d{1,2})?'
        currency_matches = []
        if preserve_currency:
            currency_matches = re.findall(currency_pattern, text)
            # Replace with placeholders
            for i, match in enumerate(currency_matches):
                text = text.replace(match, f"__CURRENCY_{i}__", 1)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Handle contractions (expand them)
        contractions = {
            "won't": "will not", "can't": "cannot", "n't": " not",
            "'re": " are", "'ve": " have", "'ll": " will",
            "'d": " would", "'m": " am"
        }
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        # Convert to lowercase if requested
        if lowercase:
            text = text.lower()
        
        # Remove numbers if requested (but not currency placeholders)
        if remove_numbers:
            text = re.sub(r'(?<!_)\d+(?!_)', '', text)
        
        # Remove punctuation if requested
        if remove_punctuation:
            # Create translation table excluding currency placeholder markers
            translator = str.maketrans('', '', string.punctuation.replace('_', ''))
            text = text.translate(translator)
        
        # Restore currency amounts
        if preserve_currency:
            for i, match in enumerate(currency_matches):
                text = text.replace(f"__CURRENCY_{i}__", match)
        
        # Remove extra whitespace again
        text = ' '.join(text.split())
        
        return text.strip()
    
    def tokenize(self, text: str, method: str = 'word') -> List[str]:
        """
        Tokenize text using various methods.
        
        Args:
            text: Input text to tokenize
            method: Tokenization method ('word', 'sentence', 'char')
            
        Returns:
            List of tokens
        """
        if not text:
            return []
        
        try:
            if method == 'word':
                return word_tokenize(text)
            elif method == 'sentence':
                return sent_tokenize(text)
            elif method == 'char':
                return list(text)
            else:
                logger.warning(f"Unknown tokenization method: {method}. Using word tokenization.")
                return word_tokenize(text)
        except Exception as e:
            logger.warning(f"NLTK tokenization failed: {e}. Using simple split.")
            if method == 'word':
                return text.split()
            elif method == 'sentence':
                return [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
            else:
                return list(text)
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remove stopwords from token list.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Filtered token list without stopwords
        """
        if not self.stop_words:
            return tokens
        
        return [token for token in tokens if token.lower() not in self.stop_words]
    
    def lemmatize(self, tokens: List[str]) -> List[str]:
        """
        Lemmatize tokens to their base form.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Lemmatized tokens
        """
        if not self.lemmatizer:
            logger.warning("Lemmatizer not available. Returning original tokens.")
            return tokens
        
        try:
            return [self.lemmatizer.lemmatize(token) for token in tokens]
        except Exception as e:
            logger.warning(f"Lemmatization failed: {e}. Returning original tokens.")
            return tokens
    
    def stem(self, tokens: List[str]) -> List[str]:
        """
        Stem tokens to their root form.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Stemmed tokens
        """
        if not self.stemmer:
            logger.warning("Stemmer not available. Returning original tokens.")
            return tokens
        
        try:
            return [self.stemmer.stem(token) for token in tokens]
        except Exception as e:
            logger.warning(f"Stemming failed: {e}. Returning original tokens.")
            return tokens
    
    def preprocess(self, text: str, 
                   lowercase: bool = True,
                   remove_stopwords: bool = False,
                   lemmatize: bool = False,
                   stem: bool = False) -> List[str]:
        """
        Complete preprocessing pipeline for text.
        
        Args:
            text: Input text
            lowercase: Convert to lowercase
            remove_stopwords: Remove stopwords
            lemmatize: Apply lemmatization
            stem: Apply stemming
            
        Returns:
            Preprocessed tokens
        """
        # Clean the text
        cleaned = self.clean_text(text, lowercase=lowercase, 
                                 remove_punctuation=False,
                                 preserve_currency=True)
        
        # Tokenize
        tokens = self.tokenize(cleaned, method='word')
        
        # Remove stopwords if requested
        if remove_stopwords:
            tokens = self.remove_stopwords(tokens)
        
        # Lemmatize if requested
        if lemmatize:
            tokens = self.lemmatize(tokens)
        
        # Stem if requested (usually don't do both lemmatize and stem)
        if stem and not lemmatize:
            tokens = self.stem(tokens)
        
        return tokens
    
    def extract_features(self, text: str) -> Dict[str, any]:
        """
        Extract various features from text for analysis.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of extracted features
        """
        tokens = self.tokenize(text, method='word')
        
        features = {
            'text_length': len(text),
            'token_count': len(tokens),
            'avg_word_length': sum(len(token) for token in tokens) / len(tokens) if tokens else 0,
            'sentence_count': len(self.tokenize(text, method='sentence')),
            'has_currency': bool(re.search(r'\$\d+', text)),
            'has_numbers': bool(re.search(r'\d+', text)),
            'has_question': '?' in text,
            'has_exclamation': '!' in text,
        }
        
        return features
    
    def batch_preprocess(self, texts: List[str], **kwargs) -> List[List[str]]:
        """
        Preprocess multiple texts in batch.
        
        Args:
            texts: List of input texts
            **kwargs: Arguments to pass to preprocess method
            
        Returns:
            List of preprocessed token lists
        """
        return [self.preprocess(text, **kwargs) for text in texts]
    
    def create_vocabulary(self, texts: List[str], min_frequency: int = 1) -> Dict[str, int]:
        """
        Create vocabulary from a list of texts.
        
        Args:
            texts: List of texts
            min_frequency: Minimum frequency for a word to be included
            
        Returns:
            Dictionary mapping words to their frequencies
        """
        word_freq = {}
        
        for text in texts:
            tokens = self.preprocess(text)
            for token in tokens:
                word_freq[token] = word_freq.get(token, 0) + 1
        
        # Filter by minimum frequency
        vocabulary = {word: freq for word, freq in word_freq.items() 
                     if freq >= min_frequency}
        
        logger.info(f"Created vocabulary with {len(vocabulary)} words (min_freq={min_frequency})")
        return vocabulary
    
    def get_ngrams(self, tokens: List[str], n: int = 2) -> List[Tuple[str, ...]]:
        """
        Generate n-grams from tokens.
        
        Args:
            tokens: List of tokens
            n: Size of n-grams
            
        Returns:
            List of n-gram tuples
        """
        if len(tokens) < n:
            return []
        
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
