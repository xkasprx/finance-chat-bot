"""
Advanced NLP Features Module
Implements sophisticated NLP capabilities:
- Sentiment Analysis: Determine sentiment (positive/negative/neutral)
- Named Entity Recognition: Extract entities (dates, amounts, categories)
- Intent confidence scoring
- Contextual understanding
"""

import re
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from .logger import get_logger
from .preprocessing import TextPreprocessor

logger = get_logger(__name__)


class SentimentAnalyzer:
    """
    Analyzes sentiment of text using lexicon-based and ML approaches.
    """
    
    def __init__(self):
        """Initialize sentiment analyzer."""
        logger.info("Initializing SentimentAnalyzer")
        
        # Sentiment lexicon (simple approach)
        self.positive_words = {
            'good', 'great', 'excellent', 'awesome', 'wonderful', 'fantastic',
            'happy', 'pleased', 'satisfied', 'love', 'perfect', 'best',
            'earned', 'gained', 'bonus', 'profit', 'income', 'received'
        }
        
        self.negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'poor', 'worst',
            'sad', 'unhappy', 'disappointed', 'hate', 'lost', 'waste',
            'spent', 'expense', 'debt', 'owe', 'bill', 'paid'
        }
        
        self.preprocessor = TextPreprocessor(download_nltk_data=False)
        logger.info("SentimentAnalyzer initialized")
    
    def analyze(self, text: str) -> Dict[str, any]:
        """
        Analyze sentiment of text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with sentiment label, score, and confidence
        """
        # Preprocess text
        tokens = self.preprocessor.preprocess(text, lowercase=True, 
                                             remove_stopwords=False)
        
        # Count positive and negative words
        positive_count = sum(1 for token in tokens if token in self.positive_words)
        negative_count = sum(1 for token in tokens if token in self.negative_words)
        
        # Calculate sentiment score (-1 to 1)
        total = positive_count + negative_count
        if total == 0:
            sentiment_score = 0.0
            sentiment_label = 'neutral'
            confidence = 0.5
        else:
            sentiment_score = (positive_count - negative_count) / total
            confidence = total / len(tokens) if tokens else 0.5
            
            if sentiment_score > 0.2:
                sentiment_label = 'positive'
            elif sentiment_score < -0.2:
                sentiment_label = 'negative'
            else:
                sentiment_label = 'neutral'
        
        return {
            'label': sentiment_label,
            'score': float(sentiment_score),
            'confidence': float(confidence),
            'positive_words': positive_count,
            'negative_words': negative_count
        }


class NamedEntityRecognizer:
    """
    Extracts named entities from text including:
    - Monetary amounts
    - Dates and times
    - Categories
    - Percentages
    """
    
    def __init__(self):
        """Initialize named entity recognizer."""
        logger.info("Initializing NamedEntityRecognizer")
        
        # Category keywords mapping
        self.category_patterns = {
            'groceries': r'\b(grocery|groceries|supermarket|market|food shopping)\b',
            'rent': r'\b(rent|landlord|housing)\b',
            'utilities': r'\b(utilities|electric|electricity|water|gas bill|internet|wifi|phone bill)\b',
            'entertainment': r'\b(movie|cinema|entertainment|netflix|spotify|game|concert|theatre)\b',
            'dining': r'\b(dinner|lunch|breakfast|food|restaurant|cafe|coffee|pizza|takeout)\b',
            'transportation': r'\b(uber|lyft|bus|train|ticket|taxi|metro|subway)\b',
            'auto': r'\b(car|fuel|gasoline|gas station|repair|maintenance|oil change)\b',
            'healthcare': r'\b(doctor|hospital|medical|pharmacy|medicine|health insurance)\b',
            'shopping': r'\b(shopping|store|mall|amazon|purchase|buy|bought)\b',
            'salary': r'\b(salary|paycheck|wages|pay)\b',
            'bonus': r'\b(bonus|incentive|commission)\b',
            'investment': r'\b(investment|dividend|stock|return)\b',
        }
        
        logger.info("NamedEntityRecognizer initialized")
    
    def extract_entities(self, text: str) -> Dict[str, any]:
        """
        Extract all entities from text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of extracted entities
        """
        entities = {
            'amounts': self._extract_amounts(text),
            'dates': self._extract_dates(text),
            'categories': self._extract_categories(text),
            'percentages': self._extract_percentages(text),
        }
        
        return entities
    
    def _extract_amounts(self, text: str) -> List[Dict[str, any]]:
        """
        Extract monetary amounts from text.
        
        Args:
            text: Input text
            
        Returns:
            List of amount entities with value and position
        """
        amounts = []
        
        # Pattern for currency amounts
        pattern = r'\$?\s*(\d+(?:,\d{3})*(?:\.\d{1,2})?)\s*(?:dollars?|bucks?|\$)?'
        
        for match in re.finditer(pattern, text, re.IGNORECASE):
            amount_str = match.group(1).replace(',', '')
            try:
                value = float(amount_str)
                amounts.append({
                    'value': value,
                    'text': match.group(0),
                    'start': match.start(),
                    'end': match.end()
                })
            except ValueError:
                continue
        
        return amounts
    
    def _extract_dates(self, text: str) -> List[Dict[str, any]]:
        """
        Extract dates from text.
        
        Args:
            text: Input text
            
        Returns:
            List of date entities
        """
        dates = []
        
        # Common date patterns
        patterns = [
            (r'\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b', 'MM/DD/YYYY'),
            (r'\b(\d{4})-(\d{1,2})-(\d{1,2})\b', 'YYYY-MM-DD'),
            (r'\b(today|yesterday|tomorrow)\b', 'relative'),
            (r'\b(last|this|next)\s+(week|month|year)\b', 'relative_period'),
        ]
        
        for pattern, date_type in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                dates.append({
                    'text': match.group(0),
                    'type': date_type,
                    'start': match.start(),
                    'end': match.end()
                })
        
        return dates
    
    def _extract_categories(self, text: str) -> List[Dict[str, str]]:
        """
        Extract financial categories from text.
        
        Args:
            text: Input text
            
        Returns:
            List of category entities
        """
        categories = []
        text_lower = text.lower()
        
        for category, pattern in self.category_patterns.items():
            if re.search(pattern, text_lower):
                categories.append({
                    'category': category,
                    'confidence': 0.8
                })
        
        return categories
    
    def _extract_percentages(self, text: str) -> List[Dict[str, any]]:
        """
        Extract percentages from text.
        
        Args:
            text: Input text
            
        Returns:
            List of percentage entities
        """
        percentages = []
        
        pattern = r'(\d+(?:\.\d+)?)\s*%'
        
        for match in re.finditer(pattern, text):
            try:
                value = float(match.group(1))
                percentages.append({
                    'value': value,
                    'text': match.group(0),
                    'start': match.start(),
                    'end': match.end()
                })
            except ValueError:
                continue
        
        return percentages


class IntentConfidenceScorer:
    """
    Calculates confidence scores for intent classification.
    """
    
    def __init__(self):
        """Initialize intent confidence scorer."""
        logger.info("Initializing IntentConfidenceScorer")
        
        # Intent-specific keywords and weights
        self.intent_keywords = {
            'add_income': {
                'strong': ['earned', 'got paid', 'salary', 'paycheck', 'bonus', 'received'],
                'medium': ['made', 'income', 'deposit'],
                'weak': ['money', 'cash']
            },
            'add_expense': {
                'strong': ['spent', 'bought', 'purchased', 'paid for'],
                'medium': ['expense', 'cost', 'price'],
                'weak': ['got', 'need']
            },
            'query_income': {
                'strong': ['total income', 'how much earned', 'what\'s my income'],
                'medium': ['show income', 'income total'],
                'weak': ['income', 'earned']
            },
            'query_expenses': {
                'strong': ['total expenses', 'how much spent', 'what\'s my spending'],
                'medium': ['show expenses', 'expense total'],
                'weak': ['expenses', 'spent']
            }
        }
        
        logger.info("IntentConfidenceScorer initialized")
    
    def score(self, text: str, predicted_intent: str) -> float:
        """
        Calculate confidence score for predicted intent.
        
        Args:
            text: Input text
            predicted_intent: Predicted intent label
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        if predicted_intent not in self.intent_keywords:
            return 0.5  # Default moderate confidence
        
        text_lower = text.lower()
        keywords = self.intent_keywords[predicted_intent]
        
        # Calculate score based on keyword matches
        score = 0.3  # Base score
        
        # Strong keywords add significant confidence
        for keyword in keywords.get('strong', []):
            if keyword in text_lower:
                score += 0.25
        
        # Medium keywords add moderate confidence
        for keyword in keywords.get('medium', []):
            if keyword in text_lower:
                score += 0.15
        
        # Weak keywords add slight confidence
        for keyword in keywords.get('weak', []):
            if keyword in text_lower:
                score += 0.05
        
        # Cap at 1.0
        return min(score, 1.0)


class AdvancedNLPProcessor:
    """
    Main class combining all advanced NLP features.
    """
    
    def __init__(self):
        """Initialize advanced NLP processor."""
        logger.info("Initializing AdvancedNLPProcessor")
        
        self.sentiment_analyzer = SentimentAnalyzer()
        self.ner = NamedEntityRecognizer()
        self.confidence_scorer = IntentConfidenceScorer()
        self.preprocessor = TextPreprocessor(download_nltk_data=False)
        
        logger.info("AdvancedNLPProcessor initialized successfully")
    
    def process(self, text: str, intent: Optional[str] = None) -> Dict[str, any]:
        """
        Perform comprehensive NLP analysis on text.
        
        Args:
            text: Input text
            intent: Optional predicted intent for confidence scoring
            
        Returns:
            Dictionary with all NLP analysis results
        """
        logger.debug(f"Processing text: {text[:50]}...")
        
        results = {
            'original_text': text,
            'cleaned_text': self.preprocessor.clean_text(text),
            'tokens': self.preprocessor.tokenize(text),
            'sentiment': self.sentiment_analyzer.analyze(text),
            'entities': self.ner.extract_entities(text),
            'features': self.preprocessor.extract_features(text)
        }
        
        # Add confidence score if intent provided
        if intent:
            results['intent_confidence'] = self.confidence_scorer.score(text, intent)
        
        return results
    
    def analyze_sentiment(self, text: str) -> Dict[str, any]:
        """
        Quick sentiment analysis.
        
        Args:
            text: Input text
            
        Returns:
            Sentiment analysis results
        """
        return self.sentiment_analyzer.analyze(text)
    
    def extract_entities(self, text: str) -> Dict[str, any]:
        """
        Quick entity extraction.
        
        Args:
            text: Input text
            
        Returns:
            Extracted entities
        """
        return self.ner.extract_entities(text)
