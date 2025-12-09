"""
Context-Aware Response System
Manages conversation context and generates contextual responses.
Features:
- Conversation history tracking
- Context-aware response generation
- Reference resolution (pronouns, "it", "that", etc.)
- Multi-turn conversation support
"""

from typing import List, Dict, Optional, Tuple
from datetime import datetime
from collections import deque
import random
from .logger import get_logger

logger = get_logger(__name__)


class ConversationContext:
    """
    Manages conversation context and history.
    """
    
    def __init__(self, max_history: int = 10):
        """
        Initialize conversation context.
        
        Args:
            max_history: Maximum number of turns to keep in history
        """
        logger.info(f"Initializing ConversationContext (max_history={max_history})")
        
        self.max_history = max_history
        self.history = deque(maxlen=max_history)
        self.last_intent = None
        self.last_entities = {}
        self.last_category = None
        self.last_amount = None
        self.conversation_state = {}
        
        logger.info("ConversationContext initialized")
    
    def add_turn(self, user_message: str, bot_response: str, 
                 intent: str, entities: Dict):
        """
        Add a conversation turn to history.
        
        Args:
            user_message: User's message
            bot_response: Bot's response
            intent: Detected intent
            entities: Extracted entities
        """
        turn = {
            'timestamp': datetime.now(),
            'user_message': user_message,
            'bot_response': bot_response,
            'intent': intent,
            'entities': entities
        }
        
        self.history.append(turn)
        self.last_intent = intent
        self.last_entities = entities
        
        # Track specific entities
        if 'category' in entities:
            self.last_category = entities['category']
        if 'amount' in entities:
            self.last_amount = entities['amount']
        
        logger.debug(f"Added turn to context: intent={intent}")
    
    def get_recent_history(self, n: int = 3) -> List[Dict]:
        """
        Get n most recent conversation turns.
        
        Args:
            n: Number of recent turns to retrieve
            
        Returns:
            List of recent conversation turns
        """
        return list(self.history)[-n:]
    
    def get_last_intent(self) -> Optional[str]:
        """Get the last detected intent."""
        return self.last_intent
    
    def get_last_category(self) -> Optional[str]:
        """Get the last mentioned category."""
        return self.last_category
    
    def get_last_amount(self) -> Optional[float]:
        """Get the last mentioned amount."""
        return self.last_amount
    
    def clear(self):
        """Clear conversation history and context."""
        self.history.clear()
        self.last_intent = None
        self.last_entities = {}
        self.last_category = None
        self.last_amount = None
        self.conversation_state = {}
        logger.info("Conversation context cleared")


class ContextAwareResponseGenerator:
    """
    Generates context-aware responses based on conversation history.
    """
    
    def __init__(self):
        """Initialize context-aware response generator."""
        logger.info("Initializing ContextAwareResponseGenerator")
        
        self.context = ConversationContext()
        
        # Response templates with context awareness
        self.context_templates = {
            'follow_up_income': [
                "Great! I've added that ${amount:.2f} to your income as well.",
                "Got it! Another ${amount:.2f} added to your income.",
                "Recorded! Your new income entry of ${amount:.2f} has been saved.",
            ],
            'follow_up_expense': [
                "I've recorded that ${amount:.2f} expense in {category} too.",
                "Got it! Another ${amount:.2f} expense in {category} has been logged.",
                "Recorded! ${amount:.2f} added to your {category} expenses.",
            ],
            'follow_up_query': [
                "Based on your question, your total {type} is ${total:.2f}.",
                "As you asked, your {type} total is ${total:.2f}.",
            ],
            'reference_resolution': [
                "Based on our conversation about {topic}, {response}",
                "Following up on {topic}, {response}",
            ]
        }
        
        # Contextual phrases for smooth conversation
        self.contextual_phrases = {
            'follow_up': ['Also', 'Additionally', 'And', 'Furthermore'],
            'confirmation': ['Got it', 'Understood', 'I see', 'Okay'],
            'continuation': ['Let me help you with that', 'Sure thing', 'Absolutely'],
        }
        
        logger.info("ContextAwareResponseGenerator initialized")
    
    def generate_response(self, intent: str, entities: Dict, 
                         user_message: str, db_result: Optional[Dict] = None) -> str:
        """
        Generate a context-aware response.
        
        Args:
            intent: Detected intent
            entities: Extracted entities
            user_message: Original user message
            db_result: Optional database query result
            
        Returns:
            Context-aware response string
        """
        # Check if this is a follow-up to a previous conversation
        is_follow_up = self._is_follow_up(user_message, intent)
        
        # Handle reference resolution (e.g., "it", "that", "same category")
        entities = self._resolve_references(entities, user_message)
        
        # Generate base response
        response = self._generate_base_response(intent, entities, db_result, is_follow_up)
        
        # Add contextual flavor
        response = self._add_contextual_flavor(response, is_follow_up)
        
        # Store this turn in context
        self.context.add_turn(user_message, response, intent, entities)
        
        return response
    
    def _is_follow_up(self, user_message: str, current_intent: str) -> bool:
        """
        Determine if current message is a follow-up to previous conversation.
        
        Args:
            user_message: Current user message
            current_intent: Current intent
            
        Returns:
            True if this is a follow-up message
        """
        # Check for follow-up indicators
        follow_up_words = ['also', 'another', 'and', 'too', 'same', 'again']
        message_lower = user_message.lower()
        
        if any(word in message_lower for word in follow_up_words):
            return True
        
        # Check if same intent as last turn
        last_intent = self.context.get_last_intent()
        if last_intent and current_intent == last_intent:
            return True
        
        return False
    
    def _resolve_references(self, entities: Dict, user_message: str) -> Dict:
        """
        Resolve references like "it", "that", "same category".
        
        Args:
            entities: Extracted entities
            user_message: User message
            
        Returns:
            Updated entities with resolved references
        """
        message_lower = user_message.lower()
        
        # If no category specified but message mentions "same" or "that"
        if not entities.get('category') or entities.get('category') == 'Misc':
            if any(word in message_lower for word in ['same', 'that', 'it']):
                last_category = self.context.get_last_category()
                if last_category:
                    entities['category'] = last_category
                    logger.debug(f"Resolved category reference to: {last_category}")
        
        # Handle pronoun references
        if 'another' in message_lower or 'more' in message_lower:
            # User likely referring to same type of transaction
            if not entities.get('category'):
                last_category = self.context.get_last_category()
                if last_category:
                    entities['category'] = last_category
        
        return entities
    
    def _generate_base_response(self, intent: str, entities: Dict,
                               db_result: Optional[Dict], is_follow_up: bool) -> str:
        """
        Generate base response based on intent and entities.
        
        Args:
            intent: Detected intent
            entities: Extracted entities
            db_result: Database result
            is_follow_up: Whether this is a follow-up
            
        Returns:
            Base response string
        """
        if intent == 'add_income':
            template_key = 'follow_up_income' if is_follow_up else 'add_income'
            if is_follow_up and template_key in self.context_templates:
                templates = self.context_templates[template_key]
            else:
                templates = [
                    "Great! I've recorded your income of ${amount:.2f}.",
                    "Income of ${amount:.2f} has been added successfully.",
                    "Got it! ${amount:.2f} added to your income.",
                ]
            return random.choice(templates).format(**entities)
        
        elif intent == 'add_expense':
            template_key = 'follow_up_expense' if is_follow_up else 'add_expense'
            if is_follow_up and template_key in self.context_templates:
                templates = self.context_templates[template_key]
            else:
                templates = [
                    "I've recorded your expense of ${amount:.2f} under {category}.",
                    "Expense of ${amount:.2f} in {category} has been logged.",
                    "Got it! ${amount:.2f} expense in {category} recorded.",
                ]
            return random.choice(templates).format(**entities)
        
        elif intent == 'query_income':
            if db_result and 'total' in db_result:
                return f"Your total income is ${db_result['total']:.2f}."
            return "Let me check your income total."
        
        elif intent == 'query_expenses':
            if db_result and 'total' in db_result:
                return f"Your total expenses are ${db_result['total']:.2f}."
            return "Let me check your expense total."
        
        elif intent == 'top_spend_category':
            if db_result and db_result.get('categories'):
                cats = db_result['categories']
                if len(cats) > 0:
                    top = cats[0]
                    response = f"Your top spending category is '{top[0]}' at ${top[1]:.2f}."
                    if len(cats) > 1:
                        others = ", ".join([f"{c} ${v:.2f}" for c, v in cats[1:3]])
                        response += f" Other top categories include: {others}."
                    return response
            return "I don't have enough expense data yet."
        
        elif intent == 'least_spend_category':
            if db_result and db_result.get('categories'):
                cats = db_result['categories']
                if len(cats) > 0:
                    least = cats[0]
                    return f"Your least spending category is '{least[0]}' at ${least[1]:.2f}."
            return "I don't have enough expense data yet."
        
        elif intent == 'avg_expense_per_txn':
            if db_result and 'average' in db_result:
                return f"Your average expense per transaction is ${db_result['average']:.2f}."
            return "I don't have enough expense data yet."
        
        elif intent == 'greeting':
            return "Hello! How can I help you with your finances today?"
        
        elif intent == 'help':
            return ("I can help you track income and expenses, show your spending patterns, "
                   "and analyze your finances. Try saying 'I spent $50 on groceries' or "
                   "'What's my total income?'")
        
        return "I understand. How else can I help you?"
    
    def _add_contextual_flavor(self, response: str, is_follow_up: bool) -> str:
        """
        Add contextual phrases to make response more natural.
        
        Args:
            response: Base response
            is_follow_up: Whether this is a follow-up
            
        Returns:
            Enhanced response
        """
        # Don't modify certain responses
        if response.startswith(('Hello', 'I can help', 'I understand')):
            return response
        
        # Add follow-up acknowledgment
        if is_follow_up:
            confirmations = self.contextual_phrases['confirmation']
            response = f"{random.choice(confirmations)}! {response}"
        
        return response
    
    def get_context_summary(self) -> str:
        """
        Get a summary of current conversation context.
        
        Returns:
            Context summary string
        """
        recent = self.context.get_recent_history(3)
        
        if not recent:
            return "No conversation history yet."
        
        summary = "Recent conversation:\n"
        for i, turn in enumerate(recent, 1):
            summary += f"{i}. User: {turn['user_message'][:50]}...\n"
            summary += f"   Bot: {turn['bot_response'][:50]}...\n"
        
        return summary
    
    def clear_context(self):
        """Clear conversation context."""
        self.context.clear()
        logger.info("Response generator context cleared")
