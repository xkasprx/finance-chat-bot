"""
Machine Learning Dataset Module
Provides comprehensive training data for the chatbot including:
- User messages and chatbot responses
- Intent classifications
- Context-aware conversation pairs
- Financial transaction examples
"""

import json
import os
import random
from typing import List, Tuple, Dict
from .logger import get_logger

logger = get_logger(__name__)


class ChatbotDataset:
    """
    Manages the chatbot's training dataset with conversation pairs,
    intents, and contextual responses.
    """
    
    def __init__(self):
        """Initialize the dataset with comprehensive training examples."""
        logger.info("Initializing ChatbotDataset")
        
        # Dataset storage
        self.conversation_pairs = []
        self.intent_examples = {}
        self.context_examples = []
        
        # Generate dataset
        self._generate_dataset()
        logger.info(f"Dataset created with {len(self.conversation_pairs)} conversation pairs")
    
    def _generate_dataset(self):
        """Generate comprehensive training data for all chatbot capabilities."""
        
        # 1. Add Income Intent Examples
        self.intent_examples['add_income'] = [
            ("I earned $500 today", "add_income", {"amount": 500.0, "category": "Income"}),
            ("I got paid $2000", "add_income", {"amount": 2000.0, "category": "Income"}),
            ("My salary of $3500 was deposited", "add_income", {"amount": 3500.0, "category": "salary"}),
            ("Received a bonus of $300", "add_income", {"amount": 300.0, "category": "bonus"}),
            ("I made 100 dollars from tips", "add_income", {"amount": 100.0, "category": "Income"}),
            ("Paycheck came in for $2800", "add_income", {"amount": 2800.0, "category": "salary"}),
            ("Got a refund of $50", "add_income", {"amount": 50.0, "category": "refund"}),
            ("I received a tip of $20", "add_income", {"amount": 20.0, "category": "Income"}),
            ("Freelance payment of $450 received", "add_income", {"amount": 450.0, "category": "Income"}),
            ("Bonus deposit $1000", "add_income", {"amount": 1000.0, "category": "bonus"}),
        ]
        
        # 2. Add Expense Intent Examples
        self.intent_examples['add_expense'] = [
            ("I spent $25 on lunch", "add_expense", {"amount": 25.0, "category": "dining"}),
            ("I bought groceries for $80", "add_expense", {"amount": 80.0, "category": "groceries"}),
            ("Paid rent $1200", "add_expense", {"amount": 1200.0, "category": "rent"}),
            ("Purchased a new phone for $900", "add_expense", {"amount": 900.0, "category": "Misc"}),
            ("I paid the electric bill $120", "add_expense", {"amount": 120.0, "category": "utilities"}),
            ("I bought gas for $40", "add_expense", {"amount": 40.0, "category": "auto"}),
            ("Purchased movie tickets $30", "add_expense", {"amount": 30.0, "category": "entertainment"}),
            ("I spent 15 on coffee", "add_expense", {"amount": 15.0, "category": "dining"}),
            ("Bought dinner at restaurant $65", "add_expense", {"amount": 65.0, "category": "dining"}),
            ("Paid for Uber ride $18", "add_expense", {"amount": 18.0, "category": "transportation"}),
            ("Internet bill $75", "add_expense", {"amount": 75.0, "category": "utilities"}),
            ("Purchased groceries at the supermarket $95", "add_expense", {"amount": 95.0, "category": "groceries"}),
        ]
        
        # 3. Query Income Intent Examples
        self.intent_examples['query_income'] = [
            ("What's my total income?", "query_income", {}),
            ("How much have I earned?", "query_income", {}),
            ("Show my income total", "query_income", {}),
            ("Total money I made", "query_income", {}),
            ("How much income do I have?", "query_income", {}),
            ("What's my total earnings?", "query_income", {}),
            ("Display my income", "query_income", {}),
        ]
        
        # 4. Query Expenses Intent Examples
        self.intent_examples['query_expenses'] = [
            ("What's my total expenses?", "query_expenses", {}),
            ("How much have I spent?", "query_expenses", {}),
            ("Show my spending total", "query_expenses", {}),
            ("Total expenses so far", "query_expenses", {}),
            ("How much did I spend?", "query_expenses", {}),
            ("What are my expenses?", "query_expenses", {}),
            ("Display my total spending", "query_expenses", {}),
        ]
        
        # 5. Top Spend Category Intent Examples
        self.intent_examples['top_spend_category'] = [
            ("What do I spend the most money on?", "top_spend_category", {}),
            ("Where do I spend the most?", "top_spend_category", {}),
            ("Show my top expense category", "top_spend_category", {}),
            ("What is my biggest expense category?", "top_spend_category", {}),
            ("Top category for my spending", "top_spend_category", {}),
            ("Which category costs me the most?", "top_spend_category", {}),
        ]
        
        # 6. Least Spend Category Intent Examples
        self.intent_examples['least_spend_category'] = [
            ("Where do I spend the least?", "least_spend_category", {}),
            ("What is my smallest expense category?", "least_spend_category", {}),
            ("Show my least expense category", "least_spend_category", {}),
            ("Where is my lowest spend?", "least_spend_category", {}),
            ("Which category do I spend the least on?", "least_spend_category", {}),
        ]
        
        # 7. Average Expense Intent Examples
        self.intent_examples['avg_expense_per_txn'] = [
            ("What is my average spending per transaction?", "avg_expense_per_txn", {}),
            ("Show my average expense per transaction", "avg_expense_per_txn", {}),
            ("What's my mean spend per purchase?", "avg_expense_per_txn", {}),
            ("Average spend per transaction", "avg_expense_per_txn", {}),
            ("On average how much do I spend each time?", "avg_expense_per_txn", {}),
        ]
        
        # 8. Greeting and Conversation Examples
        self.intent_examples['greeting'] = [
            ("Hello", "greeting", {}),
            ("Hi", "greeting", {}),
            ("Hey there", "greeting", {}),
            ("Good morning", "greeting", {}),
            ("Good evening", "greeting", {}),
        ]
        
        # 9. Help Intent Examples
        self.intent_examples['help'] = [
            ("What can you do?", "help", {}),
            ("Help me", "help", {}),
            ("What are your features?", "help", {}),
            ("How do I use you?", "help", {}),
        ]
        
        # Build conversation pairs from intent examples
        self._build_conversation_pairs()
        
        # Add context-aware conversation examples
        self._build_context_examples()
    
    def _build_conversation_pairs(self):
        """Build conversation pairs with appropriate responses."""
        
        # Response templates for each intent
        responses = {
            'add_income': [
                "Great! I've recorded your income of ${amount:.2f}.",
                "Income of ${amount:.2f} has been added successfully.",
                "Got it! ${amount:.2f} added to your income.",
            ],
            'add_expense': [
                "I've recorded your expense of ${amount:.2f} under {category}.",
                "Expense of ${amount:.2f} in {category} has been logged.",
                "Got it! ${amount:.2f} expense recorded.",
            ],
            'query_income': [
                "Your total income is ${total:.2f}.",
                "You've earned ${total:.2f} in total.",
            ],
            'query_expenses': [
                "Your total expenses are ${total:.2f}.",
                "You've spent ${total:.2f} so far.",
            ],
            'top_spend_category': [
                "Your top spending category is {category} with ${amount:.2f}.",
                "You spend the most on {category}, totaling ${amount:.2f}.",
            ],
            'least_spend_category': [
                "Your lowest spending category is {category} with ${amount:.2f}.",
                "You spend the least on {category}, totaling ${amount:.2f}.",
            ],
            'avg_expense_per_txn': [
                "Your average expense per transaction is ${avg:.2f}.",
                "On average, you spend ${avg:.2f} per transaction.",
            ],
            'greeting': [
                "Hello! How can I help you with your finances today?",
                "Hi there! What would you like to know about your finances?",
                "Hey! I'm here to help with your financial tracking.",
            ],
            'help': [
                "I can help you track income and expenses, show your spending patterns, and analyze your finances. Try saying 'I spent $50 on groceries' or 'What's my total income?'",
                "You can ask me to record income or expenses, check your totals, or analyze your spending habits. Just tell me what you need!",
            ],
        }
        
        # Create conversation pairs
        for intent, examples in self.intent_examples.items():
            if intent in responses:
                for user_msg, intent_label, entities in examples:
                    # Pick a random response template
                    response = random.choice(responses[intent])
                    
                    # Add to conversation pairs
                    self.conversation_pairs.append({
                        'user_message': user_msg,
                        'intent': intent_label,
                        'entities': entities,
                        'bot_response': response
                    })
    
    def _build_context_examples(self):
        """Build context-aware conversation examples."""
        
        # Multi-turn conversations showing context awareness
        self.context_examples = [
            {
                'conversation': [
                    {'user': 'I spent $50 on groceries', 'bot': "I've recorded your expense of $50.00 under groceries."},
                    {'user': 'And another $30 at the market', 'bot': "Got it! Another $30.00 expense in groceries recorded."},
                    {'user': 'How much did I spend on groceries?', 'bot': "You've spent $80.00 on groceries."}
                ]
            },
            {
                'conversation': [
                    {'user': 'What do I spend the most on?', 'bot': "Your top spending category is rent with $1200.00."},
                    {'user': 'And the least?', 'bot': "Your lowest spending category is entertainment with $30.00."}
                ]
            },
            {
                'conversation': [
                    {'user': 'I got paid $2000', 'bot': "Great! I've recorded your income of $2000.00."},
                    {'user': "What's my income now?", 'bot': "Your total income is $2000.00."},
                    {'user': 'I spent $100 on utilities', 'bot': "I've recorded your expense of $100.00 under utilities."},
                    {'user': "What's my balance?", 'bot': "Your total income is $2000.00 and expenses are $100.00, giving you a balance of $1900.00."}
                ]
            }
        ]
    
    def get_all_conversation_pairs(self) -> List[Dict]:
        """Get all conversation pairs for training."""
        return self.conversation_pairs
    
    def get_intent_training_data(self) -> Tuple[List[str], List[str]]:
        """
        Get training data formatted for intent classification.
        
        Returns:
            Tuple of (texts, labels) for training
        """
        texts = []
        labels = []
        
        for intent, examples in self.intent_examples.items():
            for user_msg, intent_label, _ in examples:
                texts.append(user_msg)
                labels.append(intent_label)
        
        return texts, labels
    
    def get_context_examples(self) -> List[Dict]:
        """Get context-aware conversation examples."""
        return self.context_examples
    
    def save_to_file(self, filepath: str):
        """
        Save the dataset to a JSON file for future use.
        
        Args:
            filepath: Path to save the dataset
        """
        try:
            dataset = {
                'conversation_pairs': self.conversation_pairs,
                'intent_examples': {
                    intent: [(msg, lbl, ent) for msg, lbl, ent in examples]
                    for intent, examples in self.intent_examples.items()
                },
                'context_examples': self.context_examples
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Dataset saved to {filepath}")
        except Exception as e:
            logger.exception(f"Failed to save dataset: {e}")
            raise
    
    def load_from_file(self, filepath: str):
        """
        Load dataset from a JSON file.
        
        Args:
            filepath: Path to load the dataset from
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
            
            self.conversation_pairs = dataset.get('conversation_pairs', [])
            self.intent_examples = {
                intent: [(msg, lbl, ent) for msg, lbl, ent in examples]
                for intent, examples in dataset.get('intent_examples', {}).items()
            }
            self.context_examples = dataset.get('context_examples', [])
            
            logger.info(f"Dataset loaded from {filepath}")
        except Exception as e:
            logger.exception(f"Failed to load dataset: {e}")
            raise
