import re
import os
import random
from .logger import get_logger, log_call
import tensorflow as tf
from tensorflow.keras import layers, models

# Import new advanced NLP components
try:
    from .advanced_nlp import AdvancedNLPProcessor
    from .ml_models import LSTMIntentClassifier
    ADVANCED_NLP_AVAILABLE = True
except ImportError as e:
    logger = get_logger(__name__)
    logger.warning(f"Advanced NLP features not available: {e}")
    ADVANCED_NLP_AVAILABLE = False

logger = get_logger(__name__)

INTENT_LABELS = [
    "add_income",
    "add_expense",
    "query_income",
    "query_expenses",
    "top_spend_category",
    "least_spend_category",
    "avg_expense_per_txn",
]


class NLPProcessor:
    @log_call()
    def __init__(self):
        logger.info("Initializing NLP processor with advanced features")
        # Paths and settings
        self.model_dir = os.getenv("CHATBOT_INTENT_MODEL_DIR", os.path.join("models", "tf_intent_classifier"))
        self.lstm_model_dir = os.path.join("models", "lstm_intent_classifier")
        os.makedirs(self.model_dir, exist_ok=True)

        # Initialize advanced NLP processor if available
        self.advanced_nlp = None
        if ADVANCED_NLP_AVAILABLE:
            try:
                self.advanced_nlp = AdvancedNLPProcessor()
                logger.info("Advanced NLP processor initialized")
            except Exception as e:
                logger.warning(f"Could not initialize advanced NLP: {e}")
                self.advanced_nlp = None

        # Try to load LSTM model first, fall back to simple TF model
        self.intent_model = None
        self.lstm_model = None
        
        try:
            # Try loading LSTM model if it exists
            if ADVANCED_NLP_AVAILABLE and os.path.exists(self.lstm_model_dir):
                try:
                    self.lstm_model = LSTMIntentClassifier()
                    self.lstm_model.load(self.lstm_model_dir)
                    logger.info("LSTM intent model loaded successfully")
                except Exception as e:
                    logger.warning(f"Could not load LSTM model: {e}")
                    self.lstm_model = None
            
            # Load or train simple TF model as fallback
            if self.lstm_model is None:
                self.intent_model = self._load_intent_model()
                if self.intent_model is None:
                    self.intent_model = self._train_intent_model()
                logger.info("Simple TensorFlow intent model ready")
        except Exception as e:
            logger.exception(f"Failed to initialize intent models: {e}")
            self.intent_model = None
            self.lstm_model = None
        
        logger.info("NLP processor ready")

    @log_call(log_result=True)
    def extract_entities(self, text):
        """
        Extract entities from text using enhanced NLP if available.
        Falls back to simple extraction if advanced NLP not available.
        """
        # Try advanced NER first
        if self.advanced_nlp:
            try:
                entities_dict = self.advanced_nlp.extract_entities(text)
                
                # Extract amount from advanced NER
                amount = 0.0
                if entities_dict.get('amounts'):
                    amount = entities_dict['amounts'][0]['value']
                
                # Extract category from advanced NER
                category = "Misc"
                if entities_dict.get('categories'):
                    category = entities_dict['categories'][0]['category']
                else:
                    # Fall back to keyword matching
                    category = self._extract_category_simple(text)
                
                return {"amount": amount, "category": category}
            except Exception as e:
                logger.warning(f"Advanced entity extraction failed: {e}, using simple method")
        
        # Simple extraction (original method)
        amount_match = re.search(r"\$?(\d+(?:\.\d{1,2})?)", text)
        amount = float(amount_match.group(1)) if amount_match else 0.0

        category = self._extract_category_simple(text)
        return {"amount": amount, "category": category}
    
    def _extract_category_simple(self, text):
        """Simple category detection based on keywords."""
        text_l = text.lower()
        category_map = {
            "groceries": ["grocery", "groceries", "supermarket", "market"],
            "rent": ["rent", "landlord"],
            "utilities": ["utilities", "electric", "water", "gas bill", "internet", "wifi"],
            "entertainment": ["movie", "cinema", "entertainment", "netflix", "game"],
            "dining": ["dinner", "lunch", "breakfast", "food", "restaurant", "cafe"],
            "transportation": ["uber", "lyft", "bus", "train", "ticket"],
            "auto": ["car", "fuel", "gasoline", "gas station", "repair"],
            "income": ["salary", "paycheck", "bonus", "deposit", "refund"]
        }
        category = "Misc"
        for cat, keys in category_map.items():
            if any(k in text_l for k in keys):
                category = cat
                break
        return category

    @log_call(log_result=True)
    def classify_intent(self, text):
        """
        Classify intent using LSTM model if available, otherwise use simple TF model or rules.
        """
        # Try LSTM model first (most accurate)
        if self.lstm_model is not None:
            try:
                predicted_intent, confidence = self.lstm_model.predict(text)
                logger.debug(f"LSTM prediction: {predicted_intent} (confidence: {confidence:.3f})")
                
                # Use LSTM prediction if confidence is high enough
                if confidence > 0.5:
                    return self._postprocess_intent(text, predicted_intent)
                else:
                    logger.debug(f"Low confidence ({confidence:.3f}), trying fallback methods")
            except Exception as e:
                logger.warning(f"LSTM prediction failed: {e}, trying fallback")
        
        # Try simple TF model
        if self.intent_model is not None:
            try:
                pred = self.intent_model.predict(tf.constant([text]), verbose=0)
                idx = int(pred.argmax(axis=-1)[0])
                predicted = INTENT_LABELS[idx]
                return self._postprocess_intent(text, predicted)
            except Exception as e:
                logger.exception(f"TF classify failed, falling back to rules: {e}")

        # Fallback to rule-based classification
        return self._classify_by_rules(text)
    
    def _classify_by_rules(self, text):
        """Rule-based intent classification (fallback method)."""
        t = text.lower()
        if ("how do i" in t) or ("how to" in t):
            # Treat procedural questions as FAQ/unknown so FAQ search can answer
            return "unknown"
        is_query = ("how much" in t) or ("total" in t)
        if is_query:
            if any(k in t for k in ["spend", "spent", "expenses", "expense"]):
                return "query_expenses"
            if any(k in t for k in ["earn", "earned", "income", "made", "salary", "paycheck", "bonus", "deposit"]):
                return "query_income"
        # Top spend phrasing
        if any(k in t for k in ["what do i spend the most", "biggest expense", "most money on", "top category", "where do i spend the most"]):
            return "top_spend_category"
        # Least spend phrasing
        if any(k in t for k in ["spend the least", "smallest expense", "least money on", "where do i spend the least", "lowest spend"]):
            return "least_spend_category"
        # Average spend phrasing
        if any(k in t for k in ["average spend", "average spending", "average expense", "mean spend", "average per transaction", "on average how much do i spend"]):
            return "avg_expense_per_txn"
        if "got paid" in t or "received" in t:
            return "add_income"
        if any(k in t for k in ["earned", "income", "made", "paycheck", "salary", "bonus", "deposit", "tip", "refund"]):
            return "add_income"
        if any(k in t for k in ["spent", "spend", "buy", "bought", "purchase", "purchased", "expense", "paid", "pay for", "paid for"]):
            return "add_expense"
        if "expenses" in t:
            return "query_expenses"
        if "income" in t:
            return "query_income"
        return "unknown"

    def _postprocess_intent(self, text: str, predicted: str) -> str:
        """Apply light heuristics to correct obvious misclassifications."""
        t = text.lower()
        if ("how do i" in t) or ("how to" in t):
            # Route procedural/how-to queries to FAQ handler
            return "unknown"
        if ("how much" in t) or ("total" in t) or ("show" in t and ("income" in t or "expenses" in t)):
            if any(k in t for k in ["spend", "spent", "expenses", "expense"]):
                return "query_expenses"
            if any(k in t for k in ["earn", "earned", "income", "made", "salary", "paycheck", "bonus", "deposit"]):
                return "query_income"
        if any(k in t for k in ["most money on", "spend the most", "biggest expense", "top category", "where do i spend the most"]):
            return "top_spend_category"
        if any(k in t for k in ["spend the least", "smallest expense", "least money on", "where do i spend the least", "lowest spend"]):
            return "least_spend_category"
        if any(k in t for k in ["average spend", "average spending", "average expense", "mean spend", "average per transaction", "on average how much do i spend"]):
            return "avg_expense_per_txn"
        return predicted

    # --------------- TF intent model ---------------
    def _load_intent_model(self):
        try:
            path = os.path.join(self.model_dir, "intent_classifier.keras")
            if os.path.exists(path):
                model = tf.keras.models.load_model(path)
                logger.info(f"Loaded intent model from {path}")
                # Ensure the loaded model output matches current labels; else force retrain
                try:
                    out_units = getattr(model.layers[-1], "units", None)
                    if out_units is None or out_units != len(INTENT_LABELS):
                        logger.info(
                            f"Intent label set changed (model units={out_units}, labels={len(INTENT_LABELS)}); will retrain."
                        )
                        return None
                except Exception as e:
                    logger.exception(f"Failed to validate model output units: {e}")
                    return None
                return model
        except Exception as e:
            logger.exception(f"Failed to load intent model: {e}")
        return None

    def _train_intent_model(self):
        texts, labels = self._generate_training_data()
        label_to_index = {l: i for i, l in enumerate(INTENT_LABELS)}
        y = tf.convert_to_tensor([label_to_index[l] for l in labels], dtype=tf.int32)

        # Shuffle and split
        rng = list(range(len(texts)))
        random.shuffle(rng)
        texts = [texts[i] for i in rng]
        y = tf.gather(y, rng)
        split = int(0.9 * len(texts))
        train_texts, val_texts = texts[:split], texts[split:]
        y_train, y_val = y[:split], y[split:]

        # Vectorization layer
        max_tokens = int(os.getenv("CHATBOT_TF_MAX_TOKENS", 10000))
        seq_len = int(os.getenv("CHATBOT_TF_SEQ_LEN", 24))
        vec = layers.TextVectorization(max_tokens=max_tokens, output_mode="int", output_sequence_length=seq_len)
        vec.adapt(train_texts)
        vocab_size = len(vec.get_vocabulary())
        logger.info(f"Vectorizer adapted: vocab_size={vocab_size}")

        # Build simple model
        embed_dim = int(os.getenv("CHATBOT_TF_EMBED_DIM", 64))
        model = models.Sequential([
            vec,
            layers.Embedding(input_dim=vocab_size + 2, output_dim=embed_dim, mask_zero=True),
            layers.GlobalAveragePooling1D(),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(len(INTENT_LABELS), activation="softmax"),
        ])
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

        batch = int(os.getenv("CHATBOT_TF_BATCH", 32))
        epochs = int(os.getenv("CHATBOT_TF_EPOCHS", 5))
        model.fit(x=tf.convert_to_tensor(train_texts), y=y_train, validation_data=(tf.convert_to_tensor(val_texts), y_val), batch_size=batch, epochs=epochs, verbose=0)
        logger.info("Intent model trained")

        # Save
        save_path = os.path.join(self.model_dir, "intent_classifier.keras")
        model.save(save_path)
        logger.info(f"Intent model saved to {save_path}")
        return model

    def _generate_training_data(self):
        # Synthetic but broad coverage phrases for each intent
        incomes = [
            "I earned $500 today", "I got paid 2000", "My salary was deposited", "Received a bonus of $300",
            "I made 100 dollars", "Paycheck came in", "Got a refund of $50", "I received a tip of $20",
        ]
        expenses = [
            "I spent $25 on lunch", "I bought groceries for $80", "Paid rent $1200", "Purchased a new phone for $900",
            "I paid the electric bill", "I bought gas for $40", "Purchased movie tickets", "I spent 15 on coffee",
        ]
        q_income = [
            "What's my total income?", "How much have I earned?", "Show my income total", "Total money I made",
            "How much income do I have?",
        ]
        q_expenses = [
            "What's my total expenses?", "How much have I spent?", "Show my spending total", "Total expenses so far",
            "How much did I spend?",
        ]
        top_spend = [
            "What do I spend the most money on?",
            "Where do I spend the most?",
            "Show my top expense category",
            "What is my biggest expense category?",
            "Top category for my spending",
        ]
        least_spend = [
            "Where do I spend the least?",
            "What is my smallest expense category?",
            "Show my least expense category",
            "Where is my lowest spend?",
            "Which category do I spend the least on?",
        ]
        avg_spend = [
            "What is my average spending per transaction?",
            "Show my average expense per transaction",
            "What's my mean spend per purchase?",
            "Average spend per transaction",
            "On average how much do I spend each time?",
        ]
        texts = incomes + expenses + q_income + q_expenses + top_spend + least_spend + avg_spend
        labels = (
            ["add_income"] * len(incomes)
            + ["add_expense"] * len(expenses)
            + ["query_income"] * len(q_income)
            + ["query_expenses"] * len(q_expenses)
            + ["top_spend_category"] * len(top_spend)
            + ["least_spend_category"] * len(least_spend)
            + ["avg_expense_per_txn"] * len(avg_spend)
        )
        return texts, labels
