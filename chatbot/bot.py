from typing import Optional

from .db_manager import DBManager
from .nlp import NLPProcessor
from .logger import get_logger, log_call

# Import context-aware response generator if available
try:
    from .context import ContextAwareResponseGenerator
    CONTEXT_AWARE_AVAILABLE = True
except ImportError:
    CONTEXT_AWARE_AVAILABLE = False

logger = get_logger(__name__)

class Chatbot:
    @log_call()
    def __init__(self):
        logger.info("Initializing Chatbot with enhanced features")
        self.db = DBManager()
        self.nlp = NLPProcessor()
        
        # Initialize context-aware response generator if available
        self.context_generator = None
        if CONTEXT_AWARE_AVAILABLE:
            try:
                self.context_generator = ContextAwareResponseGenerator()
                logger.info("Context-aware response generator initialized")
            except Exception as e:
                logger.warning(f"Could not initialize context generator: {e}")
                self.context_generator = None
        
        logger.info("Chatbot initialized successfully")

    @log_call()
    def start(self):
        # First-run setup
        self._ensure_first_run_setup()

        print("=" * 60)
        print("Personal Finance Chatbot - Enhanced with Advanced NLP")
        print("=" * 60)
        print("\nI can help you track income and expenses with natural language!")
        print("Features:")
        print("  • Track income and expenses")
        print("  • Analyze spending patterns")
        print("  • Context-aware conversations")
        print("  • Natural language understanding")
        print("\nType 'exit' to quit, 'help' for assistance.\n")
        
        while True:
            try:
                user_input = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break
                
            if user_input.lower() in ["exit", "quit"]:
                logger.info("Exiting chatbot")
                print("Goodbye! Have a great day!")
                break

            if not user_input:
                continue

            # Handle profile/settings commands before NLP
            if self._handle_command(user_input):
                continue

            try:
                logger.debug(f"Received input: {user_input}")
                
                # Extract entities and classify intent
                entities = self.nlp.extract_entities(user_input)
                intent = self.nlp.classify_intent(user_input)
                logger.info(f"Intent={intent} Entities={entities}")
            except Exception as e:
                logger.exception(f"NLP processing failed: {e}")
                print("Sorry, I didn't understand that.")
                continue

            try:
                # Process intent and generate response
                response = self._process_intent(intent, entities, user_input)
                print(response)
                try:
                    self.db.log_interaction(user_query=user_input, bot_response=response, intent=intent)
                except Exception as e:
                    logger.warning(f"Could not log interaction: {e}")
                
            except Exception as e:
                logger.exception(f"Intent processing failed: {e}")
                print("Sorry, something went wrong while processing your request.")
    
    def _process_intent(self, intent, entities, user_input):
        """
        Process the intent and generate appropriate response.
        Handles database operations and response generation.
        """
        db_result = None
        response = ""
        
        try:
            if intent == "unknown":
                faq_hit = self._answer_from_faq(user_input)
                if faq_hit:
                    return faq_hit
                return "I don't have an answer for that yet, but I'll keep learning."

            if intent == "add_income":
                if entities["amount"] <= 0:
                    return "Please include a positive amount for income (e.g., 'I got paid $2000')."
                self.db.add_income(entities["amount"], entities.get("category", "Income"))
                db_result = {"amount": entities["amount"], "category": entities.get("category", "Income")}
                
            elif intent == "add_expense":
                if entities["amount"] <= 0:
                    return "Please include a positive amount for the expense (e.g., 'I spent $25 on lunch')."
                self.db.add_expense(entities["amount"], entities.get("category", "Misc"))
                db_result = {"amount": entities["amount"], "category": entities.get("category", "Misc")}
                
            elif intent == "query_expenses":
                total = self.db.get_total_expenses()
                db_result = {"total": total}
                
            elif intent == "query_income":
                total = self.db.get_total_income()
                db_result = {"total": total}
                
            elif intent == "top_spend_category":
                top = self.db.get_top_expense_categories(limit=3)
                db_result = {"categories": top}
                
            elif intent == "least_spend_category":
                least = self.db.get_least_expense_categories(limit=3)
                db_result = {"categories": least}
                
            elif intent == "avg_expense_per_txn":
                avg = self.db.get_avg_expense_per_txn()
                db_result = {"average": avg}
            
            # Generate context-aware response if available
            if self.context_generator:
                response = self.context_generator.generate_response(
                    intent, entities, user_input, db_result
                )
            else:
                # Fall back to simple responses
                response = self._generate_simple_response(intent, entities, db_result)
                
        except Exception as e:
            logger.exception(f"DB operation failed: {e}")
            response = "Sorry, something went wrong while processing your request."
        
        return response

    def _answer_from_faq(self, user_input: str) -> Optional[str]:
        hits = self.db.search_faq(user_input, limit=1)
        if hits:
            return hits[0].get("answer") or ""
        return None

    def _handle_command(self, user_input: str) -> bool:
        """Handle explicit profile/settings commands. Returns True if handled."""
        text = user_input.strip()
        lower = text.lower()

        if lower in {"profile", "show profile", "show settings"}:
            profile = self.db.get_user_profile()
            print("Current profile:")
            print(f"  Name: {profile.get('name') or 'Not set'}")
            print(f"  Currency: {profile.get('currency') or 'Not set'}")
            print(f"  Timezone: {profile.get('timezone') or 'Not set'}")
            print(f"  Locale: {profile.get('locale') or 'Not set'}")
            return True

        if lower == "update profile":
            self._run_profile_wizard(prompt_header="Update profile")
            return True

        for field in ["name", "currency", "timezone", "locale"]:
            prefix = f"set {field} "
            if lower.startswith(prefix):
                value = text[len(prefix):].strip()
                if not value:
                    print("Please provide a value.")
                    return True
                self._update_profile_field(field, value)
                print(f"Updated {field} to '{value}'.")
                return True

        if lower.startswith("export faqs"):
            target = text[len("export faqs"):].strip() or "faqs_export.json"
            ok = self.db.export_faqs_json(target)
            print("FAQs exported" if ok else "Export failed")
            return True

        if lower.startswith("import faqs"):
            target = text[len("import faqs"):].strip() or "faqs_export.json"
            ok = self.db.import_faqs_json(target)
            print("FAQs imported" if ok else "Import failed")
            return True

        if lower.startswith("export history"):
            target = text[len("export history"):].strip() or "chat_history.json"
            ok = self.db.export_chat_history_json(target)
            print("History exported" if ok else "Export failed")
            return True

        if lower.startswith("import history"):
            target = text[len("import history"):].strip() or "chat_history.json"
            ok = self.db.import_chat_history_json(target)
            print("History imported" if ok else "Import failed")
            return True

        return False

    def _update_profile_field(self, field: str, value: str):
        profile = self.db.get_user_profile()
        profile[field] = value
        self.db.upsert_user_profile(
            name=profile.get("name"),
            currency=profile.get("currency"),
            timezone=profile.get("timezone"),
            locale=profile.get("locale"),
        )
        # Mirror key settings for quick access
        if field in {"currency", "timezone", "locale"}:
            self.db.set_setting(field, value, "str")

    def _ensure_first_run_setup(self):
        if self.db.is_first_run_completed():
            return
        print("First-time setup: let's capture a few preferences. Press Enter to keep defaults.")
        self._run_profile_wizard(prompt_header="First-time setup")
        self.db.set_setting("first_run_completed", "true", "bool")
        print("Setup complete!\n")

    def _run_profile_wizard(self, prompt_header: str = "Profile setup"):
        profile = self.db.get_user_profile()
        print(f"{prompt_header} (leave blank to keep current/default values)")

        def ask(label: str, key: str, default_val: str = None) -> str:
            existing = profile.get(key) or default_val or ""
            prompt = f"{label} [{existing}]: " if existing else f"{label}: "
            val = input(prompt).strip()
            return val or existing or ""

        name = ask("Name", "name", "User")
        currency = ask("Currency (e.g., USD)", "currency", "USD")
        timezone = ask("Timezone", "timezone", "UTC")
        locale = ask("Locale (e.g., en_US)", "locale", "en_US")

        self.db.upsert_user_profile(name=name, currency=currency, timezone=timezone, locale=locale)
        self.db.set_setting("currency", currency, "str")
        self.db.set_setting("timezone", timezone, "str")
        self.db.set_setting("locale", locale, "str")
        print("Profile saved.")
    
    def _generate_simple_response(self, intent, entities, db_result):
        """Generate simple responses without context awareness (fallback)."""
        if intent == "add_income":
            return f"Income of ${entities['amount']:.2f} added under category '{entities.get('category', 'Income')}'."
        elif intent == "add_expense":
            return f"Expense of ${entities['amount']:.2f} added under category '{entities.get('category', 'Misc')}'."
        elif intent == "query_expenses":
            return f"Total expenses: ${db_result['total']:.2f}"
        elif intent == "query_income":
            return f"Total income: ${db_result['total']:.2f}"
        elif intent == "top_spend_category":
            top = db_result.get('categories', [])
            if not top:
                return "I don't have any expenses recorded yet."
            head = top[0]
            if len(top) == 1:
                return f"Your top spending category is '{head[0]}' at ${head[1]:.2f}."
            else:
                others = ", ".join([f"{c} ${v:.2f}" for c, v in top[1:]])
                return f"Top categories: {head[0]} ${head[1]:.2f}; then {others}."
        elif intent == "least_spend_category":
            least = db_result.get('categories', [])
            if not least:
                return "I don't have any expenses recorded yet."
            head = least[0]
            if len(least) == 1:
                return f"Your least spending category is '{head[0]}' at ${head[1]:.2f}."
            else:
                others = ", ".join([f"{c} ${v:.2f}" for c, v in least[1:]])
                return f"Lowest categories: {head[0]} ${head[1]:.2f}; then {others}."
        elif intent == "avg_expense_per_txn":
            avg = db_result.get('average', 0)
            if avg <= 0:
                return "I don't have any expenses recorded yet."
            else:
                return f"Average expense per transaction: ${avg:.2f}"
        else:
            return "Sorry, I didn't understand that."
