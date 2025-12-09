"""
Flask web wrapper for the Personal Finance Chatbot
Provides REST API and browser-based UI without modifying core bot logic
"""

import os
import webbrowser
import threading
import time
import logging

# Set log level BEFORE importing anything else
os.environ["CHATBOT_LOG_LEVEL"] = "WARNING"

# Suppress TensorFlow and other verbose loggers to console
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("werkzeug").setLevel(logging.WARNING)

from flask import Flask, render_template, request, jsonify
from chatbot.bot import Chatbot
from chatbot.logger import get_logger

logger = get_logger(__name__)

app = Flask(__name__)
bot = None

# Global lock for thread-safe DB access
import threading
db_lock = threading.Lock()


def initialize_bot():
    """Initialize the chatbot instance (singleton pattern)"""
    global bot
    with db_lock:
        if bot is None:
            logger.info("Initializing chatbot for web interface")
            bot = Chatbot()
        return bot


@app.route('/')
def index():
    """Serve the main chat UI"""
    return render_template('chat.html')


@app.route('/api/greeting', methods=['GET'])
def greeting():
    """Get initial greeting with user profile"""
    try:
        initialize_bot()
        profile = bot.db.get_user_profile()
        user_name = profile.get('name') or 'there'
        greeting_msg = f"Hi {user_name}! 👋 I'm your personal finance assistant. How can I help you today?"
        return jsonify({'greeting': greeting_msg}), 200
    except Exception as e:
        logger.exception(f"Greeting retrieval failed: {e}")
        return jsonify({'greeting': "Hi! I'm your personal finance assistant. How can I help you today?"}), 200


@app.route('/api/chat', methods=['POST'])
def chat():
    """API endpoint to process user messages"""
    try:
        data = request.json
        user_input = data.get('message', '').strip()
        
        if not user_input:
            return jsonify({'error': 'Empty message'}), 400
        
        logger.debug(f"Received message: {user_input}")
        
        # Initialize bot if needed
        initialize_bot()
        
        # Extract entities and classify intent
        try:
            entities = bot.nlp.extract_entities(user_input)
            intent = bot.nlp.classify_intent(user_input)
            logger.debug(f"Intent={intent} Entities={entities}")
        except Exception as e:
            logger.exception(f"NLP processing failed: {e}")
            return jsonify({'response': "Sorry, I didn't understand that."}), 200
        
        # Process intent and generate response
        try:
            if user_input.lower() in ["exit", "quit"]:
                return jsonify({'response': "Goodbye! Have a great day!", 'exit': True}), 200
            
            response = bot._process_intent(intent, entities, user_input)
            
            # Ensure DB is committed after write operations
            if intent in ["add_income", "add_expense"]:
                try:
                    bot.db.conn.commit()
                except Exception as e:
                    logger.warning(f"Could not commit DB: {e}")
            
            # Log interaction to DB
            try:
                bot.db.log_interaction(user_query=user_input, bot_response=response, intent=intent)
                bot.db.conn.commit()
            except Exception as e:
                logger.warning(f"Could not log interaction: {e}")
            
            return jsonify({'response': response}), 200
            
        except Exception as e:
            logger.exception(f"Intent processing failed: {e}")
            return jsonify({'response': "Sorry, something went wrong while processing your request."}), 200
    
    except Exception as e:
        logger.exception(f"Chat endpoint error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/profile', methods=['GET'])
def get_profile():
    """Get user profile"""
    try:
        initialize_bot()
        profile = bot.db.get_user_profile()
        return jsonify(profile), 200
    except Exception as e:
        logger.exception(f"Profile retrieval failed: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/profile', methods=['POST'])
def update_profile():
    """Update user profile"""
    try:
        initialize_bot()
        data = request.json
        bot.db.upsert_user_profile(
            name=data.get('name'),
            currency=data.get('currency'),
            timezone=data.get('timezone'),
            locale=data.get('locale')
        )
        profile = bot.db.get_user_profile()
        return jsonify(profile), 200
    except Exception as e:
        logger.exception(f"Profile update failed: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/export/faqs', methods=['GET'])
def export_faqs():
    """Export FAQs as JSON"""
    try:
        initialize_bot()
        import json
        faqs = bot.db.search_faq("", limit=10000)
        return jsonify(faqs), 200
    except Exception as e:
        logger.exception(f"FAQ export failed: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/export/history', methods=['GET'])
def export_history():
    """Export chat history as JSON"""
    try:
        initialize_bot()
        import json
        history = bot.db.get_chat_history(limit=10000)
        return jsonify(history), 200
    except Exception as e:
        logger.exception(f"History export failed: {e}")
        return jsonify({'error': str(e)}), 500


def open_browser():
    """Open browser after a short delay to allow server to start"""
    time.sleep(1.5)
    webbrowser.open('http://localhost:5000')


if __name__ == '__main__':
    logger.info("Starting Personal Finance Chatbot Web Interface")
    
    # Ensure first-run setup is done before starting server
    initialize_bot()
    
    # Open browser in background thread
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()
    
    # Start Flask server
    app.run(debug=False, host='localhost', port=5000, threaded=True)
