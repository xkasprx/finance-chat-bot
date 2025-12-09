# Personal Finance Chatbot

Modern personal finance assistant with web UI that records income/expenses, tracks spending, and answers FAQs using a local SQLite database. Features include profile setup, chat history, FAQ knowledge base, and JSON import/export.

## Features
- **Web UI**: Modern browser-based chat interface with gradient design
- **Console UI**: Traditional command-line interface (optional)
- Natural-language expense and income tracking with ML intent classification
- SQLite database with transactions, categories, settings, user_profile, FAQs, and chat history
- FAQ knowledge base with smart search fallback
- Profile wizard for personalized greetings
- JSON import/export for FAQs and chat history

## Requirements
- Python 3.10+
- OS: Windows/macOS/Linux
- Modern web browser (for web UI)

## Quick Start

### 1. Create Virtual Environment
**Windows PowerShell:**
```powershell
python -m venv chatbot_env
.\chatbot_env\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
python3 -m venv chatbot_env
source chatbot_env/bin/activate
```

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 3. Run the Application

**Web UI (Recommended):**
```powershell
python app.py
```
Browser opens automatically to `http://localhost:5000`

**Console UI:**
```powershell
python main.py
```

On first launch, complete the profile setup wizard. A SQLite DB (`chatbot.db`) is created with starter categories and sample FAQs.

## Usage Examples

### Web UI
- Open browser to `http://localhost:5000`
- Chat naturally with the bot
- Personalized greeting with your name
- Modern, responsive interface

### Commands (Both UIs)
- **Record expense**: `I spent $50 on groceries`
- **Record income**: `I got paid 2000`
- **Query totals**: `How much did I spend?` / `What's my total income?`
- **Top/least category**: `Where do I spend the most?`
- **Average**: `What is my average spending per transaction?`
- **Profile**: `profile` (console only - web UI has API endpoint)
- **Export FAQs**: `export faqs faqs.json`
- **Import FAQs**: `import faqs faqs.json`
- **Export history**: `export history history.json`
- **Import history**: `import history history.json`

If the bot cannot classify an input, it searches the FAQ knowledge base.

## Project Structure
```
├── app.py                      # Flask web UI (main entry)
├── main.py                     # Console UI (alternative)
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── chatbot/                    # Core package
│   ├── bot.py                  # Conversation logic
│   ├── db_manager.py           # SQLite database layer
│   ├── nlp.py                  # NLP & intent classification
│   ├── logger.py               # Logging configuration
│   ├── transaction.py          # Transaction models
│   ├── advanced_nlp.py         # Sentiment & NER
│   ├── context.py              # Context-aware responses
│   ├── ml_models.py            # LSTM models
│   ├── preprocessing.py        # Text preprocessing
│   ├── dataset.py              # Training data
│   └── embeddings.py           # Word embeddings
├── templates/
│   └── chat.html               # Web UI template
├── database/
│   └── schema_sqlite.sql       # SQLite schema reference
├── models/
│   ├── tf_intent_classifier/   # TensorFlow model
│   ├── lstm_intent_classifier/ # LSTM model (optional)
│   └── finetuned_model/        # Fine-tuned model (optional)
├── scripts/
│   ├── make_dist.ps1           # Build distribution package
│   └── train_models.py         # Training pipeline (optional)
└── logs/                       # Application logs
```

## Configuration

Environment variables (optional):
- `CHATBOT_SQLITE_PATH`: Database file path (default: `chatbot.db`)
- `CHATBOT_LOG_LEVEL`: `WARNING`, `INFO`, `DEBUG` (default: `WARNING`)
- `CHATBOT_LOG_DIR`: Log directory (default: `logs`)

## Distribution

Build a distribution package:
```powershell
.\scripts\make_dist.ps1
```
Creates `Chatbot-dist.zip` with all necessary files (excludes venv, logs, cache).

## Troubleshooting

**Venv activation issues (Windows):**
- Use `.bat` version: `.\chatbot_env\Scripts\activate.bat`
- Or run directly: `.\chatbot_env\Scripts\python.exe app.py`

**Database issues:**
- Delete `chatbot.db` and restart to recreate with fresh schema

**Web UI not opening:**
- Manually visit `http://localhost:5000` in your browser
- Check firewall settings

**Missing dependencies:**
- Ensure venv is activated before running `pip install`
- Reinstall: `pip install -r requirements.txt --force-reinstall`

## Technologies
- **Backend**: Python, Flask, SQLite
- **ML/NLP**: TensorFlow, NLTK, scikit-learn
- **Frontend**: HTML5, CSS3, JavaScript (vanilla)
- **Data**: JSON import/export

## License
Educational project for ITEC-5025, Capella University (Fall 2025)

### Environment Variables
Set before running `python main.py`:

#### Logging
- `CHATBOT_LOG_LEVEL`: `CRITICAL` (default), `ERROR`, `WARNING`, `INFO`, or `DEBUG`
- `CHATBOT_LOG_DIR`: Log directory path (default: `logs`)
- `CHATBOT_LOG_FILE`: Full log file path (default: `logs/chatbot.log`)

#### Database - SQLite
- `CHATBOT_DB_TYPE`: `sqlite` (default) or `mysql`
- `CHATBOT_SQLITE_PATH`: Database file path (default: `chatbot.db`)

#### Database - MySQL
- `CHATBOT_MYSQL_HOST`: Host address (default: `localhost`)
- `CHATBOT_MYSQL_PORT`: Port number (default: `3306`)
- `CHATBOT_MYSQL_USER`: Username (default: `chatbot_user`)
- `CHATBOT_MYSQL_PASSWORD`: Password (default: `ChatbotStrongPass!2025`)
- `CHATBOT_MYSQL_DB`: Database name (default: `finance_chatbot`)

#### TensorFlow Model (Advanced)
- `CHATBOT_INTENT_MODEL_DIR`: Model directory (default: `models/tf_intent_classifier`)
- `CHATBOT_TF_MAX_TOKENS`: Vocabulary size (default: `10000`)
- `CHATBOT_TF_SEQ_LEN`: Max sequence length (default: `24`)
- `CHATBOT_TF_EMBED_DIM`: Embedding dimension (default: `64`)
- `CHATBOT_TF_BATCH`: Training batch size (default: `32`)
- `CHATBOT_TF_EPOCHS`: Training epochs (default: `5`)

## Usage

### Quick Start with SQLite (No Setup Required)
```powershell
python main.py
```

### MySQL Setup (Optional)

#### Step 1: Import Database Schema
**Ensure MySQL server is running**, then import:
```powershell
mysql -u root -p < database/schema_mysql.sql
```

This creates:
- Database: `finance_chatbot`
- User: `chatbot_user` with password `ChatbotStrongPass!2025`
- Table: `transactions` with indexes

**Security Note:** Change the password in `schema_mysql.sql` before importing in production.

#### Step 2: Run with MySQL
```powershell
$env:CHATBOT_DB_TYPE = "mysql"
$env:CHATBOT_MYSQL_PASSWORD = "ChatbotStrongPass!2025"
python main.py
```

### Reducing Log Verbosity
For cleaner output (recommended for normal use):
```powershell
$env:CHATBOT_LOG_LEVEL = "CRITICAL"
python main.py
```

## Interactive Commands

### Example Conversations

**Track Expenses:**
```
> I spent $25 on lunch
Expense of $25.00 added under category 'dining'.

> I bought groceries for $80
Expense of $80.00 added under category 'groceries'.

> Paid rent $1200
Expense of $1200.00 added under category 'rent'.
```

**Track Income:**
```
> I got paid $2000
Income of $2000.00 added under category 'Income'.

> Received a bonus of $300
Income of $300.00 added under category 'Income'.
```

**Query Totals:**
```
> How much have I spent?
Total expenses: $1305.00

> What's my total income?
Total income: $2300.00
```

**Analyze Spending:**
```
> What do I spend the most money on?
Top categories: rent $1200.00; then groceries $80.00, dining $25.00.

> Where do I spend the least?
Lowest categories: dining $25.00; then groceries $80.00, rent $1200.00.

> What is my average spending per transaction?
Average expense per transaction: $435.00
```

**Exit:**
```
> exit
[Chatbot closes]
```

## Supported Intents

| Intent | Example Phrases |
|--------|----------------|
| **Add Expense** | "I spent $X on Y", "I bought X", "Paid X for Y" |
| **Add Income** | "I got paid $X", "Received $X", "Earned $X" |
| **Query Expenses** | "How much have I spent?", "Total expenses?" |
| **Query Income** | "What's my total income?", "How much did I earn?" |
| **Top Spending** | "What do I spend the most on?", "Biggest expense category?" |
| **Least Spending** | "Where do I spend the least?", "Smallest expense category?" |
| **Average Spending** | "Average spending per transaction?", "Mean expense?" |

## Auto-Detected Categories

The chatbot automatically categorizes transactions based on keywords:

- **groceries**: grocery, supermarket, market
- **rent**: rent, landlord
- **utilities**: electric, water, gas bill, internet, wifi
- **entertainment**: movie, cinema, netflix, game
- **dining**: dinner, lunch, breakfast, restaurant, cafe
- **transportation**: uber, lyft, bus, train, ticket
- **auto**: car, fuel, gasoline, gas station, repair
- **income**: salary, paycheck, bonus, deposit, refund
- **Misc**: Default for unrecognized categories

## Machine Learning Model

### TensorFlow Intent Classifier
The chatbot uses a custom TensorFlow model to classify user intents:
- **Architecture**: TextVectorization → Embedding → GlobalAveragePooling → Dense(ReLU) → Softmax
- **Training**: Automatic on first run using synthetic examples (~50 phrases)
- **Persistence**: Saved to `models/tf_intent_classifier/intent_classifier.keras`
- **Auto-retrain**: Triggered when intent labels change (model output mismatch)

### Model Behavior
- **First Launch**: Trains model (~30 seconds), then ready for use
- **Subsequent Launches**: Loads pre-trained model instantly
- **Fallback**: Rule-based classification if model loading fails

## Troubleshooting

### "Module not found" errors
```powershell
# Ensure virtual environment is activated
.\chatbot_env\Scripts\Activate.ps1

# Reinstall dependencies
pip install -r requirements.txt
```

### MySQL connection fails
- Verify MySQL server is running: `mysql -u root -p`
- Check credentials match `schema_mysql.sql` or your env vars
- Ensure `finance_chatbot` database exists
- Falls back to SQLite automatically if connection fails

### TensorFlow warnings about CPU/GPU
Safe to ignore - the model runs efficiently on CPU.

### Model retrains every time
- Check file permissions on `models/tf_intent_classifier/`
- Ensure disk space available (model file ~5MB)

### "Permission denied" on log files
- Run from a directory with write permissions
- Or customize log path: `$env:CHATBOT_LOG_FILE = "C:\logs\chatbot.log"`

## Project Structure

```
├── main.py                          # Application entry point
├── requirements.txt                 # Python dependencies
├── README.md                        # Documentation (this file)
├── chatbot/                         # Main package
│   ├── __init__.py
│   ├── bot.py                       # Conversation loop & intent routing
│   ├── db_manager.py                # Database abstraction (SQLite/MySQL)
│   ├── logger.py                    # Centralized logging
│   ├── nlp.py                       # Intent classification & entity extraction
│   └── transaction.py               # Transaction utilities
├── database/
│   └── schema_mysql.sql             # MySQL bootstrap script
├── chatbot.db                       # SQLite database (auto-created)
├── logs/                            # Log files (auto-created)
│   └── chatbot.log
└── models/                          # ML models (auto-created)
    └── tf_intent_classifier/
        └── intent_classifier.keras
```

## License & Credits
Developed for educational purposes at Capella University (Fall 2025).

## Support
For issues or questions, review this README and check the `logs/chatbot.log` file for detailed diagnostics.

