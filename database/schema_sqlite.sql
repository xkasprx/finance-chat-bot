-- schema_sqlite.sql
-- SQLite bootstrap for Personal Finance Chatbot

-- Transactions
CREATE TABLE IF NOT EXISTS transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT NOT NULL,
    amount REAL NOT NULL,
    category TEXT,
    occurred_at TEXT,
    description TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Categories
CREATE TABLE IF NOT EXISTS categories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    kind TEXT DEFAULT 'expense',
    parent_id INTEGER,
    FOREIGN KEY(parent_id) REFERENCES categories(id)
);

-- Settings
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT,
    type TEXT
);

-- User profile (single row, id=1)
CREATE TABLE IF NOT EXISTS user_profile (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    name TEXT,
    currency TEXT,
    timezone TEXT,
    locale TEXT
);

-- FAQs / knowledge base
CREATE TABLE IF NOT EXISTS faqs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT UNIQUE NOT NULL,
    answer TEXT NOT NULL,
    category TEXT,
    keywords TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Chat history
CREATE TABLE IF NOT EXISTS chat_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_query TEXT NOT NULL,
    bot_response TEXT,
    intent TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_transactions_type ON transactions(type);
CREATE INDEX IF NOT EXISTS idx_transactions_category_text ON transactions(category);
CREATE INDEX IF NOT EXISTS idx_transactions_occurred_at ON transactions(occurred_at);
CREATE INDEX IF NOT EXISTS idx_faqs_question ON faqs(question);
CREATE INDEX IF NOT EXISTS idx_chat_history_ts ON chat_history(timestamp);

-- Seeds
INSERT OR IGNORE INTO categories (name, kind) VALUES
    ('Income','income'),
    ('Salary','income'),
    ('Bonus','income'),
    ('Dining','expense'),
    ('Groceries','expense'),
    ('Rent','expense'),
    ('Utilities','expense'),
    ('Transport','expense'),
    ('Misc','expense');

INSERT OR IGNORE INTO settings (key, value, type) VALUES ('first_run_completed','false','bool');

INSERT OR IGNORE INTO faqs (question, answer, category, keywords) VALUES
    ('How do I add an expense?','Say something like ''I spent 25 on lunch'' and I will record it.','help','add expense record spend'),
    ('How do I see my total expenses?','Ask ''What are my total expenses?'' and I will sum them.','help','total expenses sum');
