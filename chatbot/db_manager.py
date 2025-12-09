import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from .logger import get_logger, log_call

load_dotenv()

logger = get_logger(__name__)

class DBManager:
    @log_call()
    def __init__(self):
        self.conn = None
        self.cursor = None
        self.connect()

    @log_call()
    def connect(self):
        self._connect_sqlite()
        self.create_tables()
        self._seed_defaults()

    @log_call()
    def create_tables(self):
        try:
            self.cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS transactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    type TEXT NOT NULL,
                    amount REAL NOT NULL,
                    category TEXT,
                    occurred_at TEXT,
                    description TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            self.cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS categories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    kind TEXT DEFAULT 'expense',
                    parent_id INTEGER,
                    FOREIGN KEY(parent_id) REFERENCES categories(id)
                )
                """
            )
            self.cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    type TEXT
                )
                """
            )
            self.cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS user_profile (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    name TEXT,
                    currency TEXT,
                    timezone TEXT,
                    locale TEXT
                )
                """
            )
            self.cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS faqs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT UNIQUE NOT NULL,
                    answer TEXT NOT NULL,
                    category TEXT,
                    keywords TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            self.cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_query TEXT NOT NULL,
                    bot_response TEXT,
                    intent TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            self.conn.commit()
            self._ensure_txn_columns()
            logger.info("Tables ensured")
        except Exception as e:
            logger.exception(f"Failed to create table: {e}")

    @log_call()
    def _add_transaction(self, txn_type, amount, category):
        try:
            sql = "INSERT INTO transactions (type, amount, category) VALUES (?, ?, ?)"
            self.cursor.execute(sql, (txn_type, amount, category))
            self.conn.commit()
            logger.info(f"Added transaction: {txn_type} ${amount} category {category}")
        except Exception as e:
            logger.exception(f"Failed to insert transaction: {e}")

    @log_call()
    def add_income(self, amount, category="Income"):
        self._add_transaction("income", amount, category)

    @log_call()
    def add_expense(self, amount, category="Misc"):
        self._add_transaction("expense", amount, category)

    @log_call()
    def get_total_income(self):
        try:
            sql = "SELECT SUM(amount) FROM transactions WHERE type=?"
            self.cursor.execute(sql, ("income",))
            result = self.cursor.fetchone()[0]
            total = float(result) if result else 0.0
            logger.debug(f"Total income calculated: {total}")
            return total
        except Exception as e:
            logger.exception(f"Failed to get total income: {e}")
            return 0.0

    @log_call()
    def get_total_expenses(self):
        try:
            sql = "SELECT SUM(amount) FROM transactions WHERE type=?"
            self.cursor.execute(sql, ("expense",))
            result = self.cursor.fetchone()[0]
            total = float(result) if result else 0.0
            logger.debug(f"Total expenses calculated: {total}")
            return total
        except Exception as e:
            logger.exception(f"Failed to get total expenses: {e}")
            return 0.0

    @log_call()
    def get_top_expense_categories(self, limit: int = 1):
        try:
            sql = (
                "SELECT COALESCE(category, 'Misc') as category, SUM(amount) as total "
                "FROM transactions WHERE type=? GROUP BY category ORDER BY total DESC LIMIT ?"
            )
            self.cursor.execute(sql, ("expense", limit))
            rows = self.cursor.fetchall()
            results = [(r[0], float(r[1])) for r in rows]
            logger.debug(f"Top expense categories: {results}")
            return results
        except Exception as e:
            logger.exception(f"Failed to get top expense categories: {e}")
            return []

    @log_call()
    def get_least_expense_categories(self, limit: int = 1):
        try:
            sql = (
                "SELECT COALESCE(category, 'Misc') as category, SUM(amount) as total "
                "FROM transactions WHERE type=? GROUP BY category ORDER BY total ASC LIMIT ?"
            )
            self.cursor.execute(sql, ("expense", limit))
            rows = self.cursor.fetchall()
            results = [(r[0], float(r[1])) for r in rows]
            logger.debug(f"Least expense categories: {results}")
            return results
        except Exception as e:
            logger.exception(f"Failed to get least expense categories: {e}")
            return []

    @log_call()
    def get_avg_expense_per_txn(self) -> float:
        try:
            sql = "SELECT AVG(amount) FROM transactions WHERE type=?"
            self.cursor.execute(sql, ("expense",))
            result = self.cursor.fetchone()[0]
            avg_val = float(result) if result is not None else 0.0
            logger.debug(f"Average expense per transaction: {avg_val}")
            return avg_val
        except Exception as e:
            logger.exception(f"Failed to get average expense per transaction: {e}")
            return 0.0

    def _connect_sqlite(self):
        sqlite_path = os.getenv("CHATBOT_SQLITE_PATH", "chatbot.db")
        logger.info(f"Connecting to SQLite database at {sqlite_path}")
        self.conn = sqlite3.connect(sqlite_path, check_same_thread=False, timeout=10)
        self.cursor = self.conn.cursor()
        self.conn.row_factory = sqlite3.Row

    # ---------- Schema maintenance and seeding ----------

    def _ensure_txn_columns(self):
        """Add newer columns to transactions if they don't exist (SQLite-safe)."""
        new_cols = [
            ("occurred_at", "TEXT"),
            ("description", "TEXT"),
            ("created_at", "DATETIME DEFAULT CURRENT_TIMESTAMP"),
        ]
        for col, ddl in new_cols:
            try:
                self.cursor.execute(f"ALTER TABLE transactions ADD COLUMN {col} {ddl}")
            except Exception:
                # Column likely exists; ignore
                pass
        self.conn.commit()

    def _seed_defaults(self):
        """Seed default categories, sample FAQs, and ensure first-run flag exists."""
        try:
            defaults = [
                ("Income", "income"),
                ("Salary", "income"),
                ("Bonus", "income"),
                ("Dining", "expense"),
                ("Groceries", "expense"),
                ("Rent", "expense"),
                ("Utilities", "expense"),
                ("Transport", "expense"),
                ("Misc", "expense"),
            ]
            sql = "INSERT OR IGNORE INTO categories (name, kind) VALUES (?, ?)"
            for name, kind in defaults:
                self.cursor.execute(sql, (name, kind))

            sample_faqs = [
                (
                    "How do I add an expense?",
                    "Say something like 'I spent 25 on lunch' and I'll record it.",
                    "help",
                    "add expense record spend",
                ),
                (
                    "How do I see my total expenses?",
                    "Ask 'What are my total expenses?' and I'll sum them.",
                    "help",
                    "total expenses sum",
                ),
            ]
            sql_faq = (
                "INSERT OR IGNORE INTO faqs (question, answer, category, keywords) "
                "VALUES (?, ?, ?, ?)"
            )
            for q, a, cat, kw in sample_faqs:
                self.cursor.execute(sql_faq, (q, a, cat, kw))

            if self.get_setting("first_run_completed") is None:
                self.set_setting("first_run_completed", "false", "bool")

            self.conn.commit()
        except Exception as e:
            logger.exception(f"Failed to seed defaults: {e}")

    # ---------- Settings helpers ----------

    def get_setting(self, key: str, default: Optional[str] = None) -> Optional[str]:
        try:
            sql = "SELECT value FROM settings WHERE key = ?"
            self.cursor.execute(sql, (key,))
            row = self.cursor.fetchone()
            return row[0] if row else default
        except Exception as e:
            logger.exception(f"Failed to get setting {key}: {e}")
            return default

    def set_setting(self, key: str, value: str, type_hint: str = "str"):
        try:
            sql = "INSERT INTO settings (key, value, type) VALUES (?, ?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value, type=excluded.type"
            self.cursor.execute(sql, (key, value, type_hint))
            self.conn.commit()
        except Exception as e:
            logger.exception(f"Failed to set setting {key}: {e}")

    def is_first_run_completed(self) -> bool:
        return str(self.get_setting("first_run_completed", "false")).lower() == "true"

    # ---------- User profile helpers ----------

    def get_user_profile(self) -> Dict[str, Optional[str]]:
        try:
            sql = "SELECT name, currency, timezone, locale FROM user_profile WHERE id = 1"
            self.cursor.execute(sql)
            row = self.cursor.fetchone()
            if not row:
                return {"name": None, "currency": None, "timezone": None, "locale": None}
            return {"name": row[0], "currency": row[1], "timezone": row[2], "locale": row[3]}
        except Exception as e:
            logger.exception(f"Failed to fetch user profile: {e}")
            return {"name": None, "currency": None, "timezone": None, "locale": None}

    def upsert_user_profile(self, name: Optional[str] = None, currency: Optional[str] = None, timezone: Optional[str] = None, locale: Optional[str] = None):
        try:
            sql = (
                "INSERT INTO user_profile (id, name, currency, timezone, locale) VALUES (1, ?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET name=excluded.name, currency=excluded.currency, timezone=excluded.timezone, locale=excluded.locale"
            )
            self.cursor.execute(sql, (name, currency, timezone, locale))
            self.conn.commit()
        except Exception as e:
            logger.exception(f"Failed to upsert user profile: {e}")

    # ---------- Category helpers ----------

    def list_categories(self):
        try:
            sql = "SELECT name, kind FROM categories ORDER BY name"
            self.cursor.execute(sql)
            return self.cursor.fetchall()
        except Exception as e:
            logger.exception(f"Failed to list categories: {e}")
            return []

    # ---------- FAQ / knowledge base ----------

    def add_faq(self, question: str, answer: str, category: Optional[str] = None, keywords: Optional[str] = None):
        try:
            sql = (
                "INSERT OR REPLACE INTO faqs (question, answer, category, keywords) "
                "VALUES (?, ?, ?, ?)"
            )
            self.cursor.execute(sql, (question, answer, category, keywords))
            self.conn.commit()
        except Exception as e:
            logger.exception(f"Failed to add FAQ: {e}")

    def search_faq(self, query: str, limit: int = 1) -> List[Dict[str, Any]]:
        try:
            like = f"%{query}%"
            sql = (
                "SELECT question, answer, category, keywords FROM faqs "
                "WHERE question LIKE ? OR answer LIKE ? OR keywords LIKE ? "
                "ORDER BY id DESC LIMIT ?"
            )
            self.cursor.execute(sql, (like, like, like, limit))
            rows = self.cursor.fetchall()
            results = []
            for r in rows:
                if isinstance(r, sqlite3.Row):
                    results.append(dict(r))
                else:
                    results.append({"question": r[0], "answer": r[1], "category": r[2], "keywords": r[3]})
            return results
        except Exception as e:
            logger.exception(f"Failed to search FAQs: {e}")
            return []

    def export_faqs_json(self, filepath: str) -> bool:
        try:
            data = self.search_faq("", limit=10000)
            Path(filepath).write_text(json.dumps(data, indent=2), encoding="utf-8")
            return True
        except Exception as e:
            logger.exception(f"Failed to export FAQs: {e}")
            return False

    def import_faqs_json(self, filepath: str) -> bool:
        try:
            path = Path(filepath)
            if not path.exists():
                logger.error(f"FAQ import file not found: {filepath}")
                return False
            items = json.loads(path.read_text(encoding="utf-8"))
            for item in items:
                self.add_faq(
                    item.get("question", ""),
                    item.get("answer", ""),
                    item.get("category"),
                    item.get("keywords"),
                )
            return True
        except Exception as e:
            logger.exception(f"Failed to import FAQs: {e}")
            return False

    # ---------- Chat history ----------

    def log_interaction(self, user_query: str, bot_response: str, intent: Optional[str]):
        try:
            sql = "INSERT INTO chat_history (user_query, bot_response, intent) VALUES (?, ?, ?)"
            self.cursor.execute(sql, (user_query, bot_response, intent))
            self.conn.commit()
        except Exception as e:
            logger.exception(f"Failed to log interaction: {e}")

    def get_chat_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        try:
            sql = (
                "SELECT user_query, bot_response, intent, timestamp FROM chat_history "
                "ORDER BY id DESC LIMIT ?"
            )
            self.cursor.execute(sql, (limit,))
            rows = self.cursor.fetchall()
            results = []
            for r in rows:
                if isinstance(r, sqlite3.Row):
                    results.append(dict(r))
                else:
                    results.append({"user_query": r[0], "bot_response": r[1], "intent": r[2], "timestamp": r[3]})
            return results
        except Exception as e:
            logger.exception(f"Failed to fetch chat history: {e}")
            return []

    def export_chat_history_json(self, filepath: str) -> bool:
        try:
            data = self.get_chat_history(limit=100000)
            Path(filepath).write_text(json.dumps(data, indent=2), encoding="utf-8")
            return True
        except Exception as e:
            logger.exception(f"Failed to export chat history: {e}")
            return False

    def import_chat_history_json(self, filepath: str) -> bool:
        try:
            path = Path(filepath)
            if not path.exists():
                logger.error(f"Chat history import file not found: {filepath}")
                return False
            items = json.loads(path.read_text(encoding="utf-8"))
            sql = "INSERT INTO chat_history (user_query, bot_response, intent, timestamp) VALUES (?, ?, ?, ?)"
            for item in items:
                self.cursor.execute(
                    sql,
                    (
                        item.get("user_query", ""),
                        item.get("bot_response", ""),
                        item.get("intent"),
                        item.get("timestamp"),
                    ),
                )
            self.conn.commit()
            return True
        except Exception as e:
            logger.exception(f"Failed to import chat history: {e}")
            return False
