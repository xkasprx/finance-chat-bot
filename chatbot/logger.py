import logging
import os
import sys
import time
from functools import wraps
from logging.handlers import RotatingFileHandler

_CONFIGURED = False


def _ensure_log_dir(path: str) -> str:
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    return path


def _configure_root_logger():
    global _CONFIGURED
    if _CONFIGURED:
        return

    log_level = os.getenv("CHATBOT_LOG_LEVEL", "CRITICAL").upper()
    level = getattr(logging, log_level, logging.CRITICAL)

    # Determine log file path (default: logs/chatbot.log)
    log_dir = os.getenv("CHATBOT_LOG_DIR", os.path.join("logs"))
    log_file = os.getenv("CHATBOT_LOG_FILE", os.path.join(log_dir, "chatbot.log"))
    log_file = _ensure_log_dir(log_file)

    root = logging.getLogger()
    root.setLevel(level)

    # Avoid duplicate handlers if re-imported
    if not root.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(funcName)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # File handler with rotation
        try:
            file_handler = RotatingFileHandler(log_file, maxBytes=1_000_000, backupCount=5, encoding="utf-8")
            file_handler.setFormatter(formatter)
            file_handler.setLevel(level)
            root.addHandler(file_handler)
        except Exception:
            # If file handler fails (e.g., permission), at least have console logging
            pass

        # Console handler
        console_handler = logging.StreamHandler(stream=sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)
        root.addHandler(console_handler)

    _CONFIGURED = True


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a module-scoped logger. Ensures configuration on first use."""
    _configure_root_logger()
    return logging.getLogger(name if name else __name__)


def log_call(level: int = logging.DEBUG, log_args: bool = True, log_result: bool = False, redact: list[str] | None = None):
    """Decorator to log function entry, exit, duration, and errors.

    - level: logging level for entry/exit
    - log_args: include args/kwargs (with redaction)
    - log_result: include returned value (careful with large objects)
    - redact: list of kwarg names to mask
    """
    redact = redact or []

    def decorator(func):
        logger = get_logger(func.__module__)

        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            if log_args:
                try:
                    masked_kwargs = {k: ("***" if k in redact else v) for k, v in kwargs.items()}
                    # Avoid logging entire self objects; replace with class name
                    arg_preview = []
                    for i, a in enumerate(args):
                        if i == 0 and hasattr(a, "__class__"):
                            arg_preview.append(f"<self:{a.__class__.__name__}>")
                        else:
                            arg_preview.append(a)
                    logger.log(level, f"-> {func.__name__} args={arg_preview} kwargs={masked_kwargs}")
                except Exception:
                    logger.log(level, f"-> {func.__name__} (args unavailable)")
            else:
                logger.log(level, f"-> {func.__name__}")

            try:
                result = func(*args, **kwargs)
                duration_ms = (time.perf_counter() - start) * 1000
                if log_result:
                    logger.log(level, f"<- {func.__name__} ok in {duration_ms:.1f} ms result={result}")
                else:
                    logger.log(level, f"<- {func.__name__} ok in {duration_ms:.1f} ms")
                return result
            except Exception as e:
                duration_ms = (time.perf_counter() - start) * 1000
                logger.exception(f"x {func.__name__} failed after {duration_ms:.1f} ms: {e}")
                raise

        return wrapper

    return decorator


# Configure immediately on import so any module gets consistent logging.
_configure_root_logger()

