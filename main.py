from chatbot.bot import Chatbot
from chatbot.logger import get_logger, log_call

logger = get_logger(__name__)

if __name__ == "__main__":
    logger.info("Application starting")
    bot = Chatbot()
    try:
        bot.start()
    except Exception as e:
        logger.exception(f"Fatal error in bot: {e}")
        raise
    finally:
        logger.info("Application exiting")
