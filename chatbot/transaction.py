from .logger import get_logger, log_call

logger = get_logger(__name__)


class Transaction:
    @log_call()
    def __init__(self, amount, category):
        self.amount = amount
        self.category = category
        logger.debug(f"Transaction created: amount={amount}, category={category}")

class Income(Transaction):
    pass

class Expense(Transaction):
    pass
