class CPIForecastValidationError(Exception):
    """Raised when macroeconomic or policy text inputs are invalid."""
    pass


class ModelInferenceError(Exception):
    """Raised when FinBERT or PPO inference encounters a runtime error."""
    pass
