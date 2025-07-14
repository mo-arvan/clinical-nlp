import logging
import os

from dotenv import load_dotenv

APP_NAME = "ehr-qa"

def configure_logger() -> logging.Logger:
    """
    Configure a logger with standardized formatting.

    Returns:
        logging.Logger: Configured logger instance.
    """

    # Create a logger
    logger = logging.getLogger(APP_NAME)
    logger.setLevel(logging.INFO)

    # Create a formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Set the formatter on the root logger
    for handler in logger.handlers:
        handler.setFormatter(formatter)
        
    # set httpx and litellm log level to warning
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    os.environ["LITELLM_LOG"] = "ERROR"

    return logger

def load_env_variables() -> dict:
    """
    Load environment variables from the .env file.

    Returns:
        dict: A dictionary containing environment variables.
    """

    load_dotenv()
    return {
        "AZURE_API_KEY": os.getenv("AZURE_API_KEY"),
        "AZURE_API_BASE": os.getenv("AZURE_API_BASE"),
        "AZURE_API_VERSION": os.getenv("AZURE_API_VERSION"),
    }
