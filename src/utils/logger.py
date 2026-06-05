import os
import sys
from loguru import logger

# Define default paths and levels
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "papermind.log")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Ensure logs directory exists at import time
os.makedirs(LOG_DIR, exist_ok=True)

# Remove the default Loguru logger handler to prevent double outputs
logger.remove()

# Add customized console output handler
logger.add(
    sys.stdout,
    level=LOG_LEVEL,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    enqueue=True
)

# Add rotating file logger handler
logger.add(
    LOG_FILE,
    level=LOG_LEVEL,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    rotation="10 MB",
    retention=7,
    encoding="utf-8",
    enqueue=True
)
