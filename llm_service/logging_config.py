import sys
from loguru import logger
from config import settings

def setup_logging():
    """
    Configures the Loguru logger for the application, including daily rotation.
    """
    logger.remove()  # Remove default handler to avoid duplicate logs

    # Configure a handler for console output with color formatting
    logger.add(
        sys.stdout,
        level=settings.LOG_LEVEL.upper(),
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )
    )
    
    # Configure a file handler for daily log rotation
    logger.add(
        settings.LOG_FILE,
        rotation=settings.LOG_ROTATION,
        retention="10 days",
        compression="zip",
        level=settings.LOG_LEVEL.upper(),
        enqueue=True,  # Makes logging calls non-blocking, important for performance
        backtrace=True, # Show full stack trace on exceptions
        diagnose=True,  # Add exception variable values for easier debugging
    )
    
    logger.info("Logger configured successfully.")