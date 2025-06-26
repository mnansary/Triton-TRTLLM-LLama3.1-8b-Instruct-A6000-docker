#!/bin/bash

# This script starts the LLM service using Gunicorn for production.
# It ensures logs directory exists and loads environment variables.

# Set script to exit immediately if a command exits with a non-zero status.
set -e

# --- Setup ---
# Create log directory if it doesn't exist to prevent startup errors.
mkdir -p logs
echo "Log directory './logs' ensured."

# Load environment variables from .env file if it exists.
# This allows for easy configuration without modifying the script.
if [ -f .env ]; then
  echo "Loading environment variables from .env file..."
  export $(grep -v '^#' .env | xargs)
fi

# --- Configuration ---
# Use environment variables if set, otherwise fall back to defaults from config.py
HOST=${HOST:-$(python -c "from config import settings; print(settings.HOST)")}
PORT=${PORT:-$(python -c "from config import settings; print(settings.PORT)")}
WORKERS=${WORKERS:-$(python -c "from config import settings; print(settings.WORKERS)")}
LOG_LEVEL=${LOG_LEVEL:-$(python -c "from config import settings; print(settings.LOG_LEVEL)")}

echo "--- Starting LLM Service ---"
echo "Host: ${HOST}"
echo "Port: ${PORT}"
echo "Worker Processes: ${WORKERS}"
echo "Log Level: ${LOG_LEVEL}"
echo "----------------------------"

# --- Execution ---
# Run the FastAPI app with Gunicorn and Uvicorn workers.
# Gunicorn is a battle-tested process manager for Python WSGI applications.
# UvicornWorker allows Gunicorn to run an ASGI application like FastAPI.
exec gunicorn llm_service.main:app \\
    --workers ${WORKERS} \\
    --worker-class uvicorn.workers.UvicornWorker \\
    --bind ${HOST}:${PORT} \\
    --log-level ${LOG_LEVEL,,}