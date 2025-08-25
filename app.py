#!/usr/bin/env python3
"""FastAPI app instance for uvicorn deployment."""

import os
import logging
from src.visualization.web_server import WebServer
from src.output.output_manager import OutputManager

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

try:
    if os.getenv("DATABASE_URL"):
        logger.info("Initializing database...")
        from src.database.database import init_db
        from src.database.migrate import run_migrations
        init_db()
        run_migrations()
        logger.info("Database initialization completed")
    else:
        logger.info("No DATABASE_URL found, using file-based storage")
except Exception as e:
    logger.error(f"Database initialization failed: {e}")
    raise

output_manager = OutputManager()
web_server = WebServer(output_manager)

app = web_server.get_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
