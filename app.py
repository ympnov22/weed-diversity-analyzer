#!/usr/bin/env python3
"""FastAPI app instance for uvicorn deployment."""

from src.visualization.web_server import WebServer
from src.output.output_manager import OutputManager

output_manager = OutputManager()
web_server = WebServer(output_manager)

app = web_server.get_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
