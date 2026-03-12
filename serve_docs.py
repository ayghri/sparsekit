#!/usr/bin/env python3
"""Serve the SparseKit documentation locally.

Run:
    python serve_docs.py

Then open http://localhost:8000 in your browser.
Press Ctrl+C to stop the server.
"""
import os
import sys
import http.server
import socketserver
from pathlib import Path

DOCS_DIR = Path(__file__).parent / "docs" / "_build" / "html"
PORT = 8000

if not DOCS_DIR.is_dir():
    print(f"Documentation not found at {DOCS_DIR}")
    print("The built HTML should be included in the submission.")
    sys.exit(1)

os.chdir(DOCS_DIR)
Handler = http.server.SimpleHTTPRequestHandler
Handler.log_message = lambda *_: None   # suppress request logs

print(f"Serving SparseKit docs at  http://localhost:{PORT}")
print(f"Source directory:          {DOCS_DIR}")
print("Press Ctrl+C to stop.\n")

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
