"""
Archimedes Survey 3D Viewer - Local HTTP Server
Run: python viewer/server.py
Then open: http://localhost:8765
"""
import http.server
import socketserver
import os
import webbrowser
import threading

PORT = 8765
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class CORSHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ROOT, **kwargs)

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def log_message(self, format, *args):
        pass  # suppress logs

def open_browser():
    import time; time.sleep(0.5)
    webbrowser.open(f"http://localhost:{PORT}/viewer/index.html")

threading.Thread(target=open_browser, daemon=True).start()

print(f"Archimedes 3D Viewer → http://localhost:{PORT}/viewer/index.html")
print("Press Ctrl+C to stop.")
with socketserver.TCPServer(("", PORT), CORSHandler) as httpd:
    httpd.serve_forever()
