"""A loopback-only, read-only view of one validated comparison."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
import json


def serve(report, port):
    web = files("quantdoctor").joinpath("web")
    content = {
        "/": ("text/html; charset=utf-8", web.joinpath("index.html").read_bytes()),
        "/app.js": ("text/javascript; charset=utf-8", web.joinpath("app.js").read_bytes()),
        "/style.css": ("text/css; charset=utf-8", web.joinpath("style.css").read_bytes()),
        "/report.json": ("application/json", json.dumps(report, allow_nan=False).encode()),
    }

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.headers.get("Host") != f"127.0.0.1:{self.server.server_port}":
                self.send_error(403)
                return
            payload = content.get(self.path)
            if payload is None:
                self.send_error(404)
                return
            mime, body = payload
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Security-Policy",
                             "default-src 'self'; base-uri 'none'; frame-ancestors 'none'; object-src 'none'")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

    with ThreadingHTTPServer(("127.0.0.1", port), Handler) as http:
        print(f"http://127.0.0.1:{http.server_port}/", flush=True)
        http.serve_forever()
