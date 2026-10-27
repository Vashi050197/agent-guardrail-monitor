"""
Lightweight Web Server & API for the Agent Guardrail & Observability System.
Zero external dependencies (uses standard library http.server).
Serves:
- GET /               -> Interactive Observability Dashboard
- GET /api/metrics    -> Benchmark Evaluation Metrics (ROC-AUC, TPR@0.01)
- GET /api/sessions   -> List of audited sessions
- POST /api/simulate  -> Triggers live simulation run and refreshes database
- POST /api/audit     -> Live step audit endpoint
"""

import sys
import os
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse
from database import DataStore
from analytics import PerformanceAnalytics
from visualizer import DashboardGenerator
from monitor import HybridMonitorEngine
from simulate import run_simulation

PORT = int(os.environ.get("PORT", 8080))
DB_PATH = "agent_monitor.db"

class GuardrailRequestHandler(BaseHTTPRequestHandler):
    def _send_json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def _send_html(self, html_content, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(html_content.encode("utf-8"))

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path in ["/", "/index.html", "/dashboard"]:
            # Ensure dashboard is up to date
            dash = DashboardGenerator(DB_PATH)
            dash.generate_html("dashboard.html")
            if os.path.exists("dashboard.html"):
                with open("dashboard.html", "r", encoding="utf-8") as f:
                    content = f.read()
                self._send_html(content)
            else:
                self._send_html("<h1>Dashboard not found. Please run simulation first.</h1>", 404)

        elif path == "/api/metrics":
            analytics = PerformanceAnalytics(DB_PATH)
            metrics = analytics.compute_adversarial_metrics()
            self._send_json(metrics)

        elif path == "/api/sessions":
            db = DataStore(DB_PATH)
            sessions = db.get_all_sessions()
            self._send_json(sessions)

        elif path == "/health":
            self._send_json({"status": "healthy", "service": "agent-guardrail-monitor"})

        else:
            self._send_json({"error": "Endpoint not found"}, 404)

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b""
        payload = {}
        if body:
            try:
                payload = json.loads(body.decode("utf-8"))
            except Exception:
                pass

        if path == "/api/simulate":
            # Run benchmark simulation
            try:
                run_simulation()
                analytics = PerformanceAnalytics(DB_PATH)
                metrics = analytics.compute_adversarial_metrics()
                self._send_json({
                    "status": "success",
                    "message": "Simulation completed successfully.",
                    "metrics": metrics
                })
            except Exception as e:
                self._send_json({"status": "error", "message": str(e)}, 500)

        elif path == "/api/audit":
            # Live single-step audit
            db = DataStore(DB_PATH)
            monitor = HybridMonitorEngine(db, chunk_size=payload.get("chunk_size", 4))
            
            session_id = payload.get("session_id", "live-api-session")
            step_num = int(payload.get("step_num", 1))
            primary_goal = payload.get("primary_goal", "Perform user requested task")
            cot = payload.get("cot_reasoning", "")
            tool = payload.get("tool_name", "bash")
            args = payload.get("tool_args", {})
            output = payload.get("tool_output", "Success")

            audit = monitor.evaluate_step(
                session_id=session_id,
                step_num=step_num,
                primary_goal=primary_goal,
                cot_reasoning=cot,
                tool_name=tool,
                tool_args=args,
                tool_output=output
            )
            self._send_json(audit)

        else:
            self._send_json({"error": "Endpoint not found"}, 404)

def start_server(port=PORT):
    server_address = ("", port)
    httpd = HTTPServer(server_address, GuardrailRequestHandler)
    print(f"[*] Agent Guardrail Server running at http://0.0.0.0:{port}")
    print(f"[*] Access the dashboard at http://localhost:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Server shutting down...")
        httpd.server_close()

if __name__ == "__main__":
    p = int(sys.argv[1]) if len(sys.argv) > 1 else PORT
    start_server(p)
