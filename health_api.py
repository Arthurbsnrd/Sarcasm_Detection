"""
Optional Flask health API for MLOps monitoring (Docker / VM / sidecar).

Run locally:
    pip install flask
    python health_api.py

Then open: http://localhost:8080/health
"""

from flask import Flask, jsonify

from monitoring import get_health

app = Flask(__name__)


@app.route("/health")
def health():
    return jsonify(get_health(model_loaded=False))


@app.route("/")
def index():
    return jsonify({"service": "sarcasm-detector-health", "endpoints": ["/health"]})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
