
from flask import Flask, render_template, jsonify
import threading
import time
import random  # For demo; replace with your real metrics fetching

app = Flask(__name__)
consciousness_history = []

def fetch_metrics():
    """RETIRED 2026-07-29. Served three `random.uniform(0, 1)` values as metrics."""
    raise NotImplementedError(
        "fetch_metrics served `consciousness_score`, `memory_coherence` and "
        "`global_workspace` as `random.uniform(0.0, 1.0)`. A dashboard that renders "
        "random numbers under those labels is the most screenshot-ready way to "
        "misrepresent this project. Real per-step metrics are written to "
        "runs/<name>/metrics.csv by scripts/training/metrics_logger.py; a dashboard "
        "should read that file. Note most columns there are legitimately frozen "
        "(default-off modules), so read the audit-project degeneracy check first."
    )

@app.route("/")
def index():
    return render_template("dashboard.html")

@app.route("/metrics")
def get_metrics():
    return jsonify(consciousness_history[-50:])  # Last 50 points

def run_dashboard():
    app.run(host="0.0.0.0", port=5000, debug=False)

if __name__ == "__main__":
    # Start background thread for data collection
    metrics_thread = threading.Thread(target=fetch_metrics, daemon=True)
    metrics_thread.start()

    # Start Flask server
    run_dashboard()