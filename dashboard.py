"""
dashboard.py — Training monitor dashboard

Usage:
    python dashboard.py
    Then open http://localhost:8501 in your browser.
"""

import http.server
import json
import os
import re
import signal
import subprocess
import threading
import time

PORT = 8501
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(BASE_DIR, "training.log")
PID_FILE = os.path.join(BASE_DIR, "training.pid")

training_process = None
lock = threading.Lock()


def parse_training_log():
    """Parse the training log file for current status."""
    status = {
        "running": is_training_running(),
        "epoch": 0,
        "total_epochs": 20,
        "batch": 0,
        "total_batches": 0,
        "box_loss": 0,
        "cls_loss": 0,
        "dfl_loss": 0,
        "gpu_mem": "",
        "eta_epoch": "",
        "phase": "idle",
        "log_lines": [],
        "epochs_done": [],
    }

    if not os.path.isfile(LOG_FILE):
        return status

    try:
        with open(LOG_FILE, "r", errors="replace") as f:
            raw = f.read()
    except Exception:
        return status

    # Strip ANSI codes
    clean = re.sub(r'\x1b\[[0-9;]*m', '', raw)
    clean = re.sub(r'\[K', '', clean)

    lines = clean.split('\n')

    # Grab last 50 meaningful lines for the log panel
    meaningful = [l.strip() for l in lines if l.strip() and not l.strip().startswith('\r')]
    status["log_lines"] = meaningful[-60:]

    # Detect total epochs
    m = re.search(r'Starting training for (\d+) epochs', clean)
    if m:
        status["total_epochs"] = int(m.group(1))

    # Detect phase
    if "Downloading" in clean or "Extracting" in clean:
        status["phase"] = "downloading"
    elif "Merging" in clean or "[merge]" in clean:
        status["phase"] = "merging"
    elif "Scanning" in clean and "images" in clean.split("Scanning")[-1]:
        status["phase"] = "scanning"
    elif "Training" in clean or "Epoch" in clean.split('\n')[-1] if lines else "":
        status["phase"] = "training"

    # Parse epoch progress lines: "1/20  9.74G  1.51  2.768  1.044  328  640: 4%"
    epoch_pattern = re.compile(
        r'(\d+)/(\d+)\s+([\d.]+G?)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\d+\s+\d+:\s*(\d+)%\s*.*?(\d+)/(\d+)\s+([\d.]+)s/it\s+([\d:]+)<([\d:]+)'
    )
    # Simpler fallback pattern
    epoch_simple = re.compile(
        r'\s+(\d+)/(\d+)\s+([\d.]+G?)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\d+\s+\d+:\s*(\d+)%'
    )
    batch_pattern = re.compile(r'(\d+)/(\d+)\s+[\d.]+s/it\s+([\d:.]+)<([\d:.]+)')

    last_epoch_match = None
    last_batch_match = None

    for line in lines:
        m = epoch_simple.search(line)
        if m:
            last_epoch_match = m
        m2 = batch_pattern.search(line)
        if m2:
            last_batch_match = m2

    if last_epoch_match:
        status["epoch"] = int(last_epoch_match.group(1))
        status["total_epochs"] = int(last_epoch_match.group(2))
        status["gpu_mem"] = last_epoch_match.group(3)
        status["box_loss"] = float(last_epoch_match.group(4))
        status["cls_loss"] = float(last_epoch_match.group(5))
        status["dfl_loss"] = float(last_epoch_match.group(6))
        status["phase"] = "training"

    if last_batch_match:
        status["batch"] = int(last_batch_match.group(1))
        status["total_batches"] = int(last_batch_match.group(2))
        status["eta_epoch"] = last_batch_match.group(4)

    # Collect completed epoch validation results
    val_pattern = re.compile(
        r'(\d+)/(\d+)\s+[\d.]+G?\s+([\d.]+)\s+([\d.]+)\s+([\d.]+).*?all\s+\d+\s+\d+\s+([\d.]+)\s+([\d.]+)'
    )
    # Simpler: look for "Epoch X done" style or mAP lines
    map_pattern = re.compile(r'all\s+\d+\s+\d+\s+([\d.]+)\s+([\d.]+)')
    epoch_done_lines = []
    current_epoch_for_val = 0
    for line in lines:
        em = epoch_simple.search(line)
        if em:
            current_epoch_for_val = int(em.group(1))
        mm = map_pattern.search(line)
        if mm and current_epoch_for_val > 0:
            epoch_done_lines.append({
                "epoch": current_epoch_for_val,
                "mAP50": float(mm.group(1)),
                "mAP50_95": float(mm.group(2)),
            })

    # Deduplicate by epoch
    seen = set()
    for ed in epoch_done_lines:
        if ed["epoch"] not in seen:
            status["epochs_done"].append(ed)
            seen.add(ed["epoch"])

    # Check for completion
    if "Training complete" in clean or "training complete" in clean.lower():
        status["phase"] = "complete"
    if re.search(r'Best model copied', clean):
        status["phase"] = "complete"

    return status


def is_training_running():
    """Check if the training process is still alive."""
    global training_process
    if training_process is not None:
        if training_process.poll() is None:
            return True
        training_process = None

    # Also check PID file
    if os.path.isfile(PID_FILE):
        try:
            with open(PID_FILE) as f:
                pid = int(f.read().strip())
            os.kill(pid, 0)  # check if alive
            return True
        except (ProcessLookupError, ValueError, PermissionError):
            os.remove(PID_FILE)
    return False


def start_training(epochs=20, resume=False):
    """Start the training process."""
    global training_process
    with lock:
        if is_training_running():
            return {"ok": False, "error": "Training already running"}

        cmd = [
            "bash", "-c",
            f"cd '{BASE_DIR}' && source venv/bin/activate && "
            f"python3 train_model.py --skip-download --epochs {epochs}"
            + (" --resume" if resume else "")
        ]

        log_fh = open(LOG_FILE, "a")
        training_process = subprocess.Popen(
            cmd, stdout=log_fh, stderr=subprocess.STDOUT,
        )

        with open(PID_FILE, "w") as f:
            f.write(str(training_process.pid))

        return {"ok": True, "pid": training_process.pid}


def stop_training():
    """Stop the training process."""
    global training_process
    with lock:
        killed = False
        if training_process and training_process.poll() is None:
            training_process.send_signal(signal.SIGINT)
            try:
                training_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                training_process.kill()
            training_process = None
            killed = True

        if os.path.isfile(PID_FILE):
            try:
                with open(PID_FILE) as f:
                    pid = int(f.read().strip())
                os.kill(pid, signal.SIGINT)
                killed = True
            except Exception:
                pass
            os.remove(PID_FILE)

        return {"ok": True, "killed": killed}


HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Soccer YOLO Training Dashboard</title>
<style>
  :root {
    --bg: #0f1117;
    --card: #1a1d27;
    --border: #2a2d3a;
    --text: #e0e0e0;
    --dim: #888;
    --accent: #4ade80;
    --red: #f87171;
    --yellow: #fbbf24;
    --blue: #60a5fa;
    --purple: #a78bfa;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'SF Pro', system-ui, sans-serif;
    background: var(--bg); color: var(--text);
    padding: 24px; min-height: 100vh;
  }
  h1 { font-size: 1.6rem; font-weight: 700; margin-bottom: 20px; }
  h1 span { color: var(--accent); }

  .grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; margin-bottom: 20px; }
  .card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 12px; padding: 20px;
  }
  .card-label { font-size: 0.75rem; color: var(--dim); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px; }
  .card-value { font-size: 2rem; font-weight: 700; font-variant-numeric: tabular-nums; }
  .card-sub { font-size: 0.8rem; color: var(--dim); margin-top: 4px; }

  .status-row { display: flex; gap: 12px; align-items: center; margin-bottom: 20px; }
  .status-badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 6px 14px; border-radius: 20px; font-size: 0.85rem; font-weight: 600;
  }
  .status-badge.running { background: rgba(74,222,128,0.15); color: var(--accent); }
  .status-badge.stopped { background: rgba(248,113,113,0.15); color: var(--red); }
  .status-badge.complete { background: rgba(96,165,250,0.15); color: var(--blue); }
  .dot { width: 8px; height: 8px; border-radius: 50%; }
  .dot.pulse { animation: pulse 1.5s infinite; }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.3; } }
  .running .dot { background: var(--accent); }
  .stopped .dot { background: var(--red); }
  .complete .dot { background: var(--blue); }

  .btn {
    padding: 8px 20px; border: none; border-radius: 8px; font-size: 0.9rem;
    font-weight: 600; cursor: pointer; transition: all 0.15s;
  }
  .btn:hover { filter: brightness(1.1); transform: translateY(-1px); }
  .btn:active { transform: translateY(0); }
  .btn-start { background: var(--accent); color: #000; }
  .btn-resume { background: var(--blue); color: #000; }
  .btn-stop { background: var(--red); color: #fff; }
  .btn:disabled { opacity: 0.4; cursor: not-allowed; filter: none; transform: none; }

  .progress-wrap {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 12px; padding: 20px; margin-bottom: 20px;
  }
  .progress-label { display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 0.85rem; }
  .progress-bar { background: #2a2d3a; border-radius: 6px; height: 12px; overflow: hidden; }
  .progress-fill { height: 100%; border-radius: 6px; transition: width 0.5s ease; }
  .fill-epoch { background: linear-gradient(90deg, var(--accent), #22d3ee); }
  .fill-batch { background: linear-gradient(90deg, var(--blue), var(--purple)); }

  .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 20px; }
  .loss-chart { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 20px; }
  .loss-chart h3 { font-size: 0.9rem; margin-bottom: 12px; }
  canvas { width: 100% !important; height: 200px !important; }

  .log-panel {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 12px; padding: 20px;
  }
  .log-panel h3 { font-size: 0.9rem; margin-bottom: 12px; }
  .log-content {
    background: #12141c; border-radius: 8px; padding: 12px;
    font-family: 'SF Mono', 'Fira Code', monospace; font-size: 0.75rem;
    line-height: 1.5; max-height: 300px; overflow-y: auto;
    color: var(--dim); white-space: pre-wrap; word-break: break-all;
  }
  .log-content .highlight { color: var(--accent); }

  .epochs-table { width: 100%; border-collapse: collapse; font-size: 0.8rem; }
  .epochs-table th { color: var(--dim); font-weight: 500; text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--border); }
  .epochs-table td { padding: 6px 8px; border-bottom: 1px solid var(--border); font-variant-numeric: tabular-nums; }
  .good { color: var(--accent); }
</style>
</head>
<body>

<h1>Soccer YOLO <span>Training Dashboard</span></h1>

<div class="status-row">
  <div id="statusBadge" class="status-badge stopped">
    <div class="dot"></div>
    <span id="statusText">Stopped</span>
  </div>
  <div id="phaseText" style="color:var(--dim);font-size:0.85rem;"></div>
  <div style="flex:1"></div>
  <button class="btn btn-start" id="btnStart" onclick="startTraining(false)">Start Training</button>
  <button class="btn btn-resume" id="btnResume" onclick="startTraining(true)">Resume</button>
  <button class="btn btn-stop" id="btnStop" onclick="stopTraining()" disabled>Stop</button>
</div>

<div class="progress-wrap">
  <div class="progress-label">
    <span>Epoch <strong id="epochNum">0</strong> / <strong id="epochTotal">20</strong></span>
    <span id="epochPct">0%</span>
  </div>
  <div class="progress-bar"><div class="progress-fill fill-epoch" id="epochBar" style="width:0%"></div></div>
  <div class="progress-label" style="margin-top:12px">
    <span>Batch <strong id="batchNum">0</strong> / <strong id="batchTotal">0</strong></span>
    <span id="etaEpoch"></span>
  </div>
  <div class="progress-bar"><div class="progress-fill fill-batch" id="batchBar" style="width:0%"></div></div>
</div>

<div class="grid">
  <div class="card">
    <div class="card-label">Box Loss</div>
    <div class="card-value" id="boxLoss">—</div>
  </div>
  <div class="card">
    <div class="card-label">Class Loss</div>
    <div class="card-value" id="clsLoss">—</div>
  </div>
  <div class="card">
    <div class="card-label">DFL Loss</div>
    <div class="card-value" id="dflLoss">—</div>
    <div class="card-sub" id="gpuMem"></div>
  </div>
</div>

<div class="two-col">
  <div class="loss-chart">
    <h3>Completed Epochs</h3>
    <div style="max-height:220px;overflow-y:auto;">
      <table class="epochs-table">
        <thead><tr><th>Epoch</th><th>mAP@50</th><th>mAP@50-95</th></tr></thead>
        <tbody id="epochsBody"></tbody>
      </table>
      <div id="noEpochs" style="color:var(--dim);font-size:0.8rem;padding:12px;">No completed epochs yet</div>
    </div>
  </div>
  <div class="log-panel">
    <h3>Training Log</h3>
    <div class="log-content" id="logContent">Waiting for data...</div>
  </div>
</div>

<script>
let polling = null;

async function fetchStatus() {
  try {
    const r = await fetch('/api/status');
    const d = await r.json();
    update(d);
  } catch(e) {}
}

function update(d) {
  // Status badge
  const badge = document.getElementById('statusBadge');
  const text = document.getElementById('statusText');
  badge.className = 'status-badge ' + (d.phase === 'complete' ? 'complete' : d.running ? 'running' : 'stopped');
  const dot = badge.querySelector('.dot');
  dot.className = d.running ? 'dot pulse' : 'dot';
  text.textContent = d.phase === 'complete' ? 'Complete' : d.running ? 'Running' : 'Stopped';

  // Phase
  const phases = {downloading:'Downloading datasets...', merging:'Merging datasets...', scanning:'Scanning images...', training:'Training model', complete:'Training complete!', idle:'Ready'};
  document.getElementById('phaseText').textContent = phases[d.phase] || d.phase;

  // Buttons
  document.getElementById('btnStart').disabled = d.running;
  document.getElementById('btnResume').disabled = d.running;
  document.getElementById('btnStop').disabled = !d.running;

  // Epoch progress
  document.getElementById('epochNum').textContent = d.epoch;
  document.getElementById('epochTotal').textContent = d.total_epochs;
  const epPct = d.total_epochs > 0 ? Math.round(d.epoch / d.total_epochs * 100) : 0;
  document.getElementById('epochPct').textContent = epPct + '%';
  document.getElementById('epochBar').style.width = epPct + '%';

  // Batch progress
  document.getElementById('batchNum').textContent = d.batch;
  document.getElementById('batchTotal').textContent = d.total_batches || '?';
  const btPct = d.total_batches > 0 ? Math.round(d.batch / d.total_batches * 100) : 0;
  document.getElementById('batchBar').style.width = btPct + '%';
  document.getElementById('etaEpoch').textContent = d.eta_epoch ? 'ETA: ' + d.eta_epoch : '';

  // Losses
  document.getElementById('boxLoss').textContent = d.box_loss ? d.box_loss.toFixed(3) : '—';
  document.getElementById('clsLoss').textContent = d.cls_loss ? d.cls_loss.toFixed(3) : '—';
  document.getElementById('dflLoss').textContent = d.dfl_loss ? d.dfl_loss.toFixed(3) : '—';
  document.getElementById('gpuMem').textContent = d.gpu_mem ? 'GPU: ' + d.gpu_mem : '';

  // Epochs table
  const tbody = document.getElementById('epochsBody');
  const noEp = document.getElementById('noEpochs');
  if (d.epochs_done && d.epochs_done.length > 0) {
    noEp.style.display = 'none';
    tbody.innerHTML = d.epochs_done.map(e =>
      `<tr><td>${e.epoch}</td><td class="${e.mAP50>0.5?'good':''}">${e.mAP50.toFixed(4)}</td><td>${e.mAP50_95.toFixed(4)}</td></tr>`
    ).join('');
  } else {
    noEp.style.display = 'block';
    tbody.innerHTML = '';
  }

  // Log
  const logEl = document.getElementById('logContent');
  if (d.log_lines && d.log_lines.length > 0) {
    const filtered = d.log_lines.slice(-30).map(l =>
      l.replace(/(Epoch|mAP|best|saved)/gi, '<span class="highlight">$1</span>')
    );
    logEl.innerHTML = filtered.join('\\n');
    logEl.scrollTop = logEl.scrollHeight;
  }
}

async function startTraining(resume) {
  const ep = resume ? 0 : prompt('Number of epochs?', '20');
  if (!resume && !ep) return;
  const body = resume ? {resume:true} : {epochs: parseInt(ep)};
  await fetch('/api/start', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
  fetchStatus();
}

async function stopTraining() {
  if (!confirm('Stop training? Progress is saved — you can resume later.')) return;
  await fetch('/api/stop', {method:'POST'});
  fetchStatus();
}

// Poll every 2 seconds
fetchStatus();
polling = setInterval(fetchStatus, 2000);
</script>
</body>
</html>"""


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # suppress request logs

    def _json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def _html(self, content):
        body = content.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/status":
            self._json(parse_training_log())
        else:
            self._html(HTML)

    def do_POST(self):
        if self.path == "/api/start":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            epochs = body.get("epochs", 20)
            resume = body.get("resume", False)
            result = start_training(epochs=epochs, resume=resume)
            self._json(result)
        elif self.path == "/api/stop":
            result = stop_training()
            self._json(result)
        else:
            self._json({"error": "not found"}, 404)


if __name__ == "__main__":
    # If training is already running in background, adopt its log
    # by pointing LOG_FILE at the existing output
    print(f"Soccer YOLO Training Dashboard")
    print(f"Open http://localhost:{PORT}")
    print(f"Log file: {LOG_FILE}")
    print(f"Press Ctrl+C to stop the dashboard (training continues in background)")

    server = http.server.HTTPServer(("", PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
        server.server_close()
