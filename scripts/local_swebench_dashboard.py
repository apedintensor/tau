from __future__ import annotations

import argparse
import json
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

DEFAULT_ROOT = Path("workspace/validate/netuid-66/benchmarks/swebench-verified")


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SWE-bench King Benchmark</title>
<style>
:root {
  color-scheme: dark;
  --bg: #090a0c;
  --panel: #11151a;
  --panel-2: #171d24;
  --line: #28313c;
  --text: #eef3f8;
  --muted: #8f9ba8;
  --green: #40d98b;
  --red: #ff6874;
  --yellow: #ffd166;
  --blue: #69a7ff;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text); }
main { width: min(1180px, calc(100vw - 28px)); margin: 0 auto; padding: 18px 0 28px; }
header { display: flex; justify-content: space-between; gap: 16px; align-items: flex-start; margin-bottom: 14px; }
h1 { margin: 0; font-size: 18px; letter-spacing: .04em; text-transform: uppercase; }
.sub { color: var(--muted); margin-top: 5px; font-size: 12px; overflow-wrap: anywhere; }
.pill { border: 1px solid var(--line); border-radius: 999px; padding: 6px 9px; color: var(--muted); font-size: 11px; text-transform: uppercase; white-space: nowrap; }
.grid { display: grid; grid-template-columns: repeat(12, 1fr); gap: 10px; }
.card { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 12px; min-width: 0; }
.span-3 { grid-column: span 3; }
.span-4 { grid-column: span 4; }
.span-6 { grid-column: span 6; }
.span-8 { grid-column: span 8; }
.span-12 { grid-column: 1 / -1; }
.label { color: var(--muted); font-size: 10px; text-transform: uppercase; letter-spacing: .08em; }
.value { margin-top: 6px; font-size: 24px; line-height: 1; overflow-wrap: anywhere; }
.small { color: var(--muted); font-size: 11px; margin-top: 7px; overflow-wrap: anywhere; }
.ok { color: var(--green); }
.bad { color: var(--red); }
.warn { color: var(--yellow); }
.blue { color: var(--blue); }
.bars { display: grid; gap: 14px; }
.bar-head { display: flex; justify-content: space-between; gap: 8px; font-size: 12px; text-transform: uppercase; }
.track { height: 18px; border: 1px solid var(--line); border-radius: 4px; background: var(--panel-2); overflow: hidden; margin-top: 6px; }
.fill { height: 100%; width: 0%; background: var(--blue); min-width: 2px; }
.fill.king { background: var(--green); }
.fill.pi { background: var(--blue); }
table { width: 100%; border-collapse: collapse; font-size: 12px; }
th, td { border-bottom: 1px solid var(--line); padding: 7px 6px; text-align: left; vertical-align: top; }
th { color: var(--muted); font-size: 10px; text-transform: uppercase; letter-spacing: .08em; }
tr:last-child td { border-bottom: 0; }
pre { white-space: pre-wrap; overflow-wrap: anywhere; margin: 0; color: var(--muted); font-size: 11px; }
@media (max-width: 760px) {
  header { display: block; }
  .pill { display: inline-block; margin-top: 10px; }
  .grid { display: block; }
  .card { margin-bottom: 10px; }
}
</style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>SWE-bench King Benchmark</h1>
      <div class="sub" id="root">loading</div>
    </div>
    <div class="pill" id="updated">loading</div>
  </header>

  <section class="grid">
    <div class="card span-3"><div class="label">PM2</div><div class="value" id="pm2">--</div><div class="small" id="pm2-detail">--</div></div>
    <div class="card span-3"><div class="label">Job</div><div class="value" id="job">--</div><div class="small" id="job-detail">--</div></div>
    <div class="card span-3"><div class="label">King</div><div class="value" id="king-rate">--</div><div class="small" id="king-detail">--</div></div>
    <div class="card span-3"><div class="label">Pi</div><div class="value" id="pi-rate">--</div><div class="small" id="pi-detail">--</div></div>

    <div class="card span-8">
      <div class="label">King vs Pi</div>
      <div class="bars" style="margin-top:12px">
        <div><div class="bar-head"><span>Current king</span><span id="king-count">--</span></div><div class="track"><div class="fill king" id="king-bar"></div></div></div>
        <div><div class="bar-head"><span>Pi baseline</span><span id="pi-count">--</span></div><div class="track"><div class="fill pi" id="pi-bar"></div></div></div>
      </div>
    </div>
    <div class="card span-4"><div class="label">Run</div><div class="value" id="delta">--</div><div class="small" id="run-detail">--</div></div>

    <div class="card span-6"><div class="label">Progress</div><div id="progress"></div></div>
    <div class="card span-6"><div class="label">Usage</div><div id="usage"></div></div>
    <div class="card span-12"><div class="label">Recent Jobs</div><div id="jobs"></div></div>
    <div class="card span-12"><div class="label">Latest Error</div><pre id="error">--</pre></div>
  </section>
</main>
<script>
const $ = (id) => document.getElementById(id);
const pct = (v) => Number.isFinite(Number(v)) ? (Number(v) * 100).toFixed(1) + '%' : '--';
const num = (v) => Number.isFinite(Number(v)) ? Number(v).toLocaleString() : '--';
const ago = (iso) => {
  const t = Date.parse(iso || '');
  if (!Number.isFinite(t)) return '--';
  const s = Math.max(0, Math.floor((Date.now() - t) / 1000));
  if (s < 60) return s + 's ago';
  if (s < 3600) return Math.floor(s / 60) + 'm ago';
  if (s < 86400) return Math.floor(s / 3600) + 'h ago';
  return Math.floor(s / 86400) + 'd ago';
};
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const score = (x) => x || {};
const rateFrom = (s) => Number.isFinite(Number(s.pass_rate)) ? Number(s.pass_rate) : (Number(s.total_count) > 0 ? Number(s.resolved_count) / Number(s.total_count) : NaN);

function render(data) {
  $('root').textContent = data.root;
  $('updated').textContent = 'updated ' + new Date().toLocaleTimeString();
  const pm2 = data.pm2 || {};
  $('pm2').textContent = (pm2.status || 'unknown').toUpperCase();
  $('pm2').className = 'value ' + (pm2.status === 'online' ? 'ok' : pm2.status === 'stopped' ? 'warn' : 'bad');
  $('pm2-detail').textContent = 'restarts ' + (pm2.restarts ?? '--') + ' / pid ' + (pm2.pid || '--');

  const latest = data.latest || {};
  const job = latest.job || {};
  const comparison = latest.comparison || job.comparison || {};
  const scores = comparison.scores || {};
  const king = score(scores.king || latest.king_score);
  const pi = score(scores.pi || latest.pi_score);
  const kingRate = rateFrom(king);
  const piRate = rateFrom(pi);
  const delta = Number.isFinite(kingRate) && Number.isFinite(piRate) ? kingRate - piRate : scores.delta_pass_rate;

  $('job').textContent = String(job.status || latest.status || 'none').toUpperCase();
  $('job').className = 'value ' + (job.status === 'completed' ? 'ok' : job.status === 'failed' ? 'bad' : 'warn');
  $('job-detail').textContent = (latest.commit || '--').slice(0, 12) + ' / ' + ago(job.updated_at || job.finished_at || job.started_at);
  $('king-rate').textContent = pct(kingRate);
  $('king-detail').textContent = countText(king) + ' resolved';
  $('pi-rate').textContent = pct(piRate);
  $('pi-detail').textContent = countText(pi) + ' resolved';
  $('king-count').textContent = countText(king);
  $('pi-count').textContent = countText(pi);
  $('king-bar').style.width = barWidth(kingRate);
  $('pi-bar').style.width = barWidth(piRate);
  $('delta').textContent = Number.isFinite(Number(delta)) ? ((delta >= 0 ? '+' : '') + pct(delta)) : '--';
  $('delta').className = 'value ' + (Number(delta) >= 0 ? 'ok' : 'bad');
  $('run-detail').textContent = [
    comparison.model || '--',
    comparison.provider_only || '--',
    comparison.pi_baseline_cached ? 'pi cached' : 'pi uncached'
  ].join(' / ');
  $('progress').innerHTML = progressTable(data.progress || {});
  $('usage').innerHTML = usageTable(comparison.usage || data.usage || {});
  $('jobs').innerHTML = jobsTable(data.jobs || []);
  $('error').textContent = latest.error || job.error || '--';
}

function countText(s) {
  return Number.isFinite(Number(s.resolved_count)) && Number.isFinite(Number(s.total_count)) ? num(s.resolved_count) + '/' + num(s.total_count) : '--';
}
function barWidth(v) {
  return Number.isFinite(Number(v)) ? Math.max(0, Math.min(100, Number(v) * 100)).toFixed(1) + '%' : '0%';
}
function progressTable(progress) {
  const rows = ['king', 'pi'].map(name => {
    const p = progress[name] || {};
    return `<tr><td>${esc(name)}</td><td>${num(p.solve_results)}</td><td>${num(p.predictions)}</td><td>${num(p.errors)}</td><td>${esc(p.last_update || '--')}</td></tr>`;
  }).join('');
  return `<table><thead><tr><th>agent</th><th>solves</th><th>predictions</th><th>errors</th><th>last update</th></tr></thead><tbody>${rows}</tbody></table>`;
}
function usageTable(usage) {
  const rows = ['king', 'pi'].map(name => {
    const u = usage[name] || {};
    return `<tr><td>${esc(name)}</td><td>${num(u.request_count)}</td><td>${num(u.total_tokens)}</td><td>${u.cost == null ? 'n/a' : '$' + Number(u.cost).toFixed(4)}</td></tr>`;
  }).join('');
  return `<table><thead><tr><th>agent</th><th>requests</th><th>tokens</th><th>cost</th></tr></thead><tbody>${rows}</tbody></table>`;
}
function jobsTable(jobs) {
  if (!jobs.length) return '<div class="small">No jobs yet.</div>';
  const rows = jobs.map(j => `<tr><td>${esc((j.commit || '').slice(0, 12))}</td><td>${esc(j.status || '--')}</td><td>${esc(ago(j.updated_at || j.finished_at || j.started_at))}</td><td>${esc(j.error || '')}</td></tr>`).join('');
  return `<table><thead><tr><th>commit</th><th>status</th><th>updated</th><th>error</th></tr></thead><tbody>${rows}</tbody></table>`;
}
async function poll() {
  try {
    const res = await fetch('/api/status', { cache: 'no-store' });
    render(await res.json());
  } catch (err) {
    $('error').textContent = String(err);
  }
}
poll();
setInterval(poll, 3000);
</script>
</body>
</html>
"""


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def newest_mtime(paths: list[Path]) -> str | None:
    existing = [path for path in paths if path.exists()]
    if not existing:
        return None
    newest = max(path.stat().st_mtime for path in existing)
    return f"{newest:.0f}"


def agent_progress(job_dir: Path, agent: str) -> dict[str, Any]:
    agent_dir = job_dir / agent
    solve_path = agent_dir / "solve_results.jsonl"
    prediction_path = agent_dir / "predictions.jsonl"
    records = []
    if solve_path.exists():
        records = [
            json.loads(line)
            for line in solve_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    return {
        "solve_results": len(records),
        "predictions": read_jsonl_count(prediction_path),
        "errors": sum(1 for record in records if record.get("error")),
        "last_update": newest_mtime([solve_path, prediction_path, agent_dir / "usage_summary.json"]),
    }


def score_from_report(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = read_json(path)
    scores = payload.get("scores") if isinstance(payload, dict) else None
    return scores if isinstance(scores, dict) else None


def job_summary(job_dir: Path) -> dict[str, Any]:
    job_path = job_dir / "job.json"
    job = read_json(job_path) if job_path.exists() else {}
    comparison_path = job_dir / "comparison.json"
    comparison = read_json(comparison_path) if comparison_path.exists() else None
    scores = score_from_report(comparison_path) or {}
    return {
        "commit": job_dir.name,
        "status": job.get("status") or (comparison or {}).get("status"),
        "started_at": job.get("started_at") or (comparison or {}).get("started_at"),
        "updated_at": job.get("updated_at"),
        "finished_at": job.get("finished_at") or (comparison or {}).get("finished_at"),
        "error": job.get("error"),
        "job": job,
        "comparison": comparison,
        "king_score": scores.get("king"),
        "pi_score": scores.get("pi"),
    }


def job_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        [path for path in root.iterdir() if path.is_dir() and not path.name.startswith("_")],
        key=lambda path: (path / "job.json").stat().st_mtime if (path / "job.json").exists() else path.stat().st_mtime,
        reverse=True,
    )


def pm2_status(name: str) -> dict[str, Any]:
    try:
        result = subprocess.run(["pm2", "jlist"], capture_output=True, text=True, timeout=5, check=False)
        processes = json.loads(result.stdout or "[]")
    except Exception as exc:  # noqa: BLE001
        return {"status": "unknown", "error": repr(exc)}
    for process in processes:
        if process.get("name") == name:
            env = process.get("pm2_env") or {}
            return {
                "status": env.get("status"),
                "restarts": env.get("restart_time"),
                "pid": process.get("pid"),
                "uptime": env.get("pm_uptime"),
            }
    return {"status": "missing"}


def dashboard_payload(root: Path, process_name: str) -> dict[str, Any]:
    dirs = job_dirs(root)
    latest = job_summary(dirs[0]) if dirs else {}
    latest_dir = dirs[0] if dirs else root / "_missing"
    comparison = latest.get("comparison") or {}
    return {
        "root": str(root.resolve()),
        "pm2": pm2_status(process_name),
        "latest": latest,
        "jobs": [job_summary(path) for path in dirs[:12]],
        "progress": {
            "king": agent_progress(latest_dir, "king"),
            "pi": agent_progress(latest_dir, "pi"),
        },
        "usage": comparison.get("usage") if isinstance(comparison, dict) else {},
    }


def make_handler(root: Path, process_name: str) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_HEAD(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path in {"/", "/index.html"}:
                body = HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("content-type", "text/html; charset=utf-8")
                self.send_header("cache-control", "no-store")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                return
            if path == "/api/status":
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("cache-control", "no-store")
                self.end_headers()
                return
            self.send_error(404)

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == "/api/status":
                body = json.dumps(dashboard_payload(root, process_name), sort_keys=True).encode("utf-8")
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("cache-control", "no-store")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if path in {"/", "/index.html"}:
                body = HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("content-type", "text/html; charset=utf-8")
                self.send_header("cache-control", "no-store")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_error(404)

        def log_message(self, format: str, *args: Any) -> None:
            return

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--process-name", default="swebench-king-benchmark")
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(args.root, args.process_name))
    print(f"local SWE-bench dashboard: http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
