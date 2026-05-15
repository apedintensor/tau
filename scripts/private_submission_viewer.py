#!/usr/bin/env python3
from __future__ import annotations

import argparse
import difflib
import html
import json
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


@dataclass(frozen=True)
class ViewerConfig:
    root: Path
    base_agent: Path


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(read_text(path))
    except Exception:
        return {}


def submission_dirs(root: Path) -> list[Path]:
    return sorted(
        [path for path in root.iterdir() if path.is_dir() and (path / "agent.py").exists()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def accepted_index(root: Path) -> dict[str, dict[str, Any]]:
    payload = read_json(root / "_accepted_submissions.json")
    rows = payload if isinstance(payload, list) else payload.get("submissions", [])
    return {
        str(row.get("submission_id") or ""): row
        for row in rows
        if isinstance(row, dict) and row.get("submission_id")
    }


def check_summary(check_result: dict[str, Any]) -> dict[str, Any]:
    checks = check_result.get("checks") or check_result.get("ci_checks") or {}
    judge = checks.get("openrouter_judge") or check_result.get("llm_judge") or {}
    registration = checks.get("registration_gate") or {}
    return {
        "accepted": bool(check_result.get("accepted")),
        "agent_sha256": check_result.get("agent_sha256"),
        "judge_status": judge.get("status"),
        "judge_score": judge.get("score"),
        "judge_summary": judge.get("summary"),
        "registration_block": (registration.get("metadata") or {}).get("registration_block"),
    }


def submission_summary(path: Path, accepted: dict[str, dict[str, Any]]) -> dict[str, Any]:
    sid = path.name
    check = read_json(path / "check_result.json")
    indexed = accepted.get(sid, {})
    agent = path / "agent.py"
    return {
        "submission_id": sid,
        "hotkey": indexed.get("hotkey") or check.get("hotkey"),
        "accepted_at": indexed.get("accepted_at"),
        "size_bytes": agent.stat().st_size,
        "line_count": len(read_text(agent).splitlines()),
        **check_summary(check),
    }


def list_submissions(config: ViewerConfig) -> list[dict[str, Any]]:
    accepted = accepted_index(config.root)
    return [submission_summary(path, accepted) for path in submission_dirs(config.root)]


def safe_submission_dir(root: Path, submission_id: str) -> Path | None:
    if "/" in submission_id or "\\" in submission_id or not submission_id:
        return None
    path = (root / submission_id).resolve()
    root_resolved = root.resolve()
    if root_resolved not in path.parents or not path.is_dir():
        return None
    return path


def unified_diff(base: str, submitted: str) -> str:
    return "".join(
        difflib.unified_diff(
            base.splitlines(keepends=True),
            submitted.splitlines(keepends=True),
            fromfile="base/agent.py",
            tofile="submitted/agent.py",
        )
    )


def response_bytes(payload: Any) -> bytes:
    return json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")


def page_html() -> bytes:
    return r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Private Submission Viewer</title>
  <style>
    :root { color-scheme: dark; --bg:#101214; --panel:#171a1d; --line:#2a3036; --text:#e8edf2; --muted:#8d98a5; --good:#41d17d; --warn:#f5bd4f; --bad:#ff6b6b; --add-bg:#11381f; --add-text:#b9f7cc; --del-bg:#40191d; --del-text:#ffc1c8; --hunk-bg:#182331; --file-bg:#221b31; }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif; background: var(--bg); color: var(--text); }
    header { height: 56px; display:flex; align-items:center; justify-content:space-between; padding:0 18px; border-bottom:1px solid var(--line); background:#0c0e10; position:sticky; top:0; z-index:2; }
    h1 { font-size: 16px; margin: 0; font-weight: 700; }
    main { display:grid; grid-template-columns: 390px 1fr; min-height: calc(100vh - 56px); }
    aside { border-right:1px solid var(--line); overflow:auto; max-height:calc(100vh - 56px); }
    button { background:transparent; color:inherit; border:0; font:inherit; text-align:left; cursor:pointer; }
    .item { display:block; width:100%; padding:14px 16px; border-bottom:1px solid var(--line); }
    .item:hover, .item.active { background:#20252a; }
    .row { display:flex; gap:10px; align-items:center; justify-content:space-between; }
    .sid { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size:12px; color:#cfd8e3; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .hotkey { margin-top:6px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; color:var(--muted); font-size:12px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .score { color:var(--good); font-weight:700; }
    .meta { margin-top:8px; color:var(--muted); font-size:12px; display:flex; gap:12px; }
    section { min-width:0; }
    .toolbar { display:flex; gap:8px; align-items:center; padding:12px 14px; border-bottom:1px solid var(--line); background:#121518; position:sticky; top:56px; z-index:1; }
    .tab { padding:8px 10px; border:1px solid var(--line); border-radius:6px; }
    .tab.active { background:#25303a; border-color:#435363; }
    .summary { padding:14px; border-bottom:1px solid var(--line); color:var(--muted); }
    .content { overflow:auto; max-height:calc(100vh - 174px); }
    pre { margin:0; padding:16px; font:12px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace; white-space:pre; }
    .diff { font:12px/1.45 ui-monospace, SFMono-Regular, Menlo, monospace; min-width:max-content; }
    .diff-row { display:grid; grid-template-columns: 58px 58px minmax(720px, 1fr); border-bottom:1px solid rgba(255,255,255,0.035); }
    .diff-row:hover { filter:brightness(1.14); }
    .ln { color:#697584; text-align:right; padding:0 10px; user-select:none; border-right:1px solid rgba(255,255,255,0.05); }
    .code { white-space:pre; padding:0 12px; tab-size:2; }
    .file { background:var(--file-bg); color:#e5d5ff; font-weight:700; position:sticky; top:0; z-index:1; }
    .hunk { background:var(--hunk-bg); color:#9fc8ff; position:sticky; top:22px; z-index:1; }
    .add { background:var(--add-bg); color:var(--add-text); }
    .del { background:var(--del-bg); color:var(--del-text); }
    .ctx { background:#111417; color:#d7dde5; }
    .meta-line { background:#171a1d; color:#8793a1; }
    .empty { color:var(--muted); padding:28px; }
  </style>
</head>
<body>
  <header><h1>Private Submission Viewer</h1><div id="count"></div></header>
  <main><aside id="list"></aside><section><div class="toolbar"><button class="tab active" data-view="diff">Diff</button><button class="tab" data-view="code">Code</button><button class="tab" data-view="checks">Checks</button></div><div id="summary" class="summary">Select a submission.</div><div id="content" class="content"></div></section></main>
  <script>
    let submissions = [], selected = null, view = 'diff';
    const $ = (id) => document.getElementById(id);
    const esc = (s) => String(s ?? '').replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
    async function getJson(url) { const r = await fetch(url); if (!r.ok) throw new Error(url + ' ' + r.status); return r.json(); }
    async function getText(url) { const r = await fetch(url); if (!r.ok) throw new Error(url + ' ' + r.status); return r.text(); }
    function parseHunk(line) {
      const match = /^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@/.exec(line);
      return match ? { oldLine: Number(match[1]), newLine: Number(match[2]) } : null;
    }
    function renderUnifiedDiff(text) {
      let oldLine = '', newLine = '';
      const rows = [];
      for (const raw of text.split('\\n')) {
        let cls = 'ctx', left = '', right = '', code = raw;
        if (raw.startsWith('--- ') || raw.startsWith('+++ ') || raw.startsWith('diff --git ')) {
          cls = 'file'; code = raw;
        } else if (raw.startsWith('@@')) {
          const hunk = parseHunk(raw);
          if (hunk) { oldLine = hunk.oldLine; newLine = hunk.newLine; }
          cls = 'hunk'; code = raw;
        } else if (raw.startsWith('+')) {
          cls = 'add'; right = newLine === '' ? '' : newLine++; code = raw.slice(1);
        } else if (raw.startsWith('-')) {
          cls = 'del'; left = oldLine === '' ? '' : oldLine++; code = raw.slice(1);
        } else if (raw.startsWith('\\\\')) {
          cls = 'meta-line'; code = raw;
        } else {
          cls = 'ctx';
          left = oldLine === '' ? '' : oldLine++;
          right = newLine === '' ? '' : newLine++;
          code = raw.startsWith(' ') ? raw.slice(1) : raw;
        }
        rows.push('<div class="diff-row ' + cls + '"><span class="ln">' + esc(left) + '</span><span class="ln">' + esc(right) + '</span><span class="code">' + esc(code) + '</span></div>');
      }
      return '<div class="diff">' + rows.join('') + '</div>';
    }
    function renderPlain(text) {
      return '<pre>' + esc(text) + '</pre>';
    }
    function renderList() {
      $('count').textContent = submissions.length + ' submissions';
      $('list').innerHTML = submissions.map(s => '<button class="item ' + (selected === s.submission_id ? 'active' : '') + '" data-id="' + esc(s.submission_id) + '"><div class="row"><span class="sid">' + esc(s.submission_id) + '</span><span class="score">' + esc(s.judge_score ?? '--') + '</span></div><div class="hotkey">' + esc(s.hotkey || 'unknown hotkey') + '</div><div class="meta"><span>' + esc(s.line_count) + ' lines</span><span>' + esc(s.size_bytes) + ' bytes</span></div></button>').join('');
      document.querySelectorAll('.item').forEach(b => b.onclick = () => select(b.dataset.id));
    }
    function renderSummary(s) {
      $('summary').innerHTML = '<b>' + esc(s.submission_id) + '</b><br>' + esc(s.hotkey || 'unknown hotkey') + '<br>Judge: ' + esc(s.judge_status) + ' / ' + esc(s.judge_score) + ' | accepted: ' + esc(s.accepted) + '<br>' + esc(s.judge_summary || '');
    }
    async function renderContent() {
      if (!selected) return;
      document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.view === view));
      const url = '/api/submissions/' + encodeURIComponent(selected) + '/' + view;
      const content = view === 'checks' ? JSON.stringify(await getJson(url), null, 2) : await getText(url);
      $('content').innerHTML = view === 'diff' ? renderUnifiedDiff(content) : renderPlain(content);
    }
    async function select(id) {
      selected = id;
      renderList();
      renderSummary(submissions.find(s => s.submission_id === id));
      await renderContent();
    }
    document.querySelectorAll('.tab').forEach(t => t.onclick = async () => { view = t.dataset.view; await renderContent(); });
    (async function init() {
      submissions = await getJson('/api/submissions');
      renderList();
      if (submissions[0]) await select(submissions[0].submission_id);
    })().catch(e => { $('content').innerHTML = renderPlain(e.stack || String(e)); });
  </script>
</body>
</html>""".encode("utf-8")


def build_handler(config: ViewerConfig):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.client_address and self.client_address[0] not in {"127.0.0.1", "::1"}:
                self.send_error(403)
                return
            parsed = urlparse(self.path)
            try:
                status, body, content_type = route(parsed.path, config)
            except Exception as exc:
                status, body, content_type = 500, response_bytes({"error": str(exc)}), "application/json"
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            return

    return Handler


def route(path: str, config: ViewerConfig) -> tuple[int, bytes, str]:
    if path in {"", "/"}:
        return 200, page_html(), "text/html; charset=utf-8"
    if path == "/api/submissions":
        return 200, response_bytes(list_submissions(config)), "application/json"
    parts = [unquote(part) for part in path.split("/") if part]
    if len(parts) == 4 and parts[:2] == ["api", "submissions"]:
        sub_dir = safe_submission_dir(config.root, parts[2])
        if sub_dir is None:
            return 404, response_bytes({"error": "not found"}), "application/json"
        kind = parts[3]
        if kind == "code":
            return 200, read_text(sub_dir / "agent.py").encode("utf-8"), "text/plain; charset=utf-8"
        if kind == "checks":
            return 200, response_bytes(read_json(sub_dir / "check_result.json")), "application/json"
        if kind == "diff":
            return (
                200,
                unified_diff(read_text(config.base_agent), read_text(sub_dir / "agent.py")).encode("utf-8"),
                "text/plain; charset=utf-8",
            )
    return 404, response_bytes({"error": "not found"}), "application/json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8077)
    parser.add_argument("--root", type=Path, default=Path("workspace/validate/netuid-66/private-submissions"))
    parser.add_argument("--base-agent", type=Path, default=Path("/home/const/subnet66/ninja/agent.py"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ViewerConfig(root=args.root.expanduser().resolve(), base_agent=args.base_agent.expanduser().resolve())
    server = ThreadingHTTPServer((args.host, args.port), build_handler(config))
    print(f"Serving private submission viewer on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
