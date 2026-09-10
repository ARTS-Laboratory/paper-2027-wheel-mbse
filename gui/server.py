"""The GUI server: stdlib only, localhost only, and deliberately dumb.

IT HOLDS NO STATE.  Every answer is computed from the repo and from `gui/runs/` at the
moment it is asked, which is what lets you stop this process mid-descent and start another
one that knows exactly as much.  There is no job table in memory, no websocket, no
subscription -- the client polls, and the polling is cheap because `progress.py` caches on
mtime.

STDLIB ONLY IS THE POINT, NOT A LIMITATION.  This tree's two virtualenvs are argued for at
length in requirements-cad.txt and requirements-opt.txt, `tests/test_import_hygiene.py`
enforces what may be imported where, and `pyproject.toml` says outright that the project is
two scripts run by two interpreters rather than an installable package.  A GUI that added
a web framework would have to be argued into that, and would stop being deletable.  It
runs under `.venv-opt/bin/python` because that env can already import every module the
panels read.

BOUND TO 127.0.0.1, WITH NO AUTHENTICATION AND THEREFORE NO REMOTE BIND.  This endpoint
starts processes.  Anyone who can reach it can start a 50-hour job or read any genome on
disk, so it is reachable only from this machine; use an ssh tunnel rather than changing the
bind address.
"""

import json
import mimetypes
import os
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import catalog                                                    # noqa: E402
import jobs as J                                                  # noqa: E402
import progress                                                   # noqa: E402

STATIC = os.path.join(HERE, "static")

# `model` drags jax in (through wheel_objective's weight table) and costs a couple of
# seconds and ~200 MB. Deferred so the server answers on the port immediately and only
# pays when someone actually opens the MBSE panel.
_model = None
_model_lock = threading.Lock()


def model():
    global _model
    with _model_lock:
        if _model is None:
            import model as m
            _model = m
    return _model


_preview = None


def preview():
    global _preview
    if _preview is None:
        import preview as p
        _preview = p
    return _preview


def _targets():
    return [{"key": t.key, "label": t.label, "group": t.group, "blurb": t.blurb,
             "heavy": t.heavy, "progress": t.progress,
             "params": [{"name": p.name, "kind": p.kind, "default": p.default,
                         "choices": list(p.choices), "help": p.help}
                        for p in t.params]}
            for t in catalog.TARGETS]


class Handler(BaseHTTPRequestHandler):
    server_version = "wheelgui"

    def log_message(self, fmt, *args):
        pass                      # the terminal that started this is not a request log

    # -- plumbing -----------------------------------------------------------
    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        return json.loads(self.rfile.read(n) or b"{}")

    def _static(self, path):
        name = os.path.basename(path) or "index.html"
        full = os.path.join(STATIC, name)
        if not os.path.isfile(full) or os.path.dirname(os.path.abspath(full)) != STATIC:
            return self._send(404, {"error": "not found"})
        ctype = mimetypes.guess_type(full)[0] or "application/octet-stream"
        with open(full, "rb") as fh:
            self._send(200, fh.read(), ctype)

    # -- routes -------------------------------------------------------------
    def do_GET(self):
        url = urlparse(self.path)
        q = parse_qs(url.query)
        try:
            if url.path in ("/", "/index.html"):
                return self._static("index.html")
            if url.path.startswith("/static/"):
                return self._static(url.path)
            if url.path == "/api/targets":
                return self._send(200, _targets())
            if url.path == "/api/jobs":
                out = []
                for job in J.list_jobs():
                    row = {k: job.get(k) for k in
                           ("id", "key", "label", "group", "state", "heavy",
                            "started_at", "ended_at", "exit_code", "estimate")}
                    row["progress"] = progress.read(job)
                    row["progress"].pop("series", None)     # the list view does not plot
                    out.append(row)
                return self._send(200, out)
            if url.path.startswith("/api/jobs/"):
                job_id = url.path.split("/")[3]
                job = J.get(job_id)
                if job is None:
                    return self._send(404, {"error": "no such job"})
                job = dict(job)
                job["progress"] = progress.read(job)
                job["log"] = progress.tail(job.get("log") or "", 24000)
                return self._send(200, job)
            if url.path == "/api/status":
                return self._send(200, model().status())
            if url.path == "/api/mbse/form":
                m = model()
                return self._send(200, {"form": m.mission_form(),
                                        "calibrated": m.calibrated_points()})
            if url.path == "/api/genome":
                p = preview()
                path = (q.get("path") or [None])[0]
                return self._send(200, {"genes": p.load_genes(path),
                                        "bounds": p.bounds(),
                                        "constants": p.constants()})
            if url.path == "/api/machine":
                return self._send(200, {"available_gib": round(J.available_gib(), 1),
                                        "total_gib": round(J._total_gib(), 1),
                                        "systemd": J.HAVE_SYSTEMD,
                                        "detach": J.DETACH,
                                        "platform": sys.platform})
            return self._send(404, {"error": "not found"})
        except Exception as exc:
            traceback.print_exc()
            return self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

    def do_POST(self):
        url = urlparse(self.path)
        try:
            body = self._body()
            if url.path == "/api/plan":
                return self._send(200, J.plan(body["key"], body.get("params") or {}))
            if url.path == "/api/jobs":
                job = J.launch(body["key"], body.get("params") or {},
                               force=bool(body.get("force")))
                return self._send(200, job)
            if url.path.startswith("/api/jobs/") and url.path.endswith("/cancel"):
                return self._send(200, J.cancel(url.path.split("/")[3]))
            if url.path.startswith("/api/jobs/") and url.path.endswith("/forget"):
                J.forget(url.path.split("/")[3])
                return self._send(200, {"ok": True})
            if url.path == "/api/mbse/derive":
                return self._send(200, model().derive(body.get("mission") or {},
                                                      body.get("points")))
            if url.path == "/api/mbse/compliance":
                return self._send(200, model().compliance(
                    body.get("record"), body.get("mission"), body.get("points")))
            if url.path == "/api/mbse/save":
                path = body.get("path") or os.path.join(catalog.ROOT, "requirements.json")
                return self._send(200, model().save_requirements(
                    body.get("mission") or {}, body.get("points"), path))
            if url.path == "/api/preview":
                p = preview()
                return self._send(200, p.outline(body["genes"]))
            return self._send(404, {"error": "not found"})
        except (KeyError, ValueError, RuntimeError) as exc:
            # A refused launch, a 105-point allocation, a mission above the card's
            # service ceiling -- all of these are the model saying no, and the message
            # it says it with is the useful part. 400, with the text, not a 500.
            return self._send(400, {"error": str(exc)})
        except Exception as exc:
            traceback.print_exc()
            return self._send(500, {"error": f"{type(exc).__name__}: {exc}"})


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=8731,
                    help="0 asks the OS for a free one, which is what the desktop shell "
                         "passes: two windows on one box must not fight over 8731")
    ap.add_argument("--host", default="127.0.0.1",
                    help="localhost by default and you should leave it there: this "
                         "endpoint starts processes and has no authentication")
    ap.add_argument("--open", action="store_true", help="open a browser at it")
    args = ap.parse_args(argv)

    os.makedirs(J.RUNS, exist_ok=True)
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    # The bound port, NOT the requested one -- with `--port 0` they differ, and the
    # requested one is the useless of the two.
    port = httpd.server_address[1]
    url = f"http://{args.host}:{port}/"
    print(f"  wheel gui   {url}")
    print(f"  runs        {J.RUNS}")
    print(f"  detach      {J.DETACH}   -- jobs keep running when this process stops")
    print("  ctrl-c to stop the server (running jobs are unaffected)")
    # A LINE FOR A PROGRAM RATHER THAN A PERSON, and the desktop shell blocks on it.
    # It has to be flushed by hand: stdout is a pipe there, so it is block-buffered and
    # a shell waiting on this line would wait for the buffer to fill, which is never.
    print(f"WHEELGUI_READY {url}", flush=True)
    if args.open:
        import webbrowser
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  server stopped. running jobs continue; restart to reattach.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
