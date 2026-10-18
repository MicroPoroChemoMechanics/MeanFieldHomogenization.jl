"""The HTTP server.

Deliberately built on `http.server` from the standard library rather than a
framework: the interface must start with `python3 -m mfhstudio` on any machine
that already runs MeanFieldHomogenization, and adding a dependency tree to achieve that
would be a poor trade. The API is small — a handful of JSON endpoints and the
static files.
"""

from __future__ import annotations

import hmac
import json
import os
import secrets
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

from . import catalog as catalog_module
from . import convert as convert_module
from .codegen import cell_expression, extract_embedded, generate
from .juliabridge import PROJECT_ROOT, Bridge, SidecarError, SidecarUnavailable
from .model import Model, default_model
from .preflight import check_project
from .readback import model_from_script

WEB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "web"))
EXAMPLES_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "examples")
)

_MIME = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json",
    ".svg": "image/svg+xml",
    ".map": "application/json",
}


class Session:
    """Everything one running studio holds.

    The model lives here, in Python, which is what makes the sidecar
    disposable: it can be restarted after a wedge without losing work.
    """

    def __init__(self, project: Optional[str] = None, julia: Optional[str] = None) -> None:
        # The page must present this token with every /api call: the studio
        # runs Julia on request, so no other web page the user has open, and
        # no other user of the machine, may talk to it.
        self.token = secrets.token_urlsafe(24)
        self.host = "127.0.0.1"
        self.port = 0
        self.model: Model = default_model()
        self.path: Optional[str] = None
        self.bridge = Bridge(julia=julia, project=project or PROJECT_ROOT)
        self.lock = threading.Lock()
        self._catalog: Optional[dict] = None
        self.catalog_error: Optional[str] = None

    def catalog(self) -> dict:
        """The form definitions, upgraded with live facts once Julia answers.

        This never raises. The interface has to be usable while MeanFieldHomogenization
        is still loading — otherwise every control is dead and the only clue
        is a `TypeError` in the browser console.
        """
        if self._catalog is None:
            introspected = None
            try:
                introspected = self.bridge.catalog()
            except Exception as exc:  # noqa: BLE001
                self.catalog_error = str(exc)
            else:
                self.catalog_error = None
            cat = catalog_module.merge(introspected)
            if not cat["introspected"]:
                # Not cached: retry on the next request, so the interface
                # upgrades itself as soon as the sidecar is ready.
                return cat
            self._catalog = cat
        return self._catalog

    def script(self) -> str:
        return generate(self.model)


class Handler(BaseHTTPRequestHandler):
    session: Session  # injected by `serve`

    server_version = "MFHStudio"

    def log_message(self, fmt: str, *args) -> None:  # quieter console
        return

    # -- helpers ----------------------------------------------------------

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except BrokenPipeError:
            pass

    def _json(self, obj: Any, code: int = 200) -> None:
        self._send(code, json.dumps(obj).encode("utf-8"), "application/json")

    def _error(self, exc: Exception, code: int = 500) -> None:
        # The interface shows these verbatim: a precise message beats a
        # generic failure the user cannot act on.
        payload = {"error": str(exc), "kind": type(exc).__name__}
        if not isinstance(exc, (SidecarError, SidecarUnavailable, ValueError)):
            payload["traceback"] = traceback.format_exc()
        self._json(payload, code)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0:
            return {}
        return json.loads(self.rfile.read(n).decode("utf-8"))

    def _refusal(self, method: str) -> Optional[str]:
        """Why an /api request is refused, or None.

        A missing or wrong token; a Host that is not the studio's (a page using
        DNS rebinding); an Origin that is not the studio's page; a POST that is
        not JSON, which a form on another page could send without the browser
        asking first.
        """
        s = self.session
        token = self.headers.get("X-Studio-Token") or ""
        if not hmac.compare_digest(token.encode(), s.token.encode()):
            return "missing or wrong token"
        host = self.headers.get("Host") or ""
        if s.host in ("127.0.0.1", "localhost", "::1"):
            allowed = {f"127.0.0.1:{s.port}", f"localhost:{s.port}", f"[::1]:{s.port}"}
            if host not in allowed:
                return "unexpected Host header"
        origin = self.headers.get("Origin")
        if origin and origin != f"http://{host}":
            return "cross-origin request refused"
        if method == "POST":
            ctype = (self.headers.get("Content-Type") or "").lower()
            if not ctype.startswith("application/json"):
                return "a POST must carry application/json"
        return None

    # -- routing ----------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path.startswith("/api/"):
            refusal = self._refusal("GET")
            if refusal:
                return self._json({"error": refusal}, 403)
        try:
            if path == "/api/state":
                return self._json(self._state())
            if path == "/api/catalog":
                return self._json(self.session.catalog())
            if path == "/api/script":
                return self._json({"source": self.session.script()})
            if path == "/api/browse":
                return self._browse(parse_qs(urlparse(self.path).query))
            if path == "/api/convert/available":
                return self._json({"available": convert_module.available()})
            if path == "/api/sidecar":
                st = dict(self.session.bridge.status)
                st["catalog_error"] = self.session.catalog_error
                st["introspected"] = bool(
                    self.session._catalog and self.session._catalog.get("introspected")
                )
                return self._json(st)
            return self._static(path)
        except Exception as exc:  # noqa: BLE001
            return self._error(exc)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        refusal = self._refusal("POST")
        if refusal:
            return self._json({"error": refusal}, 403)
        try:
            body = self._body()
            if path == "/api/model":
                return self._set_model(body)
            if path == "/api/traces":
                return self._traces(body)
            if path == "/api/run":
                return self._run(body)
            if path == "/api/open":
                return self._open(body)
            if path == "/api/save":
                return self._save(body)
            if path == "/api/sidecar/restart":
                self.session.bridge.restart()
                self.session._catalog = None
                return self._json(self.session.bridge.status)
            return self._json({"error": f"unknown endpoint {path}"}, 404)
        except Exception as exc:  # noqa: BLE001
            return self._error(exc)

    # -- endpoints --------------------------------------------------------

    def _state(self) -> dict:
        s = self.session
        return {
            "model": s.model.to_dict(),
            "script": s.script(),
            "problems": s.model.validate(),
            "path": s.path,
            "multiscale": s.model.uses_multiscale(),
        }

    def _set_model(self, body: dict) -> None:
        with self.session.lock:
            model = Model.from_dict(body.get("model", {}))
            self._prune_scheme_options(model)
            self.session.model = model
        return self._json(self._state())

    def _prune_scheme_options(self, model: Model) -> None:
        """Drop solver options the chosen scheme does not read.

        Switching from `SelfConsistent` to `MoriTanaka` used to carry the old
        scheme's options along, and `MoriTanaka(; verbose = false)` is a
        MethodError — the singleton schemes take no keywords at all. Doing this
        on the model rather than in the browser also cleans a model reopened
        from a file that carries stale options.
        """
        cat = self.session.catalog()
        allowed = {
            s["name"]: {o["name"] for o in s.get("options", []) if o.get("editable")}
            for s in cat.get("schemes", [])
        }
        if not allowed:
            return  # the sidecar has not answered yet; leave the model alone

        def clean(name: str, options: dict) -> dict:
            keep = allowed.get(name)
            if keep is None:
                return options  # unknown scheme: not ours to second-guess
            return {k: v for k, v in (options or {}).items() if k in keep}

        for entry in model.sweep.schemes:
            entry["options"] = clean(entry.get("name", ""), entry.get("options"))
        for c in model.cells:
            for ph in c.phases:
                for pr in ph.properties:
                    if pr.source == "cell" and pr.scheme:
                        pr.scheme_options = clean(pr.scheme, pr.scheme_options)

    def _traces(self, body: dict) -> None:
        expr = body.get("expr") or ""
        # A laminate has no per-member shape — the geometry of the cell *is*
        # the stack — so the browser names the cell and the expression is built
        # here, by the code generator. Spelling it out a second time in
        # JavaScript would give the picture its own idea of the model.
        cid = body.get("cell")
        if not expr and cid:
            c = self.session.model.cell(cid)
            # A parametrized builder has no single value to draw.
            if c is not None and not c.params:
                expr = cell_expression(self.session.model, c)
        if not expr:
            return self._json({"data": [], "layout": {}})
        kw = {}
        if "cutaway" in body:
            kw["cutaway"] = bool(body["cutaway"])
        return self._json(self.session.bridge.traces(expr, **kw))

    def _run(self, body: dict) -> None:
        source = body.get("source") or self.session.script()
        timeout = float(body.get("timeout", 300.0))
        result = self.session.bridge.run(source, timeout=timeout)
        # A wedged sidecar cannot be trusted for the next request.
        if result.get("wedged"):
            try:
                self.session.bridge.restart()
                self.session._catalog = None
            except Exception:
                pass
        return self._json(result)

    def _browse(self, query: dict) -> None:
        """List a directory for the file picker.

        The browser cannot give a real path from `<input type="file">`, and a
        path is exactly what saving back to the same file needs — so the
        picker walks the server's filesystem instead. That also makes it work
        when the studio runs on a remote machine, which the local file dialog
        never would.
        """
        raw = (query.get("path") or [""])[0]
        start = os.path.abspath(os.path.expanduser(raw)) if raw else self._default_dir()
        if os.path.isfile(start):
            start = os.path.dirname(start)
        if not os.path.isdir(start):
            start = self._default_dir()

        dirs, files = [], []
        try:
            for name in sorted(os.listdir(start), key=str.lower):
                if name.startswith("."):
                    continue
                full = os.path.join(start, name)
                if os.path.isdir(full):
                    dirs.append({"name": name, "path": full})
                elif name.endswith((".jl", ".py")):
                    files.append({
                        "name": name, "path": full,
                        "size": os.path.getsize(full),
                        # An Echoes script is opened by translating it; saying
                        # so in the listing beats a surprise afterwards.
                        "echoes": name.endswith(".py"),
                    })
        except PermissionError as exc:
            raise ValueError(f"cannot read {start}: {exc}") from exc

        parent = os.path.dirname(start)
        return self._json({
            "path": start,
            "parent": parent if parent and parent != start else None,
            "dirs": dirs,
            "files": files,
            "places": self._places(),
            "sep": os.sep,
        })

    def _default_dir(self) -> str:
        if self.session.path:
            return os.path.dirname(os.path.abspath(self.session.path))
        # The examples are where somebody opening this for the first time
        # should land: they are the only files that come back as a filled-in
        # model rather than as preserved text.
        for p in (EXAMPLES_DIR, os.path.join(PROJECT_ROOT, "scripts")):
            if os.path.isdir(p):
                return p
        return os.getcwd()

    @staticmethod
    def _places() -> list:
        """Shortcuts worth one click, and only the ones that exist."""
        out = []
        for label, p in (
            ("examples", EXAMPLES_DIR),
            ("scripts", os.path.join(PROJECT_ROOT, "scripts")),
            ("package", PROJECT_ROOT),
            ("home", os.path.expanduser("~")),
            ("cwd", os.getcwd()),
        ):
            if os.path.isdir(p):
                out.append({"label": label, "path": os.path.abspath(p)})
        return out

    def _open(self, body: dict) -> None:
        path = body.get("path") or ""
        source = body.get("source")

        # An Echoes script is not a thing the studio can read — it is a thing
        # it can translate. Same button, one extra question.
        if convert_module.is_echoes_script(path):
            return self._convert_and_open(body, path)

        if source is None:
            if not os.path.isfile(path):
                raise ValueError(f"no such file: {path}")
            with open(path, encoding="utf-8") as fh:
                source = fh.read()

        model, report = model_from_script(source, self.session.bridge)
        if "error" in report:
            # Nothing was read: the model and the file bound stay as they were,
            # so that a Save cannot write an empty model over the script.
            return self._json({"error": report["error"], "read_report": report}, 422)
        with self.session.lock:
            self.session.model = model
            self.session.path = path or None
        out = self._state()
        out["read_report"] = report
        return self._json(out)

    def _convert_and_open(self, body: dict, path: str) -> None:
        """Translate an Echoes script, write the Julia, then open that."""
        if not os.path.isfile(path):
            raise ValueError(f"no such file: {path}")
        conv = convert_module.convert(path)

        target = body.get("output") or convert_module.suggest_output(path)
        os.makedirs(os.path.dirname(os.path.abspath(target)) or ".", exist_ok=True)
        with open(target, "w", encoding="utf-8") as fh:
            fh.write(conv.julia)

        model, report = model_from_script(conv.julia, self.session.bridge)
        if "error" in report:
            # The translation is written; reading it back needs the sidecar.
            return self._json({"error": report["error"], "read_report": report, "output": target}, 422)
        with self.session.lock:
            self.session.model = model
            self.session.path = target

        out = self._state()
        out["read_report"] = report
        out["conversion"] = {
            "source": path,
            "output": target,
            "summary": conv.summary(),
            "findings": conv.findings,
            "blocking": conv.blocking,
        }
        return self._json(out)

    def _save(self, body: dict) -> None:
        path = body.get("path") or self.session.path
        if not path:
            raise ValueError("no path given")
        source = body.get("source") or self.session.script()
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(source)
        with self.session.lock:
            self.session.path = path
        return self._json({"path": path, "bytes": len(source)})

    # -- static files -----------------------------------------------------

    def _static(self, path: str) -> None:
        rel = "index.html" if path in ("/", "") else path.lstrip("/")
        full = os.path.abspath(os.path.join(WEB_DIR, rel))
        # Never serve outside the web directory.
        if not full.startswith(WEB_DIR) or not os.path.isfile(full):
            return self._send(404, b"not found", "text/plain; charset=utf-8")
        ext = os.path.splitext(full)[1]
        with open(full, "rb") as fh:
            data = fh.read()
        return self._send(200, data, _MIME.get(ext, "application/octet-stream"))


def _open_url(url: str) -> None:
    """Open the browser without its output landing in our console.

    Over SSH or in a VS Code remote session, `BROWSER` points at a helper that
    forwards the URL to the local machine. It works — but Python's `webbrowser`
    lets that helper inherit our stdio, and VS Code's is a Node script that
    prints a `url.parse()` deprecation warning. Arriving right under "Starting
    the Julia sidecar…", it reads like a failure of ours. Running the helper
    ourselves with its streams redirected keeps the console honest.
    """
    import subprocess

    browser = os.environ.get("BROWSER")
    if browser:
        try:
            subprocess.Popen(
                [browser, url],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            return
        except Exception:  # noqa: BLE001 — fall through to the stdlib opener
            pass
    try:
        import webbrowser

        webbrowser.open(url)
    except Exception:  # noqa: BLE001
        pass


def _is_remote() -> bool:
    return bool(
        os.environ.get("SSH_CONNECTION") or os.environ.get("VSCODE_IPC_HOOK_CLI")
    )


DEFAULT_PORT = 8765
PORT_TRIES = 20


def bind(handler, host: str, port: Optional[int]) -> ThreadingHTTPServer:
    """The server on `port`, or, without one, on 8765 or the next free port.

    A port the user names is used or refused; the default one gives way, so
    that a second studio, or a forwarded port, does not stop the first start.
    """
    ports = [port] if port is not None else range(DEFAULT_PORT, DEFAULT_PORT + PORT_TRIES)
    last = None
    for p in ports:
        try:
            return ThreadingHTTPServer((host, p), handler)
        except OSError as exc:  # the port is taken
            last = exc
    if port is not None:
        raise SystemExit(f"port {port} is already in use; start with another one, --port {port + 1}") from last
    raise SystemExit(
        f"ports {DEFAULT_PORT} to {DEFAULT_PORT + PORT_TRIES - 1} are all in use; start with --port"
    ) from last


def serve(
    host: str = "127.0.0.1", port: Optional[int] = None, open_browser: bool = True,
    project: Optional[str] = None, julia: Optional[str] = None,
) -> None:
    session = Session(project=project, julia=julia)
    handler = type("BoundHandler", (Handler,), {"session": session})
    httpd = bind(handler, host, port)
    session.host = host
    session.port = httpd.server_address[1]
    if port is None and session.port != DEFAULT_PORT:
        print(f"Port {DEFAULT_PORT} is taken; the studio uses port {session.port}.", flush=True)

    shown = "localhost" if host in ("0.0.0.0", "::") else host
    url = f"http://{shown}:{session.port}/?token={session.token}"
    print(f"MFH Studio → {url}", flush=True)
    if _is_remote():
        print(
            "  (remote session: open that URL on your own machine — VS Code "
            f"forwards the port, and so does `ssh -L {session.port}:127.0.0.1:{session.port}`)",
            flush=True,
        )
    print("Starting the Julia sidecar (loading MeanFieldHomogenization takes a while)…", flush=True)

    def warm() -> None:
        # Reading the manifest is instant and names a missing local checkout
        # precisely, where the Julia failure would only say "not installed".
        if not str(session.bridge.project).startswith("@"):
            rep = check_project(session.bridge.project)
            if not rep.ok:
                print("\n" + rep.text() + "\n")
        try:
            session.bridge.start()
            session.catalog()
            print("Sidecar ready.", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"Sidecar unavailable: {exc}", flush=True)
            print(
                "\nThe interface still runs; 3-D, read-back and Run are off.\n"
                "Run `--check` for a diagnosis of the Julia environment."
            )

    threading.Thread(target=warm, daemon=True).start()

    if open_browser:
        threading.Timer(0.5, lambda: _open_url(url)).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        session.bridge.stop()
        httpd.server_close()
