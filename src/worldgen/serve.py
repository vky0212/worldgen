"""Local server: viewer, compiled worlds and the studio API (docs/studio-api.md)."""

from __future__ import annotations

import functools
import http.server
import json
import mimetypes
import urllib.parse
import webbrowser
from pathlib import Path
from typing import Any

from worldgen import __version__
from worldgen.studio import JobManager, RequestError, available_profiles, list_worlds, plan_for, region_from_request

mimetypes.add_type("model/gltf-binary", ".glb")
mimetypes.add_type("application/json", ".json")
mimetypes.add_type("text/javascript", ".js")

MAX_BODY = 64 * 1024


class _Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, viewer: Path, world: Path | None, root: Path, jobs: JobManager,  # noqa: ANN002
                 **kwargs) -> None:  # noqa: ANN003
        self._viewer = viewer
        self._world = world
        self._root = root
        self._jobs = jobs
        super().__init__(*args, directory=str(viewer), **kwargs)

    # -- static --------------------------------------------------------------------------
    @staticmethod
    def _inside(base: Path, rel: str) -> str:
        base = base.resolve()
        target = (base / urllib.parse.unquote(rel)).resolve()
        if target != base and base not in target.parents:
            return str(base / "__forbidden__")
        return str(target)

    def translate_path(self, path: str) -> str:
        clean = path.split("?", 1)[0].split("#", 1)[0]
        if clean.startswith("/world/"):
            if self._world is None:
                return str(self._root / "__no_default_world__")
            return self._inside(self._world, clean[len("/world/"):])
        if clean.startswith("/worlds/"):
            return self._inside(self._root, clean[len("/worlds/"):])
        return super().translate_path(path)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def log_message(self, fmt: str, *args) -> None:  # noqa: ANN002
        pass

    # -- API -----------------------------------------------------------------------------
    def _json(self, status: int, data: Any) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, message: str) -> None:
        self._json(status, {"error": message})

    def do_GET(self) -> None:  # noqa: N802
        url = urllib.parse.urlsplit(self.path)
        if not url.path.startswith("/api/"):
            return super().do_GET()
        q = {k: v[-1] for k, v in urllib.parse.parse_qs(url.query).items()}
        parts = url.path.strip("/").split("/")
        try:
            if url.path == "/api/health":
                return self._json(200, {"ok": True, "version": __version__, "output_root": str(self._root),
                                        "default_world": "/world/" if self._world else None,
                                        "busy": self._jobs.busy})
            if url.path == "/api/profiles":
                return self._json(200, {"profiles": available_profiles()})
            if url.path == "/api/plan":
                return self._json(200, plan_for(self._plan_region(q), q.get("profile", "basic"),
                                                float(q["tile_size"]) if q.get("tile_size") else None))
            if url.path == "/api/worlds":
                return self._json(200, {"worlds": list_worlds(self._root)})
            if url.path == "/api/jobs":
                return self._json(200, {"jobs": [j.snapshot() for j in self._jobs.list()]})
            if len(parts) >= 3 and parts[1] == "jobs":
                job = self._jobs.jobs.get(parts[2])
                if job is None:
                    return self._error(404, f"no job {parts[2]!r}")
                if len(parts) == 3:
                    return self._json(200, {"job": job.snapshot(), "events": list(job.events)})
                if len(parts) == 4 and parts[3] == "events":
                    after = q.get("after") or self.headers.get("Last-Event-ID") or "0"
                    return self._sse(job, int(after) if after.isdigit() else 0)
            return self._error(404, f"unknown endpoint {url.path}")
        except RequestError as exc:
            return self._error(400, str(exc))
        except (ValueError, KeyError) as exc:
            return self._error(400, f"bad request: {exc}")

    def do_POST(self) -> None:  # noqa: N802
        url = urllib.parse.urlsplit(self.path)
        parts = url.path.strip("/").split("/")
        try:
            if url.path == "/api/jobs":
                n = int(self.headers.get("Content-Length") or 0)
                if n <= 0 or n > MAX_BODY:
                    return self._error(400, "JSON body required (max 64 KB)")
                try:
                    body = json.loads(self.rfile.read(n).decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    return self._error(400, "body is not valid JSON")
                if not isinstance(body, dict):
                    return self._error(400, "body must be a JSON object")
                job = self._jobs.submit(body)
                return self._json(201, {"job": job.snapshot()})
            if len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "cancel":
                job = self._jobs.jobs.get(parts[2])
                if job is None:
                    return self._error(404, f"no job {parts[2]!r}")
                return self._json(200, {"job": self._jobs.cancel(job).snapshot()})
            return self._error(404, f"unknown endpoint {url.path}")
        except RequestError as exc:
            return self._error(400, str(exc))

    @staticmethod
    def _plan_region(q: dict[str, str]) -> dict[str, Any]:
        if q.get("bbox"):
            return region_from_request({"bbox": [float(v) for v in q["bbox"].split(",")]})
        return region_from_request({"center": {"lat": q.get("lat"), "lon": q.get("lon")},
                                    "radius_m": q.get("radius")})

    def _sse(self, job, after: int) -> None:  # noqa: ANN001
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Connection", "close")
        self.end_headers()
        seq = after
        try:
            while True:
                batch = job.events_after(seq, timeout=15.0)
                if not batch:
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
                    continue
                for ev in batch:
                    data = json.dumps(ev, ensure_ascii=False)
                    self.wfile.write(f"id: {ev['seq']}\nevent: {ev['type']}\ndata: {data}\n\n".encode("utf-8"))
                    seq = ev["seq"]
                self.wfile.flush()
                if any(ev["type"] == "end" for ev in batch):
                    return
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return


def create_server(port: int, viewer: Path, world: Path | None, output_root: Path, cache_dir: Path
                  ) -> tuple[http.server.ThreadingHTTPServer, JobManager]:
    jobs = JobManager(output_root, cache_dir)
    handler = functools.partial(_Handler, viewer=viewer, world=world, root=jobs.output_root, jobs=jobs)
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    httpd.daemon_threads = True
    return httpd, jobs


def serve(world: Path | None = None, port: int = 8765, viewer: Path | None = None, open_browser: bool = False,
          output_root: Path = Path("generated"), cache_dir: Path = Path(".worldgen-cache")) -> int:
    viewer = viewer or Path(__file__).resolve().parents[2] / "viewer" / "dist"
    if not (viewer / "index.html").is_file():
        print(f"viewer build not found at {viewer}; run `npm install && npm run build` in viewer/")
        return 2
    if world is not None and not (world / "manifest.json").is_file():
        print(f"{world} is not a compiled world (manifest.json missing)")
        return 2
    httpd, jobs = create_server(port, viewer, world, output_root, cache_dir)
    url = f"http://127.0.0.1:{port}/"
    print(f"WorldGen {__version__} at {url}  (viewer: {viewer})")
    print(f"  worlds: {jobs.output_root} -> {url}worlds/<name>/   generate: {url}generate.html")
    if world is not None:
        print(f"  default world: {world.resolve()} -> {url}world/")
    if open_browser:
        webbrowser.open(url if world is not None else url + "generate.html")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0
