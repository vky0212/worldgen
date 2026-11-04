"""Tiny static server: the built viewer at ``/`` and a compiled world at ``/world/``."""

from __future__ import annotations

import functools
import http.server
import mimetypes
import webbrowser
from pathlib import Path

mimetypes.add_type("model/gltf-binary", ".glb")
mimetypes.add_type("application/json", ".json")
mimetypes.add_type("text/javascript", ".js")


class _Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, viewer: Path, world: Path, **kwargs) -> None:  # noqa: ANN002, ANN003
        self._viewer = viewer
        self._world = world
        super().__init__(*args, directory=str(viewer), **kwargs)

    def translate_path(self, path: str) -> str:
        clean = path.split("?", 1)[0].split("#", 1)[0]
        if clean.startswith("/world/"):
            rel = clean[len("/world/"):]
            target = (self._world / rel).resolve()
            if self._world.resolve() not in target.parents and target != self._world.resolve():
                return str(self._world / "__forbidden__")
            return str(target)
        return super().translate_path(path)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def log_message(self, fmt: str, *args) -> None:  # noqa: ANN002
        pass


def serve(world: Path, port: int = 8765, viewer: Path | None = None, open_browser: bool = False) -> int:
    viewer = viewer or Path(__file__).resolve().parents[2] / "viewer" / "dist"
    if not (viewer / "index.html").is_file():
        print(f"viewer build not found at {viewer}; run `npm install && npm run build` in viewer/")
        return 2
    if not (world / "manifest.json").is_file():
        print(f"{world} is not a compiled world (manifest.json missing)")
        return 2
    handler = functools.partial(_Handler, viewer=viewer, world=world)
    with http.server.ThreadingHTTPServer(("127.0.0.1", port), handler) as httpd:
        url = f"http://127.0.0.1:{port}/"
        print(f"Serving viewer {viewer} and world {world.resolve()} at {url} (Ctrl+C to stop)")
        if open_browser:
            webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
    return 0
