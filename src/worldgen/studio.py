"""Build jobs for the studio API: queueing, subprocess builds and event fan-out.

Every job runs ``python -m worldgen build ... --progress jsonl`` in a child
process, so a crash or a cancel never takes the server down. Events from the
child are re-sequenced (``seq``) and re-timed (``t`` since the job was
created) and kept in memory for replay to any number of SSE clients.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from worldgen.config import load_config, profile_search_paths
from worldgen.errors import WorldGenError

NAME_RE = re.compile(r"^[a-z0-9_-]{1,48}$")
MAX_RADIUS_M = 5000.0
MAX_TILES = 1600
TERMINAL = ("succeeded", "failed", "cancelled")


class RequestError(WorldGenError):
    """Invalid API request (HTTP 400)."""


def utc_now() -> str:
    return _dt.datetime.now(_dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def available_profiles() -> list[dict[str, str]]:
    seen: dict[str, str] = {}
    for d in {p.parent for p in profile_search_paths("basic")[1:] if p.parent.is_dir()}:
        for f in sorted(d.glob("*.yaml")):
            if f.stem in seen:
                continue
            first = f.read_text(encoding="utf-8").splitlines()[:1]
            seen[f.stem] = first[0].lstrip("# ").strip() if first and first[0].startswith("#") else ""
    order = {"basic": 0, "procedural": 1, "high_detail": 2}
    return [{"name": n, "description": d} for n, d in sorted(seen.items(), key=lambda kv: (order.get(kv[0], 9), kv[0]))]


def region_from_request(body: dict[str, Any]) -> dict[str, Any]:
    """Validated config ``region`` mapping from an API request (center+radius_m or bbox)."""
    if body.get("bbox") is not None:
        bb = body["bbox"]
        if not (isinstance(bb, list) and len(bb) == 4 and all(isinstance(v, (int, float)) for v in bb)):
            raise RequestError("bbox must be [south, west, north, east]")
        s, w, n, e = (float(v) for v in bb)
        if not (-85 <= s < n <= 85 and -180 <= w < e <= 180):
            raise RequestError("bbox must satisfy south < north and west < east within valid ranges")
        diag = math.hypot((n - s) * 111_320.0, (e - w) * 111_320.0 * math.cos(math.radians((s + n) / 2)))
        if diag > 2 * math.sqrt(2) * MAX_RADIUS_M:
            raise RequestError(f"bbox is too large (diagonal {diag / 1000:.1f} km)")
        return {"bbox": [s, w, n, e]}
    c = body.get("center")
    if not isinstance(c, dict):
        raise RequestError("center {lat, lon} and radius_m, or bbox, is required")
    try:
        lat, lon, r = float(c["lat"]), float(c["lon"]), float(body.get("radius_m", 0))
    except (KeyError, TypeError, ValueError) as exc:
        raise RequestError("center needs numeric lat and lon, radius_m must be a number") from exc
    if not (-85 <= lat <= 85 and -180 <= lon <= 180):
        raise RequestError("center out of range")
    if not (50 <= r <= MAX_RADIUS_M):
        raise RequestError(f"radius_m must be between 50 and {MAX_RADIUS_M:.0f}")
    return {"center_lat": lat, "center_lon": lon, "radius_m": r}


def plan_for(region: dict[str, Any], profile: str = "basic", tile_size: float | None = None) -> dict[str, Any]:
    from worldgen.build import plan_payload
    from worldgen.spatial.region import Region

    overrides = [f"tiling.tile_size={tile_size}"] if tile_size else []
    try:
        cfg = load_config(profile, overrides=overrides, region=region)
    except WorldGenError as exc:
        raise RequestError(str(exc)) from exc
    r = Region.from_config(cfg)
    payload = plan_payload(r, cfg)
    payload["tile_count"] = len(payload["tiles"])
    payload["area_km2"] = round(r.footprint.area / 1e6, 4)
    payload["limits"] = {"max_radius_m": MAX_RADIUS_M, "max_tiles": MAX_TILES}
    payload.pop("sources", None)
    return payload


def default_name(region: dict[str, Any]) -> str:
    if "bbox" in region:
        s, w, n, e = region["bbox"]
        lat, lon = (s + n) / 2, (w + e) / 2
    else:
        lat, lon = region["center_lat"], region["center_lon"]
    tag = f"{abs(lat):.4f}{'n' if lat >= 0 else 's'}_{abs(lon):.4f}{'e' if lon >= 0 else 'w'}"
    return tag.replace(".", "-")


@dataclass
class Job:
    id: str
    name: str
    request: dict[str, Any]
    output: Path
    world_url: str
    created: float = field(default_factory=time.time)
    created_utc: str = field(default_factory=utc_now)
    started_utc: str | None = None
    finished_utc: str | None = None
    state: str = "queued"
    progress: float = 0.0
    stage: str | None = None
    message: str | None = None
    summary: dict[str, Any] | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    cond: threading.Condition = field(default_factory=threading.Condition)
    proc: subprocess.Popen | None = None
    cancel_requested: bool = False

    def add(self, event: dict[str, Any]) -> dict[str, Any]:
        with self.cond:
            ev = {k: v for k, v in event.items() if k not in ("seq", "t")}
            ev = {"seq": len(self.events) + 1, "t": round(time.time() - self.created, 3), **ev}
            self.events.append(ev)
            t = ev["type"]
            if "overall" in ev:
                self.progress = max(self.progress, float(ev["overall"]))
            if t == "stage" and ev.get("state") == "start":
                self.stage = ev["name"]
            elif t == "error":
                self.message = ev.get("message")
            elif t == "log" and ev.get("level") == "error":
                self.message = ev.get("message")
            elif t == "done":
                self.summary = ev.get("summary")
                self.progress = 1.0
            self.cond.notify_all()
            return ev

    def snapshot(self) -> dict[str, Any]:
        return {
            "id": self.id, "name": self.name, "state": self.state, "created_utc": self.created_utc,
            "started_utc": self.started_utc, "finished_utc": self.finished_utc, "request": self.request,
            "world_url": self.world_url, "progress": round(self.progress, 4), "stage": self.stage,
            "message": self.message, "summary": self.summary,
        }

    def events_after(self, seq: int, timeout: float) -> list[dict[str, Any]]:
        with self.cond:
            if len(self.events) <= seq and self.state not in TERMINAL:
                self.cond.wait(timeout)
            return self.events[seq:]


class JobManager:
    def __init__(self, output_root: Path, cache_dir: Path) -> None:
        self.output_root = output_root.resolve()
        self.cache_dir = cache_dir.resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, Job] = {}
        self._queue: deque[Job] = deque()
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._worker = threading.Thread(target=self._run_forever, name="worldgen-jobs", daemon=True)
        self._worker.start()

    # -- API -----------------------------------------------------------------------------
    def submit(self, body: dict[str, Any]) -> Job:
        fixture = body.get("fixture")
        if fixture is not None:
            from worldgen.build import fixture_region
            from worldgen.sources.synthetic import FIXTURES

            if fixture not in FIXTURES:
                raise RequestError(f"unknown fixture {fixture!r}")
            fr = fixture_region(str(fixture))
            body = {**body, "center": {"lat": fr["center_lat"], "lon": fr["center_lon"]}, "radius_m": fr["radius_m"],
                    "name": body.get("name") or f"fx_{fixture}"}
        region = region_from_request(body)
        profile = str(body.get("profile") or "basic")
        if profile not in {p["name"] for p in available_profiles()}:
            raise RequestError(f"unknown profile {profile!r}")
        name = str(body.get("name") or default_name(region)).lower()
        if not NAME_RE.match(name):
            raise RequestError("name must match [a-z0-9_-]{1,48}")
        plan = plan_for(region, profile)
        if plan["tile_count"] > MAX_TILES:
            raise RequestError(f"region needs {plan['tile_count']} tiles (limit {MAX_TILES})")
        with self._lock:
            busy = [j for j in self.jobs.values() if j.name == name and j.state not in TERMINAL]
            if busy:
                raise RequestError(f"world {name!r} is already being built by job {busy[0].id}")
            req = {"name": name, "profile": profile, "offline": bool(body.get("offline", False)),
                   "force": bool(body.get("force", False))}
            if fixture is not None:
                req["fixture"] = fixture
            if "bbox" in region:
                req["bbox"] = region["bbox"]
            else:
                req["center"] = {"lat": region["center_lat"], "lon": region["center_lon"]}
                req["radius_m"] = region["radius_m"]
            job = Job(uuid.uuid4().hex[:8], name, req, self.output_root / name, f"/worlds/{name}/")
            self.jobs[job.id] = job
            self._queue.append(job)
        job.add({"type": "job", "state": "queued"})
        self._wake.set()
        return job

    def cancel(self, job: Job) -> Job:
        with self._lock:
            if job.state == "queued":
                try:
                    self._queue.remove(job)
                except ValueError:
                    pass
                self._finish(job, "cancelled")
            elif job.state == "running":
                job.cancel_requested = True
                if job.proc is not None and job.proc.poll() is None:
                    job.proc.terminate()
        return job

    def list(self) -> list[Job]:
        return sorted(self.jobs.values(), key=lambda j: j.created, reverse=True)

    @property
    def busy(self) -> bool:
        return any(j.state == "running" for j in self.jobs.values())

    # -- worker --------------------------------------------------------------------------
    def _run_forever(self) -> None:
        while True:
            self._wake.wait(1.0)
            self._wake.clear()
            while True:
                with self._lock:
                    job = self._queue.popleft() if self._queue else None
                if job is None:
                    break
                try:
                    self._run(job)
                except Exception as exc:  # keep the worker alive whatever happens
                    job.add({"type": "error", "message": f"server error: {exc}"})
                    self._finish(job, "failed")

    def command(self, job: Job) -> list[str]:
        r = job.request
        cmd = [sys.executable, "-m", "worldgen", "build", "--output", str(job.output),
               "--cache-dir", str(self.cache_dir), "--profile", r["profile"], "--name", job.name,
               "--progress", "jsonl"]
        if r.get("fixture"):
            cmd += ["--fixture", r["fixture"]]
        elif "bbox" in r:
            cmd += ["--bbox", ",".join(f"{v:.7f}" for v in r["bbox"])]
        else:
            cmd += ["--center", f"{r['center']['lat']:.7f},{r['center']['lon']:.7f}", "--radius", f"{r['radius_m']:g}"]
        if r.get("offline"):
            cmd.append("--offline")
        if r.get("force"):
            cmd.append("--force")
        return cmd

    def _run(self, job: Job) -> None:
        job.state = "running"
        job.started_utc = utc_now()
        job.add({"type": "job", "state": "running"})
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        src = str(Path(__file__).resolve().parents[1])
        env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        job.proc = subprocess.Popen(self.command(job), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    stdin=subprocess.DEVNULL, env=env, text=True, encoding="utf-8",
                                    errors="replace", bufsize=1, creationflags=flags)
        stderr_tail: deque[str] = deque(maxlen=20)

        def pump_stderr() -> None:
            assert job.proc is not None and job.proc.stderr is not None
            for line in job.proc.stderr:
                line = line.rstrip()
                if line:
                    stderr_tail.append(line)
                    if "WARNING" in line or "ERROR" in line or "Traceback" in line:
                        job.add({"type": "log", "level": "error" if "ERROR" in line else "warning",
                                 "message": line})

        t_err = threading.Thread(target=pump_stderr, daemon=True)
        t_err.start()
        saw_done = False
        assert job.proc.stdout is not None
        for line in job.proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                job.add({"type": "log", "level": "info", "message": line})
                continue
            if ev.get("type") == "done":
                saw_done = True
                ev["world_url"] = job.world_url
                ev.pop("output", None)
            job.add(ev)
        rc = job.proc.wait()
        t_err.join(timeout=2.0)
        if job.cancel_requested:
            self._finish(job, "cancelled")
        elif rc == 0 and saw_done:
            self._finish(job, "succeeded")
        else:
            if not any(e["type"] == "error" for e in job.events):
                tail = " | ".join(list(stderr_tail)[-3:]) or f"build exited with code {rc}"
                job.add({"type": "error", "message": tail})
            self._finish(job, "failed")

    def _finish(self, job: Job, state: str) -> None:
        job.state = state
        job.finished_utc = utc_now()
        job.proc = None
        job.add({"type": "end", "state": state})


# --------------------------------------------------------------------------------------
# compiled worlds


_world_cache: dict[Path, tuple[float, dict[str, Any]]] = {}


def list_worlds(root: Path) -> list[dict[str, Any]]:
    out = []
    if not root.is_dir():
        return out
    for d in sorted(root.iterdir()):
        mp = d / "manifest.json"
        if not d.is_dir() or not mp.is_file() or not NAME_RE.match(d.name):
            continue
        mtime = mp.stat().st_mtime
        cached = _world_cache.get(mp)
        if cached and cached[0] == mtime:
            out.append(cached[1])
            continue
        try:
            info = _world_info(d.name, json.loads(mp.read_text(encoding="utf-8")))
        except (ValueError, KeyError, TypeError):
            continue
        _world_cache[mp] = (mtime, info)
        out.append(info)
    return sorted(out, key=lambda w: w.get("created_utc") or "", reverse=True)


def _world_info(name: str, m: dict[str, Any]) -> dict[str, Any]:
    from shapely.geometry import Point, box

    from worldgen.spatial.frame import GeoPoint, LocalFrame

    reg = m["region"]
    stats = m.get("statistics", {})
    o = m["origin"]
    frame = LocalFrame(GeoPoint(o["lat"], o["lon"]), m["crs"]["local"]["method"])
    if reg.get("radius_m"):
        shape = Point(0, 0).buffer(float(reg["radius_m"]), quad_segs=16)
    else:
        w, s, e, n = reg["bbox_wgs84"]
        shape = frame.geom_to_local(box(w, s, e, n).segmentize(0.002))
    lon, lat = frame.to_wgs84(*zip(*shape.exterior.coords))
    return {
        "name": name, "url": f"/worlds/{name}/", "center": reg["center"], "radius_m": reg.get("radius_m"),
        "bbox_wgs84": reg.get("bbox_wgs84"),
        "footprint": [[round(float(a), 7), round(float(b), 7)] for a, b in zip(lat, lon)],
        "created_utc": m.get("build", {}).get("created_utc"), "profile": m.get("config", {}).get("profile"),
        "tile_count": m.get("tiling", {}).get("tile_count"),
        "statistics": {k: stats.get(k) for k in ("roads", "buildings", "trees", "bridges", "tunnels",
                                                 "graph_nodes", "graph_edges", "vegetation_regions")},
        "validation": {k: m.get("validation", {}).get(k) for k in ("errors", "warnings", "info")},
    }
