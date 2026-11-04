"""Studio API: plan preview, job submission, SSE progress stream, worlds listing."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from worldgen.serve import create_server

pytestmark = pytest.mark.slow


@pytest.fixture
def server(tmp_path):  # noqa: ANN001, ANN201
    viewer = tmp_path / "viewer"
    viewer.mkdir()
    (viewer / "index.html").write_text("<html>viewer</html>")
    httpd, jobs = create_server(0, viewer, None, tmp_path / "worlds", tmp_path / "cache")
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", jobs
    httpd.shutdown()
    httpd.server_close()


def get(url: str):  # noqa: ANN201
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read())


def post(url: str, body: dict):  # noqa: ANN201
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.status, json.loads(r.read())


def sse_events(url: str) -> list[dict]:
    events: list[dict] = []
    with urllib.request.urlopen(url, timeout=300) as r:
        data: list[str] = []
        for raw in r:
            line = raw.decode("utf-8").rstrip("\r\n")
            if line.startswith("data: "):
                data.append(line[6:])
            elif line == "" and data:
                events.append(json.loads("\n".join(data)))
                data = []
                if events[-1]["type"] == "end":
                    break
    return events


def test_health_profiles_and_plan(server):
    base, _ = server
    h = get(f"{base}/api/health")
    assert h["ok"] and h["default_world"] is None
    names = [p["name"] for p in get(f"{base}/api/profiles")["profiles"]]
    assert names[:3] == ["basic", "procedural", "high_detail"]
    plan = get(f"{base}/api/plan?lat=25.033&lon=121.5654&radius=1000")
    assert plan["tile_count"] == 60 and len(plan["tiles"][0]["ring"]) == 4
    lat, lon = plan["tiles"][0]["ring"][0]
    assert 25.0 < lat < 25.1 and 121.5 < lon < 121.6
    assert plan["area_km2"] == pytest.approx(3.14, abs=0.01)
    with pytest.raises(urllib.error.HTTPError) as err:
        get(f"{base}/api/plan?lat=25&lon=121&radius=99999")
    assert err.value.code == 400


def test_job_runs_and_streams_progress(server):
    base, _ = server
    status, res = post(f"{base}/api/jobs", {"fixture": "simple_intersection"})
    assert status == 201 and res["job"]["state"] in ("queued", "running")
    job_id = res["job"]["id"]
    events = sse_events(f"{base}/api/jobs/{job_id}/events")
    types = [e["type"] for e in events]
    assert types[0] == "job" and types[-1] == "end" and events[-1]["state"] == "succeeded", events[-3:]
    assert [e["seq"] for e in events] == list(range(1, len(events) + 1))
    assert "plan" in types and "done" in types and types.count("tile") >= 4
    started = [e["name"] for e in events if e["type"] == "stage" and e["state"] == "start"]
    assert started[0] == "acquire" and started[-1] == "manifest"
    overall = [e["overall"] for e in events if "overall" in e]
    assert overall == sorted(overall) and overall[-1] == 1.0
    done = next(e for e in events if e["type"] == "done")
    assert done["world_url"] == "/worlds/fx_simple_intersection/" and done["summary"]["tiles_built"] == 4
    # Replay after a given seq.
    tail = sse_events(f"{base}/api/jobs/{job_id}/events?after={len(events) - 2}")
    assert [e["seq"] for e in tail] == [len(events) - 1, len(events)]
    snap = get(f"{base}/api/jobs/{job_id}")
    assert snap["job"]["state"] == "succeeded" and snap["job"]["progress"] == 1.0
    worlds = get(f"{base}/api/worlds")["worlds"]
    assert worlds[0]["name"] == "fx_simple_intersection" and worlds[0]["footprint"]
    manifest = get(f"{base}/worlds/fx_simple_intersection/manifest.json")
    assert manifest["schema"] == "worldgen.manifest"


def test_rejects_bad_requests_and_cancels(server):
    base, _ = server
    for body in ({"center": {"lat": 95, "lon": 0}, "radius_m": 500}, {"fixture": "nope"},
                 {"center": {"lat": 25, "lon": 121}, "radius_m": 500, "name": "../evil"},
                 {"center": {"lat": 25, "lon": 121}, "radius_m": 500, "profile": "missing"}):
        with pytest.raises(urllib.error.HTTPError) as err:
            post(f"{base}/api/jobs", body)
        assert err.value.code == 400
    # The second job waits in the queue and is cancelled there; the first is cancelled while running.
    _, a = post(f"{base}/api/jobs", {"fixture": "demo_town"})
    _, b = post(f"{base}/api/jobs", {"fixture": "bridge_river"})
    _, c = post(f"{base}/api/jobs/{b['job']['id']}/cancel", {})
    assert c["job"]["state"] in ("cancelled", "running")
    post(f"{base}/api/jobs/{a['job']['id']}/cancel", {})
    ev = sse_events(f"{base}/api/jobs/{a['job']['id']}/events")
    assert ev[-1]["type"] == "end" and ev[-1]["state"] in ("cancelled", "succeeded")
    with pytest.raises(urllib.error.HTTPError) as err:
        urllib.request.urlopen(f"{base}/worlds/..%2F..%2Fsecret.txt", timeout=5)
    assert err.value.code == 404


def test_job_options_budget_and_road_surface(server):
    base, jobs = server
    body = {"fixture": "demo_town", "name": "budgeted", "road_surface": "ribbon",
            "budget": {"enabled": True, "buildings_per_tile": 2, "trees_per_tile": 200}}
    _, res = post(f"{base}/api/jobs", body)
    req = res["job"]["request"]
    assert req["budget"] == {"enabled": True, "buildings_per_tile": 2, "trees_per_tile": 200}
    assert req["road_surface"] == "ribbon"
    cmd = jobs.command(jobs.jobs[res["job"]["id"]])
    assert "roads.surface_mode=ribbon" in cmd and "buildings.budget.max_per_tile=2" in cmd
    events = sse_events(f"{base}/api/jobs/{res['job']['id']}/events")
    done = next(e for e in events if e["type"] == "done")
    stats = done["summary"]["statistics"]
    assert stats["buildings_budget_dropped"] > 0 and stats["trees_budget_dropped"] > 0
    with pytest.raises(urllib.error.HTTPError) as err:
        post(f"{base}/api/jobs", {"fixture": "demo_town", "budget": {"trees_per_tile": 0}})
    assert err.value.code == 400
