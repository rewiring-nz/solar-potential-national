"""A work queue that is nothing but objects in the bucket.

WHY A BUCKET AND NOT A SERVICE. Google Cloud Batch and Pub/Sub both need an
API enabled and IAM granted before anything moves; the bucket already
works from every account this project uses. And the queue's state is a folder
anyone can open in the console -- `queue/`, `claims/`, `done/`, `failed/` -- so
"is the run healthy" is four object counts, not a log to read
(docs/scale-architecture.md, "Coordinating machines").

WHAT MAKES IT CORRECT. Cloud Storage's create-if-absent is atomic
(`--if-generation-match=0`): when two workers try to claim the same region in
the same instant, the storage service lets exactly one succeed. No lock
server, no race. A claim is refreshed every few minutes; one that has gone
stale is taken over by deleting it with its generation number, which again
only one taker can win. Everything here is `gcloud storage` calls, so it runs
anywhere gcloud is authenticated, with no extra Python dependency.

Layout under gs://<bucket>/<prefix>/:

    queue/<region>.json      the task: bbox, buildings, survey. Written once.
    claims/<region>          {"worker", "since", "beat"}; refreshed while working.
    done/<region>.json       the region's summary. Claim deleted.
    failed/<region>.json     {"attempts", "worker", "log"}; retried up to MAX_ATTEMPTS.
    regions/<region>/...     what publish_region uploaded.
"""

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

BUCKET = os.environ.get("SOLAR_BUCKET", "gs://rewiring-solar-data")
PREFIX = os.environ.get("SOLAR_BUILD_PREFIX", "build")
STALE_AFTER_S = 45 * 60      # a claim not refreshed for this long is abandoned
MAX_ATTEMPTS = 3

WORKER = os.environ.get("SOLAR_WORKER", socket.gethostname())


def url(*parts):
    return "/".join([BUCKET, PREFIX, *parts])


def _run(args, check=True, capture=True):
    r = subprocess.run(["gcloud", "storage", *args], capture_output=capture, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"gcloud storage {' '.join(args)}: {r.stderr.strip()[:400]}")
    return r


def ls(*parts):
    """Object names (basenames) under a prefix; [] when it does not exist."""
    r = _run(["ls", url(*parts) + "/"], check=False)
    if r.returncode != 0:
        return []
    return [line.rsplit("/", 1)[-1] for line in r.stdout.split() if not line.endswith("/")]


def read_json(*parts):
    r = _run(["cat", url(*parts)], check=False)
    return json.loads(r.stdout) if r.returncode == 0 and r.stdout.strip() else None


def write_json(obj, *parts, if_generation_match=None):
    """Returns True if written. With if_generation_match=0 this is the atomic
    create-if-absent that the whole scheme rests on."""
    args = ["cp", "-", url(*parts)]
    if if_generation_match is not None:
        args = ["cp", f"--if-generation-match={if_generation_match}", "-", url(*parts)]
    r = subprocess.run(["gcloud", "storage", *args], input=json.dumps(obj),
                       capture_output=True, text=True)
    return r.returncode == 0


def generation(*parts):
    # `gcloud storage stat` does not exist in the gcloud on this machine;
    # `objects describe` does, and it is the generation the precondition wants.
    r = _run(["objects", "describe", "--format=value(generation)", url(*parts)], check=False)
    return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else None


def delete(*parts, if_generation_match=None):
    args = ["rm", url(*parts)]
    if if_generation_match is not None:
        args = ["rm", f"--if-generation-match={if_generation_match}", url(*parts)]
    return _run(args, check=False).returncode == 0


# ------------------------------------------------------------------ claims

def claim_next():
    """Claim one region, or return None when there is nothing left to do.

    Order: never-claimed regions first, then stale claims, then failed regions
    with attempts left. Everything is one attempt at an atomic create; a lost
    race just moves on to the next candidate.
    """
    queued = {n[:-5] for n in ls("queue") if n.endswith(".json")}
    done = {n[:-5] for n in ls("done") if n.endswith(".json")}
    claimed = set(ls("claims"))
    failed = {n[:-5] for n in ls("failed") if n.endswith(".json")}
    now = time.time()
    stamp = {"worker": WORKER, "since": now, "beat": now}

    for region in sorted(queued - done - claimed - failed):
        if write_json(stamp, "claims", region, if_generation_match=0):
            return region

    for region in sorted((queued - done) & claimed):
        c = read_json("claims", region)
        if not c or now - c.get("beat", 0) < STALE_AFTER_S:
            continue
        gen = generation("claims", region)
        # take over only the exact stale object we read -- if its owner beats
        # again in between, the generation moves and this delete fails
        if gen and delete("claims", region, if_generation_match=gen):
            if write_json(stamp, "claims", region, if_generation_match=0):
                return region

    for region in sorted((queued - done - claimed) & failed):
        f = read_json("failed", region) or {}
        if f.get("attempts", 0) >= MAX_ATTEMPTS:
            continue
        if write_json(stamp, "claims", region, if_generation_match=0):
            return region
    return None


def heartbeat(region):
    c = read_json("claims", region) or {"worker": WORKER, "since": time.time()}
    c["beat"] = time.time()
    write_json(c, "claims", region)


def mark_done(region, summary):
    write_json(summary, "done", region + ".json")
    delete("claims", region)
    delete("failed", region + ".json")


def mark_failed(region, log_tail):
    f = read_json("failed", region + ".json") or {"attempts": 0}
    f["attempts"] = f.get("attempts", 0) + 1
    f["worker"] = WORKER
    f["at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    f["log"] = log_tail[-4000:]
    write_json(f, "failed", region + ".json")
    delete("claims", region)


# ------------------------------------------------------------------ status

def status():
    queued = {n[:-5] for n in ls("queue") if n.endswith(".json")}
    done = {n[:-5] for n in ls("done") if n.endswith(".json")}
    claims = ls("claims")
    failed = {n[:-5] for n in ls("failed") if n.endswith(".json")}
    exhausted = 0
    for r in failed:
        f = read_json("failed", r + ".json") or {}
        if f.get("attempts", 0) >= MAX_ATTEMPTS:
            exhausted += 1
    now = time.time()
    rows = []
    for r in sorted(claims):
        c = read_json("claims", r) or {}
        age = (now - c.get("since", now)) / 60
        beat = (now - c.get("beat", now)) / 60
        rows.append(f"    {r:28s} {c.get('worker', '?'):22s} {age:5.0f} min"
                    + ("   STALE" if beat > STALE_AFTER_S / 60 else ""))
    print(f"done {len(done)} / building {len(claims)} / queued "
          f"{len(queued - done - set(claims))} / failed {len(failed)} "
          f"({exhausted} given up)  --  {len(queued)} regions in all")
    if rows:
        print("  building now:")
        print("\n".join(rows))
    return {"done": len(done), "building": len(claims), "queued": len(queued),
            "failed": len(failed), "exhausted": exhausted}


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "claim":
        r = claim_next()
        print(r or "")
        sys.exit(0 if r else 3)
    status()
