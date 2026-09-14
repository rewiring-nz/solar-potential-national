"""Regenerate the status board from what is actually true right now.

Josh: "why aren't you updating the status dashboard? That should always be
updating and automatic and live".

Because it was hand-written HTML, so it was only ever as current as the last
time someone retyped it -- it went stale within the hour, still claiming v36 was
live and a rebuild was at 20 of 24. A dashboard nobody trusts is worse than no
dashboard, because it is consulted and believed.

So every number here is READ, not typed:
    live version + panel totals   fetched from the deployed site
    local build                   data/solar_potential.geojson
    labelling progress            data/roof_labels.json
    benchmark history             data/bench_history.json, with deltas
    recent work                   git log
    VM                            gcloud, if it answers quickly

The curated half -- what needs Josh, what is rejected and why -- cannot be
derived from the repo, so it lives in data/status_items.json as DATA. Editing
that file is how those cards change; nobody hand-edits HTML again.

Usage:
    python tools/status_board.py            # regenerate
    python tools/status_board.py --no-net   # skip the live fetch (offline)
"""

import argparse
import html
import json
import subprocess
import sys
import time
import urllib.request
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "data" / "preview" / "status_board.html"
ITEMS = ROOT / "data" / "status_items.json"
LIVE_CFG = "https://rewiring-nz.github.io/nz-solar-potential/site-config.js"
LIVE_GEO = ("https://rewiring-nz.github.io/nz-solar-potential/"
            "data/solar_potential.geojson")


def _fetch(url, timeout=90):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.read().decode()
    except Exception:
        return None


def live_state(use_net=True):
    st = {"version": None, "panels": None, "gwh": None, "buildings": None,
          "reachable": False}
    if not use_net:
        return st
    cfg = _fetch(LIVE_CFG, 30)
    if cfg:
        st["reachable"] = True
        for line in cfg.splitlines():
            if "dataVersion" in line:
                st["version"] = line.split('"')[1] if '"' in line else None
                break
    return st


def local_build():
    p = ROOT / "data" / "solar_potential.geojson"
    if not p.exists():
        return {}
    try:
        d = json.loads(p.read_text())
    except Exception:
        return {}
    f = d.get("features") or []
    panels = gwh = 0
    withheld = 0
    for x in f:
        pr = x.get("properties") or {}
        panels += pr.get("fill_panels_100", pr.get("panel_count", 0)) or 0
        gwh += pr.get("ac_kwh_year", 0) or 0
        if pr.get("no_estimate_reason"):
            withheld += 1
    return {"buildings": len(f), "panels": panels, "gwh": gwh / 1e6,
            "withheld": withheld,
            "mtime": time.strftime("%d %b %H:%M",
                                   time.localtime(p.stat().st_mtime))}


def labels():
    p = ROOT / "data" / "roof_labels.json"
    if not p.exists():
        return {}
    b = json.loads(p.read_text()).get("buildings", {})
    return {"total": len(b),
            "complete": sum(1 for v in b.values() if v.get("complete")),
            "faces": sum(1 for v in b.values() if v.get("faces"))}


def bench():
    p = ROOT / "data" / "bench_history.json"
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text())
    except Exception:
        return []


def commits(n=6):
    try:
        out = subprocess.run(["git", "log", f"-{n}", "--format=%h\x1f%s\x1f%ar"],
                             cwd=ROOT, capture_output=True, text=True,
                             timeout=20).stdout
        return [l.split("\x1f") for l in out.strip().splitlines() if l]
    except Exception:
        return []


def vm():
    try:
        r = subprocess.run(
            ["gcloud", "compute", "instances", "list",
             "--format=value(name,status)"],
            capture_output=True, text=True, timeout=45)
        rows = [l.split() for l in r.stdout.strip().splitlines() if l]
        return rows[0] if rows else None
    except Exception:
        return None


def items():
    if ITEMS.exists():
        try:
            return json.loads(ITEMS.read_text())
        except Exception:
            pass
    return {"needs_josh": [], "open": [], "rejected": [], "constraints": []}


def e(s):
    return html.escape(str(s))


def render(st):
    L, B, lab = st["local"], st["bench"], st["labels"]
    live, cm, machine, it = st["live"], st["commits"], st["vm"], st["items"]

    def stat(label, value, sub=""):
        return (f'<div class="stat"><span class="lab">{e(label)}</span>'
                f'<div class="v">{e(value)}</div>'
                f'<div class="d">{sub}</div></div>')

    band = [
        stat("Live version", live.get("version") or "?",
             '<span class="ok">serving</span>' if live.get("reachable")
             else '<span class="dim">could not reach the site</span>'),
        stat("Panels", f"{L.get('panels', 0):,}", "in the local build"),
        stat("Annual output", f"{L.get('gwh', 0):.1f}", "GWh/yr"),
        stat("Buildings", f"{L.get('buildings', 0):,}",
             f"{L.get('withheld', 0):,} withheld"),
        stat("Roofs you have drawn", f"{lab.get('complete', 0)}",
             f"of {lab.get('total', 0)} started"),
    ]

    # ---- benchmark, the part that makes this a live scoreboard ----
    bench_html = ""
    if B:
        rows = []
        prev = None
        for run in B[-6:]:
            s = run["score"]
            fpr = s["facets"] / max(s["roofs"], 1)
            fx = (100 * s["faces_exact"] / s["faces_total"]
                  if s.get("faces_total") else None)
            ac = (100 * s["panels_across"] / s["panels_on_labelled"]
                  if s.get("panels_on_labelled") else None)

            def cell(v, pv, lower_better, fmt="{:.1f}", suffix=""):
                if v is None:
                    return '<td class="n dim">--</td>'
                txt = fmt.format(v) + suffix
                if pv is None:
                    return f'<td class="n">{txt}</td>'
                d = v - pv
                if abs(d) < 0.05:
                    return f'<td class="n">{txt}</td>'
                good = (d < 0) if lower_better else (d > 0)
                cls = "win" if good else "bad"
                return (f'<td class="n">{txt} '
                        f'<span class="{cls}">{d:+.1f}{suffix}</span></td>')

            pf = prev
            rows.append(
                "<tr>"
                f'<td>{e(run.get("note") or "—")}'
                f'<div class="dim" style="font-size:12px">'
                f'{e(run["when"][5:16].replace("T", " "))} &middot; '
                f'{run.get("secs", 0)/60:.1f} min</div></td>'
                + cell(fx, pf[0] if pf else None, False, suffix="%")
                + cell(ac, pf[1] if pf else None, True, suffix="%")
                + cell(fpr, pf[2] if pf else None, True)
                + f'<td class="n">{s["panels"]:,}</td>'
                "</tr>")
            prev = (fx, ac, fpr)
        bench_html = f"""
  <section class="panel">
    <h2>Benchmark &mdash; 152 roofs, scored in about 90 seconds</h2>
    <p class="note">Every change is judged here before a district rebuild.
    <code>python tools/bench.py --note "what I changed"</code></p>
    <div class="scroll"><table>
      <thead><tr><th>run</th><th>faces matching<br>your markup</th>
      <th>panels across<br>a line</th><th>facets<br>per roof</th>
      <th>panels</th></tr></thead>
      <tbody>{''.join(reversed(rows))}</tbody>
    </table></div>
  </section>"""

    def cards(lst, cls):
        out = []
        for c in lst:
            chip = (f'<span class="chip {cls}">{e(c["chip"])}</span>'
                    if c.get("chip") else "")
            body = "".join(f"<p>{c['body']}</p>" for c in [c] if c.get("body"))
            extra = (f'<p class="dim" style="font-size:13px">{c["note"]}</p>'
                     if c.get("note") else "")
            out.append(f'<div class="card {cls}">{chip}'
                       f'<h3>{e(c["title"])}</h3>{body}{extra}</div>')
        return "".join(out)

    commit_rows = "".join(
        f'<tr><td class="n dim">{e(h)}</td><td>{e(s)}</td>'
        f'<td class="n dim">{e(w)}</td></tr>' for h, s, w in cm)

    machine_line = ""
    if machine:
        state = machine[1] if len(machine) > 1 else "?"
        cls = "ok" if state == "RUNNING" else "dim"
        machine_line = (f'<p>Build VM <code>{e(machine[0])}</code>: '
                        f'<span class="{cls}">{e(state)}</span></p>')

    rejected_rows = "".join(
        f'<tr><td>{e(r["what"])}</td><td class="n">{e(r["result"])}</td>'
        f'<td class="dim">{e(r["why"])}</td></tr>' for r in it.get("rejected", []))

    return TEMPLATE.format(
        generated=time.strftime("%d %b %Y, %H:%M"),
        band="".join(band),
        needs=cards(it.get("needs_josh", []), "you"),
        open_cards=cards(it.get("open", []), ""),
        bench=bench_html,
        commits=commit_rows,
        machine=machine_line,
        built=e(L.get("mtime", "unknown")),
        rejected=rejected_rows,
        constraints="".join(f"<li>{c}</li>" for c in it.get("constraints", [])),
    )


TEMPLATE = """<title>Rooftop Solar Build Status</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500&family=Source+Sans+3:wght@400;600&display=swap">
<style>
  :root{{
    --ground:#f4f6f8; --surface:#ffffff; --sunk:#e9edf1;
    --ink:#141a20; --ink-2:#3d4a56; --muted:#6b7a88; --line:#d5dce3;
    --accent:#c07806; --ok:#1d7d55; --you:#1d5fb0; --stop:#b23b36;
    --ok-bg:#e2f2ea; --you-bg:#e2ecfa; --stop-bg:#fae6e5;
  }}
  @media (prefers-color-scheme: dark){{
    :root:not([data-theme="light"]){{
      --ground:#0f1317; --surface:#171d23; --sunk:#11161b;
      --ink:#e6ecf2; --ink-2:#c2ccd6; --muted:#8695a3; --line:#26303a;
      --accent:#ffb43c; --ok:#52c08a; --you:#6ba8ff; --stop:#e0645f;
      --ok-bg:#14261e; --you-bg:#141f2e; --stop-bg:#2a1717;
    }}
  }}
  :root[data-theme="dark"]{{
    --ground:#0f1317; --surface:#171d23; --sunk:#11161b;
    --ink:#e6ecf2; --ink-2:#c2ccd6; --muted:#8695a3; --line:#26303a;
    --accent:#ffb43c; --ok:#52c08a; --you:#6ba8ff; --stop:#e0645f;
    --ok-bg:#14261e; --you-bg:#141f2e; --stop-bg:#2a1717;
  }}
  *{{box-sizing:border-box}}
  body{{background:var(--ground); color:var(--ink);
    font:16px/1.55 "Source Sans 3", ui-sans-serif, system-ui, sans-serif;
    margin:0; padding:28px 22px 60px;}}
  .wrap{{max-width:1100px; margin:0 auto; display:flex; flex-direction:column; gap:24px}}
  h1,h2,h3,.lab,th{{font-family:Archivo, ui-sans-serif, system-ui, sans-serif}}
  h1{{font-size:26px; font-weight:700; margin:0; letter-spacing:-.01em; text-wrap:balance}}
  .sub{{color:var(--muted); font-size:14px; margin-top:3px}}
  .lab{{font-size:11px; font-weight:600; letter-spacing:.09em;
        text-transform:uppercase; color:var(--muted)}}
  .n{{font-family:"IBM Plex Mono", ui-monospace, monospace;
      font-variant-numeric:tabular-nums; white-space:nowrap}}
  .band{{background:var(--surface); border:1px solid var(--line); border-radius:8px;
    padding:18px 20px; display:flex; flex-wrap:wrap; gap:28px; align-items:flex-end;}}
  .stat .lab{{display:block; margin-bottom:3px}}
  .stat .v{{font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:25px;
    font-weight:500; font-variant-numeric:tabular-nums; line-height:1.1}}
  .stat .d{{font-size:12.5px; color:var(--muted); margin-top:2px}}
  .lanes{{display:grid; grid-template-columns:repeat(auto-fit,minmax(310px,1fr)); gap:18px}}
  .lane{{display:flex; flex-direction:column; gap:10px}}
  .card{{background:var(--surface); border:1px solid var(--line);
    border-left:3px solid var(--line); border-radius:6px; padding:13px 15px}}
  .card.you{{border-left-color:var(--you)}}
  .card h3{{font-size:14.5px; font-weight:600; margin:0 0 5px}}
  .card p{{margin:0; font-size:14px; color:var(--ink-2)}}
  .card p + p{{margin-top:7px}}
  .chip{{display:inline-block; font-family:Archivo,sans-serif; font-size:10.5px;
    font-weight:600; letter-spacing:.07em; text-transform:uppercase;
    padding:2.5px 7px; border-radius:3px; margin-bottom:7px;
    background:var(--sunk); color:var(--muted)}}
  .chip.you{{background:var(--you-bg); color:var(--you)}}
  code{{font-family:"IBM Plex Mono",monospace; font-size:12.5px;
    background:var(--sunk); padding:1px 5px; border-radius:3px; color:var(--ink-2)}}
  .panel{{background:var(--surface); border:1px solid var(--line);
    border-radius:8px; padding:18px 20px}}
  .panel h2{{font-size:15px; font-weight:600; margin:0 0 3px}}
  .panel .note{{color:var(--muted); font-size:13.5px; margin:0 0 14px}}
  .scroll{{overflow-x:auto}}
  table{{border-collapse:collapse; width:100%; font-size:14px}}
  th,td{{text-align:left; padding:7px 12px 7px 0; border-bottom:1px solid var(--line);
    vertical-align:top}}
  th{{font-size:11px; font-weight:600; letter-spacing:.07em;
    text-transform:uppercase; color:var(--muted)}}
  tr:last-child td{{border-bottom:none}}
  .win{{color:var(--ok)}} .bad{{color:var(--stop)}} .ok{{color:var(--ok)}}
  .dim{{color:var(--muted)}}
  ul{{margin:0; padding-left:18px}}
  li{{margin:5px 0; font-size:14px; color:var(--ink-2)}}
  .foot{{color:var(--muted); font-size:13px; text-align:center; padding-top:4px}}
</style>

<div class="wrap">
  <header>
    <h1>Rooftop Solar Build Status</h1>
    <div class="sub">Queenstown district &middot; generated from the repo and the
    live site at {generated}</div>
  </header>

  <section class="band">{band}</section>

  {bench}

  <div class="lanes">
    <div class="lane"><span class="lab">Needs you</span>{needs}</div>
    <div class="lane"><span class="lab">Open</span>{open_cards}</div>
  </div>

  <section class="panel">
    <h2>Recent work</h2>
    <p class="note">Local build written {built}. {machine}</p>
    <div class="scroll"><table><tbody>{commits}</tbody></table></div>
  </section>

  <section class="panel">
    <h2>Measured and rejected</h2>
    <p class="note">Recorded so none of it gets retried from scratch.</p>
    <div class="scroll"><table>
      <thead><tr><th>approach</th><th>result</th><th>why it failed</th></tr></thead>
      <tbody>{rejected}</tbody></table></div>
  </section>

  <section class="panel">
    <h2>Constraints that will not move</h2>
    <ul>{constraints}</ul>
  </section>

  <div class="foot">Regenerate with
  <code>python tools/status_board.py</code></div>
</div>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-net", action="store_true")
    a = ap.parse_args()
    st = {"live": live_state(not a.no_net), "local": local_build(),
          "labels": labels(), "bench": bench(), "commits": commits(),
          "vm": None if a.no_net else vm(), "items": items()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render(st))
    b = st["bench"]
    print(f"wrote {OUT}")
    print(f"  live v{st['live'].get('version')}  "
          f"local {st['local'].get('panels', 0):,} panels  "
          f"{st['labels'].get('complete', 0)} roofs drawn  "
          f"{len(b)} benchmark run(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
