#!/bin/bash
# Build the browsable map of this codebase: graphify-out/graph.html
#
# WHY GRAPHIFY AND NOT THE OTHER ONE. The reader needs to identify bugs and
# methodological mistakes without reading code.
# Two candidates were evaluated -- Graphify-Labs/graphify and
# Egonex-AI/Understand-Anything. Understand-Anything has the friendlier
# output: an LLM writes a plain-English summary of every node, plus guided
# tours and a PM-facing view. That is also the reason it lost. Its summaries
# are GENERATED, and a reader who cannot check one against the code has no
# defence against a summary that is subtly wrong -- which is the opposite of
# what "help me spot mistakes" needs.
#
# Graphify parses with tree-sitter and no LLM, so nothing in the graph is
# invented, every edge is tagged EXTRACTED or INFERRED, and the plain-English
# text on each node is the DOCSTRING ITS AUTHOR WROTE. In a repo where the
# comments carry the measurements and the reasoning, that is the best
# explanation available and it cannot drift from the code.
#
# WHAT IT IS GOOD FOR: what depends on what, what everything flows through,
# where a change would ripple, and reading any function's own stated purpose
# without opening a file.
#
# WHAT IT IS NOT: a bug finder. Every real fault found on 20-21 September was
# semantic, not structural -- a value computed and never read, a rule correct
# for one building and wrong for another, a loop unpacking a shape its
# producers do not emit. None of those appear in a call graph. The tools that
# DO catch them are tools/bench.py (geometry against the markup),
# tools/cases.py (every flagged roof) and tests/run_all.sh.
#
# Usage:  ./tools/build_code_graph.sh          then open graphify-out/graph.html
set -eu
cd "$(dirname "$0")/.."
PY=.venv/bin/python
mkdir -p graphify-out
$PY -c "import graphify" 2>/dev/null || .venv/bin/pip install -q graphifyy

$PY - <<'EOF'
import collections, json
from pathlib import Path
from graphify.detect import detect
from graphify.extract import collect_files, extract
from graphify.build import build_from_json
from graphify.cluster import cluster, score_all
from graphify.analyze import god_nodes, surprising_connections, suggest_questions
from graphify.report import generate
from graphify.export import to_json

root = Path(".").resolve()
out = Path("graphify-out")

# data/ holds 7,000+ raster tiles and the built GeoJSON. They are outputs, not
# code, and they swamp the detector -- scope to what a person would read.
SKIP = {"data", ".claude", ".vscode", "graphify-out", ".venv", ".venv-sam"}
det = detect(root)
out.joinpath(".graphify_detect.json").write_text(json.dumps(det, ensure_ascii=False))
files = []
for f in det["files"]["code"]:
    p = Path(f)
    try:
        rel = p.relative_to(root)
    except ValueError:
        continue
    if rel.parts[0] in SKIP:
        continue
    files.extend(collect_files(p) if p.is_dir() else [p])
print(f"{len(files)} source files")

ex = extract(files, cache_root=root)
ex = {"nodes": ex["nodes"], "edges": ex["edges"], "hyperedges": [],
      "input_tokens": 0, "output_tokens": 0}
out.joinpath(".graphify_extract.json").write_text(json.dumps(ex, ensure_ascii=False))

G = build_from_json(ex, root=".", directed=False)
communities = cluster(G)

# NAME THE GROUPS, FROM THE FILES IN THEM. Out of the box every community is
# "Community 37", which makes the sidebar useless -- and the upstream step
# that names them uses an LLM, which is the one thing this build avoids. The
# dominant source file is a true, checkable name for a cluster of its
# functions.
members = collections.defaultdict(list)
for n in G.nodes(data=True):
    members[n[1].get("community")].append(n[1].get("source_file") or "")
labels, used = {}, collections.Counter()
# A node can carry no community at all; int(None) is not a label.
members.pop(None, None)
for cid, srcs in sorted(members.items(), key=lambda kv: -len(kv[1])):
    counts = collections.Counter(s for s in srcs if s)
    if not counts:
        labels[cid] = f"group {cid}"
        continue
    top, n = counts.most_common(1)[0]
    name = Path(top).stem.replace("_", " ")
    if n / len(srcs) < 0.6:
        rest = [Path(f).stem.replace("_", " ") for f, _ in counts.most_common(3)[1:]]
        if rest:
            name += " + " + ", ".join(rest)
    used[name] += 1
    labels[cid] = name if used[name] == 1 else f"{name} ({used[name]})"
out.joinpath(".graphify_labels.json").write_text(
    json.dumps({str(k): v for k, v in labels.items()}, indent=1, ensure_ascii=False))

cohesion = score_all(G, communities)
gods = god_nodes(G)
surprises = surprising_connections(G, communities)
questions = suggest_questions(G, communities, labels)
to_json(G, communities, "graphify-out/graph.json", force=True, community_labels=labels)
out.joinpath("GRAPH_REPORT.md").write_text(generate(
    G, communities, cohesion, labels, gods, surprises, det,
    {"input": 0, "output": 0}, ".", suggested_questions=questions))
print(f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
      f"{len(communities)} named groups")
EOF

.venv/bin/graphify export html
echo ""
echo "open graphify-out/graph.html"
echo "  or ask questions with:  .venv/bin/graphify query \"how does X work\""
echo "                          .venv/bin/graphify explain \"fit_panels_on_facet\""
echo "                          .venv/bin/graphify path \"partition_roof\" \"SolarModel\""
