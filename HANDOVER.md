# Handover prompt

Paste the block below into a fresh session. It is written to be pasted whole —
it assumes the new session knows nothing.

---

I'm Josh, at Rewiring NZ. We build public rooftop-solar-potential maps for New
Zealand. The repo is `~/Desktop/J/Website/solar-map` (Queenstown district, live)
with a hand-synced sibling `~/Desktop/J/Website/solar-wellington` (Island Bay).
Read `BACKLOG.md` first — the task queue lives there, not in your head. Run
`tools/check_repo_sync.py` before pushing anything that touches shared code.

**The architecture, which is settled — don't relitigate it.** Imagery finds the
roof lines; LiDAR fits the angles of the geometry imagery found. This split is
forced by the data: the survey is 1.7 returns/m² (~0.77 m spacing) so a hip
crease falls between samples and LiDAR cannot see it, while imagery at 0.1 m/px
can. I hand-mark ridges, valleys and roof faces in a browser tool; those labels
train the line detector and are also used directly for the roofs I've drawn.

**Where it stands.** 114 roofs marked up, 92 complete with faces. Five defects
were fixed where my markup was computed correctly and then thrown away — on my
labelled roofs, invented facet edges went 25.3% → ~9% and facets per roof 8.4 →
6.0. A district rebuild is shipping the first fix that reaches *unlabelled*
roofs too (a minimum-length bar on detected lines, `MIN_LEN_FRAC = 0.35` in
`src/roof_line_source.py`): invented edges 22.4% → 20.5% across the district.

**The open problem, stated precisely.** The detector emits stubs, not folds.
Where I draw a 5 m ridge it finds 1.5 m; 98.5% of its line endpoints dangle
against 56.4% of mine. Combined with a cutter that extends each line infinitely
across the roof cell, that produces both failure modes I keep seeing: missing
folds on sawtooth roofs, and invented lines slicing simple ones. More labelled
roofs measurably help — boundary F1 has gone 0.122 → 0.132 → 0.144 → 0.185 as
labels grew 18 → 37 → 55 → 74, about +1.1 per 100 roofs, still climbing.

**Already measured and rejected — do not retry these from scratch:** extending
stubs that share a bearing (82.5% → 81.1% found, and *more* invented lines);
routing model lines through segment subdivision (82.5% → 57.6%); feeding LiDAR
height into the detector (0.185 → 0.163); a face-predicting model (0.163). The
reasoning for each is written into the module docstrings, especially
`src/line_network.py`.

**How to work with me.** Batch your fixes and report on decisions, not per
commit. You have standing approval to deploy solar-map upgrades without asking.
Don't drive my real Chrome. Verify with numbers rather than impressions — say
what you measured and on how many roofs, and tell me plainly when something
regressed.

**The fast loop, which matters more than it sounds.** A geometry change used to
need a 4½-hour rebuild to judge; `tools/preview_sample.py --ids …` renders a
handful of roofs in 44 seconds. `tools/measure_facet_agreement.py` scores facets
against my markup in *both* directions — lines of mine you found, and edges I
never drew; either number alone is misleading. `tools/predeploy_check.py`
compares a build against the live site, and it must be run before every deploy:
facet agreement scores shape and cannot tell that a building has vanished from
the map, which is how three regressions once shipped.

Start by reading `BACKLOG.md` and telling me what you'd do first.

---

## Live status board

<https://claude.ai/code/artifact/cfc00fc1-feb2-4e42-bcc9-c5b41cacef61>

A snapshot, not a live feed — it reflects the state at the time it was last
published. Ask a session to update it (same file, `data/preview/status_board.html`)
after a rebuild lands.
