/* Hand-editing a roof's panel layout.
 *
 * Josh: "click on a panel, and then delete it with a little pop up bin icon
 * ... 4 + sign bubbles on each edge so I can add a panel in any direction (or
 * not if there are already panels there) ... add a new panel, so then many can
 * be added to it with the same plus buttons ... shift-selectable, so they can
 * be shift selected and dragged to be moved around. Maybe only selected onto
 * with double click, to avoid accidentally selecting panels by the user."
 *
 * WHY THIS SITS ON THE LIVE-LAYOUT OVERLAY rather than a new rendering path:
 * the panels on screen come from vector tiles, which cannot be edited in place.
 * The map already knows how to hide one building's tile panels
 * (setLayoutExclusion) and draw a replacement set from GeoJSON (the live-layout
 * source used by parameter tuning). Editing copies the building's panels out of
 * the tiles once, hides the originals, and from then on owns them.
 *
 * GEOMETRY. Panels are quadrilaterals in lng/lat at whatever angle the roof
 * runs. Reasoning about "the next panel to the left" in degrees is miserable,
 * so everything works in a local metric frame centred on the building -- flat
 * earth is more than good enough over 50 m.
 *
 * THE STEP BETWEEN PANELS IS MEASURED, NOT ASSUMED. Tiles carry panels shrunk
 * by 7 cm (shrink_panels_for_tiles) so they read as separate rectangles, which
 * means a panel's drawn size is NOT its pitch. Stepping by the drawn edge would
 * creep. So the true pitch is measured from an actual neighbouring pair on the
 * same roof where one exists, and only falls back to drawn-size-plus-gap when
 * the roof has a single panel to learn from.
 *
 * Edits are LOCAL to the browser. Nothing is written back to the build; the
 * point is to capture what a human says the roof should look like, which is
 * then sent as a correction.
 */
(function () {
  "use strict";

  const SHRINK_M = 0.07;        // shrink_panels_for_tiles, for the fallback pitch
  const OCCUPIED_FRAC = 0.35;   // overlap of a panel-sized box that counts as taken
  const HANDLE_HIT_PX = 30;

  const S = {
    active: false,
    buildingId: null,
    panels: [],               // [{ring:[[x,y]...] in local metres}]
    selected: new Set(),
    origin: null,             // [lng,lat] of the local frame
    mPerDegLng: 1, mPerDegLat: 111320,
    dragging: null,
    addMode: false,
    dirty: false,
  };

  // ---------- local metric frame ----------
  function setOrigin(lng, lat) {
    S.origin = [lng, lat];
    S.mPerDegLat = 111320;
    S.mPerDegLng = 111320 * Math.cos(lat * Math.PI / 180);
  }
  const toLocal = (lng, lat) => [(lng - S.origin[0]) * S.mPerDegLng,
                                 (lat - S.origin[1]) * S.mPerDegLat];
  const toLngLat = (x, y) => [S.origin[0] + x / S.mPerDegLng,
                              S.origin[1] + y / S.mPerDegLat];

  const centroid = ring => {
    let x = 0, y = 0;
    for (const p of ring) { x += p[0]; y += p[1]; }
    return [x / ring.length, y / ring.length];
  };
  const sub = (a, b) => [a[0] - b[0], a[1] - b[1]];
  const add = (a, b) => [a[0] + b[0], a[1] + b[1]];
  const scale = (a, k) => [a[0] * k, a[1] * k];
  const len = a => Math.hypot(a[0], a[1]);

  /* A panel's two edge directions and lengths, from its own corners. Panels are
   * quads but not always perfectly rectangular after clipping, so take the two
   * edges from one corner and treat them as the local axes. */
  function axes(ring) {
    const r = ring.length >= 5 ? ring.slice(0, 4) : ring.slice(0, 4);
    const e1 = sub(r[1], r[0]), e2 = sub(r[3] || r[2], r[0]);
    return { u: e1, v: e2, lu: len(e1), lv: len(e2) };
  }

  /* Centre-to-centre spacing actually used on this roof. Measured from the
   * closest neighbouring pair along each axis; this is the difference between
   * a layout that stays on the grid and one that drifts a few cm per panel. */
  function measurePitch() {
    if (S.panels.length < 2) return null;
    const a0 = axes(S.panels[0].ring);
    const cs = S.panels.map(p => centroid(p.ring));
    let bu = Infinity, bv = Infinity;
    const un = scale(a0.u, 1 / (a0.lu || 1)), vn = scale(a0.v, 1 / (a0.lv || 1));
    for (let i = 0; i < cs.length; i++) {
      for (let j = i + 1; j < cs.length; j++) {
        const d = sub(cs[j], cs[i]);
        const du = Math.abs(d[0] * un[0] + d[1] * un[1]);
        const dv = Math.abs(d[0] * vn[0] + d[1] * vn[1]);
        if (dv < 0.25 && du > 0.2 && du < bu) bu = du;   // same row, next along
        if (du < 0.25 && dv > 0.2 && dv < bv) bv = dv;   // next row
      }
    }
    return { u: isFinite(bu) ? bu : null, v: isFinite(bv) ? bv : null };
  }

  function stepFor(ring, dir) {
    const a = axes(ring);
    const pitch = measurePitch() || {};
    const su = pitch.u || (a.lu + SHRINK_M);
    const sv = pitch.v || (a.lv + SHRINK_M);
    const un = scale(a.u, 1 / (a.lu || 1)), vn = scale(a.v, 1 / (a.lv || 1));
    return [scale(un, su), scale(un, -su), scale(vn, sv), scale(vn, -sv)][dir];
  }

  const translate = (ring, d) => ring.map(p => add(p, d));

  /* Is a candidate position already taken? Compares centres rather than doing
   * real polygon intersection: panels on one roof share a grid, so centre
   * distance is both sufficient and far cheaper. */
  function occupied(ring) {
    const c = centroid(ring), a = axes(ring);
    const near = Math.min(a.lu, a.lv) * OCCUPIED_FRAC + 0.05;
    return S.panels.some(p => len(sub(centroid(p.ring), c)) < near);
  }

  // ---------- reading the building's panels out of the tiles ----------
  function readPanelsFromTiles(buildingId) {
    let feats = [];
    try {
      feats = map.querySourceFeatures("layout", {
        sourceLayer: "layout",
        filter: ["all", ["==", ["get", "kind"], "panel"],
                        ["==", ["get", "building_id"], buildingId]],
      });
    } catch (e) { feats = []; }
    // querySourceFeatures returns the same panel once per tile it touches
    const seen = new Set(), out = [];
    for (const f of feats) {
      let ring = f.geometry && f.geometry.coordinates;
      if (!ring) continue;
      while (Array.isArray(ring[0][0])) ring = ring[0];
      const c = ring.reduce((s, p) => [s[0] + p[0] / ring.length,
                                       s[1] + p[1] / ring.length], [0, 0]);
      const key = c[0].toFixed(7) + "," + c[1].toFixed(7);
      if (seen.has(key)) continue;
      seen.add(key);
      out.push({ lngLatRing: ring, props: f.properties || {} });
    }
    return out;
  }

  // ---------- rendering ----------
  function featureCollection() {
    return {
      type: "FeatureCollection",
      features: S.panels.map((p, i) => ({
        type: "Feature",
        properties: { kind: "panel", building_id: S.buildingId,
                      _edit_index: i, _selected: S.selected.has(i) ? 1 : 0 },
        geometry: { type: "Polygon",
                    coordinates: [p.ring.map(([x, y]) => toLngLat(x, y))
                                        .concat([toLngLat(p.ring[0][0], p.ring[0][1])])] },
      })),
    };
  }

  function render() {
    const src = map.getSource("live-layout");
    if (src) src.setData(featureCollection());
    renderHandles();
    updateToolbar();
  }

  // ---------- DOM handles (bin + four plus bubbles) ----------
  let handleLayer = null;
  function ensureHandleLayer() {
    if (handleLayer) return handleLayer;
    handleLayer = document.createElement("div");
    handleLayer.id = "panel-edit-handles";
    handleLayer.style.cssText =
      "position:absolute; inset:0; pointer-events:none; z-index:6;";
    map.getCanvasContainer().parentElement.appendChild(handleLayer);
    return handleLayer;
  }

  function mkHandle(cls, label, title, onClick) {
    const b = document.createElement("button");
    b.className = "pe-handle " + cls;
    b.innerHTML = label;
    b.title = title;
    b.style.pointerEvents = "auto";
    b.addEventListener("click", ev => { ev.stopPropagation(); onClick(); });
    b.addEventListener("dblclick", ev => ev.stopPropagation());
    return b;
  }

  function renderHandles() {
    const layer = ensureHandleLayer();
    layer.innerHTML = "";
    if (!S.active || S.selected.size === 0) return;

    // Bin sits above the selection's centre, and acts on ALL selected panels.
    const sel = [...S.selected];
    const cs = sel.map(i => centroid(S.panels[i].ring));
    const mid = cs.reduce((s, c) => [s[0] + c[0] / cs.length, s[1] + c[1] / cs.length], [0, 0]);
    const midLL = toLngLat(mid[0], mid[1]);
    const midPt = map.project(midLL);
    const bin = mkHandle("pe-bin", "&#128465;",
      sel.length > 1 ? `Delete ${sel.length} panels` : "Delete this panel",
      deleteSelected);
    bin.style.left = (midPt.x - 14) + "px";
    bin.style.top = (midPt.y - 46) + "px";
    layer.appendChild(bin);

    // The four plus bubbles only make sense for a single panel -- "add one to
    // the left of these nine" has no meaning.
    if (sel.length !== 1) return;
    const idx = sel[0], ring = S.panels[idx].ring;
    const c = centroid(ring);
    for (let dir = 0; dir < 4; dir++) {
      const d = stepFor(ring, dir);
      const cand = translate(ring, d);
      if (occupied(cand)) continue;            // Josh: "(or not if there are
                                               // already panels there)"
      const at = add(c, scale(d, 0.72));       // just outside the panel edge
      const pt = map.project(toLngLat(at[0], at[1]));
      const plus = mkHandle("pe-plus", "+", "Add a panel here",
                            () => addAdjacent(idx, dir));
      plus.style.left = (pt.x - 11) + "px";
      plus.style.top = (pt.y - 11) + "px";
      layer.appendChild(plus);
    }
  }

  // ---------- operations ----------
  function addAdjacent(idx, dir) {
    const ring = S.panels[idx].ring;
    const cand = translate(ring, stepFor(ring, dir));
    if (occupied(cand)) return;
    S.panels.push({ ring: cand });
    S.selected = new Set([S.panels.length - 1]);   // keep growing from the new one
    S.dirty = true;
    render();
  }

  function deleteSelected() {
    if (!S.selected.size) return;
    const keep = S.panels.filter((_, i) => !S.selected.has(i));
    S.panels = keep;
    S.selected.clear();
    S.dirty = true;
    render();
  }

  function addFreePanel(lngLat) {
    // Shape and angle copied from an existing panel so a new one lands on the
    // roof's grid rather than square to the screen. With no panel to copy, fall
    // back to the configured panel size, north-aligned.
    let ring;
    if (S.panels.length) {
      const src = S.panels[0].ring;
      const d = sub(toLocal(lngLat.lng, lngLat.lat), centroid(src));
      ring = translate(src, d);
    } else {
      const w = (window.assumptions && window.assumptions.panel_width_m) || 1.134;
      const h = (window.assumptions && window.assumptions.panel_height_m) || 1.961;
      const c = toLocal(lngLat.lng, lngLat.lat);
      ring = [[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]]
        .map(p => add(p, c));
    }
    if (occupied(ring)) return false;
    S.panels.push({ ring });
    S.selected = new Set([S.panels.length - 1]);
    S.dirty = true;
    render();
    return true;
  }

  function panelAt(point) {
    const ll = map.unproject(point);
    const p = toLocal(ll.lng, ll.lat);
    let best = -1, bd = Infinity;
    S.panels.forEach((pan, i) => {
      const a = axes(pan.ring);
      const d = len(sub(centroid(pan.ring), p));
      if (d < Math.max(a.lu, a.lv) * 0.6 && d < bd) { bd = d; best = i; }
    });
    return best;
  }

  // ---------- session ----------
  function enter(buildingId, lngLatHint) {
    const tiles = readPanelsFromTiles(buildingId);
    const seed = tiles.length ? tiles[0].lngLatRing[0]
                              : [lngLatHint.lng, lngLatHint.lat];
    setOrigin(seed[0], seed[1]);
    S.buildingId = buildingId;
    S.panels = tiles.map(t => ({
      ring: t.lngLatRing.slice(0, -1).map(([lng, lat]) => toLocal(lng, lat)),
    }));
    S.originalCount = S.panels.length;
    S.selected = new Set();
    S.active = true;
    S.dirty = false;
    if (typeof setLayoutExclusion === "function") setLayoutExclusion(buildingId);
    showToolbar();
    render();
  }

  function exit() {
    S.active = false;
    S.selected.clear();
    S.panels = [];
    S.addMode = false;
    if (typeof setLayoutExclusion === "function") setLayoutExclusion(undefined);
    const src = map.getSource("live-layout");
    if (src) src.setData({ type: "FeatureCollection", features: [] });
    if (handleLayer) handleLayer.innerHTML = "";
    hideToolbar();
  }

  // ---------- toolbar ----------
  let bar = null;
  function showToolbar() {
    if (!bar) {
      bar = document.createElement("div");
      bar.id = "panel-edit-bar";
      document.body.appendChild(bar);
    }
    bar.style.display = "";
    updateToolbar();
  }
  function hideToolbar() { if (bar) bar.style.display = "none"; }

  function updateToolbar() {
    if (!bar || !S.active) return;
    const n = S.panels.length, was = S.originalCount || 0;
    const delta = n - was;
    bar.innerHTML = `
      <div class="pe-title">Editing panels &middot; building #${S.buildingId}</div>
      <div class="pe-count"><b>${n}</b> panels
        ${delta ? `<span class="pe-delta">${delta > 0 ? "+" : ""}${delta}</span>` : ""}
      </div>
      <div class="pe-hint">Double-click a panel to select &middot; shift-double-click to add more
        &middot; drag to move &middot; <b>Esc</b> deselects</div>
      <div class="pe-actions">
        <button id="pe-add" class="${S.addMode ? "on" : ""}">${S.addMode ? "Click the roof…" : "+ New panel"}</button>
        <button id="pe-reset">Reset</button>
        <button id="pe-send" ${S.dirty ? "" : "disabled"}>Send correction</button>
        <button id="pe-done">Done</button>
      </div>`;
    bar.querySelector("#pe-add").onclick = () => {
      S.addMode = !S.addMode;
      map.getCanvas().style.cursor = S.addMode ? "copy" : "";
      updateToolbar();
    };
    bar.querySelector("#pe-reset").onclick = () => enter(S.buildingId, null);
    bar.querySelector("#pe-done").onclick = exit;
    bar.querySelector("#pe-send").onclick = sendCorrection;
  }

  /* Corrections go through the same Jotform as bug reports and flags. Sent as
   * counts plus panel centres: enough to reproduce the intended layout and to
   * find the building, without posting a whole geometry dump into a form
   * field. */
  function sendCorrection() {
    const note = document.getElementById("bug-f-note");
    const ctx = document.getElementById("bug-f-context");
    const form = document.getElementById("bug-form");
    if (!note || !ctx || !form) return;
    const cs = S.panels.map(p => {
      const c = toLngLat(...centroid(p.ring));
      return c[0].toFixed(6) + " " + c[1].toFixed(6);
    });
    note.value = `PANEL CORRECTION building ${S.buildingId}: `
      + `${S.originalCount} -> ${S.panels.length} panels`;
    ctx.value = `EDITED LAYOUT building ${S.buildingId} | `
      + (typeof bugCtx === "function" ? bugCtx() : "")
      + " | centres: " + cs.join("; ");
    form.submit();
    S.dirty = false;
    updateToolbar();
    const c = bar.querySelector(".pe-count");
    if (c) c.insertAdjacentHTML("beforeend",
      ' <span class="pe-sent">&#10003; sent</span>');
  }

  // ---------- input ----------
  function wire() {
    // DOUBLE click to select, on purpose. Josh: "only selected onto with
    // double click, to avoid accidentally selecting panels by the user" --
    // single-click already means "open this building" everywhere else.
    map.on("dblclick", ev => {
      if (!S.active) return;
      const idx = panelAt(ev.point);
      if (idx < 0) return;
      ev.preventDefault();
      if (ev.originalEvent && ev.originalEvent.shiftKey) {
        if (S.selected.has(idx)) S.selected.delete(idx); else S.selected.add(idx);
      } else {
        S.selected = new Set([idx]);
      }
      render();
    });

    map.on("click", ev => {
      if (!S.active) return;
      if (S.addMode) {
        addFreePanel(ev.lngLat);
        S.addMode = false;
        map.getCanvas().style.cursor = "";
        updateToolbar();
        return;
      }
      if (panelAt(ev.point) < 0 && S.selected.size) { S.selected.clear(); render(); }
    });

    // Dragging a selection. Bound on the canvas so it can pre-empt the map's
    // own pan before MapLibre starts one.
    const cvs = () => map.getCanvas();
    cvs().addEventListener("mousedown", e => {
      if (!S.active || !S.selected.size) return;
      const rect = cvs().getBoundingClientRect();
      const pt = { x: e.clientX - rect.left, y: e.clientY - rect.top };
      const idx = panelAt(pt);
      if (idx < 0 || !S.selected.has(idx)) return;
      const ll = map.unproject(pt);
      S.dragging = { from: toLocal(ll.lng, ll.lat),
                     start: [...S.selected].map(i => S.panels[i].ring.map(p => [...p])) };
      map.dragPan.disable();
      e.preventDefault();
    });
    window.addEventListener("mousemove", e => {
      if (!S.dragging) return;
      const rect = cvs().getBoundingClientRect();
      const ll = map.unproject({ x: e.clientX - rect.left, y: e.clientY - rect.top });
      const d = sub(toLocal(ll.lng, ll.lat), S.dragging.from);
      [...S.selected].forEach((i, k) => {
        S.panels[i].ring = S.dragging.start[k].map(p => add(p, d));
      });
      S.dirty = true;
      render();
    });
    window.addEventListener("mouseup", () => {
      if (!S.dragging) return;
      S.dragging = null;
      map.dragPan.enable();
    });

    map.on("move", () => { if (S.active) renderHandles(); });
    document.addEventListener("keydown", e => {
      if (!S.active) return;
      if (e.key === "Escape") {
        if (S.addMode) { S.addMode = false; map.getCanvas().style.cursor = ""; updateToolbar(); }
        else if (S.selected.size) { S.selected.clear(); render(); }
      }
      if ((e.key === "Delete" || e.key === "Backspace") && S.selected.size) {
        e.preventDefault(); deleteSelected();
      }
    });
  }

  window.PanelEditor = {
    enter, exit, isActive: () => S.active,
    buildingId: () => S.buildingId,
    _wire: wire,
  };
})();
