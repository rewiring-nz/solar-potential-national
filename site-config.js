// PER-DEPLOYMENT settings. This file is deliberately NOT synced between the
// Queenstown and national repos -- it is the one place they are meant to differ
// on the frontend, so preview.html can stay byte-identical between them.
//
// Why it exists: preview.html was unified across both deploys on 31 Aug, and
// its DEFAULT_VIEW was hardcoded to Island Bay. That silently pointed the
// QUEENSTOWN site at Wellington on first load -- Josh found it on 1 Sep. A
// shared file cannot carry a per-site default, so the default moved here.
//
// Loaded synchronously before the map is constructed, so there is no visible
// jump from a wrong starting view to the right one.
window.SITE = {
  name: "National",
  defaultView: { center: [174.7745, -41.3380], zoom: 15.0 },
  // Areas offered in the search box. Island Bay is the only region with data
  // so far; add towns here as regions land, NOT in preview.html.
  towns: [
    ["Wellington", 174.7772, -41.2889, 13.5],
    ["Island Bay", 174.7745, -41.3380, 15.0],
  ],
};
