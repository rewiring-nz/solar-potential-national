"""What kind of building a roof sits on: home, business, school, hospital...

A toggle for building types on the map: Homes, Schools, Hospitals,
Businesses, etc. Whatever the detailed building type categories you can find
access to."

WHAT IS ACTUALLY AVAILABLE, measured on the pilot's 1,270 outlines:

    LINZ `use`     Unknown 1,251  School 16  Supermarket 3
    LINZ `name`    six distinct, all schools and supermarkets

So LINZ names the civic and retail buildings it knows about -- schools,
hospitals, supermarkets, and a few dozen other categories nationally -- and
calls the other 98.5% "Unknown". For those, the only signal in hand is size,
and economics.js has drawn that line for weeks: a roof over 400 m2 or a
system over 100 kW is priced as a business. The same line is used here so the
map's "Businesses" and the economics' business tariff agree on which buildings
they mean.

The classification is deliberately coarse -- five types -- because a toggle
with thirty entries is a list, not a toggle, and because the honest
resolution of the underlying data is about five. It is also a single function
so the next source (a council zoning layer, OSM landuse) slots in as one more
clause with a measured effect, rather than as a second opinion somewhere else.
"""

# In display order. The frontend builds its toggle from this list, via
# assumptions.json, so adding a type here is the whole change.
BTYPES = [
    ("home",      "Homes"),
    ("business",  "Businesses"),
    ("school",    "Schools"),
    ("hospital",  "Hospitals"),
    ("community", "Community"),
]

# MIRRORS economics.js: biz_min_roof_m2 and biz_min_kw. Change both or the
# map and the money will call different buildings a business.
BIZ_MIN_ROOF_M2 = 400
BIZ_MIN_KW = 100

# LINZ `use` values, lower-cased substrings. The LINZ data dictionary lists
# these as free-text categories; matching on substrings survives the variants
# ("Hospital", "Hospital / Medical Centre").
_USE_RULES = [
    ("school", ("school", "kindergarten", "kura", "college", "university",
                "polytechnic", "education")),
    ("hospital", ("hospital", "medical", "health", "clinic", "hospice",
                  "rest home", "aged care")),
    ("community", ("church", "marae", "community", "hall", "library",
                   "museum", "gallery", "sport", "stadium", "gym", "pool",
                   "fire station", "police", "courthouse", "civic",
                   "council", "recreation", "club")),
    ("business", ("supermarket", "shop", "retail", "mall", "commercial",
                  "office", "warehouse", "industrial", "factory", "hotel",
                  "motel", "airport", "terminal", "depot", "bank", "market",
                  "petrol", "service station", "dairy", "restaurant", "cafe",
                  "bar", "brewery", "winery")),
]


def classify(use=None, name=None, roof_m2=None, kwp=None):
    """One of the BTYPES keys."""
    text = f"{use or ''} {name or ''}".lower()
    if text.strip() and "unknown" not in text.split():
        for btype, words in _USE_RULES:
            if any(w in text for w in words):
                return btype
    # No named use: size decides, on the same line the economics use.
    if (roof_m2 or 0) >= BIZ_MIN_ROOF_M2 or (kwp or 0) >= BIZ_MIN_KW:
        return "business"
    return "home"
