"""Real-world vocabularies + a corruption engine for hard, self-verifiable canonicalization.

The toy pools (4 countries, 3 category sets) taught the model nothing the heuristic
didn't already do. Here we back categorical columns with hundreds of REAL canonical
entities (countries, US states, currencies, cities, departments, job titles, statuses)
and corrupt each into realistic surface forms (aliases, codes, casing, punctuation,
typos) — recording the exact surface→canonical mapping so the example still round-trips
through scrubdata.executor (self-verified).

Design rule that keeps it learnable AND verifiable:
  - Each generated column draws only a FEW canonicals (low cardinality) so every dirty
    surface appears in the profile sample the model sees.
  - No outer whitespace on categorical cells (executor's canonicalize strips only outer
    space and would otherwise leave canonical-valued cells untrimmed) — casing/punct/typo
    noise is what creates the mapping entries.
"""

from __future__ import annotations

import random
import string

import pycountry

# --- generic surface corruption ---------------------------------------------

_KEEP = set(string.ascii_letters + string.digits + " ")


def _strip_punct(s: str) -> str:
    return "".join(ch for ch in s if ch in _KEEP).strip()


def _one_typo(rng: random.Random, s: str) -> str:
    if len(s) < 4:
        return s
    i = rng.randrange(1, len(s) - 1)
    mode = rng.random()
    if mode < 0.34:                      # drop a char
        return s[:i] + s[i + 1:]
    if mode < 0.67:                      # swap adjacent
        return s[:i] + s[i + 1] + s[i] + s[i + 2:]
    return s[:i] + s[i] + s[i:]          # duplicate a char


def make_surface(rng: random.Random, canonical: str, aliases: list[str]) -> str:
    """Produce one realistic dirty surface for a canonical value."""
    s = rng.choice([canonical, *aliases]) if aliases else canonical
    r = rng.random()
    if r < 0.28:
        s = s.lower()
    elif r < 0.42:
        s = s.upper()
    elif r < 0.52:
        s = s.title()
    if rng.random() < 0.15:
        s = _strip_punct(s)
    if rng.random() < 0.08:
        s = _one_typo(rng, s)
    return s.strip()


# --- vocabularies (canonical -> aliases) ------------------------------------

# Curated common aliases for high-frequency countries (codes are added automatically).
_COUNTRY_ALIASES = {
    "United States": ["USA", "U.S.A.", "U.S.", "America", "the US", "United States of America"],
    "United Kingdom": ["UK", "U.K.", "Britain", "Great Britain", "England"],
    "United Arab Emirates": ["UAE", "U.A.E."],
    "South Korea": ["Korea", "Republic of Korea"],
    "Russian Federation": ["Russia"],
    "Czechia": ["Czech Republic"],
    "Netherlands": ["Holland", "The Netherlands"],
}


def _country_entries(limit: int | None = None) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for c in pycountry.countries:
        name = c.name
        aliases = {c.alpha_2, c.alpha_3}
        official = getattr(c, "official_name", None)
        if official and official != name:
            aliases.add(official)
        aliases.update(_COUNTRY_ALIASES.get(name, []))
        out[name] = sorted(a for a in aliases if a and a != name)
        if limit and len(out) >= limit:
            break
    return out


# USPS-style: pycountry subdivision code is "US-CA"; the bare code "CA" is the alias.
def _subdivision_entries(country_code: str = "US") -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for s in pycountry.subdivisions.get(country_code=country_code) or []:
        bare = s.code.split("-")[-1]
        out[s.name] = [bare]
    return out


def _currency_entries(limit: int = 60) -> dict[str, list[str]]:
    # Canonical = ISO code (USD); aliases = the long name and a few symbols.
    symbols = {"USD": ["$", "US$"], "EUR": ["€"], "GBP": ["£"], "JPY": ["¥"], "INR": ["₹"]}
    out: dict[str, list[str]] = {}
    for c in pycountry.currencies:
        code = c.alpha_3
        out[code] = [c.name, *symbols.get(code, [])]
        if len(out) >= limit:
            break
    return out


_CITY_ALIASES = {
    "New York": ["NYC", "New York City", "NY"], "Los Angeles": ["LA", "L.A."],
    "San Francisco": ["SF", "San Fran"], "Las Vegas": ["Vegas"],
    "Washington": ["Washington D.C.", "DC", "Washington DC"], "Chicago": ["Chi-town"],
    "London": ["LDN"], "Mexico City": ["CDMX", "Ciudad de Mexico"],
    "Sao Paulo": ["São Paulo", "SP"], "Tokyo": ["Tōkyō"], "Mumbai": ["Bombay"],
    "Bengaluru": ["Bangalore"], "Istanbul": ["Constantinople"], "Beijing": ["Peking"],
    "Philadelphia": ["Philly"], "New Orleans": ["NOLA"], "Hong Kong": ["HK"],
    "Rio de Janeiro": ["Rio"], "Buenos Aires": ["BA"], "Amsterdam": ["A'dam"],
}


def _department_entries() -> dict[str, list[str]]:
    return {
        "Engineering": ["Eng", "Eng.", "R&D", "Dev"], "Sales": ["Biz Dev"],
        "Marketing": ["Mktg", "Mkt", "Growth"], "Human Resources": ["HR", "People", "People Ops"],
        "Finance": ["Fin", "Accounting"], "Operations": ["Ops"],
        "Customer Support": ["Support", "CS", "Cust Support"], "Legal": ["Legal & Compliance"],
        "Product": ["Prod", "PM"], "Information Technology": ["IT", "I.T."],
    }


def _job_title_entries() -> dict[str, list[str]]:
    return {
        "Senior Engineer": ["Sr. Engineer", "Sr Engineer", "Senior Eng", "Snr Engineer"],
        "Engineering Manager": ["Eng Manager", "Eng Mgr", "Engineering Mgr"],
        "Chief Executive Officer": ["CEO", "C.E.O."],
        "Chief Technology Officer": ["CTO", "C.T.O."],
        "Vice President": ["VP", "V.P.", "Vice Pres"],
        "Account Executive": ["AE", "Acct Exec"], "Sales Representative": ["Sales Rep", "Rep"],
        "Product Manager": ["PM", "Prod Manager", "Prod Mgr"],
        "Administrative Assistant": ["Admin Assistant", "Admin Asst", "Admin"],
        "Director": ["Dir", "Dir."],
    }


CITIES = [
    "New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia",
    "San Antonio", "San Diego", "Dallas", "San Francisco", "Austin", "Seattle",
    "Denver", "Boston", "Washington", "Las Vegas", "Portland", "Miami", "Atlanta",
    "New Orleans", "London", "Paris", "Berlin", "Madrid", "Rome", "Amsterdam",
    "Lisbon", "Dublin", "Vienna", "Prague", "Warsaw", "Stockholm", "Oslo",
    "Tokyo", "Osaka", "Seoul", "Beijing", "Shanghai", "Mumbai", "Delhi",
    "Bengaluru", "Singapore", "Hong Kong", "Bangkok", "Jakarta", "Manila",
    "Sydney", "Melbourne", "Toronto", "Vancouver", "Montreal", "Mexico City",
    "Sao Paulo", "Rio de Janeiro", "Buenos Aires", "Bogota", "Lima", "Santiago",
    "Cairo", "Lagos", "Nairobi", "Johannesburg", "Dubai", "Istanbul",
]


# Lazily-built, cached vocab dicts (pycountry iteration is a touch slow).
_CACHE: dict[str, dict[str, list[str]]] = {}


def _cached(key: str, builder) -> dict[str, list[str]]:
    if key not in _CACHE:
        _CACHE[key] = builder()
    return _CACHE[key]


def country_vocab() -> dict[str, list[str]]:
    return _cached("country", _country_entries)


def state_vocab() -> dict[str, list[str]]:
    return _cached("state", lambda: _subdivision_entries("US"))


def currency_vocab() -> dict[str, list[str]]:
    return _cached("currency", lambda: _currency_entries(60))


def city_vocab() -> dict[str, list[str]]:
    return _cached("city", lambda: {c: list(_CITY_ALIASES.get(c, [])) for c in CITIES})


def department_vocab() -> dict[str, list[str]]:
    return _cached("department", _department_entries)


def job_title_vocab() -> dict[str, list[str]]:
    return _cached("job_title", _job_title_entries)


# Status / categorical value sets (canonical -> messy variants).
_STATUS_SETS = [
    {"Won": ["won", "WON", "Closed Won", "closed-won"], "Lost": ["lost", "Closed Lost"],
     "In Progress": ["in progress", "in-progress", "WIP", "ongoing"], "Open": ["open", "new"]},
    {"High": ["high", "HIGH", "H", "P1"], "Medium": ["medium", "med", "M", "P2"],
     "Low": ["low", "L", "P3"]},
    {"Active": ["active", "ACTIVE"], "Churned": ["churned", "cancelled", "canceled"],
     "Trial": ["trial", "TRIAL", "free trial"], "Paused": ["paused", "on hold"]},
    {"Paid": ["paid", "PAID"], "Pending": ["pending", "unpaid", "due"],
     "Refunded": ["refunded", "refund"], "Overdue": ["overdue", "late"]},
]
