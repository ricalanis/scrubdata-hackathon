"""Generate samples/maria_crm_export.csv — the hero demo dataset.

Persona: Maria, ops coordinator, exported the company CRM on Monday morning.
The file is engineered against the SHIPPED pipeline (scrubdata.planner.mock_plan +
reconcile.grounded_mapping + pii.detect_column_pii) so that:
  - country / state / plan_tier canonicalize (auto-fix)
  - dates / mrr / phone normalize (auto-fix)
  - cp1252->utf8 mojibake repairs
  - a dedicated credit-card column (Luhn-valid) -> flag_pii + mask_pii
  - exact + near-duplicate rows -> drop_exact_duplicates / left as judgment
  - 3 GENUINE 'YOUR CALL' abstentions actually fire as review flags:
      Slovia  -> Slovakia vs Slovenia  (score 0.857, margin 0.0)  ABSTAIN
      Austrai -> Australia vs Austria   (score 0.875, margin 0.018) ABSTAIN
      Indai   -> India near-miss        (score 0.80  < 0.84)        ABSTAIN
  - PLUS the Acme Corp vs ACME Corporation entity near-tie, which the tool
    correctly does NOT auto-merge (documented as the human judgment call).

Deterministic: fixed seed, fixed planted rows. Re-runnable.
"""

from __future__ import annotations

import csv
import random

random.seed(20260612)

HEADER = ["company", "contact_name", "email", "phone", "country", "state",
          "signup_date", "plan_tier", "mrr", "status", "cc_on_file", "notes"]

# --- Luhn-valid fake cards (verified by pii.luhn_ok) -------------------------
CARDS = [
    "4539578763621486", "4916 3385 0608 2832", "5500-0055-5555-5559",
    "4111 1111 1111 1111", "6011000990139424", "3782 822463 10005",
    "4485275742308327", "5105105105105100",
]
# clearly-fake SSN-shaped values for the notes field
SSNS = ["123-45-6789", "078-05-1120", "219-09-9999", "457-55-5462"]

# canonical pools (each value carries its target frequency intent via how often we draw it)
PHONE_STYLES = [
    "(555) {a}-{b}", "555.{a}.{b}", "+1 555 {a} {b}", "555{a}{b}", "1-555-{a}-{b}",
]
DATE_STYLES = ["iso", "slash", "text", "dmy"]
PLAN_VARIANTS = ["Premium", "premium", "PREMIUM", "Prem", "Basic", "basic", "BASIC",
                 "Enterprise", "enterprise", "Ent", "Free", "free"]
STATUS_VARIANTS = ["Active", "active", "ACTIVE", "Churned", "churned", "Trial", "trial",
                   "Lapsed", "lapsed"]

# Large, distinctive pools so random First+Last pairs DON'T collide into spurious
# one-edit "typo" clusters (those wrongly trip the high-cardinality suspect path and
# pollute the demo with fake name merges). Distinctive surnames keep edit-distance high.
FIRST = ["Maria", "James", "Wei", "Aisha", "Carlos", "Priya", "Tom", "Sofia", "Liam",
         "Yuki", "Hassan", "Elena", "Jonathan", "Fatima", "Diego", "Anneliese", "Raj",
         "Chloe", "Omar", "Ingrid", "Pedro", "Gabrielle", "Ivan", "Lucia", "Samuel",
         "Zoe", "Benjamin", "Mei", "Olivier", "Tatiana", "Kwame", "Beatriz", "Soren",
         "Yasmin", "Dmitri", "Imani", "Lorenzo", "Saoirse", "Mateo", "Freya"]
LAST = ["Hawthorne", "Underwood", "Castellanos", "Okonkwo", "Lindqvist", "Patelkar",
        "Nakamura", "Rossellini", "Fitzgerald", "Abramovich", "Whitfield", "Delacroix",
        "Montgomery", "Vasquez", "Kowalski", "Brennan", "Sandoval", "Thornton",
        "Eriksson", "Bukowski", "Calloway", "Mwangi", "Domingo", "Ferreira",
        "Halvorsen", "Mackenzie", "Petrosyan", "Yamamoto", "Cavendish", "Olszewski"]

# Country surfaces. KEY: every non-canonical spelling must stay rare (freq<3 over the
# WHOLE column) so grounded_mapping folds it; the dominant clean surface carries the bulk.
# 'USA' is intentionally repeated a LOT so the 4-spellings demo is visible, but because
# freq>=3 surfaces are treated as data, the canonicalization is shown via the RARE variants.
CLEAN_COUNTRIES = ["United States", "United Kingdom", "Canada", "Germany", "France",
                   "Spain", "Italy", "Netherlands", "Brazil", "Japan", "Australia",
                   "India", "Mexico", "Sweden", "Poland"]
# rare misspellings that SHOULD auto-fold (each used exactly once or twice):
COUNTRY_TYPOS_AUTOFIX = {
    "U.S.A": "United States", "us": "United States", "USA.": "United States",
    "Nigeia": "Nigeria", "germny": "Germany", "Brasil": "Brazil",
    "Polnd": "Poland", "Swedn": "Sweden", "Frnace": "France",
}
# THE YOUR-CALL country ambiguities (each used once, must abstain):
COUNTRY_ABSTAIN = ["Slovia", "Austrai", "Indai"]

CLEAN_STATES = ["CA", "NY", "TX", "WA", "FL", "IL", "MA", "CO", "GA", "OR", "NJ", "AZ"]
STATE_TYPOS_AUTOFIX = {"Calfornia": "California", "Wahsington": "Washington",
                       "Virgina": "Virginia", "Mississipi": "Mississippi"}
STATE_CLEAN_LONG = ["California", "Texas", "New York", "Florida"]

# Company pools. Acme is the planted entity near-tie.
COMPANIES = ["Globex", "Initech", "Hooli", "Umbrella", "Stark Industries", "Wayne Enterprises",
             "Wonka", "Cyberdyne", "Soylent", "Tyrell", "Massive Dynamic", "Vandelay",
             "Pied Piper", "Aperture", "Black Mesa", "Gekko & Co", "Bluth Company",
             "Dunder Mifflin", "Prestige Worldwide", "Sterling Cooper"]


def _mk_phone() -> str:
    a, b = random.randint(200, 998), random.randint(1000, 9998)
    return random.choice(PHONE_STYLES).format(a=a, b=b)


def _mk_date() -> str:
    y, m, d = 2024, random.randint(1, 12), random.randint(1, 28)
    style = random.choice(DATE_STYLES)
    months = ["January", "February", "March", "April", "May", "June", "July", "August",
              "September", "October", "November", "December"]
    if style == "iso":
        return f"{y}-{m:02d}-{d:02d}"
    if style == "slash":
        return f"{m}/{d}/{str(y)[2:]}"
    if style == "text":
        return f"{months[m-1]} {d} {y}"
    return f"{d:02d}-{m:02d}-{y}"          # dmy: 01-03-2024


def _mk_mrr() -> str:
    base = random.choice([99, 199, 299, 499, 999, 1200, 2400, 4999])
    style = random.randint(0, 3)
    if style == 0:
        return f"${base:,}.00"
    if style == 1:
        return f"{base}.00"
    if style == 2:
        return f"{base:,} USD"
    return str(base)


def _mk_name() -> str:
    n = f"{random.choice(FIRST)} {random.choice(LAST)}"
    r = random.random()
    if r < 0.12:
        return n.upper()
    if r < 0.22:
        return n.lower()
    if r < 0.30:
        return "  " + n + " "          # whitespace chaos
    return n


def _mk_email(name: str) -> str:
    base = name.strip().lower().replace("  ", " ").replace(" ", ".")
    dom = random.choice(["example.com", "acme.co", "globex.io", "mail.com", "corp.net"])
    e = f"{base}@{dom}"
    if random.random() < 0.15:
        e = " " + e.upper() + " "       # case + whitespace chaos for normalize_email
    return e


def base_row() -> dict:
    name = _mk_name()
    return {
        "company": random.choice(COMPANIES),
        "contact_name": name,
        "email": _mk_email(name),
        "phone": _mk_phone(),
        "country": random.choice(CLEAN_COUNTRIES),
        "state": random.choice(CLEAN_STATES),
        "signup_date": _mk_date(),
        "plan_tier": random.choice(PLAN_VARIANTS),
        "mrr": _mk_mrr(),
        "status": random.choice(STATUS_VARIANTS),
        "cc_on_file": random.choice(CARDS),
        "notes": "",
    }


rows: list[dict] = []

# --- bulk filler rows --------------------------------------------------------
for _ in range(330):
    rows.append(base_row())

# --- planted: Acme entity near-tie (genuine YOUR CALL, tool should NOT merge) -
for _ in range(7):
    r = base_row(); r["company"] = "Acme Corp"; rows.append(r)
for _ in range(2):
    r = base_row(); r["company"] = "acme corp "; rows.append(r)      # case+ws -> folds to Acme Corp
r = base_row(); r["company"] = "ACME Corp"; rows.append(r)            # case -> folds
for _ in range(4):
    r = base_row(); r["company"] = "ACME Corporation"; rows.append(r)  # NEAR-TIE: should NOT auto-merge

# --- planted: country canonicalization (auto-fix) + abstentions --------------
for surf in COUNTRY_TYPOS_AUTOFIX:
    r = base_row(); r["country"] = surf; rows.append(r)
# extra visible 4-ways for USA so the demo "sees" the chaos (these are freq>=3 = data)
for _ in range(4):
    r = base_row(); r["country"] = "USA"; rows.append(r)
for _ in range(3):
    r = base_row(); r["country"] = "U.S.A."; rows.append(r)
# THE YOUR CALL country ambiguities — one row each, rare so they reach the abstain path
for surf in COUNTRY_ABSTAIN:
    r = base_row(); r["country"] = surf; rows.append(r)

# --- planted: state canonicalization -----------------------------------------
for surf in STATE_TYPOS_AUTOFIX:
    r = base_row(); r["state"] = surf; rows.append(r)
for surf in STATE_CLEAN_LONG:
    r = base_row(); r["state"] = surf; rows.append(r)

# --- planted: mojibake (cp1252 -> utf8) --------------------------------------
MOJIBAKE = [
    ("CafÃ© Noir SARL", "AndrÃ© Mercier"),       # Café Noir / André
    ("NaÃ¯ve Ventures", "BjÃ¶rn AnderssÃ¸n"),  # Naïve / Björn
    ("MÃ¼ller GmbH", "GÃ¼nther MÃ¼ller"),       # Müller / Günther
    ("CrÃ¨me BrûlÃ©e Co", "FranÃ§ois Dubois"),  # Crème / François
]
for comp, person in MOJIBAKE:
    r = base_row(); r["company"] = comp; r["contact_name"] = person; rows.append(r)

# --- planted: PII in notes (SSN-shaped) --------------------------------------
for ssn in SSNS:
    r = base_row(); r["notes"] = f"follow up re: SSN {ssn} on file"; rows.append(r)

# --- planted: exact + near duplicate rows ------------------------------------
dup_seed = base_row()
dup_seed["company"] = "Initech"; dup_seed["contact_name"] = "Peter Gibbons"
dup_seed["email"] = "peter.gibbons@initech.com"; dup_seed["country"] = "United States"
dup_seed["state"] = "TX"; dup_seed["cc_on_file"] = "4111 1111 1111 1111"
rows.append(dict(dup_seed))
rows.append(dict(dup_seed))                                  # EXACT duplicate
near = dict(dup_seed); near["phone"] = "(555) 867-5309"      # near-dup: one field differs
rows.append(near)
near2 = dict(dup_seed); near2["contact_name"] = "  Peter Gibbons "  # near-dup: whitespace
rows.append(near2)

# --- shuffle so planted rows aren't clustered (keep exact dups adjacent-able) -
random.shuffle(rows)

with open("samples/maria_crm_export.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=HEADER)
    w.writeheader()
    for r in rows:
        w.writerow(r)

print(f"wrote samples/maria_crm_export.csv  ({len(rows)} rows x {len(HEADER)} cols)")
