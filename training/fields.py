"""Field archetypes: clean-value generators + matched corruptors.

Each archetype produces clean values in the SAME canonical/typed representation
that scrubdata.executor outputs, and a `corrupt()` that dirties a clean column
while returning the exact ground-truth column-operations. Designed so that
executor(dirty, ops) == clean (verified downstream).
"""

from __future__ import annotations

import random

from . import vocab as V

# ---- shared corruption helpers ----------------------------------------------

DISGUISED = ["N/A", "na", "-", "--", "null", "None", "?", "#N/A", "TBD",
             "empty", "(empty)", "n/a", "NULL", "none", "unknown"]


def _add_whitespace(rng: random.Random, s: str) -> str:
    choice = rng.random()
    if choice < 0.4:
        return " " * rng.randint(1, 3) + s
    if choice < 0.7:
        return s + " " * rng.randint(1, 3)
    # doubled internal space
    parts = s.split(" ")
    if len(parts) > 1:
        i = rng.randrange(len(parts) - 1)
        parts[i] = parts[i] + " "
        return " ".join(parts)
    return " " + s + " "


def _inject_disguised_nulls(rng: random.Random, values, clean, p=0.12):
    """Randomly turn some cells into disguised-null tokens; clean value = None."""
    used = False
    out_dirty, out_clean = [], []
    for d, c in zip(values, clean):
        if rng.random() < p:
            out_dirty.append(rng.choice(DISGUISED))
            out_clean.append(None)
            used = True
        else:
            out_dirty.append(d)
            out_clean.append(c)
    return out_dirty, out_clean, used


# ---- archetypes --------------------------------------------------------------

class Field:
    semantic_type = "text"
    names: list[str] = []

    def gen_clean(self, rng: random.Random, n: int):
        raise NotImplementedError

    def corrupt(self, rng: random.Random, clean):
        """Return (dirty_values, clean_values, ops, issues)."""
        raise NotImplementedError


class NameField(Field):
    semantic_type = "text"
    names = ["name", "full_name", "customer", "contact", "rep"]
    FIRST = ["Alice", "Bob", "Carol", "David", "Eve", "Frank", "Grace", "Heidi",
             "Ivan", "Judy", "Karl", "Lena", "Mona", "Omar", "Priya", "Sara"]
    LAST = ["Johnson", "Smith", "Diaz", "Lee", "Adams", "Moore", "Park", "Cruz",
            "Petrov", "Wong", "Brandt", "Fischer", "Ali", "Khan", "Novak", "Reyes"]

    def gen_clean(self, rng, n):
        return [f"{rng.choice(self.FIRST)} {rng.choice(self.LAST)}" for _ in range(n)]

    def corrupt(self, rng, clean):
        dirty = [_add_whitespace(rng, c) if rng.random() < 0.5 else c for c in clean]
        ops = [{"op": "strip_whitespace",
                "rationale": "Trimmed leading/trailing and doubled spaces."}]
        return dirty, clean, ops, ["whitespace"]


class CompanyField(NameField):
    names = ["company", "organization", "account", "employer"]
    POOL = ["Acme Inc", "Globex", "Initech", "Umbrella", "Soylent Corp", "Hooli",
            "Vehement", "Stark Industries", "Wonka Co", "Cyberdyne"]

    def gen_clean(self, rng, n):
        return [rng.choice(self.POOL) for _ in range(n)]


class EmailField(Field):
    semantic_type = "email"
    names = ["email", "email_address", "contact_email"]

    def gen_clean(self, rng, n):
        out = []
        for _ in range(n):
            user = "".join(rng.choice("abcdefghijklmnop") for _ in range(rng.randint(4, 7)))
            dom = rng.choice(["example.com", "mail.com", "corp.io", "test.org"])
            out.append(f"{user}@{dom}")
        return out

    def corrupt(self, rng, clean):
        dirty = []
        for c in clean:
            v = c.upper() if rng.random() < 0.5 else c
            if rng.random() < 0.4:
                v = _add_whitespace(rng, v)
            dirty.append(v)
        ops = [{"op": "normalize_email",
                "rationale": "Lowercased and trimmed email addresses."}]
        return dirty, clean, ops, ["casing", "whitespace"]


class VocabField(Field):
    """Categorical column backed by a real vocabulary (canonical -> aliases).

    LOW-card mode (default): draws a FEW canonicals (every surface shows in the
    sample). HIGH-card mode (high_card=True): draws MANY (min_card..max_card, e.g.
    30..80) real canonicals with a DOMINANT-canonical long-tailed row distribution
    and single-char-substitution typos in the tail — replicating the hospital
    birmingham(75) + birminghxm(1) regime. Both corrupt() and record surface->
    canonical so canonicalize_categories recovers the clean value (self-verified)."""

    def __init__(self, names, semantic_type, entries, max_card=5, min_card=2,
                 high_card=False, typo_p=0.13):
        self.names = names
        self.semantic_type = semantic_type
        self.entries = entries
        self._canonicals = list(entries)
        self.max_card = max_card
        self.min_card = min_card
        self.high_card = high_card
        self.typo_p = typo_p

    def _choose(self, rng):
        lo = max(2, min(self.min_card, len(self._canonicals)))
        hi = min(self.max_card, len(self._canonicals))
        k = rng.randint(min(lo, hi), hi)
        return rng.sample(self._canonicals, k)

    def _gen_rows(self, rng, n):
        """Long-tailed row draw: a few dominant canonicals carry most of the mass,
        the rest form a sparse tail (where typo surfaces land as rare singletons).
        Falls back to uniform for low-card columns."""
        chosen = self._chosen
        if not self.high_card or len(chosen) < 6:
            return [rng.choice(chosen) for _ in range(n)]
        # Zipf-like weights: a couple of dominant values, steeply decaying tail.
        order = list(chosen)
        rng.shuffle(order)
        weights = [1.0 / ((i + 1) ** 1.6) for i in range(len(order))]
        # Boost the single top canonical so a clear dominant emerges (birmingham 75).
        weights[0] *= 3.0
        return rng.choices(order, weights=weights, k=n)

    def gen_clean(self, rng, n):
        self._chosen = self._choose(rng)
        return self._gen_rows(rng, n)

    def _surface_for(self, rng, c, force_typo):
        """One dirty surface for canonical c. force_typo guarantees a single-char
        substitution typo (rare-tail birminghxm regime)."""
        aliases = self.entries.get(c, [])
        if force_typo:
            s = V.make_substitution_typo(rng, c)
            return s
        return V.make_surface(rng, c, aliases, typo_p=self.typo_p)

    def corrupt(self, rng, clean):
        # Decide which canonicals get a guaranteed single-char typo surface (high-card
        # only): a controlled fraction of the present canonicals, applied to ONE of
        # their occurrences so it lands as a rare tail singleton.
        present = list(dict.fromkeys(clean))
        forced_typo_canon = set()
        if self.high_card:
            frac = rng.uniform(0.3, 0.6)
            k = max(1, int(len(present) * frac))
            forced_typo_canon = set(rng.sample(present, min(k, len(present))))
        # Reserve, per forced canonical, exactly one row index to carry the typo.
        forced_slot = {}
        if forced_typo_canon:
            for canon in forced_typo_canon:
                idxs = [i for i, c in enumerate(clean) if c == canon]
                if idxs:
                    forced_slot[rng.choice(idxs)] = canon

        # Build mapping collision-safely: a surface may only map to ONE canonical, and
        # a surface that equals some canonical's clean form must not be remapped.
        # Reserve all clean canonical strings as "do not remap" keys.
        reserved = {str(c).strip() for c in present}
        mapping = {}
        dirty, ws = [], False
        for i, c in enumerate(clean):
            force = i in forced_slot
            for _attempt in range(4):
                s = self._surface_for(rng, c, force_typo=force)
                key = str(s).strip()
                if key == str(c).strip():
                    break  # already canonical surface, no mapping needed
                # Skip surfaces that collide with another canonical, or that some
                # other canonical already claims (would make the mapping ambiguous).
                if key in reserved:
                    s = c       # ambiguous -> fall back to clean (still verifies)
                    break
                if key in mapping and mapping[key] != c:
                    s = c       # collision with a different canonical's surface
                    break
                break
            key = str(s).strip()
            if key != str(c).strip() and key not in reserved:
                mapping[key] = c
            cell = s
            # whitespace noise (less often on high-card to keep the tail clean)
            if rng.random() < (0.12 if self.high_card else 0.25):
                cell = _add_whitespace(rng, s)
                ws = True
            dirty.append(cell)

        ops, issues = [], ["inconsistent_categories", "casing"]
        if ws:  # strip first so canonicalize sees the bare surface (executor order)
            ops.append({"op": "strip_whitespace",
                        "rationale": "Trimmed surrounding/doubled spaces."})
            issues.append("whitespace")
        if mapping:
            ops.append({"op": "canonicalize_categories", "mapping": mapping,
                        "rationale": f"Unified {len(mapping)} variant spelling(s) "
                                     f"into canonical labels."})
        return dirty, clean, ops, issues


class StatusField(VocabField):
    """Like VocabField but picks a fresh status/category value-set each example."""

    def __init__(self):
        super().__init__(
            names=["status", "stage", "tier", "segment", "state", "payment_status"],
            semantic_type="categorical", entries={}, max_card=4)

    def gen_clean(self, rng, n):
        self.entries = rng.choice(V._STATUS_SETS)
        self._canonicals = list(self.entries)
        self._chosen = self._choose(rng)
        return self._gen_rows(rng, n)


class CurrencyField(Field):
    semantic_type = "currency"
    names = ["amount", "revenue", "price", "deal_size", "cost"]

    def gen_clean(self, rng, n):
        return [round(rng.uniform(50, 9000), 2) for _ in range(n)]

    def _fmt(self, rng, x: float) -> str:
        neg = x < 0
        a = abs(x)
        style = rng.random()
        if style < 0.4:
            s = f"${a:,.2f}"
        elif style < 0.7 and a == int(a):
            s = f"{int(a):,d}"            # grouped integer — only when no cents to lose
        else:  # EU style (comma decimal) — always preserves 2 decimals
            s = f"{a:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        return f"({s})" if neg else s

    def corrupt(self, rng, clean):
        dirty = [self._fmt(rng, c) for c in clean]
        dirty, clean2, used_null = _inject_disguised_nulls(rng, dirty, clean)
        ops, issues = [], ["numeric_stored_as_text", "currency_symbols"]
        if used_null:
            ops.append({"op": "normalize_disguised_nulls",
                        "rationale": "Converted N/A, '-', 'null' etc. to true missing."})
            issues.append("disguised_nulls")
        ops.append({"op": "parse_currency",
                    "rationale": "Stripped currency symbols/grouping; parsed to number."})
        return dirty, clean2, ops, issues


class DateField(Field):
    semantic_type = "date"
    names = ["signup_date", "created_at", "close_date", "date", "order_date"]

    def gen_clean(self, rng, n):
        out = []
        for _ in range(n):
            y, m, d = 2023, rng.randint(1, 12), rng.randint(1, 28)
            out.append(f"{y:04d}-{m:02d}-{d:02d}")
        return out

    def _fmt(self, rng, iso: str) -> str:
        y, m, d = iso.split("-")
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug",
                  "Sep", "Oct", "Nov", "Dec"]
        style = rng.random()
        if style < 0.3:
            return iso
        if style < 0.55:
            return f"{int(m)}/{int(d)}/{y}"          # US slash (m<=12, d<=28 -> unambiguous-ish)
        if style < 0.8:
            return f"{int(d)} {months[int(m)-1]} {y}"  # 5 Jan 2023
        # Excel serial
        import datetime
        base = datetime.date(1899, 12, 30)
        serial = (datetime.date(int(y), int(m), int(d)) - base).days
        return str(serial)

    def corrupt(self, rng, clean):
        dirty = [self._fmt(rng, c) for c in clean]
        ops = [{"op": "parse_date",
                "rationale": "Unified mixed date formats to ISO YYYY-MM-DD."}]
        return dirty, clean, ops, ["mixed_date_formats"]


class BooleanField(Field):
    semantic_type = "boolean"
    names = ["is_active", "subscribed", "verified", "opted_in"]
    TRUE = ["Yes", "Y", "TRUE", "true", "1", "T"]
    FALSE = ["No", "N", "FALSE", "false", "0", "F"]

    def gen_clean(self, rng, n):
        return [rng.random() < 0.5 for _ in range(n)]

    def corrupt(self, rng, clean):
        dirty = [rng.choice(self.TRUE if c else self.FALSE) for c in clean]
        ops = [{"op": "standardize_boolean",
                "rationale": "Mapped Yes/Y/1/TRUE → true, No/N/0/FALSE → false."}]
        return dirty, clean, ops, ["inconsistent_booleans"]


class PhoneField(Field):
    semantic_type = "phone"
    names = ["phone", "phone_number", "mobile", "contact_number"]

    def gen_clean(self, rng, n):
        # Canonical = executor's output for a plain 10-digit US number.
        out, self._digits = [], []
        for _ in range(n):
            d = "".join(str(rng.randint(0, 9)) for _ in range(10))
            d = "5" + d[1:]  # keep it phone-ish
            self._digits.append(d)
            out.append(f"({d[0:3]}) {d[3:6]}-{d[6:]}")
        return out

    def corrupt(self, rng, clean):
        dirty = []
        for d in self._digits:
            style = rng.random()
            if style < 0.25:
                dirty.append(f"{d[0:3]}.{d[3:6]}.{d[6:]}")
            elif style < 0.5:
                dirty.append(f"{d[0:3]}-{d[3:6]}-{d[6:]}")
            elif style < 0.75:
                dirty.append(d)
            else:
                dirty.append(f"({d[0:3]}){d[3:6]}-{d[6:]}")
        ops = [{"op": "standardize_phone",
                "rationale": "Standardized phone formatting."}]
        return dirty, clean, ops, ["inconsistent_formats"]


class PercentField(Field):
    semantic_type = "percent"
    names = ["rate", "discount", "completion", "margin", "growth", "conversion"]

    def gen_clean(self, rng, n):
        self._pct = [round(rng.uniform(0, 100), 1) for _ in range(n)]
        return [p / 100 for p in self._pct]

    def corrupt(self, rng, clean):
        dirty = [f"{p}%" for p in self._pct]
        ops = [{"op": "parse_percent", "rationale": "Parsed percent text to a fraction."}]
        return dirty, clean, ops, ["numeric_stored_as_text"]


ARCHETYPES: list[Field] = [
    NameField(), CompanyField(), EmailField(), PercentField(),
    VocabField(["country", "nation", "country_name"], "country", V.country_vocab(), max_card=5),
    VocabField(["state", "province", "region"], "state", V.state_vocab(), max_card=5),
    VocabField(["currency", "currency_code", "ccy"], "categorical", V.currency_vocab(), max_card=4),
    VocabField(["city", "location", "hq_city"], "city", V.city_vocab(), max_card=5),
    VocabField(["department", "dept", "team"], "categorical", V.department_vocab(), max_card=4),
    VocabField(["job_title", "title", "role", "position"], "categorical", V.job_title_vocab(), max_card=4),
    # real O*NET occupations (alternate title -> canonical, CC BY 4.0): 1,016 canonicals
    *([VocabField(["job_title", "occupation", "role"], "categorical",
                  V._cached("onet", lambda: V._alias_file("onet_jobtitle_aliases.jsonl", limit=1016)),
                  max_card=5),
       VocabField(["job_title", "occupation"], "categorical",
                  V._cached("onet", lambda: V._alias_file("onet_jobtitle_aliases.jsonl", limit=1016)),
                  min_card=25, max_card=60, high_card=True)]
      if V._alias_file("onet_jobtitle_aliases.jsonl", limit=2) else []),
    # real nickname->formal first names (Bill -> William; Apache-2.0)
    *([VocabField(["first_name", "given_name", "contact_first"], "categorical",
                  V.nickname_vocab(), max_card=5),
       VocabField(["first_name", "given_name"], "categorical",
                  V.nickname_vocab(), min_card=25, max_card=60, high_card=True)]
      if V.nickname_vocab() else []),
    VocabField(["industry", "sector", "vertical"], "categorical", V.industry_vocab(), max_card=4),
    # real Wikidata companies (alias -> canonical: 'AB InBev' -> 'Anheuser-Busch InBev')
    *([VocabField(["company", "vendor", "account", "supplier"], "categorical",
                  V.company_vocab(), max_card=5),
       VocabField(["company", "vendor", "account"], "categorical",
                  V.company_vocab(), min_card=25, max_card=60, high_card=True)]
      if V.company_vocab() else []),
    # real ROR organizations (alias/acronym -> canonical): both low-card and the
    # hospital-style high-cardinality long-tail regime. Skipped if harvest absent.
    *([VocabField(["organization", "institution", "affiliation", "employer"], "categorical",
                  V.org_vocab(), max_card=5),
       VocabField(["organization", "institution", "affiliation"], "categorical",
                  V.org_vocab(), min_card=25, max_card=60, high_card=True)]
      if V.org_vocab() else []),
    VocabField(["unit", "uom", "measure_unit"], "categorical", V.unit_vocab(), max_card=4),
    StatusField(),
    CurrencyField(), DateField(), BooleanField(), PhoneField(),
]
