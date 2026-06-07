"""Field archetypes: clean-value generators + matched corruptors.

Each archetype produces clean values in the SAME canonical/typed representation
that scrubdata.executor outputs, and a `corrupt()` that dirties a clean column
while returning the exact ground-truth column-operations. Designed so that
executor(dirty, ops) == clean (verified downstream).
"""

from __future__ import annotations

import random

# ---- shared corruption helpers ----------------------------------------------

DISGUISED = ["N/A", "na", "-", "--", "null", "None", "?", "#N/A", "TBD"]


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


class CountryField(Field):
    semantic_type = "country"
    names = ["country", "nation", "country_name"]
    VARIANTS = {
        "United States": ["USA", "U.S.A.", "us", "united states", "U.S.", "United States"],
        "United Kingdom": ["UK", "U.K.", "united kingdom", "Great Britain"],
        "Canada": ["Canada", "canada", "CA"],
        "Germany": ["Germany", "germany", "DE", "Deutschland"],
    }

    def gen_clean(self, rng, n):
        canon = list(self.VARIANTS)
        return [rng.choice(canon) for _ in range(n)]

    def corrupt(self, rng, clean):
        dirty, mapping = [], {}
        for c in clean:
            surface = rng.choice(self.VARIANTS[c])
            dirty.append(surface)
            if surface.strip() != c:
                mapping[surface.strip()] = c
        ops = []
        if mapping:
            ops.append({"op": "canonicalize_categories", "mapping": mapping,
                        "rationale": f"Unified {len(mapping)} spellings into canonical names."})
        return dirty, clean, ops, ["inconsistent_categories", "casing"]


class CategoricalField(Field):
    semantic_type = "categorical"
    names = ["status", "priority", "stage", "tier", "segment"]
    SETS = [
        {"Won": ["Won", "won", "WON"], "Lost": ["Lost", "lost"],
         "In Progress": ["In Progress", "in progress", "in-progress"]},
        {"High": ["High", "high", "HIGH"], "Medium": ["Medium", "medium", "med"],
         "Low": ["Low", "low"]},
        {"Active": ["Active", "active"], "Churned": ["Churned", "churned"],
         "Trial": ["Trial", "trial", "TRIAL"]},
    ]

    def gen_clean(self, rng, n):
        self._set = rng.choice(self.SETS)
        canon = list(self._set)
        return [rng.choice(canon) for _ in range(n)]

    def corrupt(self, rng, clean):
        mapping, dirty = {}, []
        for c in clean:
            surface = rng.choice(self._set[c])
            dirty.append(surface)
            if surface.strip() != c:
                mapping[surface.strip()] = c
        ops = []
        if mapping:
            ops.append({"op": "canonicalize_categories", "mapping": mapping,
                        "rationale": f"Unified {len(mapping)} label variants."})
        return dirty, clean, ops, ["inconsistent_categories", "casing"]


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


ARCHETYPES: list[Field] = [
    NameField(), CompanyField(), EmailField(), CountryField(), CategoricalField(),
    CurrencyField(), DateField(), BooleanField(), PhoneField(),
]
