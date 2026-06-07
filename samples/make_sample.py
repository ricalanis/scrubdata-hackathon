"""Generate a realistically *dirty* office-style leads export.

Deterministic (no RNG) so the demo and tests are reproducible. Every row is
hand-crafted to exhibit specific problems from PRODUCT.md section 2, so the
profiler/planner/executor each have something real to chew on.

Run:  uv run samples/make_sample.py
Out:  samples/dirty_contacts.csv
"""

from __future__ import annotations

import csv
from pathlib import Path

# Columns:
#   name, email, company, country, signup_date, amount, phone, status, is_active, age, notes2
# Problems baked in per column:
#   email     -> casing, whitespace, a typo domain
#   company   -> casing, whitespace, smart quotes
#   country   -> USA/U.S.A./united states/us, UK/U.K., whitespace, casing
#   signup_date -> ISO, US slash, EU slash, "5 Jan 2023", Excel serial
#   amount    -> "$1,200.50", "1.200,50", "(500)", "12", blank
#   phone     -> every format imaginable
#   status    -> Won/won/WON, Lost/lost, "In Progress"/"in-progress"
#   is_active -> Yes/Y/TRUE/1 vs No/N/FALSE/0
#   age       -> ints, one 999 outlier, one blank, one "N/A"
#   notes2    -> entirely empty column (should be dropped)
ROWS = [
    ["  Alice Johnson", "ALICE@EXAMPLE.COM ", "Acme Inc", "USA", "2023-01-05", "$1,200.50", "(555) 123-4567", "Won", "Yes", "34", ""],
    ["Bob Smith", "bob@example.com", "acme inc ", "U.S.A.", "01/06/2023", "950", "555.234.5678", "won", "Y", "41", ""],
    ["Carol  Diaz", "carol@gmial.com", "Globex", "united states", "07/01/2023", "1.200,50", "+1 555 345 6789", "WON", "TRUE", "29", ""],
    ["David Lee", "  david@example.com", "globex corp", "US", "5 Jan 2023", "(500)", "5553456789", "Lost", "1", "52", ""],
    ["Eve Adams", "eve@example.com", "Initech", "Canada", "44931", "2,300", "555-456-7890", "lost", "No", "999", ""],
    ["Frank Moore", "FRANK@example.com ", "initech", "canada", "2023-02-14", "", "(555)567-8901", "In Progress", "N", "", ""],
    ["Grace Park", "grace@example.com", "Umbrella", "UK", "14/02/2023", "$3,000", "+44 20 7946 0958", "in-progress", "FALSE", "38", ""],
    ["Heidi Cruz", "heidi@example.com", "umbrella corp ", "U.K.", "2023-03-01", "1,750.00", "020 7946 0958", "Won", "0", "N/A", ""],
    ["  Ivan Petrov", "IVAN@EXAMPLE.COM", "“Soylent”", "Germany", "03/02/2023", "4.500,00", "555 678 9012", "WON", "yes", "45", ""],
    ["Judy Wong", "judy@example.com ", "Soylent Corp", "germany", "2 Mar 2023", "2200", "(555) 789-0123", "Lost", "no", "33", ""],
    # exact duplicate of row 0 (appears 3x total with the next one)
    ["  Alice Johnson", "ALICE@EXAMPLE.COM ", "Acme Inc", "USA", "2023-01-05", "$1,200.50", "(555) 123-4567", "Won", "Yes", "34", ""],
    ["  Alice Johnson", "ALICE@EXAMPLE.COM ", "Acme Inc", "USA", "2023-01-05", "$1,200.50", "(555) 123-4567", "Won", "Yes", "34", ""],
    # an entirely empty row
    ["", "", "", "", "", "", "", "", "", "", ""],
    ["Karl Brandt", "karl@example.com", "Hooli", "United States", "2023-04-10", "$5,400.00", "+1 (555) 890-1234", "Won", "TRUE", "47", ""],
    ["Lena Fischer", "lena@example.com", "hooli inc", "USA ", "10/04/2023", "-", "555.901.2345", "in progress", "T", "31", ""],
    ["Mona Ali", "mona@example.com", "Vehement", "U.S.A", "2023-05-22", "6,000", "5559012345", "Won", "F", "39", ""],
]

HEADER = ["name", "email", "company", "country", "signup_date",
          "amount", "phone", "status", "is_active", "age", "notes2"]


def main() -> None:
    out = Path(__file__).parent / "dirty_contacts.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(ROWS)
    print(f"Wrote {out} ({len(ROWS)} rows x {len(HEADER)} cols)")


if __name__ == "__main__":
    main()
