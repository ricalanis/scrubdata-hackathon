"""P0 gate: does a sentence-trained PII token classifier transfer to bare CSV cells?

OpenMed-PII models report ~96% F1 on sentence-level clinical text (Nemotron-PII). Our
regime is context-free cell values ("John Smith", "742 Evergreen Terrace"). arXiv
2504.12308 documents how PII F1 collapses out-of-distribution, so before adopting a
tier-2 model we measure, per concept:

  * detection rate on BARE values vs a minimal CONTEXT TEMPLATE ("Name: {v}.")
  * false-positive rate on negatives (cities, products, job titles)

Tier-2's value-add over our checksum tier is names/addresses/orgs — so that's what we
probe. Decision rule: adopt the variant (bare vs templated) that detects >=0.8 on
positives with low FP; otherwise tier-2 stays off and we ship tier-1 only.

    uv run --with transformers --with torch python scripts/pii_transfer_check.py
"""

import random

MODEL = "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1"

FIRST = ["James", "Maria", "Wei", "Aisha", "Carlos", "Yuki", "Priya", "Omar",
         "Elena", "Kwame", "Ingrid", "Rahul", "Fatima", "Liam", "Sofia", "Chen"]
LAST = ["Smith", "Garcia", "Johnson", "Patel", "Kim", "Müller", "Rossi", "Khan",
        "Andersen", "Okafor", "Tanaka", "Novak", "Silva", "Dubois", "Lopez", "Wong"]
STREETS = ["Evergreen Terrace", "Oak Street", "Maple Avenue", "Main St", "Elm Drive",
           "Sunset Blvd", "5th Avenue", "Cedar Lane", "Park Road", "Hillcrest Court"]
NEGATIVES = ["Boston", "Chicago", "San Francisco", "Blue Widget XL", "Stainless Bolt M6",
             "Account Manager", "Senior Engineer", "Operations", "Pending Review",
             "Closed Won", "North Region", "Premium Tier", "Q3 Forecast", "Warehouse B"]


def make_probes(n=40, seed=11):
    rng = random.Random(seed)
    names = [f"{rng.choice(FIRST)} {rng.choice(LAST)}" for _ in range(n)]
    addrs = [f"{rng.randint(12, 9899)} {rng.choice(STREETS)}" for _ in range(n)]
    return {"person_name": names, "address": addrs, "NEGATIVE": NEGATIVES}


def main():
    from transformers import pipeline
    print(f"loading {MODEL} ...")
    ner = pipeline("token-classification", model=MODEL, aggregation_strategy="simple")
    probes = make_probes()
    templates = {"bare": "{v}", "templated": "Name: {v}."}
    print(f"\n{'concept':<14}{'variant':<11}{'detect%':>9}  top entity labels seen")
    print("-" * 70)
    for concept, values in probes.items():
        for tname, tmpl in templates.items():
            hits, labels = 0, {}
            for v in values:
                ents = ner(tmpl.format(v=v))
                # count a hit if any predicted span overlaps the VALUE (not the template)
                real = [e for e in ents if e["word"].strip(" .:#").lower() not in ("name",)]
                if real:
                    hits += 1
                    for e in real:
                        labels[e["entity_group"]] = labels.get(e["entity_group"], 0) + 1
            top = sorted(labels.items(), key=lambda x: -x[1])[:3]
            rate = hits / len(values)
            print(f"{concept:<14}{tname:<11}{rate:>8.0%}  {top}")
    print("\nDecision rule: positives need >=80% detection; NEGATIVE rows are the "
          "false-positive rate (lower is better). If bare fails but templated passes, "
          "tier-2 wraps cell values in the template.")


if __name__ == "__main__":
    main()
