# Field notes — building ScrubData small, on purpose

*Build Small Hackathon, June 2026. A ≤4B model, a Gradio Space, and two weeks of
finding out what "small but honest" actually costs.*

## The bet

The person who most needs data cleaning — the ops coordinator with a messy CRM export
and a Monday deadline — will never write a pandas script, and shouldn't have to ship
her customer data to a frontier API either. The bet: a 4B model running locally is
enough, **if you stop asking it to edit data and start asking it to plan**.

So the model never touches a cell. It reads an aggregated profile (per-value frequency
counts — a million rows profile like a hundred) and emits a JSON plan; deterministic
pandas executes it. Every change is named, reversible, and logged. Silent edits are
impossible by construction. That decomposition turned out to be the whole project.

## Things that broke, in order

**The fine-tune that aced the test and failed the job.** v4 hit canonicalization F1
0.90 on held-out synthetic data — and scored exactly 0.000 on real hospital typos. It
had never seen a high-cardinality real column. Fix: derive training pairs from real
dirty/clean benchmark tables by cell alignment, keeping only *learnable*
canonicalizations (a surface form that's a string variant of its target and never a
legitimate value elsewhere). Real repair recall: 0.00 → 0.42. Synthetic data teaches
the format; real data teaches the job.

**The GGUF that lobotomized the model.** Same adapter, two exports: Q8_0 worked
perfectly, Q4_K_M degenerated into `<tool_call>` loops. Hours of template debugging
later: the quantization itself was corrupting the export. Then the bf16 path had its
own version — training converged (loss 0.16) but free-running generation *still*
emitted tool-call loops, because Qwen3's tool-calling prior dominates the first token.
The fix is two tokens long: `suppress_tokens=[151657, 151658]`.

**The model that invented cities.** Asked for canonical forms, a generative model
generates — including `guntxrsvillx → huntsville` (wrong town). Frequency clustering
can't fix this either: a lone column has no signal to vote against the error (GARF
proves this structurally). The fix came from the literature: never free-generate a
canonical. Retrieve candidates from a reference taxonomy (GeoNames, ISO), require a
similarity threshold *and* an ambiguity margin, and **abstain** when unsure. `boxz` is
equally close to `Box` and `Boaz` — so the system declines and asks. We measured the
abstention: precision rises monotonically with the threshold (90% at the default, 95%
at 0.91). Knowing when not to act turned out to be the most valuable feature.

**The eval that graded itself too kindly — twice.** Our own ablations caught two metric
artifacts: (1) convention-tolerant scoring counted bulk case-rewrites as "good
changes," inflating precision — removing case-matching *gained* +0.12 until we made
the metric churn-neutral; (2) our adversarial traps included `Boazz`, which grounding
correctly maps to the real city Boaz — the trap was punishing correct behavior. Both
fixes are reported in the paper as results, because an eval you haven't tried to break
is an eval you can't trust.

**The honest negative result.** On *injected* typos, classical frequency clustering
remains a strong baseline — by construction: injection puts the canonical in the
column, which is clustering's ideal regime. Grounding's edge is real errors, tail
entities, and not wrong-merging. We report both slices separately rather than
averaging the difference away.

**The verifier that made the model shippable.** The fine-tune's hospital numbers told
an awkward story: recall 0.475 (best we'd measured for a local model) at precision
0.185 — it fixed errors *and* invented merges. Instead of retraining, we scored every
proposed mapping with three deterministic gates distilled from its actual failures: a
value occurring ≥3 times is data, not a typo (*errors are rare*); a repair target must
dominate its source in frequency (no mapping one typo onto another); digit-bearing
codes only repair when the letter part is near-identical (`amix-2 → ami-2` yes,
`ak_ → al_` no). The gated model plan alone: **0.993 precision at 0.287 coverage** —
146 of 147 changes correct. Union it with the grounded heuristic and you get **0.905
precision at 0.413 coverage** on hospital's 509 real errors. Every dropped mapping
becomes a review flag, not a silent skip. That composition — verify the model's
output, never trust it — is what the app now ships as its default planner.

## The PII turn

A friend pointed at the OpenMed project (small Apache-2.0 token classifiers; their
paper is the sister result to our thesis — small specialized beats big generic). Their
44M PII model, trained on clinical *sentences*, turned out to transfer perfectly to
bare CSV cells: 100% on names and addresses, no prompt template needed. We put it
behind a sensitive-type allowlist and a column-level vote, added a deterministic
checksum tier (Luhn, IBAN mod-97 — math, not vibes), and made masking an executor
operation. Leak test: 0/360 residual detectable PII after masking. OOD type detection:
5/5 with 0/7 false positives. The privacy ribbon at the top of the app — "nothing
leaves this machine" — now describes the PII handling too, not just the inference.

## What we'd tell the next person

1. **Planner/executor is the trust unlock.** Auditability isn't a feature you add;
   it's a decomposition you choose.
2. **Verify supervision by executing it.** Every training example we kept provably
   recovers the clean table. Bad plans can't become labels.
3. **Ground generation in references and budget for abstention.** A small model that
   declines correctly beats a big model that guesses confidently.
4. **Attack your own eval before reviewers do.** Both of our metric bugs were found by
   ablations we almost didn't run.
5. **Small models are enough more often than you think** — and roughly $35 of GPU
   credit covers an embarrassing number of mistakes if each one teaches you something.

— Built with a ≤4B planner, a 44M PII classifier, checksums, and a reference gazetteer.
Total model weight: under 4.1B parameters. Total cloud spend: about $35.
