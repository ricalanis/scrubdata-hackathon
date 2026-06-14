"""Pin / unpin a warm A100 container on the DEPLOYED Modal endpoint — no redeploy.

Flip this on for the live judging window (zero cold starts) and off the rest of the
time (scale-to-zero, $0 idle). Uses Modal's runtime autoscaler update, so it takes
effect in seconds against the already-deployed `scrubdata-serve` app.

    uv run python scripts/modal_warm.py on     # min_containers=1 -> always warm, no cold start
    uv run python scripts/modal_warm.py off    # min_containers=0 -> scale-to-zero, $0 idle

Cost math (A100 40GB @ $0.000583/s ~= $2.10/hr):
  - on  : ~$2.10/hr continuous while pinned. $137 credit ~= ~63h always-warm.
  - off : ~$0.05/clean bursty (~$0.22 isolated, incl. the 5-min scaledown tail); $0 idle.
Typical use: leave OFF; switch ON only for the hours judges are actively reviewing,
then back OFF. The Space's page-load pre-warm already hides most cold starts when off.
"""

import sys

import modal

APP, FUNC = "scrubdata-serve", "serve"


def main() -> int:
    mode = (sys.argv[1] if len(sys.argv) > 1 else "").lower()
    if mode not in ("on", "warm", "1", "off", "cold", "0"):
        print(__doc__)
        print("usage: modal_warm.py on|off")
        return 2

    fn = modal.Function.from_name(APP, FUNC)
    if mode in ("on", "warm", "1"):
        fn.update_autoscaler(min_containers=1)
        print("WARM pinned: min_containers=1 — no cold starts (~$2.10/hr while pinned).")
        print("Run `uv run python scripts/modal_warm.py off` when judging is done.")
    else:
        fn.update_autoscaler(min_containers=0)
        print("SCALE-TO-ZERO: min_containers=0 — $0 idle (~$0.05-0.22/clean).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
