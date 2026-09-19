"""How much of the next-day fire task is answered by "it stays where it is"?

Section 15 measures channels against each other on a spread-only target -- tomorrow's
*newly* burning pixels -- rather than WildfireSpreadTS's published target of tomorrow's
whole active fire. That choice needs a number behind it rather than an assertion, and the
number is what a rule with no model and no channels at all scores on each.

Two trivial rules, on the same test splits the networks use:

    copy today    tomorrow's fire is exactly today's fire
    ring          tomorrow's new fire is the ring just outside today's fire

Run:  python scripts/gen_wsts_pairs.py --target new
      python scripts/gen_wsts_pairs.py --target full
      python scripts/wsts_trivial.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"

C_FIRE = 22
TARGETS = {"new-fire (used in section 15)": "wsts_pairs",
           "published (whole next-day fire)": "wsts_pairs_full"}


def ap(score, y):
    return float(average_precision_score(np.asarray(y, np.int8).ravel(),
                                         np.asarray(score, np.float64).ravel()))


def main():
    out = {}
    print("=" * 78)
    print("what a rule with no model and no channels scores on each target")
    print("=" * 78)
    print(f"\n  {'target':>34} {'positives':>10} {'copy today':>12} {'ring':>9} "
          f"{'chance':>9}")
    for label, folder in TARGETS.items():
        p = ROOT / "data" / folder / "test.npz"
        if not p.exists():
            print(f"  {label:>34}  missing {p}")
            continue
        d = np.load(p)
        x, y = d["x"], d["y"].astype(np.int8)
        today = x[:, C_FIRE].astype(np.float32) > 0.5
        # the ring: a soft band just outside today's fire
        ring = np.zeros(today.shape, np.float32)
        for i in range(len(today)):
            ring[i] = np.exp(-ndimage.distance_transform_edt(~today[i]) / 2.0) * (~today[i])
        a_copy, a_ring, chance = ap(today, y), ap(ring, y), float(y.mean())
        out[label] = {"n": int(len(y)), "positives": chance,
                      "copy_today": a_copy, "ring": a_ring, "chance": chance}
        print(f"  {label:>34} {chance*100:9.2f}% {a_copy:12.4f} {a_ring:9.4f} "
              f"{chance:9.4f}")

    (RESULTS / "wsts_trivial.json").write_text(json.dumps(out, indent=2),
                                               encoding="utf-8")
    print(f"\nwrote {RESULTS/'wsts_trivial.json'}")

    pub = out.get("published (whole next-day fire)")
    new = out.get("new-fire (used in section 15)")
    if pub and new:
        print("\n  On the published target, copying today's fire unchanged already scores")
        print(f"  {pub['copy_today']:.4f}. The best network in section 15 scores 0.9922, so "
              f"everything a model")
        print(f"  contributes -- every channel, every parameter -- is worth "
              f"{0.9922 - pub['copy_today']:.3f} of AP on top of")
        print("  a rule that ignores the data.")
        print(f"\n  On the spread-only target the same rule scores {new['copy_today']:.4f}, "
              f"which is chance")
        print(f"  ({new['chance']:.4f}) by construction, and the best network scores 0.3046. "
              f"That is the")
        print("  whole reason section 15 measures channels there: the published framing")
        print("  leaves them fighting over the last few percent of a saturated metric.")


if __name__ == "__main__":
    main()
