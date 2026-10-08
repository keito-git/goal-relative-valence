"""Analysis for prereg v10 H22. Writes results/sweep5_summary.json.

Usage: python analyze_sweep5.py tag1 tag2 ...
"""
import os
import json
import sys
from pathlib import Path

import numpy as np

from analyze_sweep3 import conflict_stats

R = Path(os.environ.get("GOALVAL_ROOT", "."))
GOAL = [json.loads(l) for l in open(R / "data/goal_stimuli.jsonl")]


def main():
    f = R / "results" / "sweep5_summary.json"
    out = json.load(open(f)) if f.exists() else {}
    for tag in sys.argv[1:]:
        A = R / "acts5" / tag
        if not (A / "tg2_beh_binding.npy").exists():
            print(tag, "missing"); continue
        rows = [GOAL[i] for i in np.load(A / "tg2_ids.npy")]
        res = {k: conflict_stats(np.load(A / f"tg2_{k}.npy"), rows) for k in ("eval_self", "outcome_factual", "binding_factual")}
        ids = np.load(A / "tg2_beh_ids.npy")
        conf = np.array([GOAL[i]["split"] == "confirmation" for i in ids])
        o = np.array([GOAL[i]["outcome"] for i in ids]); m = np.array([GOAL[i]["match"] for i in ids])
        go, gb = np.load(A / "tg2_beh_outcome.npy"), np.load(A / "tg2_beh_binding.npy")
        res["beh_outcome_acc"] = float(((go > 0).astype(int) == o)[conf].mean())
        res["beh_binding_acc"] = float(((gb > 0).astype(int) == m)[conf].mean())
        out[tag] = res
        print(f"{tag:16s} " + " | ".join(f"{k}={res[k]['late']:.2f}[{res[k]['late_ci'][0]:.2f},{res[k]['late_ci'][1]:.2f}] v={res[k]['validity_late']:.2f}"
                                       for k in ("eval_self", "outcome_factual", "binding_factual"))
              + f" | acc won/lost={res['beh_outcome_acc']:.2f} yes/no={res['beh_binding_acc']:.2f}", flush=True)
    json.dump(out, open(f, "w"))


if __name__ == "__main__":
    main()
