"""Analyses for prereg v9 H18 (task gating) and H19 (goal-explicit key control). Writes results/sweep4_summary.json.

Usage: python analyze_sweep4.py tag1 tag2 ...
"""
import os
import json
import sys
from pathlib import Path

import numpy as np

from analyze_sweep3 import conflict_stats, beh_stats

R = Path(os.environ.get("GOALVAL_ROOT", "."))
GOAL = [json.loads(l) for l in open(R / "data/goal_stimuli.jsonl")]
GN2 = [json.loads(l) for l in open(R / "data/goalneutral2_stimuli.jsonl")]
SUFFIXES = ("eval_self", "eval_plain", "factual", "irrelevant", "none")


def main():
    f = R / "results" / "sweep4_summary.json"
    out = json.load(open(f)) if f.exists() else {}
    for tag in sys.argv[1:]:
        A = R / "acts4" / tag
        res = dict(out.get(tag, {}))
        if (A / "tg_ids.npy").exists():
            rows = [GOAL[i] for i in np.load(A / "tg_ids.npy")]
            for name in SUFFIXES:
                res[f"tg_{name}"] = conflict_stats(np.load(A / f"tg_{name}.npy"), rows)
        if (A / "gn2_behaviour.npy").exists():
            for kind in ("sst", "inf"):
                P = np.load(A / f"gn2_{kind}_gen.npy")
                res[f"gn2_{kind}_gen"] = conflict_stats(P, GN2)
                res[f"gn2_{kind}_gen_other"] = conflict_stats(P, GN2, "other")
            res["gn2_behaviour"] = beh_stats(np.load(A / "gn2_behaviour.npy"), np.load(A / "gn2_behaviour_ids.npy"), GN2)
        out[tag] = res
        show = [f"{k[3:]}={res[k]['late']:.2f}[{res[k]['late_ci'][0]:.2f},{res[k]['late_ci'][1]:.2f}] v={res[k]['validity_late']:.2f}"
                for k in (f"tg_{n}" for n in SUFFIXES) if k in res]
        if "gn2_sst_gen" in res:
            g = res["gn2_sst_gen"]; b = res["gn2_behaviour"]
            show.append(f"gn2 sst={g['late']:.2f}[{g['late_ci'][0]:.2f},{g['late_ci'][1]:.2f}] other={res['gn2_sst_gen_other']['late']:.2f} "
                        f"inf={res['gn2_inf_gen']['late']:.2f} acc={b['acc']:.2f}")
        print(f"{tag:16s} " + " | ".join(show), flush=True)
    json.dump(out, open(f, "w"))


if __name__ == "__main__":
    main()
