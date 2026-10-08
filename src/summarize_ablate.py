"""H4 summary: change in interference ratio R under valence-direction ablation vs in-plane random ablation.

Evaluated on self items of CONFIRMATION domains (primary) and reported for discovery as well.
Usage: python summarize_ablate.py tag1 tag2 ...
"""
import os
import json
import sys
from pathlib import Path

import numpy as np

R = Path(os.environ.get("GOALVAL_ROOT", "."))
rows = [json.loads(l) for l in open(R / "data/goal_stimuli.jsonl")]


def metrics(g, o, s):
    con = s == o
    pred = (g > 0).astype(int)
    b_surf = np.mean([g[(o == v) & (s == 1)].mean() - g[(o == v) & (s == 0)].mean() for v in (0, 1)])
    b_out = np.mean([g[(s == v) & (o == 1)].mean() - g[(s == v) & (o == 0)].mean() for v in (0, 1)])
    return dict(R=b_surf / b_out, B_surf=b_surf, B_out=b_out, acc_conflict=(pred[~con] == o[~con]).mean(),
                acc_congruent=(pred[con] == o[con]).mean())


def main() -> None:
    out = {}
    for tag in sys.argv[1:]:
        f = R / "results" / f"ablate_{tag}.json"
        if not f.exists():
            print(tag, "missing")
            continue
        A = json.load(open(f))
        ids = np.array(A["ids"])
        res = {}
        for split in ("confirmation", "discovery"):
            m = np.array([rows[i]["perspective"] == "self" and rows[i]["split"] == split for i in ids])
            o = np.array([rows[i]["outcome"] for i in ids[m]])
            s = np.array([rows[i]["surface"] for i in ids[m]])
            base = metrics(np.array(A["none"])[m], o, s)
            res[split] = {"none": {k: float(v) for k, v in base.items()}}
            for band in ("mid", "all"):
                rnd = [metrics(np.array(A[f"{band}/inplane_{k}"])[m], o, s) for k in range(20)]
                red_rnd = np.array([base["R"] - r["R"] for r in rnd])
                for src in ("valence", "goemo"):
                    v = metrics(np.array(A[f"{band}/{src}"])[m], o, s)
                    red = base["R"] - v["R"]
                    res[split][f"{band}/{src}"] = {**{k: float(x) for k, x in v.items()},
                                                   "R_reduction": float(red),
                                                   "rand_reduction_p95": float(np.percentile(red_rnd, 95)),
                                                   "rand_reduction_mean": float(red_rnd.mean()),
                                                   "H4": bool(red > np.percentile(red_rnd, 95))}
        out[tag] = res
        for band in ("mid", "all"):
            c = res["confirmation"]
            v = c[f"{band}/valence"]
            print(f"{tag:16s} {band:3s} R: none={c['none']['R']:.3f} -> valence={v['R']:.3f} (red {v['R_reduction']:+.3f}, "
                  f"rand p95 {v['rand_reduction_p95']:+.3f}) H4={v['H4']} | conflict acc {c['none']['acc_conflict']:.3f}"
                  f" -> {v['acc_conflict']:.3f}, congruent {c['none']['acc_congruent']:.3f} -> {v['acc_congruent']:.3f}"
                  f" | goemo H4={c[f'{band}/goemo']['H4']}")
    json.dump(out, open(R / "results" / "ablation_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
