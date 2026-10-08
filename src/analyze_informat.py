"""H12 (prereg v7): in-format directions with a validity gate. Writes results/informat_summary.json.

Usage: python analyze_informat.py tag1 tag2 ...
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

R = Path(os.environ.get("GOALVAL_ROOT", "."))
SEED = 20261007
FILES = {"valence": "goal_stimuli.jsonl", "location": "control_stimuli.jsonl", "material": "control2_stimuli.jsonl"}
ROWS = {k: [json.loads(l) for l in open(R / "data" / v)] for k, v in FILES.items()}


def band_mean(vals, lo, hi):
    n = len(vals) - 1
    return float(np.nanmean([v for L, v in enumerate(vals) if lo <= L / n <= hi and L < n]))


def analyse(P, rows):
    conf = np.array([r["split"] == "confirmation" for r in rows])
    selfm = conf & np.array([r["perspective"] == "self" for r in rows])
    nonem = conf & np.array([r["perspective"] == "none" for r in rows])
    o = np.array([r["outcome"] for r in rows]); s = np.array([r["surface"] for r in rows])
    c = selfm & (s != o)
    prof, valid = [], []
    for L in range(P.shape[1]):
        p = P[:, L]
        prof.append(float(roc_auc_score(o[c], p[c])) if p[c].std() > 0 else float("nan"))
        valid.append(float(roc_auc_score(s[nonem], p[nonem])) if p[nonem].std() > 0 else float("nan"))
    # stimulus bootstrap of late-minus-mid conflict AUC
    rng = np.random.default_rng(SEED)
    idx = np.where(c)[0]
    n = P.shape[1] - 1
    mid = [L for L in range(n) if 0.2 <= L / n <= 0.45]; late = [L for L in range(n) if 0.6 <= L / n <= 0.9]
    def stat(ii):
        a = lambda Ls: np.mean([roc_auc_score(o[ii], P[ii, L]) for L in Ls])
        return a(late) - a(mid)
    bs = [stat(rng.choice(idx, len(idx))) for _ in range(500)]
    return {"profile": prof, "validity": valid, "mid": band_mean(prof, 0.2, 0.45), "late": band_mean(prof, 0.6, 0.9),
            "validity_late": band_mean(valid, 0.6, 0.9), "diff_ci": list(np.percentile(bs, [2.5, 97.5]))}


def main():
    f = R / "results" / "informat_summary.json"
    out = json.load(open(f)) if f.exists() else {}
    for tag in sys.argv[1:]:
        A = R / "acts2" / tag
        if not (A / "inf_material_gen.npy").exists():
            print(tag, "missing"); continue
        res = {}
        for task in FILES:
            for pos in ("raw", "gen"):
                res[f"{task}_{pos}"] = analyse(np.load(A / f"inf_{task}_{pos}.npy"), ROWS[task])
        out[tag] = res
        line = " | ".join(f"{task[:3]} {pos}: valid={res[f'{task}_{pos}']['validity_late']:.2f} mid={res[f'{task}_{pos}']['mid']:.2f} late={res[f'{task}_{pos}']['late']:.2f}"
                          for pos in ("gen",) for task in FILES)
        line2 = " | ".join(f"{task[:3]} raw: valid={res[f'{task}_raw']['validity_late']:.2f} late={res[f'{task}_raw']['late']:.2f}" for task in FILES)
        print(f"{tag:16s} {line}\n{'':16s} {line2}", flush=True)
    json.dump(out, open(f, "w"))


if __name__ == "__main__":
    main()
