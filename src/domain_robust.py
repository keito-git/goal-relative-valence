"""R2 (prereg v11): per-domain and leave-one-domain-out robustness on the confirmation domains (SST-2 directions, self).
End of text from saved states; response position from acts6/<tag>/goal_sst_gen.npy. Writes results/domain_robust.json.

Usage: python domain_robust.py tag1 tag2 ...
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
ROWS = [json.loads(l) for l in open(R / "data/goal_stimuli.jsonl")]
DOMS = ["lottery", "race", "court", "hiring"]


def raw_proj(tag):
    A = R / "acts" / tag
    G = np.load(A / "goal_raw.npy", mmap_mode="r"); S = np.load(A / "sst2_raw.npy", mmap_mode="r"); y = np.load(A / "sst2_labels.npy")
    P = np.zeros(G.shape[:2], np.float32)
    for L in range(G.shape[1]):
        x = np.asarray(S[:, L], np.float32); w = x[y == 1].mean(0) - x[y == 0].mean(0)
        P[:, L] = (np.asarray(G[:, L], np.float32) - x.mean(0)) @ w
    return P


def bands_of(P, idx):
    n = P.shape[1] - 1
    o = np.array([ROWS[i]["outcome"] for i in idx]); s = np.array([ROWS[i]["surface"] for i in idx]); c = s != o
    auc = lambda L: roc_auc_score(o[c], P[idx[c], L])
    mid = float(np.mean([auc(L) for L in range(n) if 0.2 <= L / n <= 0.45]))
    late = float(np.mean([auc(L) for L in range(n) if 0.6 <= L / n <= 0.9]))
    return {"mid": mid, "late": late, "diff": late - mid}


def bsurf(tag, doms):
    A = R / "acts" / tag
    g = np.load(A / "behaviour_gb.npy"); ids = np.load(A / "behaviour_ids.npy")
    m = np.array([ROWS[i]["perspective"] == "self" and ROWS[i]["domain"] in doms for i in ids])
    o = np.array([ROWS[i]["outcome"] for i in ids[m]]); s = np.array([ROWS[i]["surface"] for i in ids[m]]); y = g[m]
    return float(np.mean([y[(o == v) & (s == 1)].mean() - y[(o == v) & (s == 0)].mean() for v in (0, 1)]))


def main():
    f = R / "results" / "domain_robust.json"
    out = json.load(open(f)) if f.exists() else {}
    for tag in sys.argv[1:]:
        res = {}
        sources = {"raw": raw_proj(tag)}
        g = R / "acts6" / tag / "goal_sst_gen.npy"
        if g.exists():
            sources["gen"] = np.load(g)
        for pos, P in sources.items():
            for d in DOMS:
                idx = np.array([i for i, r in enumerate(ROWS) if r["domain"] == d and r["perspective"] == "self"])
                res[f"{pos}_dom_{d}"] = bands_of(P, idx)
                keep = [x for x in DOMS if x != d]
                idx = np.array([i for i, r in enumerate(ROWS) if r["domain"] in keep and r["perspective"] == "self"])
                res[f"{pos}_lodo_{d}"] = bands_of(P, idx)
        for d in DOMS:
            res[f"bsurf_dom_{d}"] = bsurf(tag, [d])
            res[f"bsurf_lodo_{d}"] = bsurf(tag, [x for x in DOMS if x != d])
        out[tag] = res
        show = " ".join(f"{k}={v['diff']:+.2f}" for k, v in res.items() if isinstance(v, dict) and "_dom_" in k)
        print(tag, show, "| bsurf", " ".join(f"{res[f'bsurf_dom_{d}']:+.1f}" for d in DOMS), flush=True)
    json.dump(out, open(f, "w"))


if __name__ == "__main__":
    main()
