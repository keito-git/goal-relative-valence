"""H20 (prereg v9): per-layer additive decomposition of valence projections into surface (s), label match (m), and
stake outcome (o), coded +/-1 (o = s*m, orthogonal in the balanced design). Writes results/decompose_summary.json.

Usage: python decompose.py tag1 tag2 ...
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import json
import sys
from pathlib import Path

import numpy as np

R = Path(os.environ.get("GOALVAL_ROOT", "."))
SEED = 20261010
GOAL = [json.loads(l) for l in open(R / "data/goal_stimuli.jsonl")]


def bands(n):
    return ([L for L in range(n) if 0.2 <= L / n <= 0.45], [L for L in range(n) if 0.6 <= L / n <= 0.9])


def coefs(P, X):
    """P: items x layers (z-scored per layer); X: items x 4 (intercept, s, m, o). Returns layers x 3."""
    B, *_ = np.linalg.lstsq(X, P, rcond=None)
    return B[1:].T


def decompose(P, rows, boot=500):
    idx = np.array([i for i, r in enumerate(rows) if r["split"] == "confirmation" and r["perspective"] == "self"])
    s = np.array([2 * rows[i]["surface"] - 1 for i in idx]); m = np.array([2 * rows[i]["match"] - 1 for i in idx])
    o = np.array([2 * rows[i]["outcome"] - 1 for i in idx])
    assert np.all(o == s * m)
    X = np.c_[np.ones(len(idx)), s, m, o]
    Z = P[idx].astype(np.float64)
    Z = (Z - Z.mean(0)) / np.where(Z.std(0) > 0, Z.std(0), 1)
    C = coefs(Z, X)  # layers x (s, m, o)
    n = P.shape[1] - 1; mid, late = bands(n)
    rng = np.random.default_rng(SEED)
    bs = []
    for _ in range(boot):
        k = rng.integers(0, len(idx), len(idx))
        Cb = coefs(Z[k], X[k])
        bs.append([Cb[late, 0].mean(), Cb[late, 2].mean(), Cb[mid, 2].mean(), Cb[late, 2].mean() - Cb[late, 0].mean()])
    bs = np.array(bs)
    ci = lambda j: list(np.percentile(bs[:, j], [2.5, 97.5]))

    def half_depth(v):
        v = np.asarray(v[:n]); mx = v.max()
        L = next((L for L in range(n) if L / n >= 0.1 and v[L] >= mx / 2), None)
        return None if L is None or mx <= 0 else L / n
    return {"beta_s": C[:, 0].tolist(), "beta_m": C[:, 1].tolist(), "beta_o": C[:, 2].tolist(),
            "late_s": float(C[late, 0].mean()), "late_s_ci": ci(0), "late_o": float(C[late, 2].mean()), "late_o_ci": ci(1),
            "mid_o": float(C[mid, 2].mean()), "mid_o_ci": ci(2), "late_o_minus_s_ci": ci(3),
            "mid_s": float(C[mid, 0].mean()), "late_m": float(C[late, 1].mean()), "mid_m": float(C[mid, 1].mean()),
            "d_m": half_depth(np.abs(C[:, 1])), "d_o": half_depth(C[:, 2])}


def sst_raw(tag):
    A = R / "acts" / tag
    G = np.load(A / "goal_raw.npy", mmap_mode="r"); S = np.load(A / "sst2_raw.npy", mmap_mode="r")
    y = np.load(A / "sst2_labels.npy")
    P = np.zeros(G.shape[:2], np.float32)
    for L in range(G.shape[1]):
        x = S[:, L].astype(np.float32)
        w = x[y == 1].mean(0) - x[y == 0].mean(0)
        P[:, L] = (G[:, L].astype(np.float32) - x.mean(0)) @ w
    return P


def main():
    f = R / "results" / "decompose_summary.json"
    out = json.load(open(f)) if f.exists() else {}
    for tag in sys.argv[1:]:
        res = {}
        A2, A3 = R / "acts2" / tag, R / "acts3" / tag
        for pos in ("raw", "gen"):
            if (A2 / f"inf_valence_{pos}.npy").exists():
                res[f"inf_{pos}"] = decompose(np.load(A2 / f"inf_valence_{pos}.npy"), GOAL)
        if (A3 / "traj_ids.npy").exists():
            rows = [GOAL[i] for i in np.load(A3 / "traj_ids.npy")]
            for name in ("text_end", "question_end", "response"):
                res[f"traj_{name}"] = decompose(np.load(A3 / f"traj_{name}.npy"), rows)
        if (R / "acts" / tag / "goal_raw.npy").exists():
            res["sst_raw"] = decompose(sst_raw(tag), GOAL)
        out[tag] = res
        for k, v in res.items():
            print(f"{tag:16s} {k:18s} late s={v['late_s']:+.2f} [{v['late_s_ci'][0]:+.2f},{v['late_s_ci'][1]:+.2f}] "
                  f"o={v['late_o']:+.2f} mid o={v['mid_o']:+.2f} m(late)={v['late_m']:+.2f} d_m={v['d_m']} d_o={v['d_o']}", flush=True)
    json.dump(out, open(f, "w"))


if __name__ == "__main__":
    main()
