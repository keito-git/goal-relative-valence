"""Analyses for prereg v8 (H13-H17). Writes results/sweep3_summary.json (merging per tag).

Usage: python analyze_sweep3.py tag1 tag2 ...
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
SEED = 20261009
GN = [json.loads(l) for l in open(R / "data/goalneutral_stimuli.jsonl")]
GOAL = [json.loads(l) for l in open(R / "data/goal_stimuli.jsonl")]


def bands(n):
    return ([L for L in range(n) if 0.2 <= L / n <= 0.45], [L for L in range(n) if 0.6 <= L / n <= 0.9])


def conflict_stats(P, rows, persp="self", boot=500):
    """P: projections for `rows` (same order). Returns profile, mid/late means, bootstrap CI of the late mean."""
    idx = np.array([i for i, r in enumerate(rows) if r["split"] == "confirmation" and r["perspective"] == persp])
    o = np.array([rows[i]["outcome"] for i in idx]); s = np.array([rows[i]["surface"] for i in idx]); c = s != o
    ii = idx[c]; oo = o[c]
    prof = [float(roc_auc_score(oo, P[ii, L])) if P[ii, L].std() > 0 else float("nan") for L in range(P.shape[1])]
    n = P.shape[1] - 1; mid, late = bands(n)
    rng = np.random.default_rng(SEED)
    bs = []
    for _ in range(boot):
        k = rng.integers(0, len(ii), len(ii))
        bs.append(np.mean([roc_auc_score(oo[k], P[ii[k], L]) for L in late]))
    none = np.array([i for i, r in enumerate(rows) if r["split"] == "confirmation" and r["perspective"] == "none"])
    sv = np.array([rows[i]["surface"] for i in none])
    valid = [float(roc_auc_score(sv, P[none, L])) if len(none) and P[none, L].std() > 0 else float("nan") for L in range(P.shape[1])]
    return {"profile": prof, "mid": float(np.nanmean([prof[L] for L in mid])), "late": float(np.nanmean([prof[L] for L in late])),
            "late_ci": list(np.percentile(bs, [2.5, 97.5])), "validity_late": float(np.nanmean([valid[L] for L in late])) if len(none) else None}


def beh_stats(g, ids, rows, persp="self"):
    m = np.array([rows[i]["perspective"] == persp and rows[i]["split"] == "confirmation" for i in ids])
    o = np.array([rows[i]["outcome"] for i in ids[m]]); s = np.array([rows[i]["surface"] for i in ids[m]]); y = g[m]
    bs = np.mean([y[(o == v) & (s == 1)].mean() - y[(o == v) & (s == 0)].mean() for v in (0, 1)])
    bo = np.mean([y[(s == v) & (o == 1)].mean() - y[(s == v) & (o == 0)].mean() for v in (0, 1)])
    pred = (y > 0).astype(int); c = s != o
    return {"acc": float((pred == o).mean()), "acc_conflict": float((pred[c] == o[c]).mean()), "B_surf": float(bs), "B_out": float(bo)}


def main():
    f = R / "results" / "sweep3_summary.json"
    out = json.load(open(f)) if f.exists() else {}
    for tag in sys.argv[1:]:
        A = R / "acts3" / tag
        res = dict(out.get(tag, {}))
        if (A / "gn_behaviour.npy").exists():
            for pos in ("raw", "gen"):
                for kind in ("sst", "inf"):
                    P = np.load(A / f"gn_{kind}_{pos}.npy")
                    res[f"gn_{kind}_{pos}"] = conflict_stats(P, GN)
                    res[f"gn_{kind}_{pos}_other"] = conflict_stats(P, GN, "other")
            res["gn_behaviour"] = beh_stats(np.load(A / "gn_behaviour.npy"), np.load(A / "gn_behaviour_ids.npy"), GN)
        if (A / "traj_response.npy").exists():
            ids = np.load(A / "traj_ids.npy"); rows = [GOAL[i] for i in ids]
            for name in ("text_end", "question_end", "response"):
                res[f"traj_{name}"] = conflict_stats(np.load(A / f"traj_{name}.npy"), rows)
        if (A / "patchmap.json").exists():
            pm = [p for p in json.load(open(A / "patchmap.json")) if abs(p["g_corrupt"] - p["g_clean"]) > 1]
            if pm:
                res["patchmap"] = {"n_pairs": len(pm), "n_layers": pm[0]["n_layers"],
                                   "effects": {k: np.mean([[(x - p["g_clean"]) / (p["g_corrupt"] - p["g_clean"]) for x in p["effects"][k]]
                                                           for p in pm], 0).tolist() for k in pm[0]["effects"]}}
        if (A / "base_beh.npy").exists():
            ids = np.load(A / "base_ids.npy"); rows = [GOAL[i] for i in ids]
            res["base_cont"] = conflict_stats(np.load(A / "base_proj.npy"), rows)
            res["base_beh"] = beh_stats(np.load(A / "base_beh.npy"), np.load(A / "base_beh_ids.npy"), GOAL)
        if (A / "natval.json").exists():
            res["natval"] = json.load(open(A / "natval.json"))
        out[tag] = res
        show = []
        for k in ("gn_sst_gen", "gn_inf_gen", "traj_text_end", "traj_question_end", "traj_response", "base_cont"):
            if k in res:
                show.append(f"{k}: late={res[k]['late']:.2f} [{res[k]['late_ci'][0]:.2f},{res[k]['late_ci'][1]:.2f}]")
        if "gn_behaviour" in res:
            show.append(f"gn beh acc={res['gn_behaviour']['acc']:.2f} Bsurf={res['gn_behaviour']['B_surf']:+.2f}")
        if "natval" in res:
            show.append(f"natval={res['natval']['agree']:.3f}")
        print(f"{tag:16s} " + " | ".join(show), flush=True)
    json.dump(out, open(f, "w"))


if __name__ == "__main__":
    main()
