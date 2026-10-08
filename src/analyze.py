"""Goal-relative vs stimulus-bound valence: kill-switch and main metrics for one model.

Usage: python analyze.py --tag qwen25_7b_it --split discovery [--pos raw]
"""
import os

# keep BLAS threads small so that GPU jobs are not starved
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(os.environ.get("GOALVAL_ROOT", "."))
SEED = 20261006
N_NULL = 500


def strat_d(s, factor, stratum):
    """Cohen's d of s between factor levels, averaged over levels of stratum."""
    ds = []
    for v in (0, 1):
        m = stratum == v
        a, b = s[m & (factor == 1)], s[m & (factor == 0)]
        sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        ds.append((a.mean() - b.mean()) / sd)
    return float(np.mean(ds))


def directions(Xs, ys, rng):
    """Valence directions estimated on a source set; returns dict name -> (w, mu)."""
    mu = Xs.mean(0)
    Xc = Xs - mu
    out = {"diffmeans": Xc[ys == 1].mean(0) - Xc[ys == 0].mean(0)}
    sc = StandardScaler().fit(Xs)
    lr = LogisticRegression(C=0.01, max_iter=2000).fit(sc.transform(Xs), ys)
    out["logistic"] = lr.coef_[0] / sc.scale_
    pca = PCA(50, random_state=SEED).fit(Xc)
    lr50 = LogisticRegression(C=1.0, max_iter=2000).fit(pca.transform(Xc), ys)
    out["logistic_pca50"] = pca.components_.T @ lr50.coef_[0]
    # in-plane random floor: uniform directions within the top-50 PC span of the source
    coefs = rng.standard_normal((N_NULL, 50))
    null = coefs @ pca.components_
    return {k: v / np.linalg.norm(v) for k, v in out.items()}, mu, null / np.linalg.norm(null, axis=1, keepdims=True)


def lodo_acc(X, y, dom):
    accs = []
    for d in np.unique(dom):
        tr, te = dom != d, dom == d
        sc = StandardScaler().fit(X[tr])
        lr = LogisticRegression(C=0.01, max_iter=2000).fit(sc.transform(X[tr]), y[tr])
        accs.append(lr.score(sc.transform(X[te]), y[te]))
    return float(np.mean(accs))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--split", default="discovery")
    ap.add_argument("--pos", default="raw")
    ap.add_argument("--skip_lodo", type=int, default=0)
    args = ap.parse_args()
    A = ROOT / "acts" / args.tag
    rows = [json.loads(l) for l in open(ROOT / "data/goal_stimuli.jsonl")]
    G = np.load(A / f"goal_{args.pos}.npy", mmap_mode="r")
    rng = np.random.default_rng(SEED)

    sel = {p: np.array([r["split"] == args.split and r["perspective"] == p for r in rows]) for p in ("self", "other", "none")}
    surf = np.array([r["surface"] for r in rows])
    outc = np.array([r["outcome"] for r in rows])
    dom = np.array([r["domain"] for r in rows])

    # behavioural positive control
    beh = np.load(A / "behaviour_gb.npy")
    bid = np.load(A / "behaviour_ids.npy")
    beh_res = {}
    for p in ("self", "other"):
        m = np.array([sel[p][i] for i in bid])
        beh_res[p] = float(((beh[m] > 0).astype(int) == outc[bid[m]]).mean())

    n_layers = G.shape[1]
    res = {"tag": args.tag, "split": args.split, "pos": args.pos, "behaviour_acc": beh_res, "layers": []}
    for L in range(n_layers):
        X = np.asarray(G[:, L, :], dtype=np.float32)
        lr = {"layer": L}
        if not args.skip_lodo:
            m = sel["self"]
            lr["lodo_outcome_self"] = lodo_acc(X[m], outc[m], dom[m])
        for src in ("sst2", "goemo"):
            Xs = np.load(A / f"{src}_{args.pos}.npy", mmap_mode="r")[:, L, :].astype(np.float32)
            ys = np.load(A / f"{src}_labels.npy")
            if float(Xs.std(0).max()) == 0.0:  # source identical across items at this layer (e.g. layer 0 at the gen position)
                for name in ("diffmeans", "logistic", "logistic_pca50"):
                    lr[f"{src}/{name}"] = {k: float("nan") for k in ("auc_surface_none", "d_outcome_self", "d_surface_self",
                                           "auc_outcome_conflict_self", "d_outcome_other", "d_surface_other",
                                           "auc_outcome_conflict_other")}
                lr[f"{src}/null_absd_outcome_self_p95"] = float("nan")
                lr[f"{src}/null_absd_outcome_self_mean"] = float("nan")
                continue
            dirs, mu, null = directions(Xs, ys, rng)
            for name, w in dirs.items():
                key = f"{src}/{name}"
                d = {}
                s_none = (X[sel["none"]] - mu) @ w
                d["auc_surface_none"] = float(roc_auc_score(surf[sel["none"]], s_none))
                for p in ("self", "other"):
                    m = sel[p]
                    s = (X[m] - mu) @ w
                    d[f"d_outcome_{p}"] = strat_d(s, outc[m], surf[m])
                    d[f"d_surface_{p}"] = strat_d(s, surf[m], outc[m])
                    conf = m & (surf != outc)
                    d[f"auc_outcome_conflict_{p}"] = float(roc_auc_score(outc[conf], (X[conf] - mu) @ w))
                lr[key] = d
            # in-plane random floor for the outcome effect (self), shared by all direction estimators of this source
            m = sel["self"]
            S = (X[m] - mu) @ null.T
            nd = np.array([abs(strat_d(S[:, j], outc[m], surf[m])) for j in range(N_NULL)])
            lr[f"{src}/null_absd_outcome_self_p95"] = float(np.percentile(nd, 95))
            lr[f"{src}/null_absd_outcome_self_mean"] = float(nd.mean())
        res["layers"].append(lr)
        print(L, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in lr.items() if not isinstance(v, dict)}),
              "sst2/diffmeans", {k: round(v, 3) for k, v in lr["sst2/diffmeans"].items()}, flush=True)
    outp = ROOT / "results" / f"{args.tag}_{args.split}_{args.pos}.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(outp, "w"))
    print("behaviour", beh_res)


if __name__ == "__main__":
    main()
