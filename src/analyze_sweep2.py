"""Analyses for prereg v4: H6 (control), H7 (transition fit), H8 (prompt robustness), H9 (patching).

Usage: python analyze_sweep2.py tag1 tag2 ...   (writes results/sweep2_summary.json)
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import roc_auc_score

R = Path(os.environ.get("GOALVAL_ROOT", "."))
SEED = 20261007
BANDS = {"mid": (0.20, 0.45), "late": (0.60, 0.90)}
goal_rows = [json.loads(l) for l in open(R / "data/goal_stimuli.jsonl")]
ctrl_rows = [json.loads(l) for l in open(R / "data/control_stimuli.jsonl")]
ctrl2_path = R / "data/control2_stimuli.jsonl"
ctrl2_rows = [json.loads(l) for l in open(ctrl2_path)] if ctrl2_path.exists() else []


def band_layers(n, lo, hi):
    return [L for L in range(n) if lo <= L / n <= hi]  # n = index of final layer; final excluded


def conflict_auc_profile(G, S, ys, rows, persp="self"):
    """Per-layer conflict-cell AUC for the outcome; returns profile and per-layer projections for bootstrap."""
    idx = np.array([i for i, r in enumerate(rows) if r["split"] == "confirmation" and r["perspective"] == persp])
    o = np.array([rows[i]["outcome"] for i in idx]); s = np.array([rows[i]["surface"] for i in idx])
    prof, proj = [], []
    for L in range(G.shape[1]):
        x = np.asarray(S[:, L], np.float32)
        w = x[ys == 1].mean(0) - x[ys == 0].mean(0)
        p = (np.asarray(G[idx, L], np.float32) - x.mean(0)) @ w
        c = s != o
        try:
            prof.append(float(roc_auc_score(o[c], p[c])))
        except ValueError:
            prof.append(float("nan"))
        proj.append(p)
    return np.array(prof), proj, o, s


def none_auc_profile(G, S, ys, rows):
    idx = np.array([i for i, r in enumerate(rows) if r["split"] == "confirmation" and r["perspective"] == "none"])
    s = np.array([rows[i]["surface"] for i in idx]); out = []
    for L in range(G.shape[1]):
        x = np.asarray(S[:, L], np.float32); w = x[ys == 1].mean(0) - x[ys == 0].mean(0)
        p = (np.asarray(G[idx, L], np.float32) - x.mean(0)) @ w
        out.append(float(roc_auc_score(s, p)) if p.std() > 0 else float("nan"))
    return out



def profile_from_proj(Pm, rows, persp="self"):
    """Same as conflict_auc_profile but from saved projections (N x L) of all rows."""
    idx = np.array([i for i, r in enumerate(rows) if r["split"] == "confirmation" and r["perspective"] == persp])
    o = np.array([rows[i]["outcome"] for i in idx]); s = np.array([rows[i]["surface"] for i in idx]); c = s != o
    prof, proj = [], []
    for L in range(Pm.shape[1]):
        p = Pm[idx, L]
        try:
            prof.append(float(roc_auc_score(o[c], p[c])))
        except ValueError:
            prof.append(float("nan"))
        proj.append(p)
    return np.array(prof), proj, o, s


def none_auc_from_proj(Pm, rows):
    idx = np.array([i for i, r in enumerate(rows) if r["split"] == "confirmation" and r["perspective"] == "none"])
    s = np.array([rows[i]["surface"] for i in idx])
    return [float(roc_auc_score(s, Pm[idx, L])) if Pm[idx, L].std() > 0 else float("nan") for L in range(Pm.shape[1])]


def band_boot(proj, o, s, n, B=1000):
    rng = np.random.default_rng(SEED); c = s != o
    def stat(sel):
        r = {}
        for b, (lo, hi) in BANDS.items():
            r[b] = np.mean([roc_auc_score(o[sel][c[sel]], proj[L][sel][c[sel]]) for L in band_layers(n, lo, hi)])
        return r
    pt = stat(np.arange(len(o)))
    bs = [stat(rng.integers(0, len(o), len(o))) for _ in range(B)]
    return {"mid": pt["mid"], "late": pt["late"],
            "diff_ci": list(np.percentile([b["late"] - b["mid"] for b in bs], [2.5, 97.5]))}


def sigmoid(d, lo, hi, k, d0):
    return lo + (hi - lo) / (1 + np.exp(-k * (d - d0)))


def fit_transition(prof):
    n = len(prof) - 1
    d = np.arange(len(prof)) / n
    valid = np.isfinite(prof)
    start = int(np.nanargmin(np.where(np.arange(len(prof)) < n, prof, np.nan)))
    m = (np.arange(len(prof)) >= start) & (np.arange(len(prof)) < n) & valid
    x, y = d[m], prof[m]
    if len(x) < 5:
        return None
    try:
        p, _ = curve_fit(sigmoid, x, y, p0=[y.min(), y.max(), 20.0, float(np.median(x))],
                         bounds=([-0.1, -0.1, 0.1, 0.0], [1.1, 1.1, 200.0, 1.0]), maxfev=20000)
        pred = sigmoid(x, *p)
        r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
        return {"lo": float(p[0]), "hi": float(p[1]), "k": float(p[2]), "d_star": float(p[3]), "r2": float(r2),
                "start_depth": float(start / n)}
    except Exception as e:  # report failures explicitly
        return {"error": repr(e)}


def strat_effect(g, o, s):
    bs = np.mean([g[(o == v) & (s == 1)].mean() - g[(o == v) & (s == 0)].mean() for v in (0, 1)])
    bo = np.mean([g[(s == v) & (o == 1)].mean() - g[(s == v) & (o == 0)].mean() for v in (0, 1)])
    return float(bs), float(bo)


def main():
    f = R / "results" / "sweep2_summary.json"
    summary = json.load(open(f)) if f.exists() else {}  # update only the tags given on the command line
    for tag in sys.argv[1:]:
        A1, A2 = R / "acts" / tag, R / "acts2" / tag
        if not (A2 / "done").exists():
            print(tag, "sweep2 not done"); continue
        res = dict(summary.get(tag, {}))  # keep earlier results for parts whose inputs are not stored
        # H6/H7 control + H7 valence
        for pos in ("raw", "gen"):
            if (A2 / f"ctrl_proj_{pos}.npy").exists():
                Pc = np.load(A2 / f"ctrl_proj_{pos}.npy")
                prof_c, proj_c, oc, sc = profile_from_proj(Pc, ctrl_rows)
                none_c, n = none_auc_from_proj(Pc, ctrl_rows), Pc.shape[1] - 1
            else:
                Gc, Sc = np.load(A2 / f"ctrl_goal_{pos}.npy", mmap_mode="r"), np.load(A2 / f"ctrl_src_{pos}.npy", mmap_mode="r")
                yc = np.load(A2 / "ctrl_src_labels.npy")
                prof_c, proj_c, oc, sc = conflict_auc_profile(Gc, Sc, yc, ctrl_rows)
                none_c, n = none_auc_profile(Gc, Sc, yc, ctrl_rows), Gc.shape[1] - 1
            res[f"control_{pos}"] = {"profile": prof_c.tolist(), "none_auc": none_c,
                                     **band_boot(proj_c, oc, sc, n), "fit": fit_transition(prof_c)}
            # valence profile from the saved confirmation results (identical metric)
            C = json.load(open(R / "results" / f"{tag}_confirmation_{pos}.json"))["layers"]
            prof_v = np.array([l["sst2/diffmeans"]["auc_outcome_conflict_self"] for l in C], float)
            res[f"valence_{pos}"] = {"profile": prof_v.tolist(), "fit": fit_transition(prof_v)}
        # second control (wooden/metal), projection files only
        if (A2 / "ctrl2_behaviour.npy").exists():
            for pos in ("raw", "gen"):
                Pc2 = np.load(A2 / f"ctrl2_proj_{pos}.npy")
                prof2, proj2, o2, s2 = profile_from_proj(Pc2, ctrl2_rows)
                res[f"control2_{pos}"] = {"profile": prof2.tolist(), "none_auc": none_auc_from_proj(Pc2, ctrl2_rows),
                                          **band_boot(proj2, o2, s2, Pc2.shape[1] - 1), "fit": fit_transition(prof2)}
            g2 = np.load(A2 / "ctrl2_behaviour.npy"); i2 = np.load(A2 / "ctrl2_behaviour_ids.npy")
            m2 = np.array([ctrl2_rows[i]["perspective"] == "self" and ctrl2_rows[i]["split"] == "confirmation" for i in i2])
            o2 = np.array([ctrl2_rows[i]["outcome"] for i in i2[m2]]); s2 = np.array([ctrl2_rows[i]["surface"] for i in i2[m2]])
            bs2, bo2 = strat_effect(g2[m2], o2, s2); c2 = s2 != o2; pr2 = (g2[m2] > 0).astype(int)
            res["control2_behaviour"] = {"acc": float((pr2 == o2).mean()), "acc_conflict": float((pr2[c2] == o2[c2]).mean()),
                                         "acc_congruent": float((pr2[~c2] == o2[~c2]).mean()), "B_surf": bs2, "B_out": bo2}
        # control behaviour
        gb = np.load(A2 / "ctrl_behaviour_lr.npy"); ids = np.load(A2 / "ctrl_behaviour_ids.npy")
        m = np.array([ctrl_rows[i]["perspective"] == "self" and ctrl_rows[i]["split"] == "confirmation" for i in ids])
        o = np.array([ctrl_rows[i]["outcome"] for i in ids[m]]); s = np.array([ctrl_rows[i]["surface"] for i in ids[m]])
        bs_, bo_ = strat_effect(gb[m], o, s)
        c = s != o
        pred = (gb[m] > 0).astype(int)
        res["control_behaviour"] = {"acc": float((pred == o).mean()), "acc_conflict": float((pred[c] == o[c]).mean()),
                                    "acc_congruent": float((pred[~c] == o[~c]).mean()), "B_surf": bs_, "B_out": bo_}
        # H8
        P = json.load(open(A2 / "prompt_robustness.json")); pid = np.array(P["ids"])
        o = np.array([goal_rows[i]["outcome"] for i in pid]); s = np.array([goal_rows[i]["surface"] for i in pid])
        res["prompts"] = {}
        for k, v in P.items():
            if k.startswith("t") and isinstance(v, list):
                g = np.array(v); bs_, bo_ = strat_effect(g, o, s); c = s != o; pred = (g > 0).astype(int)
                res["prompts"][k] = {"B_surf": bs_, "B_out": bo_, "acc": float((pred == o).mean()),
                                     "acc_conflict": float((pred[c] == o[c]).mean())}
        # H9
        Pt = json.load(open(A2 / "patching.json"))
        keep = [p for p in Pt if abs(p["g_corrupt"] - p["g_clean"]) > 1]
        if keep:
            nL = keep[0]["n_layers"]
            El = np.mean([[(x - p["g_clean"]) / (p["g_corrupt"] - p["g_clean"]) for x in p["patch_label"]] for p in keep], 0)
            Ef = np.mean([[(x - p["g_clean"]) / (p["g_corrupt"] - p["g_clean"]) for x in p["patch_final"]] for p in keep], 0)
            cross = next((i for i in range(nL) if Ef[i] > El[i]), None)
            res["patching"] = {"n_pairs": len(keep), "E_label": El.tolist(), "E_final": Ef.tolist(),
                               "crossing_depth": None if cross is None else (cross + 1) / nL}
        # H10 natural replication
        nat_inputs = all((A2 / f"nat_proj_{p}.npy").exists() or ((A2 / f"nat_{p}.npy").exists() and (A1 / f"sst2_{p}.npy").exists())
                         for p in ("raw", "gen"))
        if (A2 / "nat_behaviour_gb.npy").exists() and nat_inputs:
            nat = [json.loads(l) for l in open(R / "data/natural_stimuli.jsonl")]
            for r in nat:
                r["split"] = "confirmation"  # reuse helpers (natural items are all evaluation items)
            for pos in ("raw", "gen"):
                if (A2 / f"nat_proj_{pos}.npy").exists():
                    Pn = np.load(A2 / f"nat_proj_{pos}.npy")
                    prof_n, proj_n, on, sn = profile_from_proj(Pn, nat); nn = Pn.shape[1] - 1
                else:
                    Gn = np.load(A2 / f"nat_{pos}.npy", mmap_mode="r")
                    Sv = np.load(A1 / f"sst2_{pos}.npy", mmap_mode="r"); yv = np.load(A1 / "sst2_labels.npy")
                    prof_n, proj_n, on, sn = conflict_auc_profile(Gn, Sv, yv, nat); nn = Gn.shape[1] - 1
                res[f"natural_{pos}"] = {"profile": prof_n.tolist(), **band_boot(proj_n, on, sn, nn)}
            g = np.load(A2 / "nat_behaviour_gb.npy"); ids = np.load(A2 / "nat_behaviour_ids.npy")
            o = np.array([nat[i]["outcome"] for i in ids]); s = np.array([nat[i]["surface"] for i in ids])
            bs_, bo_ = strat_effect(g, o, s); c = s != o; pred = (g > 0).astype(int)
            res["natural_behaviour"] = {"B_surf": bs_, "B_out": bo_, "acc": float((pred == o).mean()),
                                        "acc_conflict": float((pred[c] == o[c]).mean()),
                                        "acc_congruent": float((pred[~c] == o[~c]).mean())}
        summary[tag] = res
        cr, cg = res["control_raw"], res["control_gen"]
        vg = res["valence_gen"]["fit"] or {}
        print(f"{tag:16s} CONTROL raw mid={cr['mid']:.2f} late={cr['late']:.2f} d*={((cr['fit'] or {}).get('d_star', np.nan)):.2f} | "
              f"gen mid={cg['mid']:.2f} late={cg['late']:.2f} d*={((cg['fit'] or {}).get('d_star', np.nan)):.2f} | "
              f"VALENCE d* raw={((res['valence_raw']['fit'] or {}).get('d_star', np.nan)):.2f} gen={vg.get('d_star', np.nan):.2f} | "
              f"ctrl beh conf={res['control_behaviour']['acc_conflict']:.2f} Bsurf={res['control_behaviour']['B_surf']:+.2f} | "
              f"cross={res.get('patching', {}).get('crossing_depth')}", flush=True)
        if "control2_raw" in res:
            a, b, cb2 = res["control2_raw"], res["control2_gen"], res["control2_behaviour"]
            print(f"{'':16s} CONTROL2 raw mid={a['mid']:.2f} late={a['late']:.2f} d*={((a['fit'] or {}).get('d_star', np.nan)):.2f} | "
                  f"gen mid={b['mid']:.2f} late={b['late']:.2f} | beh conf={cb2['acc_conflict']:.2f} Bsurf={cb2['B_surf']:+.2f}", flush=True)
        if "natural_raw" in res:
            nr, ng, nb = res["natural_raw"], res["natural_gen"], res["natural_behaviour"]
            print(f"{'':16s} NATURAL raw mid={nr['mid']:.2f} late={nr['late']:.2f} CI={nr['diff_ci'][0]:+.2f},{nr['diff_ci'][1]:+.2f} | "
                  f"gen mid={ng['mid']:.2f} late={ng['late']:.2f} | beh acc={nb['acc']:.2f} conf={nb['acc_conflict']:.2f} Bsurf={nb['B_surf']:+.2f}", flush=True)
    json.dump(summary, open(R / "results" / "sweep2_summary.json", "w"))


if __name__ == "__main__":
    main()
