"""Confirmatory summary (prereg v1 H1/H2, v2 H3/H4) across models.

L* per model is read from the DISCOVERY results; H1/H2 are evaluated on the CONFIRMATION results at L*.
Usage: python summarize.py tag1 tag2 ...
"""
import os
import json
import sys
from pathlib import Path

import numpy as np

R = Path(os.environ.get("GOALVAL_ROOT", "."))
rows = [json.loads(l) for l in open(R / "data/goal_stimuli.jsonl")]
EST = ["diffmeans", "logistic", "logistic_pca50"]


def beh_metrics(gb, ids, split, persp):
    m = np.array([rows[i]["perspective"] == persp and rows[i]["split"] == split for i in ids])
    o = np.array([rows[i]["outcome"] for i in ids[m]])
    s = np.array([rows[i]["surface"] for i in ids[m]])
    g = gb[m]
    pred = (g > 0).astype(int)
    con = s == o
    b_surf = np.mean([g[(o == v) & (s == 1)].mean() - g[(o == v) & (s == 0)].mean() for v in (0, 1)])
    b_out = np.mean([g[(s == v) & (o == 1)].mean() - g[(s == v) & (o == 0)].mean() for v in (0, 1)])
    return dict(acc=float((pred == o).mean()), acc_congruent=float((pred[con] == o[con]).mean()),
                acc_conflict=float((pred[~con] == o[~con]).mean()), B_surf=float(b_surf), B_out=float(b_out),
                R=float(b_surf / b_out))


def main() -> None:
    out = {}
    for tag in sys.argv[1:]:
        disc = R / "results" / f"{tag}_discovery_raw.json"
        conf = R / "results" / f"{tag}_confirmation_raw.json"
        if not (disc.exists() and conf.exists()):
            print(tag, "missing results")
            continue
        D, C = json.load(open(disc)), json.load(open(conf))
        Ls = int(np.nanargmax([l["sst2/diffmeans"]["d_outcome_self"] for l in D["layers"]]))
        c = C["layers"][Ls]
        floor = c["sst2/null_absd_outcome_self_p95"]
        res = {"L*": Ls, "n_layers": len(D["layers"]) - 1, "floor_p95": floor}
        for e in EST:
            k = c[f"sst2/{e}"]
            res[e] = {kk: round(k[kk], 3) for kk in ("d_outcome_self", "d_surface_self", "d_outcome_other",
                                                     "d_surface_other", "auc_outcome_conflict_self")}
        dm = c["sst2/diffmeans"]
        res["H1"] = bool(dm["d_outcome_self"] > 0 and abs(dm["d_outcome_self"]) > floor)
        res["H2"] = bool(abs(dm["d_outcome_self"]) > abs(dm["d_surface_self"]))
        res["sign_agree_estimators"] = bool(len({np.sign(c[f"sst2/{e}"]["d_outcome_self"]) for e in EST}) == 1)
        res["goemo_diffmeans"] = {kk: round(c["goemo/diffmeans"][kk], 3) for kk in ("d_outcome_self", "d_surface_self")}
        # layer profile (confirmation) for figures
        res["profile"] = [(l["layer"], round(l["sst2/diffmeans"]["d_outcome_self"], 3),
                           round(l["sst2/diffmeans"]["d_surface_self"], 3),
                           round(l["sst2/null_absd_outcome_self_p95"], 3)) for l in C["layers"]]
        A = R / "acts" / tag
        gb, ids = np.load(A / "behaviour_gb.npy"), np.load(A / "behaviour_ids.npy")
        res["behaviour_conf"] = {p: beh_metrics(gb, ids, "confirmation", p) for p in ("self", "other")}
        res["H3"] = bool(res["behaviour_conf"]["self"]["B_surf"] > 0 and
                         res["behaviour_conf"]["self"]["acc_conflict"] < res["behaviour_conf"]["self"]["acc_congruent"])
        out[tag] = res
        b = res["behaviour_conf"]["self"]
        print(f"{tag:18s} L*={Ls:2d}/{res['n_layers']} dOut={dm['d_outcome_self']:+.2f} dSurf={dm['d_surface_self']:+.2f} "
              f"floor={floor:.2f} H1={res['H1']} H2={res['H2']} agree={res['sign_agree_estimators']} | "
              f"beh acc={b['acc']:.3f} cong={b['acc_congruent']:.3f} conf={b['acc_conflict']:.3f} "
              f"Bsurf={b['B_surf']:+.2f} Bout={b['B_out']:+.2f} H3={res['H3']}")
    json.dump(out, open(R / "results" / "confirmatory_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
