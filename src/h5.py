"""H5 (prereg v3): depth transition from stimulus-bound to goal-relative valence, confirmation domains.

Usage: python h5.py tag1 tag2 ...
"""
import os
import json
import sys
from pathlib import Path

import numpy as np

R = Path(os.environ.get("GOALVAL_ROOT", "."))
rows = [json.loads(l) for l in open(R / "data/goal_stimuli.jsonl")]


def beh_acc(tag):
    A = R / "acts" / tag
    gb, ids = np.load(A / "behaviour_gb.npy"), np.load(A / "behaviour_ids.npy")
    m = np.array([rows[i]["perspective"] == "self" and rows[i]["split"] == "confirmation" for i in ids])
    o = np.array([rows[i]["outcome"] for i in ids[m]])
    return float(((gb[m] > 0).astype(int) == o).mean())


def share(d_out, d_surf):
    return d_out / (abs(d_out) + abs(d_surf))


def main() -> None:
    out = {}
    for tag in sys.argv[1:]:
        f = R / "results" / f"{tag}_confirmation_raw.json"
        if not f.exists():
            print(f"{tag:16s} missing")
            continue
        C = json.load(open(f))["layers"]
        res = {"behaviour_acc_self_conf": beh_acc(tag)}
        for est in ("diffmeans", "logistic", "logistic_pca50"):
            dsurf = np.array([l[f"sst2/{est}"]["d_surface_self"] for l in C], dtype=float)
            dout = np.array([l[f"sst2/{est}"]["d_outcome_self"] for l in C], dtype=float)
            Ls, Lf = int(np.nanargmax(dsurf)), len(C) - 1
            rs, rf = share(dout[Ls], dsurf[Ls]), share(dout[Lf], dsurf[Lf])
            # first layer after L_s where the appraisal share exceeds 0.5 (descriptive)
            cross = next((L for L in range(Ls, len(C)) if share(dout[L], dsurf[L]) > 0.5), None)
            res[est] = dict(L_s=Ls, L_f=Lf, r_Ls=round(float(rs), 3), r_Lf=round(float(rf), 3),
                            H5a=bool(rf > rs), H5b=bool(rf > 0.5), crossover_layer=cross)
        out[tag] = res
        d = res["diffmeans"]
        print(f"{tag:16s} beh={res['behaviour_acc_self_conf']:.3f} L_s={d['L_s']:2d} r(L_s)={d['r_Ls']:+.3f} "
              f"L_f={d['L_f']:2d} r(L_f)={d['r_Lf']:+.3f} cross={d['crossover_layer']} H5a={d['H5a']} H5b={d['H5b']} | "
              f"logistic H5a={res['logistic']['H5a']} H5b={res['logistic']['H5b']} | "
              f"pca50 H5a={res['logistic_pca50']['H5a']} H5b={res['logistic_pca50']['H5b']}")
    json.dump(out, open(R / "results" / "h5_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
