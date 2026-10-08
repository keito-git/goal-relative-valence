"""R1 (prereg v11): band-boundary sensitivity of the late-minus-middle conflict-cell AUC (SST-2 directions, confirmation,
self), from saved per-layer profiles. Writes results/band_sensitivity.json."""
import json
from pathlib import Path

import numpy as np

RES = Path(__file__).resolve().parents[1] / "results"
INSTRUCT = ["qwen25_7b_it", "mistral_7b_it", "falcon3_7b_it", "granite31_8b_it", "olmo2_7b_it", "phi4", "qwen25_14b_it", "qwen25_32b_it"]
ALL = INSTRUCT + ["qwen25_7b_base", "olmo2_7b_base"]
MID_HI = [0.35, 0.40, 0.45, 0.50]
LATE_LO = [0.50, 0.55, 0.60, 0.65]


def profile(tag, pos):
    f = RES / f"{tag}_confirmation_{pos}.json"
    if not f.exists():
        return None
    L = json.load(open(f))["layers"]
    return np.array([l["sst2/diffmeans"]["auc_outcome_conflict_self"] for l in L], float)


def diff(prof, mid_hi, late_lo):
    n = len(prof) - 1
    mid = [L for L in range(n) if 0.20 <= L / n <= mid_hi]
    late = [L for L in range(n) if late_lo <= L / n <= 0.90]
    if not mid or not late or mid_hi >= late_lo:
        return None
    return float(np.nanmean(prof[late]) - np.nanmean(prof[mid]))


S = json.load(open(RES / "confirmatory_summary.json"))
elig = [t for t in INSTRUCT if S[t]["behaviour_conf"]["self"]["acc"] >= 0.85]
out = {"mid_hi": MID_HI, "late_lo": LATE_LO, "eligible": elig}
for pos, tags in (("raw", ALL), ("gen", INSTRUCT)):
    grid = {}
    for mh in MID_HI:
        for ll in LATE_LO:
            d = {t: diff(profile(t, pos), mh, ll) for t in tags if profile(t, pos) is not None}
            if any(v is None for v in d.values()):
                continue
            e = [d[t] for t in elig if t in d]
            grid[f"{mh:.2f}_{ll:.2f}"] = {"per_model": d, "mean_elig": float(np.mean(e)), "n_pos_elig": int(sum(x > 0 for x in e)),
                                         "n_elig": len(e), "n_pos_all": int(sum(x > 0 for x in d.values())), "n_all": len(d)}
    out[pos] = grid
    print(pos, {k: (round(v["mean_elig"], 2), f"{v['n_pos_elig']}/{v['n_elig']}", f"{v['n_pos_all']}/{v['n_all']}") for k, v in grid.items()})
json.dump(out, open(RES / "band_sensitivity.json", "w"), indent=1)
