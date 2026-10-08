"""J = b0 + bA*A + bL*L + bAL*A*L (A = outcome, L = surface, both coded +-0.5), confirmation self items, per model.
J is the good-minus-bad log-probability. Item-bootstrap 95% CIs (1,000)."""
import os
import json, sys
from pathlib import Path
import numpy as np
R = Path(os.environ.get("GOALVAL_ROOT", ".")); rows = [json.loads(l) for l in open(R / "data/goal_stimuli.jsonl")]
out = {}
for tag in sys.argv[1:]:
    A = R / "acts" / tag
    g, ids = np.load(A / "behaviour_gb.npy"), np.load(A / "behaviour_ids.npy")
    m = np.array([rows[i]["perspective"] == "self" and rows[i]["split"] == "confirmation" for i in ids])
    a = np.array([rows[i]["outcome"] for i in ids[m]]) - 0.5; l = np.array([rows[i]["surface"] for i in ids[m]]) - 0.5; y = g[m]
    X = np.column_stack([np.ones_like(a), a, l, a * l])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    rng = np.random.default_rng(20261008); bs = []
    for _ in range(1000):
        k = rng.integers(0, len(y), len(y)); bs.append(np.linalg.lstsq(X[k], y[k], rcond=None)[0])
    ci = np.percentile(np.array(bs), [2.5, 97.5], axis=0)
    r2 = 1 - np.sum((y - X @ beta) ** 2) / np.sum((y - y.mean()) ** 2)
    out[tag] = {"b0": beta[0], "bA": beta[1], "bL": beta[2], "bAL": beta[3], "ci_bL": ci[:, 2].tolist(), "ci_bAL": ci[:, 3].tolist(), "r2": r2}
    print(f"{tag:16s} bA={beta[1]:+6.2f} bL={beta[2]:+5.2f} [{ci[0,2]:+.2f},{ci[1,2]:+.2f}] bAL={beta[3]:+5.2f} [{ci[0,3]:+.2f},{ci[1,3]:+.2f}] R2={r2:.2f}")
json.dump({k: {kk: (float(vv) if not isinstance(vv, list) else [float(x) for x in vv]) for kk, vv in v.items()} for k, v in out.items()},
          open(R / "results" / "prior_regression.json", "w"), indent=1)
