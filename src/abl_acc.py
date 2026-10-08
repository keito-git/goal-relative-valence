"""EXPLORATORY: conflict-cell accuracy under valence vs in-plane random ablation (mid band, confirmation, self)."""
import os
import json, sys
from pathlib import Path
import numpy as np
R = Path(os.environ.get("GOALVAL_ROOT", ".")); rows = [json.loads(l) for l in open(R / "data/goal_stimuli.jsonl")]
out = {}
for tag in sys.argv[1:]:
    A = json.load(open(R / "results" / f"ablate_{tag}.json")); ids = np.array(A["ids"])
    m = np.array([rows[i]["perspective"] == "self" and rows[i]["split"] == "confirmation" for i in ids])
    o = np.array([rows[i]["outcome"] for i in ids[m]]); s = np.array([rows[i]["surface"] for i in ids[m]]); c = s != o
    acc = lambda g: float(((np.array(g)[m] > 0).astype(int)[c] == o[c]).mean())
    rnd = [acc(A[f"mid/inplane_{k}"]) for k in range(20)]
    out[tag] = {"none": acc(A["none"]), "valence": acc(A["mid/valence"]), "random_max": max(rnd), "random_mean": float(np.mean(rnd))}
    print(tag, out[tag])
json.dump(out, open(R / "results" / "ablation_conflict_acc_explore.json", "w"))
