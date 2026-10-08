"""EXPLORATORY (not pre-registered): fractional change of B_surf and B_out under valence vs in-plane random ablation."""
import os
import json, sys
from pathlib import Path
import numpy as np
R = Path(os.environ.get("GOALVAL_ROOT", "."))
rows = [json.loads(l) for l in open(R / "data/goal_stimuli.jsonl")]
def eff(g, o, s):
    bs = np.mean([g[(o == v) & (s == 1)].mean() - g[(o == v) & (s == 0)].mean() for v in (0, 1)])
    bo = np.mean([g[(s == v) & (o == 1)].mean() - g[(s == v) & (o == 0)].mean() for v in (0, 1)])
    return bs, bo
for tag in sys.argv[1:]:
    f = R / "results" / f"ablate_{tag}.json"
    if not f.exists(): continue
    A = json.load(open(f)); ids = np.array(A["ids"])
    for split in ("discovery", "confirmation"):
        m = np.array([rows[i]["perspective"] == "self" and rows[i]["split"] == split for i in ids])
        o = np.array([rows[i]["outcome"] for i in ids[m]]); s = np.array([rows[i]["surface"] for i in ids[m]])
        bs0, bo0 = eff(np.array(A["none"])[m], o, s)
        for band in ("mid",):
            rs = np.array([eff(np.array(A[f"{band}/inplane_{k}"])[m], o, s) for k in range(20)])
            line = f"{tag:14s} {split:12s} {band}: "
            for src in ("valence", "goemo"):
                bs, bo = eff(np.array(A[f"{band}/{src}"])[m], o, s)
                line += f"{src}: Bsurf x{bs/bs0:.2f} Bout x{bo/bo0:.2f} | "
            line += f"random Bsurf x{np.mean(rs[:,0])/bs0:.2f} [min {rs[:,0].min()/bs0:.2f}] Bout x{np.mean(rs[:,1])/bo0:.2f} [min {rs[:,1].min()/bo0:.2f}]"
            print(line)
