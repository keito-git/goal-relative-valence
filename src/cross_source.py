"""Cross-source validity: GoEmotions direction -> SST-2 AUC and SST-2 direction -> GoEmotions AUC, per layer (raw position)."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import json, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
R = Path(os.environ.get("GOALVAL_ROOT", "."))
out = {}
for tag in sys.argv[1:]:
    A = R / "acts" / tag
    S, ys = np.load(A / "sst2_raw.npy", mmap_mode="r"), np.load(A / "sst2_labels.npy")
    G, yg = np.load(A / "goemo_raw.npy", mmap_mode="r"), np.load(A / "goemo_labels.npy")
    g2s, s2g = [], []
    for L in range(S.shape[1]):
        s, g = np.asarray(S[:, L], np.float32), np.asarray(G[:, L], np.float32)
        ws = s[ys == 1].mean(0) - s[ys == 0].mean(0); wg = g[yg == 1].mean(0) - g[yg == 0].mean(0)
        try:
            g2s.append(float(roc_auc_score(ys, (s - g.mean(0)) @ wg))); s2g.append(float(roc_auc_score(yg, (g - s.mean(0)) @ ws)))
        except ValueError:
            g2s.append(float("nan")); s2g.append(float("nan"))
    out[tag] = {"goemo_to_sst2": g2s, "sst2_to_goemo": s2g}
    n = len(g2s) - 1
    print(f"{tag:16s} late GoEmo->SST2={np.nanmean(g2s[3*n//4:]):.3f} final={g2s[-1]:.3f} | late SST2->GoEmo={np.nanmean(s2g[3*n//4:]):.3f} final={s2g[-1]:.3f}")
json.dump(out, open(R / "results" / "cross_source_auc.json", "w"))
