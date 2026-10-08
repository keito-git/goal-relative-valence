"""Output-level baseline (post hoc): off-the-shelf sentiment classifiers on the goal stimuli (confirmation, self).
Reports conflict-cell AUC for the stake (0 = follows the word, 1 = follows the stake) and B_surf/B_out in logit units."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
import json
from pathlib import Path
import numpy as np, torch
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
torch.set_num_threads(4)
R = Path(os.environ.get("GOALVAL_ROOT", ".")); rows = [json.loads(l) for l in open(R / "data/goal_stimuli.jsonl")]
sel = [r for r in rows if r["split"] == "confirmation" and r["perspective"] == "self"]
o = np.array([r["outcome"] for r in sel]); s = np.array([r["surface"] for r in sel]); c = s != o
out = {}
for repo in ["distilbert/distilbert-base-uncased-finetuned-sst-2-english", "siebert/sentiment-roberta-large-english"]:
    tok = AutoTokenizer.from_pretrained(repo); m = AutoModelForSequenceClassification.from_pretrained(repo).cuda().eval()
    pos_id = [i for i, l in m.config.id2label.items() if l.upper().startswith("POS")][0]
    sc = []
    with torch.no_grad():
        for b in range(0, len(sel), 128):
            enc = tok([r["text"] for r in sel[b:b + 128]], return_tensors="pt", padding=True, truncation=True).to("cuda")
            lg = m(**enc).logits.float(); sc.append((lg[:, pos_id] - lg[:, 1 - pos_id]).cpu())
    g = torch.cat(sc).numpy()
    bs = np.mean([g[(o == v) & (s == 1)].mean() - g[(o == v) & (s == 0)].mean() for v in (0, 1)])
    bo = np.mean([g[(s == v) & (o == 1)].mean() - g[(s == v) & (o == 0)].mean() for v in (0, 1)])
    out[repo] = {"conflict_auc": float(roc_auc_score(o[c], g[c])), "B_surf": float(bs), "B_out": float(bo),
                 "acc_vs_outcome": float(((g > 0).astype(int) == o).mean())}
    print(repo, out[repo], flush=True)
    del m; torch.cuda.empty_cache()
json.dump(out, open(R / "results" / "clf_baseline.json", "w"), indent=1)
