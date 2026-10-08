"""R3 (prereg v11): relabel the surface polarity of the naturalistic sentences with two further sentiment classifiers
and re-run the naturalistic conflict-cell analysis. Writes results/natural_relabel.json.

Usage: python natural_relabel.py tag1 tag2 ...
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from analyze_sweep2 import profile_from_proj, conflict_auc_profile, band_boot

R = Path(os.environ.get("GOALVAL_ROOT", "."))
CLFS = {"distilbert": "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        "twitter_roberta": "cardiffnlp/twitter-roberta-base-sentiment-latest"}


@torch.no_grad()
def positive_label(repo, sents):
    tok = AutoTokenizer.from_pretrained(repo); m = AutoModelForSequenceClassification.from_pretrained(repo).cuda().eval()
    lab = {v.lower(): k for k, v in m.config.id2label.items()}
    out = []
    for i in range(0, len(sents), 32):
        e = tok(sents[i:i + 32], return_tensors="pt", padding=True, truncation=True).to("cuda")
        p = torch.softmax(m(**e).logits.float(), -1)
        out += (p[:, lab["positive"]] > p[:, lab["negative"]]).int().tolist()
    return out


def main():
    nat = [json.loads(l) for l in open(R / "data/natural_stimuli.jsonl")]
    def sent_of(r):
        if r["perspective"] != "self":
            return r["text"]
        return r["text"][len(f"You supported {r['stake']}. "):]
    assert all(r["text"].startswith(f"You supported {r['stake']}. ") for r in nat if r["perspective"] == "self")
    sents = sorted({sent_of(r) for r in nat})
    labels = {"siebert": {s: None for s in sents}}
    for r in nat:
        labels["siebert"][sent_of(r)] = r["surface"]
    for k, repo in CLFS.items():
        labels[k] = dict(zip(sents, positive_label(repo, sents)))
    agree = {s for s in sents if labels["siebert"][s] == labels["distilbert"][s] == labels["twitter_roberta"][s]}
    out = {"n_sentences": len(sents), "n_agree": len(agree),
           "agreement": {k: float(np.mean([labels[k][s] == labels["siebert"][s] for s in sents])) for k in CLFS}}
    for tag in sys.argv[1:]:
        A1, A2 = R / "acts" / tag, R / "acts2" / tag
        res = {}
        for pos in ("raw", "gen"):
            for scheme in ("distilbert", "twitter_roberta", "agree"):
                rows = []
                for r in nat:
                    s = sent_of(r); r2 = dict(r, split="confirmation")
                    if scheme == "agree":
                        if s not in agree:
                            r2["split"] = "excluded"
                    else:
                        r2["surface"] = labels[scheme][s]
                    rows.append(r2)
                if (A2 / f"nat_proj_{pos}.npy").exists():
                    prof, proj, o, sv = profile_from_proj(np.load(A2 / f"nat_proj_{pos}.npy"), rows); n = len(prof) - 1
                else:
                    G = np.load(A2 / f"nat_{pos}.npy", mmap_mode="r")
                    if (A1 / f"sst2_{pos}.npy").exists():
                        S = np.load(A1 / f"sst2_{pos}.npy", mmap_mode="r"); ys = np.load(A1 / "sst2_labels.npy")
                        prof, proj, o, sv = conflict_auc_profile(G, S, ys, rows); n = G.shape[1] - 1
                    else:  # if the response-position SST-2 states are not stored, use the directions saved by sweep6
                        d = np.load(R / "acts6" / tag / "sst_dir_gen.npz")
                        Pm = np.stack([(np.asarray(G[:, L], np.float32) - d["mu"][L].astype(np.float32)) @ d["w"][L].astype(np.float32)
                                       for L in range(G.shape[1])], 1)
                        prof, proj, o, sv = profile_from_proj(Pm, rows); n = G.shape[1] - 1
                res[f"{scheme}_{pos}"] = {"n_conflict": int((sv != o).sum()), **band_boot(proj, o, sv, n, B=500)}
        out[tag] = res
        print(tag, {k: (round(v["mid"], 2), round(v["late"], 2), [round(x, 2) for x in v["diff_ci"]]) for k, v in res.items()}, flush=True)
        json.dump(out, open(R / "results" / "natural_relabel.json", "w"))


if __name__ == "__main__":
    main()
