"""H21 (prereg v9): lexical-leakage audit of the canonical probing sets (SST-2, GoEmotions valence).
5-fold CV accuracy of a unigram bag-of-words logistic regression and of per-layer L2 logistic probes on saved
last-token states (raw position). Probes are fitted on the GPU (full-batch L-BFGS, C = 1 as in sklearn's default).
Writes results/leak_audit.json.

Usage: python leak_audit.py tag1 tag2 ...
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract import valence_sources  # noqa: E402

torch.set_num_threads(2)
R = Path(os.environ.get("GOALVAL_ROOT", "."))
SEED = 20261010
C = 1.0


def folds(y):
    return list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))


def bow_oof(texts, y):
    pred = np.zeros(len(y), int)
    for tr, te in folds(y):
        v = CountVectorizer(ngram_range=(1, 1), lowercase=True).fit([texts[i] for i in tr])
        lr = LogisticRegression(C=C, max_iter=5000).fit(v.transform([texts[i] for i in tr]), y[tr])
        pred[te] = lr.predict(v.transform([texts[i] for i in te]))
    return pred


def probe_oof(X, y):
    """X: n x d float32. L2 logistic regression (sum loss + ||w||^2 / (2C)), standardised on the training fold."""
    pred = np.zeros(len(y), int)
    for tr, te in folds(y):
        Xt = torch.tensor(X[tr], device="cuda", dtype=torch.float32)
        mu, sd = Xt.mean(0), Xt.std(0).clamp_min(1e-6)
        Xt = (Xt - mu) / sd
        yt = torch.tensor(y[tr], device="cuda", dtype=torch.float32)
        w = torch.zeros(X.shape[1], device="cuda", requires_grad=True); b = torch.zeros(1, device="cuda", requires_grad=True)
        opt = torch.optim.LBFGS([w, b], lr=1, max_iter=500, tolerance_grad=1e-6, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(Xt @ w + b, yt, reduction="sum") + (w @ w) / (2 * C)
            loss.backward()
            return loss
        opt.step(closure)
        Xe = (torch.tensor(X[te], device="cuda", dtype=torch.float32) - mu) / sd
        pred[te] = ((Xe @ w + b) > 0).int().cpu().numpy()
    return pred


def main():
    f = R / "results" / "leak_audit.json"
    out = json.load(open(f)) if f.exists() else {}
    src = valence_sources()
    bow = {}
    for ds in ("sst2", "goemo"):
        texts = [r["text"] for r in src[ds]]; y = np.array([r["label"] for r in src[ds]])
        bow[ds] = bow_oof(texts, y)
        out.setdefault("_bow", {})[ds] = float((bow[ds] == y).mean())
        print(ds, "BoW acc", out["_bow"][ds], flush=True)
    for tag in sys.argv[1:]:
        res = {}
        for ds in ("sst2", "goemo"):
            A = R / "acts" / tag
            y = np.load(A / f"{ds}_labels.npy")
            assert np.array_equal(y, np.array([r["label"] for r in src[ds]]))
            H = np.load(A / f"{ds}_raw.npy", mmap_mode="r")
            accs, hard = [], []
            wrong = bow[ds] != y
            for L in range(H.shape[1]):
                p = probe_oof(np.asarray(H[:, L], dtype=np.float32), y)
                accs.append(float((p == y).mean())); hard.append(float((p[wrong] == y[wrong]).mean()))
            n = H.shape[1] - 1
            mid = [L for L in range(n) if 0.2 <= L / n <= 0.45]
            best = int(np.argmax(accs[:n]))
            res[ds] = {"acc": accs, "acc_bow_wrong": hard, "best_layer": best, "best": accs[best], "mid": float(np.mean([accs[L] for L in mid])),
                       "bow_ratio": out["_bow"][ds] / accs[best], "n_bow_wrong": int(wrong.sum())}
            print(f"{tag:16s} {ds}: best={accs[best]:.3f}@{best}/{n} mid={res[ds]['mid']:.3f} BoW/best={res[ds]['bow_ratio']:.3f} "
                  f"probe on BoW-wrong (n={wrong.sum()}): best-layer {hard[best]:.3f}", flush=True)
        out[tag] = res
        json.dump(out, open(f, "w"))


if __name__ == "__main__":
    main()
