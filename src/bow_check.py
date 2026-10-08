"""Supplementary lexical-leak check: LODO logistic on unigram+bigram counts predicting outcome."""
import os
import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
rows = [json.loads(l) for l in open(os.path.join(os.environ.get("GOALVAL_ROOT", "."), "data", "goal_stimuli.jsonl"))]
for persp in ("self", "other"):
    for split in ("discovery", "confirmation"):
        R = [r for r in rows if r["perspective"] == persp and r["split"] == split]
        txt = [r["text"] for r in R]; y = np.array([r["outcome"] for r in R]); dom = np.array([r["domain"] for r in R])
        for ng in ((1, 1), (1, 2)):
            accs = []
            for d in np.unique(dom):
                tr, te = dom != d, dom == d
                v = CountVectorizer(ngram_range=ng, lowercase=False, token_pattern=r"\S+").fit([t for t, m in zip(txt, tr) if m])
                lr = LogisticRegression(C=1.0, max_iter=3000).fit(v.transform([t for t, m in zip(txt, tr) if m]), y[tr])
                accs.append(lr.score(v.transform([t for t, m in zip(txt, te) if m]), y[te]))
            # in-domain 5-fold as a stricter check (same vocabulary)
            from sklearn.model_selection import cross_val_score
            v = CountVectorizer(ngram_range=ng, lowercase=False, token_pattern=r"\S+")
            Xb = v.fit_transform(txt)
            cv = cross_val_score(LogisticRegression(C=1.0, max_iter=3000), Xb, y, cv=5).mean()
            print(persp, split, ng, "LODO=%.3f" % np.mean(accs), "in-domain5fold=%.3f" % cv)
