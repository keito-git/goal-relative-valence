"""Validate generated sentences with NLI (no human annotation), label surface polarity with an off-the-shelf
sentiment classifier, and build self-perspective natural stimuli. Writes data/natural_stimuli.jsonl."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
torch.set_num_threads(4)
D = Path(os.environ.get("GOALVAL_ROOT", ".")) / "data"
raw = [json.loads(l) for l in open(D / "natural_raw.jsonl")]

def probs(repo, pairs, bs=64):
    tok = AutoTokenizer.from_pretrained(repo); m = AutoModelForSequenceClassification.from_pretrained(repo).cuda().eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(pairs), bs):
            a = [p[0] for p in pairs[i:i + bs]]; b = [p[1] for p in pairs[i:i + bs]] if pairs[0][1] is not None else None
            enc = tok(a, b, return_tensors="pt", padding=True, truncation=True).to("cuda") if b else tok(a, return_tensors="pt", padding=True, truncation=True).to("cuda")
            out.append(torch.softmax(m(**enc).logits.float(), -1).cpu())
    return torch.cat(out), m.config.id2label

nli_repo = "FacebookAI/roberta-large-mnli"
win = [(r["sentence"], f"{r['winner']} won.") for r in raw]
lose = [(r["sentence"], f"{[p for p in r['parties'] if p != r['winner']][0]} won.") for r in raw]
pw, lab = probs(nli_repo, win); pl, _ = probs(nli_repo, lose)
ent = [k for k, v in lab.items() if v.upper().startswith("ENTAIL")][0]
keep = [r for r, a, b in zip(raw, pw[:, ent], pl[:, ent]) if a > 0.8 and b < 0.2]
print("NLI kept", len(keep), "of", len(raw))
ps, lab2 = probs("siebert/sentiment-roberta-large-english", [(r["sentence"], None) for r in keep])
pos = [k for k, v in lab2.items() if v.upper().startswith("POS")][0]
rows, i = [], 0
for r, p in zip(keep, ps[:, pos]):
    s = int(p > 0.5)
    for X in r["parties"]:
        o = int(X == r["winner"])
        rows.append({"id": i, "domain": r["domain"], "perspective": "self", "text": f"You supported {X}. {r['sentence']}",
                     "stake": X, "surface": s, "surface_prob": float(p), "outcome": o, "split": "natural",
                     "subject_role": r["subject_role"], "stakeholder": "you"}); i += 1
    rows.append({"id": i, "domain": r["domain"], "perspective": "none", "text": r["sentence"], "surface": s,
                 "surface_prob": float(p), "outcome": -1, "split": "natural", "stakeholder": None}); i += 1
with open(D / "natural_stimuli.jsonl", "w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
selfr = [r for r in rows if r["perspective"] == "self"]
print("items", len(rows), "self", len(selfr), "surface positive share", sum(r["surface"] for r in selfr) / len(selfr),
      "conflict share", sum(r["surface"] != r["outcome"] for r in selfr) / len(selfr))
