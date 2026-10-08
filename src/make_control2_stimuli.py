"""Second non-affective relational control (material: wooden vs metal), same XOR structure as the valence stimuli.

surface s = 1 if the event word is "wooden"; relational outcome o = 1 if the stakeholder's object is wooden = 1[s == m].
Writes data/control2_stimuli.jsonl and data/control2_source.jsonl (stake-free wood/metal sentences).
"""
import os
import itertools
import json
import random
from pathlib import Path

from make_stimuli import DOMAINS, DISCOVERY, OTHERS
from make_control_stimuli import SUBJECT

SEED = 20261008
OUT = Path(os.environ.get("GOALVAL_ROOT", ".")) / "data"
OBJECTS = {"sports": "trophy case", "election": "podium", "auction": "storage cabinet", "chess": "chess board",
           "grant": "bookshelf", "lottery": "prize chest", "race": "feeding trough", "court": "filing cabinet",
           "hiring": "desk"}


def build_items() -> list:
    rows = []
    for dom, (labels, _ctx, s_self, s_other, _ep, _en) in DOMAINS.items():
        split = "discovery" if dom in DISCOVERY else "confirmation"
        obj, subj = OBJECTS[dom], SUBJECT[dom]
        for A, X in itertools.permutations(labels, 2):
            for order in (0, 1):
                P, Q = (A, X) if order == 0 else (X, A)
                q_subj = subj.format(X=Q)
                q_subj = q_subj[0].lower() + q_subj[1:] if subj.startswith("The") else q_subj
                c = f"{subj.format(X=P)} and {q_subj} were each given a {obj}, one wooden and one metal."
                for B, match in ((A, 1), (X, 0)):
                    for mat, word in ((1, "wooden"), (0, "metal")):
                        e = f"{subj.format(X=B)} received the {word} {obj}."
                        outcome = int(mat == match)  # stakeholder's object is wooden
                        base = dict(domain=dom, split=split, A=A, X=X, B=B, order=order, surface=mat, match=match,
                                    outcome=outcome, event=e, obj=obj)
                        rows.append(dict(base, perspective="self", text=f"{c} {s_self.format(A=A)} {e}", stakeholder="you"))
                        O = OTHERS[(labels.index(A) + order) % 2]
                        rows.append(dict(base, perspective="other", text=f"{c} {s_other.format(O=O, A=A)} {e}", stakeholder=O))
                        rows.append(dict(base, perspective="none", text=f"{c} {e}", stakeholder=None))
    seen, out = set(), []
    for r in rows:
        if r["perspective"] == "none":
            if r["text"] in seen:
                continue
            seen.add(r["text"])
        out.append(r)
    random.Random(SEED).shuffle(out)
    for i, r in enumerate(out):
        r["id"] = i
    return out


def build_source(n_per_side: int = 1000) -> list:
    things = ["table", "chair", "bench", "door", "spoon", "box", "fence", "bridge", "gate", "shelf", "bowl", "frame",
              "ladder", "desk", "bucket", "cabinet", "railing", "sign", "toy", "boat", "crate", "stool", "cup", "handle",
              "panel", "tray", "comb", "whistle", "barrel", "shed"]
    frames = ["The {t} {l} was made of {m}.", "The {t} {l} was a {a} one.", "Someone had built the {t} {l} out of {m}.",
              "The old {t} {l} was {a}.", "A {a} {t} stood {l}.", "They put a {a} {t} {l}.",
              "The new {t} {l} was made entirely of {m}.", "There was a {a} {t} {l}."]
    locs = ["in the garden", "by the window", "in the hall", "near the door", "in the shed", "on the porch",
            "in the kitchen", "at the station", "in the yard", "by the river"]
    rng = random.Random(SEED)
    combos = list(itertools.product(things, frames, locs))
    rng.shuffle(combos)
    rows = []
    for side, m, a in ((1, "wood", "wooden"), (0, "metal", "metal")):
        seen = set()
        for t, f, l in combos:
            s = f.format(t=t, m=m, a=a, l=l)
            if s in seen:
                continue
            seen.add(s)
            rows.append({"text": s, "label": side})
            if len(seen) >= n_per_side:
                break
    return rows


if __name__ == "__main__":
    items = build_items()
    with open(OUT / "control2_stimuli.jsonl", "w") as f:
        for r in items:
            f.write(json.dumps(r) + "\n")
    src = build_source()
    with open(OUT / "control2_source.jsonl", "w") as f:
        for r in src:
            f.write(json.dumps(r) + "\n")
    from collections import Counter
    print(len(items), Counter((r["split"], r["perspective"]) for r in items), "source", Counter(r["label"] for r in src))
    for r in items[:2]:
        print(r["perspective"], "o=", r["outcome"], "s=", r["surface"], "|", r["text"])
