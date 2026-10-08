"""Goal-relevant but non-valenced control: the word is neutral (brass vs steel) but one option serves the stakeholder's goal.

Context: "<P> and <q> were each given a key, one brass and one steel; only the brass key opens the <obj>."
Event:   "<B> received the brass key." / "<B> received the steel key."
surface s = 1 if the event word is "brass" (lexically neutral); outcome o = 1 if the stakeholder got the brass key
(the goal-serving one) = 1[s == m]. Writes data/goalneutral_stimuli.jsonl.
"""
import os
import itertools
import json
import random
from pathlib import Path

from make_stimuli import DOMAINS, DISCOVERY, OTHERS
from make_control_stimuli import SUBJECT

SEED = 20261009
OUT = Path(os.environ.get("GOALVAL_ROOT", ".")) / "data"
OBJECTS = {"sports": "equipment room", "election": "ballot box", "auction": "vault", "chess": "trophy cabinet",
           "grant": "laboratory", "lottery": "prize safe", "race": "stable", "court": "evidence room", "hiring": "office"}


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
                c = (f"{subj.format(X=P)} and {q_subj} were each given a key, one brass and one steel; "
                     f"only the brass key opens the {obj}.")
                for B, match in ((A, 1), (X, 0)):
                    for side, word in ((1, "brass"), (0, "steel")):
                        e = f"{subj.format(X=B)} received the {word} key."
                        outcome = int(side == match)
                        base = dict(domain=dom, split=split, A=A, X=X, B=B, order=order, surface=side, match=match,
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


if __name__ == "__main__":
    items = build_items()
    with open(OUT / "goalneutral_stimuli.jsonl", "w") as f:
        for r in items:
            f.write(json.dumps(r) + "\n")
    from collections import Counter
    print(len(items), Counter((r["split"], r["perspective"]) for r in items))
    for r in items[:2]:
        print(r["perspective"], "o=", r["outcome"], "s=", r["surface"], "|", r["text"])
