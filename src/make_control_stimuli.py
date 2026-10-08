"""Non-affective relational control: same XOR structure as the valence stimuli, with left/right instead of win/loss.

Item = context (two labels, each assigned an object, one on the left and one on the right) + stake sentence
(identical to the valence stimuli) + event sentence ("<B> was assigned the <object> on the left/right.").
surface s = 1 if the event word is "left"; relational outcome o = 1 if the stakeholder's object is on the left
= 1[s == m]. Also writes stake-free left/right source sentences for estimating a left/right direction.
"""
import os
import itertools
import json
import random
from pathlib import Path

from make_stimuli import DOMAINS, DISCOVERY, OTHERS

SEED = 20261007
OUT = Path(os.environ.get("GOALVAL_ROOT", ".")) / "data"

OBJECTS = {"sports": "locker room", "election": "campaign office", "auction": "parking spot", "chess": "table",
           "grant": "meeting room", "lottery": "booth", "race": "stall", "court": "conference room", "hiring": "desk"}
# how each domain refers to a label as a sentence subject (taken from the valence event templates)
SUBJECT = {"sports": "The {X} team", "election": "{X}", "auction": "Bidder {X}", "chess": "The {X} side",
           "grant": "Proposal {X}", "lottery": "Ticket {X}", "race": "{X}", "court": "{X}", "hiring": "Applicant {X}"}


def build_items() -> list:
    rows = []
    for dom, (labels, _ctx, s_self, s_other, _ep, _en) in DOMAINS.items():
        split = "discovery" if dom in DISCOVERY else "confirmation"
        obj, subj = OBJECTS[dom], SUBJECT[dom]
        for A, X in itertools.permutations(labels, 2):
            for order in (0, 1):
                P, Q = (A, X) if order == 0 else (X, A)
                c = (f"{subj.format(X=P)} and {subj.format(X=Q)[0].lower() + subj.format(X=Q)[1:] if subj.startswith('The') else subj.format(X=Q)}"
                     f" were each assigned a {obj}, one on the left and one on the right.")
                for B, match in ((A, 1), (X, 0)):
                    for side, word in ((1, "left"), (0, "right")):
                        e = f"{subj.format(X=B)} was assigned the {obj} on the {word}."
                        outcome = int(side == match)  # stakeholder's object is on the left
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


def build_source(n_per_side: int = 1000) -> list:
    """Stake-free left/right sentences (no agent, no stake) for estimating a left/right direction."""
    things = ["cup", "lamp", "chair", "painting", "window", "bicycle", "tree", "car", "sign", "door", "box", "bench",
              "clock", "shelf", "plant", "statue", "fountain", "gate", "mailbox", "pillar", "vase", "book", "printer",
              "fridge", "bed", "desk", "mirror", "bottle", "basket", "ladder"]
    places = ["table", "room", "street", "hallway", "garden", "stage", "kitchen", "office", "road", "bridge", "screen",
              "page", "square", "platform", "corridor", "courtyard", "counter", "parking lot", "aisle", "photo"]
    frames = ["The {t} was on the {s} side of the {p}.", "There was a {t} on the {s} side of the {p}.",
              "A {t} stood at the {s} end of the {p}.", "The {t} sat to the {s} of the {p}.",
              "On the {s} side of the {p} was a {t}.", "The {t} had been placed on the {s} of the {p}.",
              "Near the {p}, the {t} was on the {s}.", "The {t} could be seen on the {s} side of the {p}."]
    rng = random.Random(SEED)
    combos = list(itertools.product(things, places, frames))
    rng.shuffle(combos)
    rows = []
    for side, word in ((1, "left"), (0, "right")):
        for t, p, f in combos[:n_per_side]:
            rows.append({"text": f.format(t=t, s=word, p=p), "label": side})
    return rows


if __name__ == "__main__":
    items = build_items()
    with open(OUT / "control_stimuli.jsonl", "w") as f:
        for r in items:
            f.write(json.dumps(r) + "\n")
    with open(OUT / "control_source.jsonl", "w") as f:
        for r in build_source():
            f.write(json.dumps(r) + "\n")
    from collections import Counter
    print(len(items), Counter((r["split"], r["perspective"]) for r in items))
    for r in items[:3]:
        print(r["perspective"], "o=", r["outcome"], "s=", r["surface"], "|", r["text"])
