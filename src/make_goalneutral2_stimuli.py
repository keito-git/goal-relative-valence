"""Goal-explicit variant of the goal-relevant neutral control (prereg v9, H19).

Context: "<P> and <q> each had to open the <obj>. They were given one brass and one steel key; only the brass key
opens the <obj>." Everything else as make_goalneutral_stimuli.py. Writes data/goalneutral2_stimuli.jsonl.
"""
import json
import random

import make_goalneutral_stimuli as gn

SEED = 20261010


def build_items() -> list:
    rows = gn.build_items()
    out = []
    for r in rows:
        r = dict(r)
        old_ctx = r["text"].split(" only the brass key opens the ")[0]  # "<P> and <q> were each given a key, one brass and one steel;"
        pq = old_ctx.split(" were each given a key")[0]
        rest = r["text"][len(old_ctx) + len(f" only the brass key opens the {r['obj']}."):]
        ctx = (f"{pq} each had to open the {r['obj']}. They were given one brass and one steel key; "
               f"only the brass key opens the {r['obj']}.")
        r["text"] = ctx + rest
        out.append(r)
    random.Random(SEED).shuffle(out)
    for i, r in enumerate(out):
        r["id"] = i
    return out


if __name__ == "__main__":
    items = build_items()
    with open(gn.OUT / "goalneutral2_stimuli.jsonl", "w") as f:
        for r in items:
            f.write(json.dumps(r) + "\n")
    from collections import Counter
    print(len(items), Counter((r["split"], r["perspective"]) for r in items))
    for r in items[:3]:
        print(r["perspective"], "o=", r["outcome"], "s=", r["surface"], "|", r["text"])
