"""Goal-relative valence stimuli.

Each item = context (two neutral labels, counterbalanced order) + stake sentence (who is bound to which label)
+ event sentence (label B with a positive or negative outcome verb).
Factors: surface polarity of the event verb (pos/neg) x match (B == bound label) -> outcome for the stakeholder
= XOR. Every label token appears equally often in every cell, so outcome-for-stakeholder has no bag-of-words cue.
Perspectives: self ("You"), other (a named third party), none (no stake sentence).
"""
import os
import itertools
import json
import random
from pathlib import Path

SEED = 20261006
OUT = Path(os.environ.get("GOALVAL_ROOT", ".")) / "data"

# domain: (labels, context template, self stake, other stake, event pos, event neg)
DOMAINS = {
    # discovery domains
    "sports": (["Blue", "Red", "Green", "Yellow", "Orange", "Purple"],
               "The {P} team played the {Q} team in the final.",
               "You are on the {A} team.", "{O} is on the {A} team.",
               "The {B} team won the final.", "The {B} team lost the final."),
    "election": (["Morgan", "Taylor", "Jordan", "Casey", "Riley", "Avery"],
                 "{P} and {Q} ran against each other for mayor.",
                 "You voted for {A}.", "{O} voted for {A}.",
                 "{B} won the election.", "{B} lost the election."),
    "auction": (["A", "B", "C", "D", "E", "F"],
                "Bidder {P} and bidder {Q} placed bids on the same house.",
                "You are bidder {A}.", "{O} is bidder {A}.",
                "Bidder {B} won the auction.", "Bidder {B} lost the auction."),
    "chess": (["White", "Black", "North", "South", "East", "West"],
              "The {P} side and the {Q} side met in the last round of the tournament.",
              "You played on the {A} side.", "{O} played on the {A} side.",
              "The {B} side won the game.", "The {B} side lost the game."),
    "grant": (["Alpha", "Beta", "Gamma", "Delta", "Sigma", "Omega"],
              "Proposal {P} and proposal {Q} were sent to the review committee.",
              "You wrote proposal {A}.", "{O} wrote proposal {A}.",
              "Proposal {B} was approved.", "Proposal {B} was rejected."),
    # confirmation domains (held out until confirmation)
    "lottery": (["17", "42", "58", "63", "71", "96"],
                "Ticket {P} and ticket {Q} reached the final draw.",
                "You hold ticket {A}.", "{O} holds ticket {A}.",
                "Ticket {B} won the draw.", "Ticket {B} lost the draw."),
    "race": (["Comet", "Maple", "Harbor", "Cedar", "Pepper", "Willow"],
             "Two horses, {P} and {Q}, ran in the last race of the day.",
             "You bet on {A}.", "{O} bet on {A}.",
             "{B} won the race.", "{B} lost the race."),
    "court": (["Hale", "Price", "Moss", "Lane", "Grant", "Reed"],
              "{P} and {Q} went to court over a contract.",
              "You represent {A}.", "{O} represents {A}.",
              "{B} won the case.", "{B} lost the case."),
    "hiring": (["one", "two", "three", "four", "five", "six"],
               "Applicant {P} and applicant {Q} interviewed for the same position.",
               "You are applicant {A}.", "{O} is applicant {A}.",
               "Applicant {B} was offered the position.", "Applicant {B} was turned down for the position."),
}
DISCOVERY = ["sports", "election", "auction", "chess", "grant"]
OTHERS = ["Sam", "Robin"]


def build() -> list:
    rows = []
    for dom, (labels, ctx, s_self, s_other, e_pos, e_neg) in DOMAINS.items():
        split = "discovery" if dom in DISCOVERY else "confirmation"
        for A, X in itertools.permutations(labels, 2):  # A = bound label, X = rival
            for order in (0, 1):
                P, Q = (A, X) if order == 0 else (X, A)
                c = ctx.format(P=P, Q=Q)
                for B, match in ((A, 1), (X, 0)):
                    for pol, etmp in ((1, e_pos), (0, e_neg)):
                        e = etmp.format(B=B)
                        outcome = int(pol == match)  # good for the stakeholder
                        base = dict(domain=dom, split=split, A=A, X=X, B=B, order=order,
                                    surface=pol, match=match, outcome=outcome, event=e)
                        rows.append(dict(base, perspective="self", text=f"{c} {s_self.format(A=A)} {e}",
                                         stakeholder="you"))
                        O = OTHERS[(labels.index(A) + order) % 2]
                        rows.append(dict(base, perspective="other", text=f"{c} {s_other.format(O=O, A=A)} {e}",
                                         stakeholder=O))
                        rows.append(dict(base, perspective="none", text=f"{c} {e}", stakeholder=None))
    # 'none' rows are duplicated across A/X bindings; keep unique texts for that perspective
    seen, out = set(), []
    for r in rows:
        key = (r["perspective"], r["text"]) if r["perspective"] == "none" else None
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        out.append(r)
    random.Random(SEED).shuffle(out)
    for i, r in enumerate(out):
        r["id"] = i
    return out


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    rows = build()
    with open(OUT / "goal_stimuli.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    from collections import Counter
    print(len(rows), Counter((r["split"], r["perspective"]) for r in rows))
    for r in rows[:4]:
        print(r["perspective"], r["outcome"], r["surface"], "|", r["text"])
