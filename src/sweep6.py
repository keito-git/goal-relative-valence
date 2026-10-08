"""Sixth sweep (prereg v11, R2 and R6): SST-2 projections of the valence stimuli at the response position, and a
patching placebo that swaps the order of the two parties without changing the stake.

Usage: python sweep6.py --repo R --tag T --model_dir D --parts goalproj,placebo
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from sweep2 import Runner, first_ids, project, label_positions
from extract import valence_sources
from make_stimuli import DOMAINS

ROOT = Path(os.environ.get("GOALVAL_ROOT", "."))
SEED = 20261011
Q = "\nFor you, was this outcome good or bad? Answer with one word: good or bad."


def goalproj_part(R, out):
    rows = [json.loads(l) for l in open(ROOT / "data/goal_stimuli.jsonl")]
    sst = valence_sources()["sst2"]; ys = np.array([r["label"] for r in sst])
    S = R.last_states([R.chat(r["text"]) for r in sst])
    W = np.stack([S[ys == 1, L].astype(np.float32).mean(0) - S[ys == 0, L].astype(np.float32).mean(0) for L in range(S.shape[1])])
    MU = np.stack([S[:, L].astype(np.float32).mean(0) for L in range(S.shape[1])])
    np.savez(out / "sst_dir_gen.npz", w=W.astype(np.float16), mu=MU.astype(np.float16))
    if not (out / "goal_sst_gen.npy").exists():
        G = R.last_states([R.chat(r["text"]) for r in rows])
        np.save(out / "goal_sst_gen.npy", project(G, S, ys))


@torch.no_grad()
def placebo_part(R, out, n_items=80):
    rows = [json.loads(l) for l in open(ROOT / "data/goal_stimuli.jsonl")]
    sel = [r for r in rows if r["split"] == "confirmation" and r["perspective"] == "self"]
    key = {(r["domain"], r["A"], r["X"], r["order"], r["B"], r["surface"]): r for r in sel}
    items = []
    for r in sel:
        flip = key.get((r["domain"], r["X"], r["A"], 1 - r["order"], r["B"], r["surface"]))      # stake label changed
        plac = key.get((r["domain"], r["A"], r["X"], 1 - r["order"], r["B"], r["surface"]))      # same stake, order swapped
        if flip is not None and plac is not None:
            items.append((r, flip, plac))
    random.Random(SEED).shuffle(items)
    gid, bid = first_ids(R.tok, "good"), first_ids(R.tok, "bad")

    def gb(logits):
        lp = torch.log_softmax(logits[0, -1].float(), -1)
        return float(torch.logsumexp(lp[gid], -1) - torch.logsumexp(lp[bid], -1))
    res = []
    for r, f, pl in items:
        if len(res) >= n_items:
            break
        s_self = DOMAINS[r["domain"]][2]
        enc = {}
        for name, it in (("clean", r), ("flip", f), ("placebo", pl)):
            enc[name] = label_positions(R, R.chat(it["text"] + Q), s_self.format(A=it["A"]), it["A"])
        lens = {len(v[0]) for v in enc.values()}; poss = {tuple(v[1]) for v in enc.values()}
        if len(lens) != 1 or len(poss) != 1 or not enc["clean"][1]:
            continue
        T = lens.pop(); pos_label = enc["clean"][1]
        x = {k: torch.tensor([v[0]], device="cuda") for k, v in enc.items()}
        g_clean = gb(R.model(x["clean"]).logits)
        o_f = R.model(x["flip"], output_hidden_states=True); g_flip = gb(o_f.logits)
        o_p = R.model(x["placebo"], output_hidden_states=True); g_plac = gb(o_p.logits)
        if abs(g_flip - g_clean) <= 1:
            continue
        eff = {"flip_label": [], "flip_final": [], "placebo_label": [], "placebo_final": []}
        for i in range(len(R.layers)):
            for src, hs in (("flip", o_f.hidden_states), ("placebo", o_p.hidden_states)):
                for where, pos in (("label", pos_label), ("final", [T - 1])):
                    R.patch = (i, {0: (pos, hs[i + 1][0, pos, :])})
                    eff[f"{src}_{where}"].append((gb(R.model(x["clean"]).logits) - g_clean) / (g_flip - g_clean))
                    R.patch = None
        res.append({"id": r["id"], "g_clean": g_clean, "g_flip": g_flip, "g_placebo": g_plac, "effects": eff})
    json.dump(res, open(out / "placebo.json", "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--model_dir", default=None); ap.add_argument("--parts", required=True)
    a = ap.parse_args()
    out = ROOT / "acts6" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    R = Runner(a.repo, a.model_dir)
    parts = a.parts.split(",")
    if "goalproj" in parts and not ((out / "goal_sst_gen.npy").exists() and (out / "sst_dir_gen.npz").exists()):
        goalproj_part(R, out); print("goalproj done", flush=True)
    if "placebo" in parts and not (out / "placebo.json").exists():
        placebo_part(R, out); print("placebo done", flush=True)
    (out / f"done_{'_'.join(parts)}").touch()


if __name__ == "__main__":
    main()
