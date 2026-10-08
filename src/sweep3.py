"""Third sweep (prereg v8), projection-only: goal-relevant neutral control, position trajectory, token x layer patching
map, base-vs-instruct plain continuation, and multi-LLM validation of the naturalistic sentences.

Usage: python sweep3.py --repo R --tag T --model_dir D --parts goalneutral,trajectory,patchmap,basecont,natval
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

from sweep2 import Runner, first_ids, project
from extract import valence_sources
from make_stimuli import DOMAINS

ROOT = Path(os.environ.get("GOALVAL_ROOT", "."))
SEED = 20261009
Q = "\nFor you, was this outcome good or bad? Answer with one word: good or bad."


@torch.no_grad()
def states_at(R, prompts, char_ends, bs=16):
    """Hidden states (N, n_pos, L+1, d) float16 at the tokens covering char positions char_ends[i][k]-1."""
    out = []
    for b in range(0, len(prompts), bs):
        P = prompts[b:b + bs]
        enc = R.tok(P, return_tensors="pt", padding=True, add_special_tokens=False, return_offsets_mapping=True)
        offs = enc.pop("offset_mapping").numpy()
        enc = {k: v.to("cuda") for k, v in enc.items()}
        hs = R.model(**enc, output_hidden_states=True).hidden_states
        H = torch.stack(hs, 2)  # B x T x (L+1) x d
        for j in range(len(P)):
            idxs = []
            for c in char_ends[b + j]:
                cand = [t for t in range(offs.shape[1]) if offs[j, t, 0] < c and offs[j, t, 1] >= c and offs[j, t, 1] > offs[j, t, 0]]
                idxs.append(cand[-1] if cand else offs.shape[1] - 1)
            out.append(H[j, idxs].float().cpu().to(torch.float16))
    return torch.stack(out).numpy()


def informat_project(G, rows, train_mask):
    y = np.array([r["surface"] for r in rows])
    P = np.zeros(G.shape[:2], np.float32)
    for L in range(G.shape[1]):
        x = G[train_mask, L].astype(np.float32)
        w = x[y[train_mask] == 1].mean(0) - x[y[train_mask] == 0].mean(0)
        P[:, L] = (G[:, L].astype(np.float32) - x.mean(0)) @ w
    return P


def goalneutral_part(R, out, bos):
    rows = [json.loads(l) for l in open(ROOT / "data/goalneutral_stimuli.jsonl")]
    sst = valence_sources()["sst2"]; ys = np.array([r["label"] for r in sst])
    tr = np.array([r["split"] == "discovery" and r["perspective"] == "none" for r in rows])
    for pos, fmt in (("raw", lambda t: bos + t), ("gen", R.chat)):
        G = R.last_states([fmt(r["text"]) for r in rows])
        S = R.last_states([fmt(r["text"]) for r in sst])
        np.save(out / f"gn_sst_{pos}.npy", project(G, S, ys))
        np.save(out / f"gn_inf_{pos}.npy", informat_project(G, rows, tr))
        del G, S
    beh = [r for r in rows if r["perspective"] != "none"]
    def q(r):
        who = "you" if r["stakeholder"] == "you" else r["stakeholder"]
        return f"{r['text']}\nFor {who}, was this outcome good or bad? Answer with one word: good or bad."
    np.save(out / "gn_behaviour.npy", R.score([R.chat(q(r)) for r in beh], first_ids(R.tok, "good"), first_ids(R.tok, "bad")))
    np.save(out / "gn_behaviour_ids.npy", np.array([r["id"] for r in beh]))


def trajectory_part(R, out):
    """Valence stimuli with the evaluative question; in-format directions at (end of stimulus, end of question, response)."""
    rows = [json.loads(l) for l in open(ROOT / "data/goal_stimuli.jsonl")]
    sel = [r for r in rows if r["perspective"] in ("self", "none")]
    prompts, ends = [], []
    for r in sel:
        p = R.chat(r["text"] + Q)
        i_text = p.index(r["text"]) + len(r["text"])
        i_q = p.index(Q) + len(Q)
        prompts.append(p); ends.append([i_text, i_q, len(p)])
    G = states_at(R, prompts, ends)  # N x 3 x L+1 x d
    tr = np.array([r["split"] == "discovery" and r["perspective"] == "none" for r in sel])
    for k, name in enumerate(("text_end", "question_end", "response")):
        np.save(out / f"traj_{name}.npy", informat_project(G[:, k], sel, tr))
    np.save(out / "traj_ids.npy", np.array([r["id"] for r in sel]))


@torch.no_grad()
def patchmap_part(R, out, n_pairs=80):
    rows = [json.loads(l) for l in open(ROOT / "data/goal_stimuli.jsonl")]
    sel = [r for r in rows if r["split"] == "confirmation" and r["perspective"] == "self"]
    key = {(r["domain"], r["A"], r["X"], r["order"], r["B"], r["surface"]): r for r in sel}
    pairs = [(r, key[(r["domain"], r["X"], r["A"], 1 - r["order"], r["B"], r["surface"])]) for r in sel
             if (r["domain"], r["X"], r["A"], 1 - r["order"], r["B"], r["surface"]) in key]
    pairs = [(a, b) for a, b in pairs if a["id"] < b["id"]]
    random.Random(SEED).shuffle(pairs)
    gid, bid = first_ids(R.tok, "good"), first_ids(R.tok, "bad")

    def spans(r, prompt):
        s_self = DOMAINS[r["domain"]][2].format(A=r["A"])
        e = r["event"]
        i_s = prompt.index(s_self); i_e = prompt.rindex(e)
        ep, en = DOMAINS[r["domain"]][4].split(), DOMAINS[r["domain"]][5].split()
        k = next(i for i, (a, b) in enumerate(zip(ep, en)) if a != b)  # first word that differs = outcome verb
        words = e.split(); verb = words[k]; i_v = i_e + len(" ".join(words[:k])) + (1 if k else 0)
        return {"stake_label": (i_s + s_self.index(r["A"]), i_s + s_self.index(r["A"]) + len(r["A"])),
                "stake_end": (i_s + len(s_self) - 1, i_s + len(s_self)),
                "event_label": (i_e + e.index(r["B"]), i_e + e.index(r["B"]) + len(r["B"])),
                "event_verb": (i_v, i_v + len(verb)),
                "event_end": (i_e + len(e) - 1, i_e + len(e)),
                "response": None}

    def tok_pos(offs, span, T):
        if span is None:
            return [T - 1]
        a, b = span
        return [t for t in range(T) if offs[t][0] < b and offs[t][1] > a and offs[t][1] > offs[t][0]]

    def g(logits):
        lp = torch.log_softmax(logits[0, -1].float(), -1)
        return float(torch.logsumexp(lp[gid], -1) - torch.logsumexp(lp[bid], -1))

    res = []
    for r, p in pairs:
        if len(res) >= n_pairs:
            break
        pr, pp = R.chat(r["text"] + Q), R.chat(p["text"] + Q)
        er = R.tok(pr, return_offsets_mapping=True, add_special_tokens=False)
        ep_ = R.tok(pp, return_offsets_mapping=True, add_special_tokens=False)
        if len(er["input_ids"]) != len(ep_["input_ids"]):
            continue
        T = len(er["input_ids"])
        sr, sp = spans(r, pr), spans(p, pp)
        pos_r = {k: tok_pos(er["offset_mapping"], v, T) for k, v in sr.items()}
        pos_p = {k: tok_pos(ep_["offset_mapping"], v, T) for k, v in sp.items()}
        if any(pos_r[k] != pos_p[k] or not pos_r[k] for k in pos_r):
            continue
        xr = torch.tensor([er["input_ids"]], device="cuda"); xp = torch.tensor([ep_["input_ids"]], device="cuda")
        op = R.model(xp, output_hidden_states=True); hs = op.hidden_states
        gc, gk = g(R.model(xr).logits), g(op.logits)
        eff = {k: [] for k in pos_r}
        for i in range(len(R.layers)):
            for k, pos in pos_r.items():
                R.patch = (i, {0: (pos, hs[i + 1][0, pos, :])})
                eff[k].append(g(R.model(xr).logits))
                R.patch = None
        res.append({"id_clean": r["id"], "g_clean": gc, "g_corrupt": gk, "effects": eff, "n_layers": len(R.layers)})
    json.dump(res, open(out / "patchmap.json", "w"))


def basecont_part(R, out, bos):
    """Plain-text continuation available to base and instruct models alike."""
    rows = [json.loads(l) for l in open(ROOT / "data/goal_stimuli.jsonl")]
    sel = [r for r in rows if r["perspective"] in ("self", "none")]
    suffix = "\nFor you, this outcome was"
    G = R.last_states([bos + r["text"] + suffix for r in sel])
    tr = np.array([r["split"] == "discovery" and r["perspective"] == "none" for r in sel])
    np.save(out / "base_proj.npy", informat_project(G, sel, tr))
    np.save(out / "base_ids.npy", np.array([r["id"] for r in sel]))
    beh = [r for r in sel if r["perspective"] == "self"]
    np.save(out / "base_beh.npy", R.score([bos + r["text"] + suffix for r in beh], first_ids(R.tok, " good"), first_ids(R.tok, " bad")))
    np.save(out / "base_beh_ids.npy", np.array([r["id"] for r in beh]))


@torch.no_grad()
def natval_part(R, out):
    """Which party won? Scored as the summed log-probability of each party name as the answer."""
    nat = [json.loads(l) for l in open(ROOT / "data/natural_raw.jsonl")]
    kept = {json.loads(l)["text"] for l in open(ROOT / "data/natural_stimuli.jsonl") if json.loads(l)["perspective"] == "none"}
    nat = [r for r in nat if r["sentence"] in kept]
    correct = 0
    for r in nat:
        a, b = r["parties"]
        prompt = R.chat(f"{r['sentence']}\nWhich party won, {a} or {b}? Answer with the name only.")
        scores = []
        for name in (a, b):
            ids_p = R.tok(prompt, add_special_tokens=False)["input_ids"]
            ids_a = R.tok(name, add_special_tokens=False)["input_ids"]
            x = torch.tensor([ids_p + ids_a], device="cuda")
            lp = torch.log_softmax(R.model(x).logits[0].float(), -1)
            scores.append(float(sum(lp[len(ids_p) - 1 + k, t] for k, t in enumerate(ids_a))))
        correct += int((a if scores[0] > scores[1] else b) == r["winner"])
    json.dump({"n": len(nat), "agree": correct / len(nat)}, open(out / "natval.json", "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--model_dir", default=None); ap.add_argument("--parts", required=True)
    a = ap.parse_args()
    out = ROOT / "acts3" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    R = Runner(a.repo, a.model_dir)
    bos = R.tok.bos_token if (R.tok.bos_token and getattr(R.tok, "add_bos_token", False)) else ""
    parts = a.parts.split(",")
    has_chat = R.tok.chat_template is not None
    if "goalneutral" in parts and has_chat and not (out / "gn_behaviour.npy").exists():
        goalneutral_part(R, out, bos); print("goalneutral done", flush=True)
    if "trajectory" in parts and has_chat and not (out / "traj_response.npy").exists():
        trajectory_part(R, out); print("trajectory done", flush=True)
    if "patchmap" in parts and has_chat and not (out / "patchmap.json").exists():
        patchmap_part(R, out); print("patchmap done", flush=True)
    if "basecont" in parts and not (out / "base_beh.npy").exists():
        basecont_part(R, out, bos); print("basecont done", flush=True)
    if "natval" in parts and has_chat and not (out / "natval.json").exists():
        natval_part(R, out); print("natval done", flush=True)
    (out / f"done_{'_'.join(parts)}").touch()


if __name__ == "__main__":
    main()
