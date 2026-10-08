"""Causal test: does the stake-free valence direction mediate surface-valence interference in appraisal answers?

Directional ablation (project out a unit direction from every decoder-layer output, all positions), with the
direction for layer i estimated from SST-2 hidden_states[i+1] at the chat-generation position.
Conditions: none | valence (sst2 diff-of-means) | goemo (GoEmotions diff-of-means) | inplane_k (random unit
directions in the top-50 PC span of SST-2 activations, per layer, 20 draws).
Outputs per condition: good-minus-bad logit for every stake-bearing item (self/other, both splits).
"""
import os

# limit BLAS threads
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_num_threads(4)

ROOT = Path(os.environ.get("GOALVAL_ROOT", "."))
SEED = 20261006
N_RAND = 20


def unit(v):
    return v / np.linalg.norm(v)


def dirs_from(A: Path, src: str, n_layers: int):
    X = np.load(A / f"{src}_gen.npy", mmap_mode="r")
    y = np.load(A / f"{src}_labels.npy")
    return [unit(X[y == 1, i + 1].astype(np.float64).mean(0) - X[y == 0, i + 1].astype(np.float64).mean(0))
            for i in range(n_layers)]


def inplane(A: Path, n_layers: int, rng):
    X = np.load(A / "sst2_gen.npy", mmap_mode="r")
    out = []
    for i in range(n_layers):
        Z = torch.tensor(np.asarray(X[:, i + 1]), dtype=torch.float32, device="cuda")
        Z = Z - Z.mean(0)
        _, _, Vt = torch.linalg.svd(Z, full_matrices=False)
        out.append(Vt[:50].double().cpu().numpy())
    draws = []
    for _ in range(N_RAND):
        draws.append([unit(rng.standard_normal(50) @ out[i]) for i in range(n_layers)])
    return draws


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model_dir", default=None)
    args = ap.parse_args()
    A = ROOT / "acts" / args.tag
    meta = json.load(open(A / "meta.json"))
    tok = AutoTokenizer.from_pretrained(args.repo)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_dir or args.repo, torch_dtype=torch.bfloat16,
                                                 device_map="cuda").eval()
    layers = model.model.layers
    nL = len(layers)
    rng = np.random.default_rng(SEED)
    conds = {"none": None, "valence": dirs_from(A, "sst2", nL), "goemo": dirs_from(A, "goemo", nL)}
    for k, d in enumerate(inplane(A, nL, rng)):
        conds[f"inplane_{k}"] = d

    current = {"D": None, "band": set()}
    bands = {"all": set(range(nL)), "mid": set(range(nL // 4, (3 * nL) // 4))}

    def hook(i):
        def f(_m, _inp, out):
            if current["D"] is None or i not in current["band"]:
                return out
            h = out[0] if isinstance(out, tuple) else out
            d = current["D"][i]
            h = h - (h @ d).unsqueeze(-1) * d
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        return f

    for i, l in enumerate(layers):
        l.register_forward_hook(hook(i))

    rows = [json.loads(l) for l in open(ROOT / "data/goal_stimuli.jsonl")]
    beh = [r for r in rows if r["perspective"] != "none"]

    def q(r):
        who = "you" if r["stakeholder"] == "you" else r["stakeholder"]
        return f"{r['text']}\nFor {who}, was this outcome good or bad? Answer with one word: good or bad."

    prompts = [tok.apply_chat_template([{"role": "user", "content": q(r)}], tokenize=False, add_generation_prompt=True)
               for r in beh]
    good, bad = meta["good_ids"], meta["bad_ids"]
    out = {"ids": [r["id"] for r in beh]}
    plan = [("none", None, "all")] + [(n, D, b) for b in ("all", "mid") for n, D in conds.items() if D is not None]
    for name, D, band in plan:
        name = name if D is None else f"{band}/{name}"
        current["band"] = bands[band]
        current["D"] = None if D is None else [torch.tensor(v, dtype=torch.bfloat16, device="cuda") for v in D]
        scores = []
        with torch.no_grad():
            for b in range(0, len(prompts), 64):
                enc = tok(prompts[b:b + 64], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
                lp = torch.log_softmax(model(**enc).logits[:, -1, :].float(), -1)
                scores.append((torch.logsumexp(lp[:, good], -1) - torch.logsumexp(lp[:, bad], -1)).cpu())
        out[name] = torch.cat(scores).tolist()
        print(name, flush=True)
    json.dump(out, open(ROOT / "results" / f"ablate_{args.tag}.json", "w"))


if __name__ == "__main__":
    main()
