"""Extract hidden states (all layers) and good/bad behavioural logits for one model.

Positions:
  raw : last token of the plain text (primary, defined for every model)
  gen : last token of the chat-templated prompt with the generation header (instruct models only)
Usage: python extract.py --repo Qwen/Qwen2.5-7B-Instruct --tag qwen25_7b_it
"""
import os

# limit BLAS threads
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_num_threads(4)

ROOT = Path(os.environ.get("GOALVAL_ROOT", "."))
SEED = 20261006


def valence_sources() -> dict:
    rng = random.Random(SEED)
    sst = load_dataset("stanfordnlp/sst2", split="train")
    pos = [r["sentence"].strip() for r in sst if r["label"] == 1 and len(r["sentence"].split()) >= 6]
    neg = [r["sentence"].strip() for r in sst if r["label"] == 0 and len(r["sentence"].split()) >= 6]
    rng.shuffle(pos), rng.shuffle(neg)
    sst_rows = [{"text": t, "label": 1} for t in pos[:1000]] + [{"text": t, "label": 0} for t in neg[:1000]]
    ge = [json.loads(l) for l in open(ROOT / "data" / "goemotions_stimuli.jsonl")]
    parts = json.load(open(ROOT / "data" / "goemotions_valence_partition.json"))
    posi, negi = set(parts["valence"][0]), set(parts["valence"][1])
    gp = [r["text"] for r in ge if r["eid"] in posi]
    gn = [r["text"] for r in ge if r["eid"] in negi]
    rng.shuffle(gp), rng.shuffle(gn)
    ge_rows = [{"text": t, "label": 1} for t in gp[:1000]] + [{"text": t, "label": 0} for t in gn[:1000]]
    return {"sst2": sst_rows, "goemo": ge_rows}


def question(row) -> str:
    who = "you" if row["stakeholder"] == "you" else row["stakeholder"]
    return f"{row['text']}\nFor {who}, was this outcome good or bad? Answer with one word: good or bad."


@torch.no_grad()
def run(model, tok, prompts, bs=32):
    """Return last-token hidden states (n, L+1, d) fp16 and final-position logits for given token ids."""
    hs_all = []
    for i in range(0, len(prompts), bs):
        enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        out = model(**enc, output_hidden_states=True)
        # left padding -> last position is the final real token
        hs = torch.stack([h[:, -1, :] for h in out.hidden_states], 1)
        hs_all.append(hs.float().cpu().to(torch.float16))
    return torch.cat(hs_all, 0).numpy()


@torch.no_grad()
def behaviour(model, tok, prompts, good_ids, bad_ids, bs=32):
    out_scores = []
    for i in range(0, len(prompts), bs):
        enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        logits = model(**enc).logits[:, -1, :].float()
        lp = torch.log_softmax(logits, -1)
        g = torch.logsumexp(lp[:, good_ids], -1)
        b = torch.logsumexp(lp[:, bad_ids], -1)
        out_scores.append((g - b).cpu())
    return torch.cat(out_scores).numpy()


def first_ids(tok, words):
    ids = set()
    for w in words:
        t = tok.encode(w, add_special_tokens=False)
        if t:
            ids.add(t[0])
    return sorted(ids)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model_dir", default=None)
    args = ap.parse_args()
    out = ROOT / "acts" / args.tag
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.repo)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_dir or args.repo, torch_dtype=torch.bfloat16,
                                                 device_map="cuda").eval()
    has_chat = tok.chat_template is not None
    bos = tok.bos_token if (tok.bos_token and getattr(tok, "add_bos_token", False)) else ""

    def chat(text):
        return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                       add_generation_prompt=True)

    goal = [json.loads(l) for l in open(ROOT / "data/goal_stimuli.jsonl")]
    srcs = valence_sources()
    sets = {"goal": [r["text"] for r in goal], **{k: [r["text"] for r in v] for k, v in srcs.items()}}
    meta = {"repo": args.repo, "has_chat": has_chat, "n_layers": model.config.num_hidden_layers}
    for name, texts in sets.items():
        np.save(out / f"{name}_raw.npy", run(model, tok, [bos + t for t in texts]))
        if has_chat:
            np.save(out / f"{name}_gen.npy", run(model, tok, [chat(t) for t in texts]))
    for k, v in srcs.items():
        np.save(out / f"{k}_labels.npy", np.array([r["label"] for r in v]))

    # behavioural positive control on stake-bearing items
    beh_rows = [r for r in goal if r["perspective"] != "none"]
    if has_chat:
        prompts = [chat(question(r)) for r in beh_rows]
        good = first_ids(tok, ["good", "Good", " good", " Good"])
        bad = first_ids(tok, ["bad", "Bad", " bad", " Bad"])
    else:
        prompts = [bos + question(r).replace("\nFor", "\nQuestion: For").split(" Answer with")[0] + "\nAnswer:" for r in beh_rows]
        good, bad = first_ids(tok, [" good", " Good"]), first_ids(tok, [" bad", " Bad"])
    np.save(out / "behaviour_gb.npy", behaviour(model, tok, prompts, good, bad))
    np.save(out / "behaviour_ids.npy", np.array([r["id"] for r in beh_rows]))
    meta.update(good_ids=good, bad_ids=bad, example_prompt=prompts[0])
    json.dump(meta, open(out / "meta.json", "w"), indent=1)
    print("done", args.tag, meta["n_layers"], has_chat)


if __name__ == "__main__":
    main()
