"""Second sweep (one model load): non-affective control, prompt robustness, activation patching.

Usage: python sweep2.py --repo Qwen/Qwen2.5-7B-Instruct --tag qwen25_7b_it --model_dir <local model directory>
Outputs under $GOALVAL_ROOT/acts2/<tag>/ and results/sweep2_<tag>_*.json
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
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_num_threads(4)
ROOT = Path(os.environ.get("GOALVAL_ROOT", "."))
SEED = 20261007
PROJECT_ONLY = False

TEMPLATES = [  # (question with {who}, positive word, negative word); each also run with the option order reversed
    ("For {who}, was this outcome good or bad? Answer with one word: {a} or {b}.", "good", "bad"),
    ("Did this outcome benefit {who} or harm {who}? Answer with one word: {a} or {b}.", "benefit", "harm"),
    ("Was this outcome favorable or unfavorable for {who}? Answer with one word: {a} or {b}.", "favorable", "unfavorable"),
    ("From your perspective, was the result positive or negative? Answer with one word: {a} or {b}.", "positive", "negative"),
    ("Should {who} feel happy or sad about this outcome? Answer with one word: {a} or {b}.", "happy", "sad"),
]


def first_ids(tok, word):
    ids = set()
    for w in (word, word.capitalize(), " " + word, " " + word.capitalize()):
        t = tok.encode(w, add_special_tokens=False)
        if t:
            ids.add(t[0])
    return sorted(ids)


class Runner:
    def __init__(self, repo, model_dir):
        self.tok = AutoTokenizer.from_pretrained(repo)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_dir or repo, torch_dtype=torch.bfloat16,
                                                          device_map="cuda").eval()
        self.layers = self.model.model.layers
        self.patch = None  # (layer_index, {batch_row: (positions, tensor[npos, d])})
        for i, l in enumerate(self.layers):
            l.register_forward_hook(self._hook(i))

    def _hook(self, i):
        def f(_m, _inp, out):
            if self.patch is None or self.patch[0] != i:
                return out
            h = out[0] if isinstance(out, tuple) else out
            h = h.clone()
            for row, (pos, val) in self.patch[1].items():
                h[row, pos, :] = val.to(h.dtype)
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        return f

    def chat(self, text):
        return self.tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)

    @torch.no_grad()
    def last_states(self, prompts, bs=32):
        out = []
        for i in range(0, len(prompts), bs):
            enc = self.tok(prompts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            o = self.model(**enc, output_hidden_states=True)
            out.append(torch.stack([h[:, -1, :] for h in o.hidden_states], 1).float().cpu().to(torch.float16))
        return torch.cat(out).numpy()

    @torch.no_grad()
    def score(self, prompts, pos_ids, neg_ids, bs=32):
        out = []
        for i in range(0, len(prompts), bs):
            enc = self.tok(prompts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            lp = torch.log_softmax(self.model(**enc).logits[:, -1, :].float(), -1)
            out.append((torch.logsumexp(lp[:, pos_ids], -1) - torch.logsumexp(lp[:, neg_ids], -1)).cpu())
        return torch.cat(out).numpy()



def project(goal_states, src_states, labels):
    """Per-layer difference-of-means direction from the source; projections of goal items (N x L), float32."""
    out = np.zeros(goal_states.shape[:2], np.float32)
    for L in range(goal_states.shape[1]):
        x = src_states[:, L].astype(np.float32)
        w = x[labels == 1].mean(0) - x[labels == 0].mean(0)
        out[:, L] = (goal_states[:, L].astype(np.float32) - x.mean(0)) @ w
    return out


def control_part(R, out_dir, bos):
    rows = [json.loads(l) for l in open(ROOT / "data/control_stimuli.jsonl")]
    src = [json.loads(l) for l in open(ROOT / "data/control_source.jsonl")]
    ys = np.array([r["label"] for r in src])
    np.save(out_dir / "ctrl_src_labels.npy", ys)
    for pos, fmt in (("raw", lambda t: bos + t), ("gen", R.chat)):
        G = R.last_states([fmt(r["text"]) for r in rows])
        Ssrc = R.last_states([fmt(r["text"]) for r in src])
        if PROJECT_ONLY:  # disk-light mode: keep projections only
            np.save(out_dir / f"ctrl_proj_{pos}.npy", project(G, Ssrc, ys))
        else:
            np.save(out_dir / f"ctrl_goal_{pos}.npy", G)
            np.save(out_dir / f"ctrl_src_{pos}.npy", Ssrc)
    beh = [r for r in rows if r["perspective"] != "none"]

    def q(r):
        whose = "your" if r["stakeholder"] == "you" else f"{r['stakeholder']}'s"
        return f"{r['text']}\nIs {whose} {r['obj']} on the left or on the right? Answer with one word: left or right."
    gb = R.score([R.chat(q(r)) for r in beh], first_ids(R.tok, "left"), first_ids(R.tok, "right"))
    np.save(out_dir / "ctrl_behaviour_lr.npy", gb)
    np.save(out_dir / "ctrl_behaviour_ids.npy", np.array([r["id"] for r in beh]))


def prompt_part(R, out_dir):
    rows = [json.loads(l) for l in open(ROOT / "data/goal_stimuli.jsonl")]
    sel = [r for r in rows if r["split"] == "confirmation" and r["perspective"] == "self"]
    res = {"ids": [r["id"] for r in sel]}
    for k, (tmpl, pw, nw) in enumerate(TEMPLATES):
        pid, nid = first_ids(R.tok, pw), first_ids(R.tok, nw)
        if set(pid) & set(nid):  # shared first token makes the contrast ill-defined
            res[f"t{k}_skipped"] = True
            continue
        for rev in (0, 1):
            a, b = (pw, nw) if rev == 0 else (nw, pw)
            prompts = [R.chat(f"{r['text']}\n" + tmpl.format(who="you", a=a, b=b)) for r in sel]
            res[f"t{k}_rev{rev}"] = R.score(prompts, pid, nid).tolist()
    json.dump(res, open(out_dir / "prompt_robustness.json", "w"))


def label_positions(R, prompt, sentence, label):
    """Token positions (in the left-padded batch row, computed later) covering `label` inside `sentence`."""
    enc = R.tok(prompt, return_offsets_mapping=True, add_special_tokens=False)
    s0 = prompt.index(sentence)
    c0 = s0 + sentence.index(label)
    c1 = c0 + len(label)
    pos = [i for i, (a, b) in enumerate(enc["offset_mapping"]) if a < c1 and b > c0]
    return enc["input_ids"], pos


@torch.no_grad()
def patch_part(R, out_dir, n_pairs=160):
    rows = [json.loads(l) for l in open(ROOT / "data/goal_stimuli.jsonl")]
    sel = [r for r in rows if r["split"] == "confirmation" and r["perspective"] == "self"]
    from make_stimuli import DOMAINS
    key = {(r["domain"], r["A"], r["X"], r["order"], r["B"], r["surface"]): r for r in sel}
    pairs = []
    for r in sel:
        # partner: stake bound to the other label, same context sentence (order flipped), same event
        p = key.get((r["domain"], r["X"], r["A"], 1 - r["order"], r["B"], r["surface"]))
        if p is not None and r["id"] < p["id"]:
            pairs.append((r, p))
    rng = random.Random(SEED)
    rng.shuffle(pairs)
    q = "\nFor you, was this outcome good or bad? Answer with one word: good or bad."
    gid, bid = first_ids(R.tok, "good"), first_ids(R.tok, "bad")
    nL = len(R.layers)
    out = []
    for r, p in pairs:
        if len(out) >= n_pairs:
            break
        s_self = DOMAINS[r["domain"]][2]
        pr_r, pr_p = R.chat(r["text"] + q), R.chat(p["text"] + q)
        ids_r, pos_r = label_positions(R, pr_r, s_self.format(A=r["A"]), r["A"])
        ids_p, pos_p = label_positions(R, pr_p, s_self.format(A=p["A"]), p["A"])
        if len(ids_r) != len(ids_p) or pos_r != pos_p or not pos_r:
            continue
        T = len(ids_r)
        x_r = torch.tensor([ids_r], device="cuda")
        x_p = torch.tensor([ids_p], device="cuda")
        o_p = R.model(x_p, output_hidden_states=True)
        hs_p = o_p.hidden_states  # hs_p[i+1] = output of layer i

        def gb(logits):
            lp = torch.log_softmax(logits[0, -1].float(), -1)
            return float(torch.logsumexp(lp[gid], -1) - torch.logsumexp(lp[bid], -1))
        g_clean, g_corr = gb(R.model(x_r).logits), gb(o_p.logits)
        eff_label, eff_final = [], []
        for i in range(nL):
            for pos, store in ((pos_r, eff_label), ([T - 1], eff_final)):
                R.patch = (i, {0: (pos, hs_p[i + 1][0, pos, :])})
                store.append(gb(R.model(x_r).logits))
                R.patch = None
        out.append({"id_clean": r["id"], "id_corrupt": p["id"], "domain": r["domain"], "g_clean": g_clean,
                    "g_corrupt": g_corr, "patch_label": eff_label, "patch_final": eff_final, "n_layers": nL})
    json.dump(out, open(out_dir / "patching.json", "w"))



def natural_part(R, out_dir, bos):
    rows = [json.loads(l) for l in open(ROOT / "data/natural_stimuli.jsonl")]
    for pos, fmt in (("raw", lambda t: bos + t), ("gen", R.chat)):
        G = R.last_states([fmt(r["text"]) for r in rows])
        if PROJECT_ONLY:
            A1 = ROOT / "acts" / out_dir.name
            Ssst = np.load(A1 / f"sst2_{pos}.npy"); ysst = np.load(A1 / "sst2_labels.npy")
            np.save(out_dir / f"nat_proj_{pos}.npy", project(G, Ssst, ysst))
        else:
            np.save(out_dir / f"nat_{pos}.npy", G)
    beh = [r for r in rows if r["perspective"] == "self"]
    q = "\nFor you, was this outcome good or bad? Answer with one word: good or bad."
    g = R.score([R.chat(r["text"] + q) for r in beh], first_ids(R.tok, "good"), first_ids(R.tok, "bad"))
    np.save(out_dir / "nat_behaviour_gb.npy", g)
    np.save(out_dir / "nat_behaviour_ids.npy", np.array([r["id"] for r in beh]))



def control2_part(R, out_dir, bos):
    """Second control (wooden vs metal); always projection-only to save disk."""
    rows = [json.loads(l) for l in open(ROOT / "data/control2_stimuli.jsonl")]
    src = [json.loads(l) for l in open(ROOT / "data/control2_source.jsonl")]
    ys = np.array([r["label"] for r in src])
    np.save(out_dir / "ctrl2_src_labels.npy", ys)
    for pos, fmt in (("raw", lambda t: bos + t), ("gen", R.chat)):
        G = R.last_states([fmt(r["text"]) for r in rows])
        Ssrc = R.last_states([fmt(r["text"]) for r in src])
        np.save(out_dir / f"ctrl2_proj_{pos}.npy", project(G, Ssrc, ys))
    beh = [r for r in rows if r["perspective"] != "none"]

    def q(r):
        whose = "your" if r["stakeholder"] == "you" else f"{r['stakeholder']}'s"
        return f"{r['text']}" + "\n" + f"Is {whose} {r['obj']} made of wood or metal? Answer with one word: wood or metal."
    g = R.score([R.chat(q(r)) for r in beh], first_ids(R.tok, "wood"), first_ids(R.tok, "metal"))
    np.save(out_dir / "ctrl2_behaviour.npy", g)
    np.save(out_dir / "ctrl2_behaviour_ids.npy", np.array([r["id"] for r in beh]))



def informat_part(R, out_dir, bos):
    """In-format directions (prereg v7): for each task, a per-layer difference-of-means direction for the surface label is
    estimated from stake-free ('none') items of the DISCOVERY domains, and all items are projected onto it.
    Saves projections only (N x L) per task and position."""
    tasks = {"valence": "goal_stimuli.jsonl", "location": "control_stimuli.jsonl", "material": "control2_stimuli.jsonl"}
    for task, fn in tasks.items():
        rows = [json.loads(l) for l in open(ROOT / "data" / fn)]
        tr = np.array([r["split"] == "discovery" and r["perspective"] == "none" for r in rows])
        y = np.array([r["surface"] for r in rows])
        for pos, fmt in (("raw", lambda t: bos + t), ("gen", R.chat)):
            G = R.last_states([fmt(r["text"]) for r in rows])
            P = np.zeros(G.shape[:2], np.float32)
            for L in range(G.shape[1]):
                x = G[tr, L].astype(np.float32)
                w = x[y[tr] == 1].mean(0) - x[y[tr] == 0].mean(0)
                P[:, L] = (G[:, L].astype(np.float32) - x.mean(0)) @ w
            np.save(out_dir / f"inf_{task}_{pos}.npy", P)
            del G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model_dir", default=None)
    ap.add_argument("--parts", default="control,prompt,patch,natural")
    ap.add_argument("--project_only", type=int, default=0)
    a = ap.parse_args()
    global PROJECT_ONLY
    PROJECT_ONLY = bool(a.project_only)
    out_dir = ROOT / "acts2" / a.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    R = Runner(a.repo, a.model_dir)
    bos = R.tok.bos_token if (R.tok.bos_token and getattr(R.tok, "add_bos_token", False)) else ""
    parts = a.parts.split(",")
    if "control" in parts and not (out_dir / "ctrl_behaviour_lr.npy").exists():
        control_part(R, out_dir, bos)
        print("control done", flush=True)
    if "prompt" in parts and not (out_dir / "prompt_robustness.json").exists():
        prompt_part(R, out_dir)
        print("prompt done", flush=True)
    if "patch" in parts and not (out_dir / "patching.json").exists():
        patch_part(R, out_dir)
        print("patch done", flush=True)
    if "natural" in parts and not (out_dir / "nat_behaviour_gb.npy").exists():
        natural_part(R, out_dir, bos)
        print("natural done", flush=True)
    if "control2" in parts and not (out_dir / "ctrl2_behaviour.npy").exists():
        control2_part(R, out_dir, bos)
        print("control2 done", flush=True)
    if "informat" in parts and not (out_dir / "inf_material_gen.npy").exists():
        informat_part(R, out_dir, bos)
        print("informat done", flush=True)
    (out_dir / "done").touch()


if __name__ == "__main__":
    main()
