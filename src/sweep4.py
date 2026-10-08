"""Fourth sweep (prereg v9), projection-only: task gating at the response boundary (H18) and the goal-explicit key
control (H19).

Usage: python sweep4.py --repo R --tag T --model_dir D --parts taskgate,gn2
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse
import json
from pathlib import Path

import numpy as np

from sweep2 import Runner, first_ids, project
from sweep3 import informat_project
from extract import valence_sources

ROOT = Path(os.environ.get("GOALVAL_ROOT", "."))
SUFFIXES = {
    "eval_self": "\nFor you, was this outcome good or bad? Answer with one word: good or bad.",
    "eval_plain": "\nWas this outcome good or bad? Answer with one word: good or bad.",
    "factual": "\nWhich party is named in the last sentence? Answer with the name only.",
    "irrelevant": "\nHow many sentences does the text contain? Answer with a number.",
    "none": "",
}


def taskgate_part(R, out):
    rows = [json.loads(l) for l in open(ROOT / "data/goal_stimuli.jsonl")]
    sel = [r for r in rows if r["perspective"] in ("self", "none")]
    tr = np.array([r["split"] == "discovery" and r["perspective"] == "none" for r in sel])
    for name, suf in SUFFIXES.items():
        G = R.last_states([R.chat(r["text"] + suf) for r in sel])
        np.save(out / f"tg_{name}.npy", informat_project(G, sel, tr))
        del G
    np.save(out / "tg_ids.npy", np.array([r["id"] for r in sel]))


def gn2_part(R, out):
    rows = [json.loads(l) for l in open(ROOT / "data/goalneutral2_stimuli.jsonl")]
    sst = valence_sources()["sst2"]; ys = np.array([r["label"] for r in sst])
    tr = np.array([r["split"] == "discovery" and r["perspective"] == "none" for r in rows])
    G = R.last_states([R.chat(r["text"]) for r in rows])
    S = R.last_states([R.chat(r["text"]) for r in sst])
    np.save(out / "gn2_sst_gen.npy", project(G, S, ys))
    np.save(out / "gn2_inf_gen.npy", informat_project(G, rows, tr))
    del G, S
    beh = [r for r in rows if r["perspective"] != "none"]

    def q(r):
        who = "you" if r["stakeholder"] == "you" else r["stakeholder"]
        return f"{r['text']}\nFor {who}, was this outcome good or bad? Answer with one word: good or bad."
    np.save(out / "gn2_behaviour.npy", R.score([R.chat(q(r)) for r in beh], first_ids(R.tok, "good"), first_ids(R.tok, "bad")))
    np.save(out / "gn2_behaviour_ids.npy", np.array([r["id"] for r in beh]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--model_dir", default=None); ap.add_argument("--parts", required=True)
    a = ap.parse_args()
    out = ROOT / "acts4" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    R = Runner(a.repo, a.model_dir)
    parts = a.parts.split(",")
    if "taskgate" in parts and not (out / "tg_ids.npy").exists():
        taskgate_part(R, out); print("taskgate done", flush=True)
    if "gn2" in parts and not (out / "gn2_behaviour.npy").exists():
        gn2_part(R, out); print("gn2 done", flush=True)
    (out / f"done_{'_'.join(parts)}").touch()


if __name__ == "__main__":
    main()
