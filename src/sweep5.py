"""Fifth sweep (prereg v10, H22), projection-only: stake-dependent but non-evaluative questions at the response boundary.

Usage: python sweep5.py --repo R --tag T --model_dir D --parts taskgate2
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse
import json
from pathlib import Path

import numpy as np

from sweep2 import Runner, first_ids
from sweep3 import informat_project

ROOT = Path(os.environ.get("GOALVAL_ROOT", "."))
SUFFIXES = {
    "eval_self": "\nFor you, was this outcome good or bad? Answer with one word: good or bad.",
    "outcome_factual": "\nDid your side win or lose? Answer with one word: won or lost.",
    "binding_factual": "\nIs the party named in the last sentence on your side? Answer with one word: yes or no.",
}


def taskgate2_part(R, out):
    rows = [json.loads(l) for l in open(ROOT / "data/goal_stimuli.jsonl")]
    sel = [r for r in rows if r["perspective"] in ("self", "none")]
    tr = np.array([r["split"] == "discovery" and r["perspective"] == "none" for r in sel])
    for name, suf in SUFFIXES.items():
        G = R.last_states([R.chat(r["text"] + suf) for r in sel])
        np.save(out / f"tg2_{name}.npy", informat_project(G, sel, tr))
        del G
    np.save(out / "tg2_ids.npy", np.array([r["id"] for r in sel]))
    beh = [r for r in sel if r["perspective"] == "self"]
    np.save(out / "tg2_beh_ids.npy", np.array([r["id"] for r in beh]))
    np.save(out / "tg2_beh_outcome.npy", R.score([R.chat(r["text"] + SUFFIXES["outcome_factual"]) for r in beh],
                                                 first_ids(R.tok, "won"), first_ids(R.tok, "lost")))
    np.save(out / "tg2_beh_binding.npy", R.score([R.chat(r["text"] + SUFFIXES["binding_factual"]) for r in beh],
                                                 first_ids(R.tok, "yes"), first_ids(R.tok, "no")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--model_dir", default=None); ap.add_argument("--parts", required=True)
    a = ap.parse_args()
    out = ROOT / "acts5" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    R = Runner(a.repo, a.model_dir)
    if "taskgate2" in a.parts and not (out / "tg2_beh_binding.npy").exists():
        taskgate2_part(R, out); print("taskgate2 done", flush=True)
    (out / f"done_{a.parts.replace(',', '_')}").touch()


if __name__ == "__main__":
    main()
