"""Download (parallel ranged HTTP, one model ahead) -> extract -> analyse -> delete weights, model by model.

Usage: python orchestrate.py tag=repo [tag=repo ...]
"""
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from queue import Queue

from huggingface_hub import HfApi, snapshot_download  # noqa: E402

ROOT = Path(os.environ.get("GOALVAL_ROOT", "."))
MODELS = ROOT / "models"
CODE = os.path.dirname(os.path.abspath(__file__))
api = HfApi()


def log(m: str) -> None:
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)
    with open(ROOT / "logs/orchestrate.log", "a") as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] {m}\n")


DISK_CAP_GB = 182  # do not start a download if the working directory would exceed this size


def root_usage_gb() -> float:
    out = subprocess.run(["du", "-sk", str(ROOT)], capture_output=True, text=True).stdout.split()
    return int(out[0]) / 1e6 if out else 0.0


def fetch(repo: str, tag: str) -> Path:
    """Download the safetensors weights and tokenizer files of `repo` into MODELS/tag (one model at a time)."""
    d = MODELS / tag
    if (d / ".done").exists():
        return d
    files = api.list_repo_files(repo)
    st = [f for f in files if f.endswith(".safetensors")]
    if any(f.startswith("model") for f in st):  # skip duplicate consolidated weights (e.g. Mistral)
        st = [f for f in st if not f.startswith("consolidated")]
    info = api.model_info(repo, files_metadata=True)
    need = sum((x.size or 0) for x in info.siblings if x.rfilename in st) / 1e9
    shutil.rmtree(d, ignore_errors=True)  # discard partial downloads
    while root_usage_gb() + need > DISK_CAP_GB:
        log(f"disk guard: waiting to fetch {tag} ({need:.0f} GB), working directory at {root_usage_gb():.0f} GB")
        time.sleep(120)
    keep = [f for f in files if f.endswith((".json", ".txt", ".model", ".tiktoken", ".py"))] + st
    snapshot_download(repo, local_dir=d, allow_patterns=keep)
    (d / ".done").touch()
    return d


def main() -> None:
    jobs = [a.split("=", 1) for a in sys.argv[1:]]
    q: Queue = Queue(maxsize=1)

    def producer() -> None:
        for tag, repo in jobs:
            if (ROOT / "acts" / tag / "meta.json").exists():
                continue
            t0 = time.time()
            try:
                d = fetch(repo, tag)
                log(f"fetched {tag} in {time.time() - t0:.0f}s")
                q.put((tag, repo, d))
            except Exception as e:  # keep going with the next model
                log(f"FETCH FAIL {tag}: {e!r}")
        q.put(None)

    threading.Thread(target=producer, daemon=True).start()
    while (item := q.get()) is not None:
        tag, repo, d = item
        t0 = time.time()
        r = subprocess.run([sys.executable, "extract.py", "--repo", repo, "--tag", tag, "--model_dir", str(d)],
                           cwd=CODE, capture_output=True, text=True)
        (ROOT / "logs" / f"extract_{tag}.log").write_text(r.stdout + r.stderr)
        log(f"extract {tag} rc={r.returncode} {time.time() - t0:.0f}s")
        if r.returncode == 0:
            shutil.rmtree(d, ignore_errors=True)
            # analysis runs in the background on CPU; discovery split only until the kill switch is recorded
            subprocess.Popen(f"nice -n 10 {sys.executable} analyze.py --tag {tag} --split discovery "
                             f"> {ROOT}/logs/analyze_{tag}_discovery.log 2>&1", shell=True, cwd=CODE)
    log("all done")


if __name__ == "__main__":
    main()
