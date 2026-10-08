"""Fetch each model (disk-guarded, one ahead), run sweep2.py, delete weights."""
import os
import shutil, subprocess, sys, threading, time
from pathlib import Path
from queue import Queue
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from orchestrate import fetch, log  # noqa: E402
ROOT = Path(os.environ.get("GOALVAL_ROOT", "."))
jobs = [a.split("=", 1) for a in sys.argv[1:]]
q: Queue = Queue(maxsize=1)

def producer():
    for tag, repo in jobs:
        if (ROOT / "acts2" / tag / "done").exists():
            continue
        try:
            q.put((tag, repo, fetch(repo, "s2_" + tag)))
        except Exception as e:
            log(f"S2 FETCH FAIL {tag}: {e!r}")
    q.put(None)

threading.Thread(target=producer, daemon=True).start()
while (item := q.get()) is not None:
    tag, repo, d = item
    t0 = time.time()
    r = subprocess.run([sys.executable, "sweep2.py", "--repo", repo, "--tag", tag, "--model_dir", str(d)],
                       cwd=os.path.dirname(os.path.abspath(__file__)), capture_output=True, text=True)
    (ROOT / "logs" / f"sweep2_{tag}.log").write_text(r.stdout + r.stderr)
    log(f"sweep2 {tag} rc={r.returncode} {time.time() - t0:.0f}s")
    if r.returncode == 0:
        shutil.rmtree(d, ignore_errors=True)
log("sweep2 all done")
