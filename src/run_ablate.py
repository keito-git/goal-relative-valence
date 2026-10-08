"""Fetch weights again and run ablate.py for each model after its extraction is finished."""
import os
import shutil, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from orchestrate import fetch  # noqa: E402
ROOT = Path(os.environ.get("GOALVAL_ROOT", "."))
for pair in sys.argv[1:]:
    tag, repo = pair.split("=", 1)
    if (ROOT / "results" / f"ablate_{tag}.json").exists():
        continue
    while not (ROOT / "acts" / tag / "meta.json").exists():
        time.sleep(60)
    d = fetch(repo, "abl_" + tag)
    t0 = time.time()
    r = subprocess.run([sys.executable, "ablate.py", "--repo", repo, "--tag", tag, "--model_dir", str(d)],
                       cwd=os.path.dirname(os.path.abspath(__file__)), capture_output=True, text=True)
    (ROOT / "logs" / f"ablate_{tag}.log").write_text(r.stdout + r.stderr)
    with open(ROOT / "logs/orchestrate.log", "a") as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] ablate {tag} rc={r.returncode} {time.time()-t0:.0f}s\n")
    shutil.rmtree(d, ignore_errors=True)
