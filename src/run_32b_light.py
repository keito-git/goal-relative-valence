"""32B sweep in projection-only mode (disk-light): guard cap 190 GB (decimal); peak = weights only."""
import os
import shutil, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import orchestrate  # noqa: E402
orchestrate.DISK_CAP_GB = 190
ROOT = Path(os.environ.get("GOALVAL_ROOT", "."))
tag, repo = "qwen25_32b_it", "Qwen/Qwen2.5-32B-Instruct"
d = orchestrate.fetch(repo, "s2_" + tag)
t0 = time.time()
r = subprocess.run([sys.executable, "sweep2.py", "--repo", repo, "--tag", tag, "--model_dir", str(d), "--project_only", "1"],
                   cwd=os.path.dirname(os.path.abspath(__file__)), capture_output=True, text=True)
(ROOT / "logs" / f"sweep2_{tag}.log").write_text(r.stdout + r.stderr)
orchestrate.log(f"sweep2(light) {tag} rc={r.returncode} {time.time() - t0:.0f}s")
if r.returncode == 0:
    shutil.rmtree(d, ignore_errors=True)
