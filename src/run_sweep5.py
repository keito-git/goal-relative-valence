"""Run sweep5 parts per model: fetch (guarded, cap 190) -> sweep5.py -> delete weights."""
import os
import shutil, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import orchestrate  # noqa: E402
orchestrate.DISK_CAP_GB = 190
ROOT = Path(os.environ.get("GOALVAL_ROOT", "."))
for spec in sys.argv[1:]:
    tag, repo, parts = spec.split("=", 2)
    if (ROOT / "acts5" / tag / f"done_{parts.replace(',', '_')}").exists():
        continue
    d = orchestrate.fetch(repo, "s5_" + tag)
    t0 = time.time()
    r = subprocess.run([sys.executable, "sweep5.py", "--repo", repo, "--tag", tag, "--model_dir", str(d), "--parts", parts],
                       cwd=os.path.dirname(os.path.abspath(__file__)), capture_output=True, text=True)
    (ROOT / "logs" / f"sweep5_{tag}.log").write_text(r.stdout + r.stderr)
    orchestrate.log(f"sweep5 {tag} rc={r.returncode} {time.time() - t0:.0f}s")
    if r.returncode == 0:
        shutil.rmtree(d, ignore_errors=True)
orchestrate.log("sweep5 all done")
