"""Second control (wooden/metal) for all instruct models: fetch (guarded), sweep2 --parts control2 --project_only 1, delete."""
import os
import shutil, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import orchestrate  # noqa: E402
orchestrate.DISK_CAP_GB = 190
ROOT = Path(os.environ.get("GOALVAL_ROOT", "."))
for pair in sys.argv[1:]:
    tag, repo = pair.split("=", 1)
    if (ROOT / "acts2" / tag / "ctrl2_behaviour.npy").exists():
        continue
    d = orchestrate.fetch(repo, "c2_" + tag)
    t0 = time.time()
    r = subprocess.run([sys.executable, "sweep2.py", "--repo", repo, "--tag", tag, "--model_dir", str(d), "--parts", "control2",
                        "--project_only", "1"], cwd=os.path.dirname(os.path.abspath(__file__)), capture_output=True, text=True)
    (ROOT / "logs" / f"control2_{tag}.log").write_text(r.stdout + r.stderr)
    orchestrate.log(f"control2 {tag} rc={r.returncode} {time.time() - t0:.0f}s")
    if r.returncode == 0:
        shutil.rmtree(d, ignore_errors=True)
orchestrate.log("control2 all done")
