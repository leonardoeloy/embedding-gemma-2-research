import json
import os
import subprocess
import time
from pathlib import Path

root = Path(__file__).resolve().parent
os.chdir(root)
source = json.loads(root.joinpath("model-discovery.json").read_text())
pending = ["wemm", "ovis"]
last_report = 0
while pending:
    progressed = False
    for model in list(pending):
        files = [f for f in source[model]["files"] if f["name"].endswith(".safetensors")]
        ready = all((root / "models" / model / f["name"]).exists() and (root / "models" / model / f["name"]).stat().st_size == f["size"] for f in files)
        if not ready:
            continue
        for dataset, output in [("data", "results"), ("data/nevir", "results/nevir")]:
            receipt = root / output / model / "receipt.json"
            if receipt.exists() and json.loads(receipt.read_text())["status"] == "complete":
                continue
            print(time.strftime("%H:%M:%S"), "encoding", model, dataset, flush=True)
            subprocess.run([str(root / ".venv/bin/python"), "-u", "encode.py", model, "--dataset-dir", dataset, "--results-dir", output], check=True, env={**os.environ, "ZE_AFFINITY_MASK": "0"})
        pending.remove(model)
        progressed = True
    if pending and not progressed:
        if time.time() - last_report > 50:
            print(time.strftime("%H:%M:%S"), "waiting for model files:", ", ".join(pending), flush=True)
            last_report = time.time()
        time.sleep(5)
for script in ["audit_models.py", "analyze.py", "analyze_nevir.py"]:
    print("running", script, flush=True)
    subprocess.run([str(root / ".venv/bin/python"), script], check=True)
print("All model runs, source verification and analyses complete.", flush=True)
