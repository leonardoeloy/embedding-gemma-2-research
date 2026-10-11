import json
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parent
discovery = json.loads(root.joinpath("model-discovery.json").read_text())
for key in ["gemma", "wemm", "ovis"]:
    model = discovery[key]
    subprocess.run(["hf", "download", model["repo"], "--revision", model["revision"], "--local-dir", str(root / "models" / key)], check=True)
