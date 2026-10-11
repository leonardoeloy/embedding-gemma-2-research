import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi

root = Path(__file__).resolve().parent
discovery = json.loads(root.joinpath("model-discovery.json").read_text())
audit = {}
for key in ["gemma", "wemm", "ovis"]:
    source = discovery[key]
    info = HfApi().model_info(source["repo"], revision=source["revision"], files_metadata=True)
    files = []
    for entry in info.siblings:
        if not entry.rfilename.endswith(".safetensors"):
            continue
        path = root / "models" / key / entry.rfilename
        h = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
                h.update(block)
        expected = entry.lfs.sha256
        record = dict(file=entry.rfilename, bytes=path.stat().st_size, expected_sha256=expected, actual_sha256=h.hexdigest())
        if record["actual_sha256"] != expected or record["bytes"] != entry.size:
            raise RuntimeError(f"Source mismatch: {key} {entry.rfilename}")
        files.append(record)
    audit[key] = dict(repo=source["repo"], revision=info.sha, verified=True, weights=files)
root.joinpath("results/weight-integrity.json").write_text(json.dumps(audit, indent=2) + "\n")
print({key: entry["verified"] for key, entry in audit.items()})
