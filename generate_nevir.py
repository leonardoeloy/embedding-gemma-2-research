import hashlib
import json
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parent
repo, revision = "orionweller/NevIR", "6263585072ce3b435ed09658613553fbf4e74184"
source = root / "sources/nevir"
if not source.joinpath("test.jsonl").exists():
    subprocess.run(["hf", "download", repo, "test.jsonl", "README.md", "--type", "dataset", "--revision", revision, "--local-dir", str(source)], check=True)
records = [json.loads(x) for x in source.joinpath("test.jsonl").read_text().splitlines()]
documents, queries = [], []
for pair, record in enumerate(records):
    ids = [f"nevir-{pair:05}-d{side}" for side in [1, 2]]
    for side in [1, 2]:
        documents.append(dict(id=ids[side - 1], pair=pair, text=record[f"doc{side}"]))
        queries.append(dict(id=f"nevir-{pair:05}-q{side}", pair=pair, side=side, text=record[f"q{side}"], candidates=ids, gold=[ids[side - 1]]))
out = root / "data/nevir"
out.mkdir(parents=True, exist_ok=True)
for name, rows in [("documents", documents), ("queries", queries)]:
    out.joinpath(name + ".jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
manifest = dict(source=repo, revision=revision, split="test", source_sha256=hashlib.sha256(source.joinpath("test.jsonl").read_bytes()).hexdigest(), license="MIT (source dataset card)", pairs=len(records), documents=len(documents), queries=len(queries), sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob("*.jsonl")})
out.joinpath("manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2))
