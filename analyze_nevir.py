import hashlib
import json
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

root = Path(__file__).resolve().parent
data, out = root / "data/nevir", root / "results/nevir"
docs = [json.loads(x) for x in data.joinpath("documents.jsonl").read_text().splitlines()]
queries = [json.loads(x) for x in data.joinpath("queries.jsonl").read_text().splitlines()]
index = {d["id"]: i for i, d in enumerate(docs)}
candidate_indices = np.array([[index[x] for x in q["candidates"]] for q in queries])
tokenize = lambda text: re.findall(r"[a-z0-9]+", text.lower())
bm25 = BM25Okapi([tokenize(d["text"]) for d in docs])
scores = {"bm25": np.stack([bm25.get_batch_scores(tokenize(q["text"]), candidate_indices[i]) for i, q in enumerate(queries)]), "random": np.random.default_rng(20261011).random((len(queries), 2))}
for name in ["gemma", "wemm", "ovis"]:
    path = out / name
    receipt = json.loads(path.joinpath("receipt.json").read_text())
    if receipt["status"] != "complete" or receipt["dataset"] != json.loads(data.joinpath("manifest.json").read_text()):
        raise RuntimeError("External dataset mismatch")
    for side, records in [("queries", queries), ("documents", docs)]:
        if json.loads(path.joinpath(side + "-ids.json").read_text()) != [r["id"] for r in records]:
            raise RuntimeError("Row ordering mismatch")
        if hashlib.sha256(path.joinpath(side + ".npy").read_bytes()).hexdigest() != receipt[side]["sha256"]:
            raise RuntimeError("Vector checksum mismatch")
    q, d = np.load(path / "queries.npy"), np.load(path / "documents.npy")
    scores[name] = np.einsum("ij,ikj->ik", q, d[candidate_indices])
summary = {}
for name, s in scores.items():
    correct = np.array([s[i, q["side"] - 1] > s[i, 2 - q["side"]] for i, q in enumerate(queries)])
    pairs = correct.reshape(-1, 2)
    paired = pairs.all(axis=1)
    rng = np.random.default_rng(20261011)
    indices = rng.integers(0, len(pairs), (2000, len(pairs)))
    summary[name] = dict(queries=len(queries), pairs=len(pairs), strict_query_accuracy=float(correct.mean()), strict_pair_accuracy=float(paired.mean()), ties=int(np.isclose(s[:, 0], s[:, 1], rtol=0, atol=0).sum()), query_accuracy_95ci=np.quantile(pairs.mean(axis=1)[indices].mean(axis=1), [0.025, 0.975]).tolist(), pair_accuracy_95ci=np.quantile(paired[indices].mean(axis=1), [0.025, 0.975]).tolist())
    np.save(out / f"pair-scores-{name}.npy", s)
    out.joinpath(f"predictions-{name}.jsonl").write_text("".join(json.dumps(dict(query_id=q["id"], pair=q["pair"], candidates=q["candidates"], gold=q["gold"], scores=s[i].tolist(), strict_correct=bool(correct[i]))) + "\n" for i, q in enumerate(queries)))
out.joinpath("summary.json").write_text(json.dumps(summary, indent=2) + "\n")
out.joinpath("analysis-receipt.json").write_text(json.dumps(dict(status="complete", script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), dataset=json.loads(data.joinpath("manifest.json").read_text()), bootstrap_unit="original document/query pair", bootstrap_resamples=2000), indent=2) + "\n")
print(json.dumps(summary, indent=2))
