import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from rank_bm25 import BM25Okapi

from generate import eligible

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"


def summarize(rows):
    result = {}
    for kind in ["constraint", "control"]:
        subset = [r for r in rows if r["kind"] == kind]
        metrics = ["hit1", "hit3", "mrr", "topical1", "violation1", "prefilter_hit1", "postfilter1_hit1", "postfilter3_hit1", "postfilter3_empty"]
        result[kind] = {key: float(np.mean([r[key] for r in subset])) for key in metrics}
        result[kind]["n"] = len(subset)
    constraints = [r for r in rows if r["kind"] == "constraint"]
    grouped = defaultdict(list)
    for r in constraints:
        grouped[(r["world"], r["family"], r["style"], r["phrasing"])].append(r["hit1"])
    result["paired_variant_success"] = float(np.mean([all(x) for x in grouped.values()]))
    margins = [r["failure_margin"] for r in constraints if not r["hit1"]]
    result["failure_margins"] = {"n": len(margins), "median": float(np.median(margins)) if margins else None, "over_0.01": sum(m > 0.01 for m in margins), "ties": sum(abs(m) < 1e-7 for m in margins)}
    result["breakdowns"] = {}
    for field in ["family", "style", "phrasing", "polarity"]:
        result["breakdowns"][field] = {str(value): {key: float(np.mean([r[key] for r in constraints if r[field] == value])) for key in metrics} for value in sorted({r[field] for r in constraints})}
    worlds = sorted({r["world"] for r in constraints})
    rng = np.random.default_rng(20261011)
    indices = rng.integers(0, len(worlds), (2000, len(worlds)))
    result["world_bootstrap_95ci"] = {}
    for metric in metrics:
        averages = np.array([np.mean([r[metric] for r in constraints if r["world"] == w]) for w in worlds])
        result["world_bootstrap_95ci"][metric] = np.quantile(averages[indices].mean(axis=1), [0.025, 0.975]).tolist()
    return result


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.results_dir
    OUT.mkdir(exist_ok=True)
    docs = [json.loads(x) for x in ROOT.joinpath("data/documents.jsonl").read_text().splitlines()]
    queries = [json.loads(x) for x in ROOT.joinpath("data/queries.jsonl").read_text().splitlines()]
    dindex = {d["id"]: i for i, d in enumerate(docs)}
    qids = [q["id"] for q in queries]
    dids = [d["id"] for d in docs]
    tokenizer = lambda text: re.findall(r"[a-z0-9]+", text.lower())
    bm25 = BM25Okapi([tokenizer(d["text"]) for d in docs])
    score_sets = {"bm25": np.stack([bm25.get_scores(tokenizer(q["text"])) for q in queries]).astype(np.float32)}
    score_sets["random"] = np.random.default_rng(20261011).random((len(queries), len(docs))).astype(np.float32)
    for model in ["gemma", "wemm", "ovis"]:
        path = OUT / model
        receipt = json.loads(path.joinpath("receipt.json").read_text())
        if receipt["status"] != "complete" or receipt["dataset"] != json.loads(ROOT.joinpath("data/manifest.json").read_text()):
            raise RuntimeError("Incomplete or mismatched encoder receipt")
        if json.loads(path.joinpath("queries-ids.json").read_text()) != qids or json.loads(path.joinpath("documents-ids.json").read_text()) != dids:
            raise RuntimeError("Vector row mismatch")
        q = np.load(path / "queries.npy")
        d = np.load(path / "documents.npy")
        for side in ["queries", "documents"]:
            if hashlib.sha256(path.joinpath(side + ".npy").read_bytes()).hexdigest() != receipt[side]["sha256"]:
                raise RuntimeError("Embedding checksum mismatch")
        dims = [q.shape[1]]
        if model == "gemma":
            dims += [512, 256, 128]
        elif model == "wemm":
            supported = receipt["supported_dimensions"]
            dims += [n for n in [512, 256, 128] if n in supported]
        for dim in dims:
            qs, ds = q[:, :dim].copy(), d[:, :dim].copy()
            qs /= np.linalg.norm(qs, axis=1, keepdims=True)
            ds /= np.linalg.norm(ds, axis=1, keepdims=True)
            score_sets[f"{model}-{dim}"] = qs @ ds.T
        if model == "ovis":
            for dim in [512, 256, 128]:
                matrix_path = ROOT / "models/ovis/Adaptor" / f"M_d{dim}.npy"
                qpath, dpath = path / f"queries-{dim}.npy", path / f"documents-{dim}.npy"
                projection_receipt = OUT / f"ovis-projection-{dim}.json"
                if qpath.exists() and dpath.exists() and projection_receipt.exists():
                    meta = json.loads(projection_receipt.read_text())
                    for side, vpath in [("queries", qpath), ("documents", dpath)]:
                        if hashlib.sha256(vpath.read_bytes()).hexdigest() != meta[side + "_sha256"]:
                            raise RuntimeError("Projected vector checksum mismatch")
                    qs, ds = np.load(qpath), np.load(dpath)
                else:
                    matrix = np.load(matrix_path)
                    qs, ds = (q @ matrix)[:, :dim], (d @ matrix)[:, :dim]
                    qs /= np.linalg.norm(qs, axis=1, keepdims=True)
                    ds /= np.linalg.norm(ds, axis=1, keepdims=True)
                    np.save(qpath, qs)
                    np.save(dpath, ds)
                    projection_receipt.write_text(json.dumps({"file": str(matrix_path.relative_to(ROOT)), "sha256": hashlib.sha256(matrix_path.read_bytes()).hexdigest(), "shape": list(matrix.shape), "recipe": "(embedding @ M)[:, :D], then L2 normalize", "dtype": str(matrix.dtype), "queries_sha256": hashlib.sha256(qpath.read_bytes()).hexdigest(), "documents_sha256": hashlib.sha256(dpath.read_bytes()).hexdigest()}, indent=2))
                score_sets[f"ovis-{dim}"] = qs @ ds.T
    all_summary, example_rows = {}, []
    for method, scores in score_sets.items():
        if not np.isfinite(scores).all():
            raise RuntimeError(f"Non-finite scores: {method}")
        rows = []
        np.save(OUT / f"scores-{method}.npy", scores)
        for qi, q in enumerate(queries):
            candidate_idx = [dindex[x] for x in q["candidates"]]
            order = np.argsort(-scores[qi, candidate_idx], kind="stable")
            ranked = [docs[candidate_idx[j]] for j in order]
            gold = set(q["gold"])
            expected = {d["id"] for d in ranked if d["topic"] == q["topic"] and eligible(d["facts"], q["predicate"])}
            if gold != expected or not gold:
                raise RuntimeError(f"Ground-truth inconsistency: {q['id']}")
            satisfying = [d for d in ranked if eligible(d["facts"], q["predicate"])]
            post3 = [d for d in ranked[:3] if eligible(d["facts"], q["predicate"])]
            first_relevant = next(i + 1 for i, d in enumerate(ranked) if d["id"] in gold)
            row = {k: q[k] for k in ["id", "world", "family", "style", "kind"]}
            row.update({k: q.get(k) for k in ["phrasing", "polarity"]})
            row.update(hit1=int(ranked[0]["id"] in gold), hit3=int(any(d["id"] in gold for d in ranked[:3])), mrr=1 / first_relevant, topical1=int(ranked[0]["topic"] == q["topic"]), violation1=int(not eligible(ranked[0]["facts"], q["predicate"])), prefilter_hit1=int(bool(satisfying) and satisfying[0]["id"] in gold), postfilter1_hit1=int(ranked[0]["id"] in gold and eligible(ranked[0]["facts"], q["predicate"])), postfilter3_hit1=int(bool(post3) and post3[0]["id"] in gold), postfilter3_empty=int(not post3), ranking=[d["id"] for d in ranked], scores=[float(scores[qi, dindex[d["id"]]]) for d in ranked], prefilter_candidates=len(satisfying), failure_margin=float(scores[qi, candidate_idx[order[0]]] - scores[qi, dindex[ranked[first_relevant - 1]["id"]]]))
            rows.append(row)
            if q["kind"] == "constraint" and not row["hit1"] and method in ["gemma-768", "wemm-2048", "ovis-2048"]:
                first_gold = next(d for d in ranked if d["id"] in gold)
                example_rows.append(dict(method=method, query=q["text"], family=q["family"], query_id=q["id"], selected=ranked[0]["text"], selected_score=row["scores"][0], valid=first_gold["text"], valid_score=float(scores[qi, dindex[first_gold["id"]]]), predicate=q["predicate"]))
        OUT.joinpath(f"predictions-{method}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        all_summary[method] = summarize(rows)
    OUT.joinpath("summary.json").write_text(json.dumps(all_summary, indent=2) + "\n")
    OUT.joinpath("failure-examples.json").write_text(json.dumps(example_rows, indent=2) + "\n")
    with OUT.joinpath("summary.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "constraint_hit1", "topical_hit1", "violation1", "prefilter_hit1", "postfilter3_hit1", "postfilter3_empty", "control_hit1", "paired_variant_success"])
        for method, s in all_summary.items():
            c = s["constraint"]
            writer.writerow([method, *[c[k] for k in ["hit1", "topical1", "violation1", "prefilter_hit1", "postfilter3_hit1", "postfilter3_empty"]], s["control"]["hit1"], s["paired_variant_success"]])
    native = ["bm25", "gemma-768", "wemm-2048", "ovis-2048"]
    labels = ["BM25", "EmbeddingGemma 2", "WeMM 2B", "Ovis Omni 3B"]
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    x = np.arange(len(native))
    for shift, metric, label, color in [(-0.25, "topical1", "Topical relevance", "#8798ad"), (0, "hit1", "Relevant + satisfies requirements", "#d17a33"), (0.25, "prefilter_hit1", "With structured prefilter", "#27836a")]:
        ax.bar(x + shift, [100 * all_summary[k]["constraint"][metric] for k in native], 0.24, label=label, color=color)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Hit@1 (%)")
    ax.set_title("Relevant does not always mean eligible")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3, frameon=False)
    fig.savefig(OUT / "constraint-gap.png", dpi=180)
    fig.savefig(OUT / "constraint-gap.svg")
    fig, ax = plt.subplots(figsize=(10, 4.8), layout="constrained")
    families = list(all_summary["gemma-768"]["breakdowns"]["family"])
    matrix = np.array([[100 * all_summary[m]["breakdowns"]["family"][f]["hit1"] for f in families] for m in native[1:]])
    image = ax.imshow(matrix, vmin=0, vmax=100, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(families)), [f.replace("_", " ") for f in families], rotation=20, ha="right")
    ax.set_yticks(range(3), labels[1:])
    for i in range(3):
        for j in range(len(families)):
            ax.text(j, i, f"{matrix[i, j]:.1f}%", ha="center", va="center", color="white" if matrix[i, j] > 65 else "black")
    ax.set_title("Constraint-satisfying Hit@1 by requirement family")
    fig.colorbar(image, ax=ax, label="Hit@1 (%)")
    fig.savefig(OUT / "families.png", dpi=180)
    fig.savefig(OUT / "families.svg")
    fig, ax = plt.subplots(figsize=(8.5, 4.8), layout="constrained")
    for model, label, color in [("gemma", "EmbeddingGemma 2", "#d17a33"), ("wemm", "WeMM 2B", "#4668b0"), ("ovis", "Ovis Omni 3B", "#27836a")]:
        dimensions = sorted(int(k.split("-")[1]) for k in all_summary if k.startswith(model + "-"))
        ax.plot(dimensions, [100 * all_summary[f"{model}-{n}"]["constraint"]["hit1"] for n in dimensions], marker="o", color=color, label=label)
    ax.set_xlabel("Vector dimensions (FP32 index: 4 bytes per dimension per item)")
    ax.set_ylabel("Constraint-satisfying Hit@1 (%)")
    ax.set_title("Validate constraints when reducing vector storage")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.savefig(OUT / "dimensions.png", dpi=180)
    fig.savefig(OUT / "dimensions.svg")
    OUT.joinpath("analysis-receipt.json").write_text(json.dumps({"status": "complete", "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "dataset": json.loads(ROOT.joinpath("data/manifest.json").read_text()), "methods": list(all_summary), "world_bootstrap_resamples": 2000, "seed": 20261011}, indent=2) + "\n")
    print(OUT.joinpath("summary.csv").read_text())


if __name__ == "__main__":
    main()
