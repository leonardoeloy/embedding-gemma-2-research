import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["gemma", "wemm", "ovis"])
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    out = ROOT / "results" / args.model
    out.mkdir(parents=True, exist_ok=True)
    if out.joinpath("receipt.json").exists():
        raise RuntimeError("Completed result exists; preserve it rather than overwrite.")
    discovery = json.loads(ROOT.joinpath("model-discovery.json").read_text())
    info = discovery[args.model]
    path = ROOT / "models" / args.model
    docs = [json.loads(line) for line in ROOT.joinpath("data/documents.jsonl").read_text().splitlines()]
    queries = [json.loads(line) for line in ROOT.joinpath("data/queries.jsonl").read_text().splitlines()]
    torch.set_num_threads(8)
    torch.manual_seed(20261011)
    if not torch.xpu.is_available():
        raise RuntimeError("An Intel GPU is required for this run.")
    device = "xpu:0"
    torch.xpu.reset_peak_memory_stats()
    started = time.time()
    receipt = dict(status="running", model=args.model, source=info, device=device, affinity=os.environ.get("ZE_AFFINITY_MASK"), torch=torch.__version__, packages={k: importlib.metadata.version(k) for k in ["transformers", "sentence-transformers", "huggingface-hub", "numpy"]}, host=platform.node(), dataset=json.loads(ROOT.joinpath("data/manifest.json").read_text()), script_sha256=digest(Path(__file__)), protocol_sha256=digest(ROOT / "PROTOCOL.md"), batch_size=args.batch_size, model_weight_files={str(p.relative_to(path)): digest(p) for p in path.rglob("*.safetensors")})
    out.joinpath("progress.json").write_text(json.dumps(receipt, indent=2))
    if args.model in ("gemma", "wemm"):
        kwargs = dict(device=device, model_kwargs={"dtype": torch.bfloat16, "attn_implementation": "eager"})
        if args.model == "gemma":
            kwargs["config_kwargs"] = {"vision_config": None, "audio_config": None}
        else:
            kwargs["trust_remote_code"] = True
        model = SentenceTransformer(str(path), **kwargs).eval()
        receipt["prompts"] = model.prompts
        receipt["module_configuration"] = str(model)
        receipt["parameter_devices"] = sorted({str(p.device) for p in model.parameters()})
        receipt["parameter_dtypes"] = sorted({str(p.dtype) for p in model.parameters()})
        receipt["loaded_parameters"] = sum(p.numel() for p in model.parameters())

        def encode(texts, side):
            method = model.encode_query if side == "queries" else model.encode_document
            if args.model == "gemma":
                method = model.encode
                kw = {"prompt_name": "SearchQuery" if side == "queries" else "Document"}
            else:
                kw = {}
            return method(texts, batch_size=args.batch_size, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False, **kw).float().cpu().numpy()

        receipt["max_input_tokens"] = max(len(model.tokenizer.encode(x["text"])) for x in docs + queries)
    else:
        from transformers import AutoProcessor, Qwen2_5OmniForConditionalGeneration
        path = path / "model"
        processor = AutoProcessor.from_pretrained(path)
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(path, dtype=torch.bfloat16, attn_implementation="eager", enable_audio_output=False).to(device).eval()
        thinker = model.thinker
        receipt["parameter_devices"] = sorted({str(p.device) for p in thinker.parameters()})
        receipt["parameter_dtypes"] = sorted({str(p.dtype) for p in thinker.parameters()})
        receipt["loaded_parameters"] = sum(p.numel() for p in thinker.parameters())
        receipt["query_instruction"] = "Retrieve relevant passages that answer the query."
        receipt["pooling"] = "last non-padding token of final Thinker hidden state, FP32 L2 normalization"
        processor.tokenizer.padding_side = "right"

        def encode(texts, side):
            rendered = []
            for t in texts:
                messages = []
                if side == "queries":
                    messages.append({"role": "system", "content": receipt["query_instruction"]})
                messages.append({"role": "user", "content": [{"type": "text", "text": t}]})
                rendered.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))
            batch = processor(text=rendered, padding=True, return_tensors="pt").to(device)
            with torch.inference_mode():
                result = thinker(**batch, output_hidden_states=True, use_cache=False, logits_to_keep=1)
                hidden = result.hidden_states[-1]
                last = batch.attention_mask.sum(dim=1) - 1
                vectors = hidden[torch.arange(len(texts), device=device), last].float()
                vectors = torch.nn.functional.normalize(vectors, dim=-1)
            return vectors.cpu().numpy()

        receipt["max_input_tokens"] = max(len(processor.tokenizer.encode(x["text"])) for x in docs + queries)
    if receipt["parameter_devices"] != [device]:
        raise RuntimeError(f"Unexpected model placement: {receipt['parameter_devices']}")
    for side, records in [("documents", docs), ("queries", queries)]:
        blocks = []
        for start in range(0, len(records), args.batch_size):
            block = encode([r["text"] for r in records[start:start + args.batch_size]], side)
            if not np.isfinite(block).all() or np.any(np.linalg.norm(block, axis=1) < 0.9):
                raise RuntimeError("Non-finite or collapsed embedding.")
            block /= np.linalg.norm(block, axis=1, keepdims=True)
            blocks.append(block)
            if start % 80 == 0:
                print(args.model, side, start, "/", len(records), flush=True)
                out.joinpath("progress.json").write_text(json.dumps({**receipt, "side": side, "encoded": start + len(block)}, indent=2))
        vectors = np.concatenate(blocks)
        np.save(out / f"{side}.npy", vectors)
        out.joinpath(f"{side}-ids.json").write_text(json.dumps([r["id"] for r in records]))
        receipt[side] = dict(shape=list(vectors.shape), sha256=digest(out / f"{side}.npy"), norm_min=float(np.linalg.norm(vectors, axis=1).min()), norm_max=float(np.linalg.norm(vectors, axis=1).max()))
    receipt.update(status="complete", seconds=time.time() - started, peak_allocated_bytes=torch.xpu.max_memory_allocated(), completed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    out.joinpath("receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(args.model, "complete", receipt["documents"], receipt["queries"], flush=True)


if __name__ == "__main__":
    main()
