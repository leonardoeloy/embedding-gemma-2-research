# When a relevant search result violates the requirement

A controlled, text-only study of EmbeddingGemma 2, WeMM-Embedding-2B and Ovis-Omni-Embedding-3B for on-device deployment-catalog search. The question is whether semantic retrieval can also enforce requirements such as offline operation, memory limits, valid dates, transfer direction and compatible runtime versions.

The dataset contains 1,152 fictional procedure documents and 1,008 queries: 864 requirement queries and 144 ordinary topical controls. Each query ranks the same eight-candidate cohort across methods. Twelve topic worlds, six requirement families, two document representations, two requirement variants and three query paraphrases expose sensitivity without tuning prompts to outcomes. This is a diagnostic stress test, not an estimate of production failure rates.

The study uses full-dimensional native embeddings as its primary comparison, with seeded random and BM25 baselines. It also compares structured prefiltering against filtering only the top three retrieved results, and measures supported vector-dimension reductions. Exact structured constraints are supplied by the dataset; constraint extraction from natural language is outside this experiment.

## Results

Read the [research report](REPORT.md) for measured results and limitations, or the [blog draft](BLOG_DRAFT.md) for the architecture narrative. Native requirement-satisfying Hit@1 was 74.88% for EmbeddingGemma 2, 78.70% for WeMM and 40.28% for Ovis. Gemma and WeMM had 100% topical controls; Ovis had 52.08%, limiting interpretation of its logical failures.

Gemma at 128 dimensions kept 100% topical controls while requirement accuracy fell to 66.67%. This is the main on-device evaluation finding. All GPU runs and the complete NevIR external check are finished and checksum-audited.

## Files

- `PROTOCOL.md`: frozen protocol; `PROTOCOL_ADDENDUM.md`: disclosed exploratory Ovis projection extension.
- `SOURCES.md`: model selection, release dates, prior work and claim boundaries.
- `data/`: generated documents, queries, relevance labels, structured facts and checksum manifest.
- `generate.py`: deterministic dataset construction.
- `download.py`: downloads immutable model revisions recorded in `model-discovery.json`.
- `encode.py`: native GPU model inference and runtime receipts, including exact executed source snapshots.
- `analyze.py`: cosine/BM25/random rankings, filters, breakdowns, uncertainty, raw predictions and figures.
- `audit_models.py`: compares downloaded weights with the pinned Hub SHA-256 values.
- `results/`: embeddings, score matrices, every ranking, runtime receipts and analysis artifacts.

The large model weights and Python environment stay on Darkstar and are excluded from Git. No data, scores or scripts from the paused LittleBit work are used.

## Reproduce on an Intel GPU

The recorded environment is Linux, Python 3.12, Torch 2.8 XPU, Transformers 5.19.0 and Sentence Transformers 6.1.0. The original run exposes physical GPU 0 only. There is no serving process, distributed inference, approximate search, finetuning or quantization.

From a fresh copy of this repository:

```bash
uv venv --python 3.12
uv pip install 'torch==2.8.0+xpu' 'torchvision==0.23.0+xpu' --index-url https://download.pytorch.org/whl/xpu
uv pip install -r requirements.txt
source .venv/bin/activate
python generate.py
HF_HUB_DISABLE_XET=1 python download.py
```

The direct HTTP transport setting worked around observed Xet transfer failures. It does not alter model files. Run the models sequentially on one visible GPU, preserving the published results:

```bash
ZE_AFFINITY_MASK=0 python encode.py gemma --results-dir repeat-results
ZE_AFFINITY_MASK=0 python encode.py wemm --results-dir repeat-results
ZE_AFFINITY_MASK=0 python encode.py ovis --results-dir repeat-results
python analyze.py --results-dir repeat-results
python audit_models.py
```

`encode.py` refuses to overwrite a completed model result. Its default results directory is `results`; use another directory for repeated runs. `audit_models.py` writes the weight integrity record to `results/weight-integrity.json`. Pinning versions does not promise bitwise parity across different GPU drivers or runtimes; the saved vectors allow CPU-only reanalysis of the original experiment.

EmbeddingGemma uses its documented text-only configuration, search/document prefixes, native mean pooling and BF16. WeMM uses its released custom Sentence Transformers module and embedding method. Ovis uses the released processor/chat template, a fixed generic query instruction and normalized last-valid-token Thinker states. Ovis's task-specific instruction is disclosed rather than presented as an exact reproduction of an authors' benchmark. Model-native BF16 inference is followed by FP32 vector normalization and dot products.

## Architecture implication

When an application has explicit constraints and reliable catalog fields, filter eligible candidates before semantic ranking. Embeddings still choose the relevant topic among eligible candidates. A small retrieved list can already have discarded the valid document before a postfilter sees it.

```mermaid
flowchart LR
  A[Query and explicit requirements] --> B[Structured eligibility filter]
  B --> C[Semantic ranking of eligible candidates]
  C --> D[Relevant result satisfying requirements]
```

This is conditional on the quality and completeness of metadata. An application with uncertain facts, missing metadata or free-form constraints needs additional validation and abstention design. This study does not measure those systems, rerankers, end-to-end RAG answers, mobile latency or energy.
