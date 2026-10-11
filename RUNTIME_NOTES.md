# Runtime details affecting reproduction

All model parameters used for inference reside on logical `xpu:0`, with `ZE_AFFINITY_MASK=0` exposing only physical GPU 0. Models run sequentially. The hardware inventory is saved in `sources/hardware.json`; inference receipts record versions, dtypes, parameter counts, tensor integrity and the exact executed encoder source.

WeMM uses Transformers' reference implementations for causal convolution and gated delta attention. Torch 2.8 XPU also falls back to CPU for `torch.linalg.solve_triangular`. This is the framework's supported reference path, not an intentional approximation. It affects execution speed, which this study does not compare. No custom kernel, compilation or performance tuning was added.

Ovis's released checkpoint includes speech-generation tensors and a generative configuration, although the model card describes an embedding release. Loading `Qwen2_5OmniForConditionalGeneration` requests a missing `spk_dict.pt`, even with speech output disabled, and exits before inference. The successful loader uses the native `Qwen2_5OmniThinkerForConditionalGeneration`, its released Thinker configuration and an explicit `thinker.` checkpoint-prefix mapping. It loads the same Thinker weights and extracts the same final hidden states. Missing/mismatched active parameters cause a hard failure. Unused Talker/token2wav tensors are recorded as unexpected keys; no substitute speech file or separately pretrained backbone is introduced.

The Ovis query instruction is fixed to “Retrieve relevant passages that answer the query.” in the system message, with the released native chat template. Candidate documents use that template's default system message. The authors do not publish an exact instruction for our fictional catalog task. This is a disclosed usable baseline, not a claim of exact benchmark-prompt reproduction. No prompt was chosen by comparing Ovis scores.

Some inherited Ovis speech configuration fields also produce vocabulary-range warnings. The speech generator is not instantiated or called. They do not describe the Thinker text tokenizer or retrieval output.

Initial Xet downloads logged transfer errors and retries. The competitor files were downloaded using standard direct HTTP instead, from the same immutable Hub revisions. `results/weight-integrity.json` compares every completed safetensors file against its upstream SHA-256. Old transfer partials are not model inputs.

The original EmbeddingGemma catalog run preceded small source changes for provenance/output paths and the Ovis loader. Its exact source is preserved as `results/gemma/encode-source.py` and matches the source hash in its receipt. Every other successful run also retains its executed source. The dataset and the original protocol hashes remain fixed.
