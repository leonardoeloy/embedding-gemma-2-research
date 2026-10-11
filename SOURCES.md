# Sources and boundaries

Retrieved 2026-10-11. Model source revisions and file lists are in `model-discovery.json`; full weight hashes are in runtime receipts. Public release dates below describe the announcement/report, not repository creation time.

## Selected contemporary models

- [Google EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2): launched October 6, 2026. [Official developer guide](https://developers.googleblog.com/embeddinggemma-2-the-developer-guide/). Native 768-dimensional vectors, supported prefix truncation at 512/256/128. Text-only loading uses approximately 270M parameters; the full multimodal model has approximately 740M. Use BF16/FP32 and search/document prefixes. This study uses text only.
- [Tencent WeMM-Embedding-2B](https://huggingface.co/tencent/WeMM-Embedding-2B): August 2026 [technical report](https://arxiv.org/abs/2608.24053), checkpoint last modified September 3. Native 2048-dimensional normalized vectors and supported Matryoshka truncation. Its released custom Sentence Transformers module applies the embedding chat template and calls the model's embedding method. This study uses that module, including its last-token pooling and input handling.
- [Alibaba Ovis-Omni-Embedding-3B](https://huggingface.co/ATH-MaaS/Ovis-Omni-Embedding-3B): September 2026 [technical report](https://arxiv.org/abs/2609.25165) and [official repository](https://github.com/ATH-MaaS/Ovis-Omni-Embedding). Native 2048-dimensional final Thinker hidden states. The model card specifies task instructions, native chat formatting, last-valid-token pooling, normalization and independent encoding. It does not publish a unique instruction for our fictional deployment-catalog task. Our fixed generic passage-retrieval instruction is recorded in the receipt; it is not tuned on scores. This limits any claim of exact reproduction of the authors' benchmark settings.

The comparison is between supported text-retrieval paths of multimodal embedding models. It does not evaluate multimodal retrieval, and the parameter counts differ. It is a diagnosis of a use case, not a size-controlled leaderboard.

Omni-Embed-Mini was considered because of its October report, but the attempted download of its v1 release required repository approval. No gate was bypassed, no weights were obtained, and it is not included in the comparison. Qwen3-VL and Jina v5 were considered earlier but were replaced with closer contemporaries before inference or scoring.

## Prior work: the broad weakness is known

- [NevIR](https://arxiv.org/abs/2305.07614) compares documents differing by negation and reports substantial architecture-dependent differences. Our study is not the first observation that dense retrieval can miss negation.
- [Reproducing NevIR](https://arxiv.org/abs/2502.13506) extends that analysis to newer systems and discusses weak transfer between negation datasets. It motivates preserving a narrow claim about this dataset.
- [MultiConIR](https://arxiv.org/abs/2503.08046) studies multi-condition retrieval, ranking robustness and query format sensitivity. Our six-family experiment is a small deployment-oriented diagnostic, not a replacement for MultiConIR.
- [BRIGHT](https://brightbenchmark.github.io/) evaluates reasoning-intensive retrieval. Our task concerns explicit constraints and arithmetic/composition, rather than the broad reasoning relations of BRIGHT.

The intended contribution is fresh empirical evidence on three recent releases, with ordinary semantic controls, whole-topic uncertainty estimates, representation/query sensitivity and the loss from filtering only a short retrieved list. Do not claim an architectural impossibility theorem: these models can succeed on many logical queries, and a different embedding model, training recipe or reranker may improve results.

## Publication and licenses

Dataset texts are newly generated fictional procedures. Model weights are not part of the shareable research directory. Consult each model's actual license before redistribution/deployment; model-card labels alone do not establish identical terms. WeMM's repository includes its own license text. The study contains no company/user data and uses no paid inference service.
