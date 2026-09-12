# 2. Materials and Methods

## 2.1 Materials

### 2.1.1 Hardware

All experiments were run on a single Dell XPS 15 9560 laptop under Microsoft Windows 10 Home (64-bit, version 10.0.19045), equipped with an Intel Core i7-7700HQ processor (4 cores, 8 threads, 2.80 GHz base clock, 6 MB L3 cache), 16 GB of DDR4-2400 memory (15.86 GB usable) and an NVIDIA GeForce GTX 1050 discrete graphics card with 4 GB of dedicated memory alongside the integrated Intel HD Graphics 630. Storage was a 477 GB Samsung NVMe solid-state drive (MZVLB512). The discrete GPU executed both small language model (SLM) inference through Ollama and neural network inference through CUDA, while CPU, system memory and GPU counters were sampled during every generation to obtain the computational efficiency metrics defined in Section 2.2.6. The complete machine-generated specification report, produced by `src/Hardware_spec.ps1`, is provided as supplementary material (`results/Hardware_Specifications.txt`) so that the execution environment can be verified exactly.

**Table 1.** Hardware specifications of the experimental platform.

| Component | Key details |
|:---|:---|
| Operating system | Windows 10 Home (64-bit), version 10.0.19045, locale en-GB, time zone UTC+08:00 |
| System | Dell XPS 15 9560 |
| CPU | Intel Core i7-7700HQ, 2.80 GHz, 4 cores / 8 threads, 6 MB L3 cache |
| GPU (discrete) | NVIDIA GeForce GTX 1050, 4 GB VRAM, driver 528.79 (Windows driver version 31.0.15.2879) |
| GPU (integrated) | Intel HD Graphics 630, 1 GB shared memory |
| RAM | 16 GB (2 × 8 GB SODIMM, 2400 MHz), 15.86 GB usable |
| Storage | 477 GB NVMe SSD (Samsung MZVLB512) |
| Inference runtime | Ollama (GPU offloading enabled), PyTorch with CUDA for the quality predictor |

### 2.1.2 Datasets

Ten publicly available reasoning benchmarks were used. For every benchmark, the first 50 records of the official test split were evaluated, giving 500 questions per model and mechanism and 7,000 generated responses in total (10 datasets × 50 questions × 7 models × 2 mechanisms). Only the test split file of each benchmark was read, so that bundled training or development splits could not enter the evaluation set.

- **AQUA**: algebraic word problems with five multiple-choice options (A–E), testing arithmetic, fractions, equations and geometry (https://huggingface.co/datasets/nguyen-brat/aqua).
- **ASDiv**: linguistically diverse math word problems spanning addition, subtraction, multiplication and division at grades 1–6 (https://huggingface.co/datasets/nguyen-brat/asdiv).
- **CLUTRR**: kinship stories requiring compositional relational inference, with a single relation term as the reference answer (https://huggingface.co/datasets/CLUTRR/v1).
- **Date**: temporal reasoning items in which a calendar date must be inferred and returned in MM/DD/YYYY form (https://huggingface.co/datasets/fanshiyu/date).
- **GSM8K**: multi-step grade-school arithmetic problems; the reference answer includes the worked solution and a final numeric value (https://huggingface.co/datasets/openai/gsm8k).
- **MultiArith**: elementary 2–3 step arithmetic problems with a numeric reference answer (https://huggingface.co/datasets/ChilleD/MultiArith).
- **QASports**: sports statements labelled plausible (1) or implausible (0) (https://huggingface.co/datasets/PedroCJardim/QASports).
- **SayCan**: natural-language instructions mapped to executable robot action sequences, with several acceptable plans per instruction (https://huggingface.co/datasets/chiayewken/saycan).
- **StrategyQA**: yes/no questions requiring implicit multi-hop commonsense reasoning (https://huggingface.co/datasets/voidful/StrategyQA).
- **SVAMP**: elementary math word problems built by perturbing existing problems to defeat pattern matching (https://huggingface.co/datasets/ChilleD/SVAMP).

Questions were submitted exactly as distributed, without dataset-specific instructions. For QASports and SayCan this means that the plausibility question and the robot action inventory, respectively, are not part of the prompt; both benchmarks are therefore reported but interpreted with caution (Section 4).

## 2.2 Methods

### 2.2.1 Text Quality Metrics

Eleven computational text quality metrics are used throughout this study. The same eleven metrics define the neural network's input and output space (Section 2.2.3), the target profile injected into MGCoT prompts (Section 2.2.2), and the profiles reported for generated and reference answers. Each metric is a deterministic function of a text, implemented in `src/metrics/profile/` and applied after lower-casing and word tokenisation; *WC* denotes the number of alphabetic words, *S* the number of sentence delimiters, and *content words* the alphabetic words that are not English stopwords.

1. **Readability** — a bounded transformation of average word length and sentence density, computed as 206.835 − 1.015 × AWL − 84.6 × (S / WC) and clipped to [0, 100], where AWL is the mean word length in characters. The coefficients follow the Flesch Reading Ease formula, but the arguments are word length and sentence density rather than syllable and sentence counts; for texts of the length generated here the expression saturates at its upper bound, which is reported as a limitation in Section 4 [29].
2. **Coherence** — 1 − (S / WC), clipped to [0, 1]; higher values indicate fewer sentence breaks per word, i.e. longer connected sentences [30].
3. **Relevance** — the proportion of content words among all words, in [0, 1]; higher values indicate denser, more on-topic content [31].
4. **Specificity** — the proportion of distinct content words among all content words, in [0, 1]; higher values indicate less repetition of the same content vocabulary [32].
5. **Engagement** — (question marks + 0.5 × interrogatives) / WC × 10, capped at 1.0, where interrogatives are *what, how, why, when, where, who*; higher values indicate a more conversational, reader-directed tone [33].
6. **Concise** — 1 / (1 + ln(words per sentence)), in [0, 1]; higher values indicate shorter sentences on average [34].
7. **Length** — 1 − exp(−0.05 × WC / 100), in [0, 1). The transformation is monotonically increasing and deliberately gradual: a 100-word answer scores approximately 0.05, so the metric orders responses by length rather than saturating within the range observed here [37].
8. **Zipf** — 1 / (1 + σ), where σ is the standard deviation of the products of word frequency and frequency rank; higher values indicate a rank–frequency distribution closer to Zipf's law [35].
9. **Hapax** — the proportion of words occurring exactly once, in [0, 1]; higher values indicate less repetitive vocabulary [36].
10. **Entropy** — the Shannon entropy of the word-frequency distribution, computed in nats (natural logarithm); higher values indicate more uniform, less predictable word choice [38].
11. **Perplexity** — 2 raised to the power of the entropy above. Because the entropy is computed in nats while the exponentiation uses base 2, the metric is a monotone transformation of entropy rather than perplexity in its classical form, and is used here only as such [39].

### 2.2.2 Metric-Guided Chain of Thoughts

MGCoT constructs each prompt from a predicted target quality profile instead of a fixed instruction, contrasting with Standard Chain-of-Thought (SCoT), which issues the same reasoning instruction to every question. The workflow has six steps:

1. **Quality metric generation.** Every question and its reference answer in the benchmark suite are profiled with the eleven metrics of Section 2.2.1, producing paired question and answer profiles (`src/nn/build_dataset.py`).
2. **Neural network training.** A dense neural network is trained on these pairs to map a question profile to the profile of an adequate answer (Section 2.2.3).
3. **Target answer quality prediction.** For a new question, the eleven metrics are computed, scaled with the fitted scaler, passed through the trained network, and inverse-transformed to the natural scale of each metric.
4. **MGCoT prompt construction.** The predicted values are rendered as explicit numeric targets, preceded by a definition and reading direction for each metric, and followed by an analyse–plan–write–validate instruction (`src/prompts/mgcot.py`).
5. **Language model generation.** The SLM receives the system instruction, the eleven targets and the unmodified question.
6. **Answer evaluation.** Generated answers are scored against the reference answer with the metrics of Sections 2.2.6–2.2.8 and compared with SCoT under the protocol of Section 2.2.9.

The SCoT baseline uses the same template, generation settings and evaluation pipeline, with a system instruction that requests step-by-step reasoning and a clearly stated final answer, and no quality targets (`src/prompts/scot.py`). The two mechanisms therefore differ only in the content of the system instruction.

### 2.2.3 Dense Neural Network

The quality predictor is a four-layer fully connected network mapping an eleven-dimensional question profile to an eleven-dimensional answer profile: an input layer of 11 neurons, three hidden layers of 128, 64 and 32 neurons, and an output layer of 11 neurons. Hidden layers use ReLU activations with batch normalisation on the first two, and dropout of 0.2, 0.2 and 0.1 respectively. Training minimises mean squared error with the AdamW optimiser (learning rate 1 × 10⁻³, weight decay 1 × 10⁻⁴), batch size 64, gradient-norm clipping at 1.0, for 300 epochs, with an 80/20 train–validation split at seed 42. Inputs and outputs are scaled to [0, 1] with MinMax scalers fitted on the training data only; the fitted scalers are persisted with the weights so that inference applies the identical transform and returns predictions on each metric's natural scale (`src/nn/train.py`, `src/nn/predict.py`).

### 2.2.4 Small Language Models

Seven instruction-tuned SLMs between 1 and 8 billion parameters were evaluated, all served locally through Ollama in their default quantised form: **phi3:mini** (3.8B, Microsoft), **llama3.2:1b** (1.2B, Meta), **gemma2:2b** (2.6B, Google DeepMind), **qwen2:1.5b** (1.5B, Alibaba), **mistral:7b** (7.3B, Mistral AI), **openchat:7b** (7B, OpenChat), and **deepseek-r1:8b** (8B, DeepSeek-AI, a reasoning-distilled Llama-3.1 variant). Every model answered every question under both mechanisms.

### 2.2.5 Experimental Protocol

Generation used identical settings for both mechanisms and all models: temperature 0.0 (greedy decoding, for reproducibility), a generation budget of 300 tokens (`num_predict`), and full GPU offloading where memory permitted (`num_gpu = -1`). Responses were streamed, and CPU, system memory and GPU counters were sampled once per streamed token, with a 0.5 s interval between samples; consequently the reported wall-clock time increases with the number of generated tokens and is interpreted as a length-sensitive cost measure rather than a pure hardware latency.

The 300-token budget exceeds the length of every reference answer in the suite (95% are at most 57 words; the longest single reference answer is 122 words) and therefore defines a **budget-compliance criterion**: a model is considered budget-compliant if it returns a visible answer within the budget. Six models satisfied this criterion in at least 97% of runs. deepseek-r1:8b, whose distilled reasoning trace consumes the budget before the answer is emitted, returned an empty visible answer in 97–98% of runs; it is reported throughout, but its quality scores measure budget compliance rather than answer quality.

For every record, SCoT was generated first and MGCoT second, so each question yields a matched pair of responses from the same model under identical conditions (`main.py`).

### 2.2.6 Computational Efficiency Evaluation Metrics

Resource use was measured between the start of generation and the end of the response. Let *t*₀ and *t*₁ denote these instants and *n* the number of samples collected in between.

- **Generation time (Equation 1).** Δt = *t*₁ − *t*₀, reported in minutes.
- **Average CPU utilisation (Equation 2).** The mean of system-wide CPU utilisation samples, in per cent.
- **Average RAM use (Equations 3 and 4).** The mean of system-wide memory samples, reported both in gigabytes and as a percentage of installed memory.
- **Average GPU utilisation, PGPU (Equation 5).** The mean of device utilisation samples reported by NVML, in per cent.
- **Average GPU memory, MGPU (Equation 6).** The mean of device memory-in-use samples reported by NVML, in gigabytes.

Each quantity is the arithmetic mean over the *n* samples of one generation; maxima are not reported. CPU, RAM and GPU counters are system-wide rather than process-specific, so they include the constant background load of the host and, for GPU memory, the resident model weights.

### 2.2.7 Semantic Similarity Evaluation Metrics

- **BERTScore F1 (Equations 7–9).** Token-level cosine similarity between contextual embeddings of the generated and reference answers, combining greedy-matched precision and recall into an F1 [48].
- **Semantic similarity (Equation 10).** Cosine similarity between sentence embeddings of the generated and reference answers, obtained with all-MiniLM-L6-v2 [49]; the value lies in [−1, 1], where 1 indicates maximal alignment.

### 2.2.8 Syntactic Similarity Evaluation Metrics

Let *P* and *R* be the token sets of the prediction and the reference.

- **ROUGE-L F1, RLF1 (Equations 11–13).** F1 over the longest common subsequence of prediction and reference [50].
- **Token precision, TP (Equation 14).** |*P* ∩ *R*| / |*P*|, the share of generated tokens that appear in the reference.
- **Token recall, TR (Equation 15).** |*P* ∩ *R*| / |*R*|, the share of reference tokens that appear in the generated answer.
- **Token F1, TF1 (Equation 16).** The harmonic mean of TP and TR [52].
- **Normalised edit similarity, NED (Equation 17).** 1 − (Levenshtein distance / length of the longer string); despite the conventional name, higher values indicate greater similarity.
- **Character F1, CF1 (Equation 18).** The harmonic mean of precision and recall over the sets of distinct characters of the two strings.

Exact match, BLEU, ROUGE-1 F1, ROUGE-2 F1 and the raw token overlap count were also computed for every response but are not reported in the main text. Because reference answers are short labels or numbers while generated answers contain the reasoning, exact match is 0 for every model, dataset and mechanism, BLEU never exceeds 0.05, and ROUGE-2 is 0.00 on most datasets; ROUGE-1 duplicates ROUGE-L, and the overlap count restates token recall on an unnormalised scale. All five are provided per model and dataset in the supplementary material.

### 2.2.9 Statistical Analysis

Because both mechanisms answer the same questions on the same machine, all comparisons are paired at the question level. For every metric, model and dataset, the 50 paired values (500 for the all-dataset aggregate) were compared with a two-sided **Wilcoxon signed-rank test**, which requires no normality assumption; pairs with a zero difference were discarded. Within each table, the resulting p-values were corrected for multiple comparisons with the **Benjamini–Hochberg** false-discovery-rate procedure, and a difference is reported as significant when the corrected p-value is below 0.05.

Effect magnitude is reported in two ways: a **95% percentile bootstrap confidence interval** for the mean paired difference (5,000 resamples, seed 42) and the **matched-pairs rank-biserial correlation**, signed so that positive values favour MGCoT irrespective of whether the metric is better when higher or lower. Every test performed, with means, standard deviations, mean difference, confidence interval, effect size, raw and corrected p-value, is provided as supplementary material (`results/analysis/supplementary_tests.xlsx`).

### 2.2.10 Deep Neural Network Evaluation Metrics

- **Mean squared error (Equation 19).** The mean squared difference between predicted and observed answer-quality vectors over the eleven metrics; 0 indicates perfect prediction [53].
- **Cosine similarity (Equation 20).** The directional alignment between predicted and observed quality vectors, in [−1, 1] [54].

### 2.2.11 Reproducibility

The evaluation pipeline, prompt templates, metric implementations, neural network training and the analysis scripts are released together with the generated result files. Greedy decoding, a fixed record order, fixed random seeds and persisted scalers make the pipeline deterministic up to the non-determinism of GPU inference; the machine-generated hardware report, the complete per-response result table, the per-model result tables and the full statistical test table are provided as supplementary material.
