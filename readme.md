# MGCoT vs SCoT Evaluation Pipeline

Evaluates Small Language Models (SLMs) on reasoning benchmarks under two mechanisms:
**SCoT** (plain step-by-step Chain-of-Thought) and **MGCoT** (Metric-Guided CoT, which predicts
target quality metrics for the answer and asks the model to match them).

## Version

**v1.0.1**

- **Automatic NLTK Resource Management (`src/utils/text.py`)**: Preprocessing and stopword loaders now automatically detect and download missing NLTK corpora (`stopwords`, `punkt`, `punkt_tab`) on demand, preventing runtime `LookupError`/`RuntimeError` exceptions.
- **Embedded CUDA Index URL (`requirements.txt`)**: Added PyTorch CUDA index repository (`--extra-index-url https://download.pytorch.org/whl/cu121`) directly into `requirements.txt` for one-step installation via `pip install -r requirements.txt`.
- **Dataset Filtering & Multi-Model Evaluation Defaults (`main.py`)**: Added `_filter_datasets()` helper to `BenchmarkRunner` with dataset subset support for `process_one()` and `process_many()`, and updated default execution to run 4 SLM models (`phi3:mini`, `llama3.2:1b`, `gemma2:2b`, `qwen2:1.5b`) across 10 benchmark datasets.

**v1.0.0**

- `BenchmarkRunner.process_one()`/`process_many()` accept a `datasets` argument to run only a
  chosen subset of dataset folders (case-insensitive name match), instead of always evaluating
  every dataset found under `dataset_dir`.
- `main()`'s active `process_many()` call now uses inline `model_names`/`max_records` literals
  instead of reading them from `app.yaml`, so they're visible and editable directly in the code.
- MGCoT's prompt (`src/prompts/mgcot.py`) now includes a metric-definitions glossary explaining
  what each of the 11 target metrics means and its range, so the SLM has grounding for the raw
  numeric targets it's asked to match.
- Fixed a hardware-profiling bug (`src/metrics/hardware/cpu.py`): CPU sampling used
  `psutil.cpu_percent(interval=1)`, which blocks for a full second on every call. Since it was
  called once per streamed response chunk, this added up to minutes of artificial delay per
  generation and dominated `delta_time_seconds`. Now uses non-blocking incremental sampling.
- `stream` argument added through `evaluate()` / `process_one()` / `process_many()` to print the
  SLM's response to the terminal live as it generates.

## 1. Setup

Create and activate a virtual environment, then install dependencies:

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

The `--extra-index-url` is required for the CUDA build of `torch`.

You also need [Ollama](https://ollama.com) running locally with the models you intend to
evaluate already pulled (e.g. `ollama pull phi3:mini`).

## 2. Download the benchmark datasets

Datasets and their source URLs are listed in `src/config/becnmarks.yaml`. Download them all with:

```
python src/utils/benchmarks.py
```

This saves each dataset's splits as JSONL under `benchmarks/<name>/<split>.jsonl`. If a
particular source is broken or gated, copy the equivalent files from `dataset/` instead
(the original data behind the paper) rather than editing the yaml to point elsewhere.

## 3. Build the DNN training data

MGCoT's quality-metric predictor is a small neural net trained to map a question's quality
profile to a target answer quality profile. Build its training CSVs from the benchmarks:

```
python src/nn/build_dataset.py --dataset-dir benchmarks --q-df-out results/training/final_q_df.csv --a-df-out results/training/final_a_df.csv --test-filename test.jsonl --max-records 50
```

This profiles every question/answer pair in each dataset's `test.jsonl` with the 11 metrics in
`src/metrics/profile/`, and writes one row per record to each CSV.

## 4. Train the quality-metric predictor

```
python src/nn/train.py --q-df results/training/final_q_df.csv --a-df results/training/final_a_df.csv --model-out results/weights/model.pth --q-scaler-out results/weights/q_scaler.pkl --a-scaler-out results/weights/a_scaler.pkl
```

Saves the trained model plus its fitted input/output scalers. All three files are required at
inference time — the scalers convert between each metric's natural scale (e.g. Readability
0-100) and the network's internal [0,1] training scale.

Model architecture and training hyperparameters live in `src/config/nn.yaml`.

## 5. Configure the evaluation run

Edit `src/config/app.yaml`:

- `model_names`: list of Ollama model tags to evaluate (e.g. `phi3:mini`, `llama3.2:1b`, ...).
- `dataset_dir`: directory of dataset subfolders to evaluate on (`benchmarks` by default).
- `output_dir`: where result CSVs are saved (`results/evaluation` by default).
- `max_records`: cap on records per dataset; `null` processes the full test split.
- `sample_interval`: seconds between hardware-usage samples while a response streams.
- `nnet`: paths to the trained model and scalers from step 4.
- `llm`: Ollama generation settings (`num_gpu: -1` offloads as many layers as fit; `num_predict`
  is the max tokens per answer; `temperature`).

## 6. Run the evaluation

```
python main.py
```

For each model in `model_names`, `BenchmarkRunner.process_many()` runs `process_one()`, which
evaluates every dataset under `dataset_dir`. For each record, it runs SCoT then MGCoT on the
same question, scores both against the gold answer, and appends both rows to that dataset's
result. Per-dataset CSVs and one combined CSV per model are saved under `output_dir`, named
`<timestamp>-<model>-<dataset>.csv` and `combined_<timestamp>-<model>-ALL.csv`.

To run a single model or override `max_records`/streaming from code instead of editing the
yaml, use the class directly:

```python
from main import BenchmarkRunner

runner = BenchmarkRunner()
runner.process_one("phi3:mini", max_records=10, stream=True)   # one model
runner.process_many(["phi3:mini", "llama3.2:1b"], stream=True)  # several models
```

`stream=True` prints each chunk of the SLM's response to the terminal as it arrives.

## Output layout

```
benchmarks/            downloaded dataset JSONL files
dataset/               original paper data (fallback source for broken downloads)
results/
  training/            final_q_df.csv, final_a_df.csv (step 3)
  weights/              model.pth, q_scaler.pkl, a_scaler.pkl (step 4)
  evaluation/           per-dataset and combined result CSVs (step 6)
logs/                   one timestamped log file per main.py run
```

## Metric definitions

The 11 quality-profile metrics MGCoT predicts and targets (see `src/metrics/profile/`):

Readability: Readability measures text complexity using the Flesch-Kincaid formula, scoring 0–100 where higher values indicate easier comprehension. Formula: FK = 206.835 - 1.015 × AWL - 84.6 × (S/WC), where AWL is average word length, S is sentence count, and WC is word count. The formula subtracts weighted word length and sentence density from a baseline, penalizing complex sentences [29].

Coherence: Coherence evaluates logical flow between ideas, scoring 0.0–1.0 with higher values indicating smoother connections. Formula: C = 1 - (S/WC), where S is sentence count and WC is word count. Lower sentence-to-word ratios (fewer sentences per word, implying longer sentences) yield higher coherence scores, suggesting better connectivity between ideas [30].

Relevance (Lexical Density): assesses how closely content aligns with the topic, scoring 0.0–1.0 where higher values indicate tighter focus. Formula: Relevance = content_words / total_words, where content_words excludes stopwords. The ratio measures the proportion of meaningful words relative to total words, with higher ratios indicating more focused, on-topic content [31].

Specificity (Lexical Diversity): measures detail level and precision, scoring 0.0–1.0 with higher values indicating greater detail. Specificity = unique_content_words / content_words, where content_words excludes stopwords. Higher ratios indicate more diverse vocabulary within content words, reflecting detailed, concrete responses (Johansson, 2008; McCarthy & Jarvis, 2010).

Engagement: evaluates conversational dynamism, scoring 0.0–1.0 with higher values indicating more interactive tone. Engagement = [(question_marks + interrogatives × 0.5) / word_count] × 10, where interrogatives count words like 'what', 'how', 'why'. The formula weights question marks and interrogative words relative to text length, measuring how frequently the text prompts reader interaction [33].

Concise (Conciseness): measures brevity and efficiency in writing, with higher values indicating more compact language. We operationalize conciseness as a bounded, monotonically decreasing transformation of average sentence length: Conciseness = 1 / (1 + log(max(words_per_sentence, 1))), where words_per_sentence = word_count / sentence_count. This transformation penalizes verbosity (long sentences) while rewarding more concise phrasing [34].

Zipf: evaluates how closely a text's vocabulary distribution follows Zipf's law, with higher values indicating more natural rank-frequency behavior. We operationalize Zipf compliance as: Zipf = 1 / (1 + std(freq × rank)), where words are ranked by frequency and freq × rank products are computed. Lower standard deviation in these products indicates more Zipf-consistent vocabulary distribution [35].

Hapax: measures the proportion of unique words appearing once (hapax legomena), scoring 0.0–1.0 with higher values indicating more diverse vocabulary. Hapax = words_appearing_once / total_words. Higher ratios indicate more unique, non-repetitive vocabulary, reflecting lexical diversity and avoiding formulaic language [36].

Length: measures the extent of a response, with higher values indicating longer text (peaking around ~100 words as a reasonable conversational target). We operationalize length as Length = 1 - exp(-0.05 × normalized_word_count), where normalized_word_count = word_count / 100. This exponential saturation curve rewards longer texts up to about 100 words, then plateaus [37].

Entropy: measures word choice unpredictability, scoring 0.0+ with higher values indicating more uniform, varied vocabulary distribution. Formula: Entropy = -sum(p_i × log2(p_i)), where p_i is the probability of word i occurring. Higher entropy indicates more uniform word distribution (less predictable), while lower entropy suggests repetitive or predictable word choices [38].

Perplexity: measures sentence structure predictability, scoring 1.0+ with lower values indicating more predictable, simpler structures. Formula: Perplexity = 2^entropy. Lower values indicate more predictable text (simpler structures); higher values suggest complex, less predictable sentence patterns [39].
