# Evidence map: findings to literature

Each finding from `discussion_points.md` is paired with the retrieved work that
explains or contextualises it. Abstracts are stored in `arxiv_abstracts.json`;
BibTeX in the `*_arxiv.bib` and `*_crossref.bib` files of this folder.

## A. Why MGCoT shortens generation for compliant models (T1, T10)

- **Token-Budget-Aware LLM Reasoning** (Han et al., arXiv:2412.18547, 2024). Reasoning is
  "unnecessarily lengthy" and can be compressed by stating a token budget in the prompt;
  the value of the budget determines how much compression is achieved. MGCoT's predicted
  Length target acts as exactly such an implicit budget, which is why compliant models
  (gemma2-2b −49.1%, openchat-7b −40.8%) shorten their answers.
- **Activation Steering for CoT Compression** (arXiv:2507.04742, 2025). Verbose and concise
  reasoning occupy distinct regions of activation space; steering between them compresses
  reasoning. Supports the claim that brevity can be induced without changing the task.

## B. Why the same instruction lengthens generation for weaker models (T1, T7)

- **From Long to Lean / MACC** (arXiv:2509.22144, 2025). Documents *token elasticity*:
  "overly small token budgets can paradoxically increase output length". This is the
  mechanism behind llama3.2-1b (+14.1% time, budget exhausted in 84% of runs) and
  mistral-7b (+29.3%) under targets that ask for very short answers.
- **The Coupling Tax** (arXiv:2605.07686, 2026). When the reasoning trace and the final
  answer share one output budget, long traces "crowd out the answer they are meant to
  support"; the authors decompose the loss into truncation waste. Explains why truncated
  MGCoT runs lose accuracy: the answer never appears within the budget.

## C. Why some models comply and others do not (T7)

- **Task Competence Is Not Instruction Following** (arXiv:2607.19608, 2026). In small
  models, instruction compliance is a behavioural tendency separable from task competence;
  compliance varies by model rather than by task. Directly supports ordering models by
  compliance rather than by parameter count.
- **MulDimIF** (arXiv:2505.07591, 2025), **LsrIF** (arXiv:2601.06431, 2026) and
  **From Complex to Simple** (2024). Instruction following degrades as the number and
  logical structure of simultaneous constraints grows. MGCoT imposes eleven numeric
  constraints at once, which is the regime these papers identify as hardest.
- **POSIX** (arXiv:2410.02185, 2024) and **Benchmarking Prompt Sensitivity**
  (arXiv:2502.06065, 2025). Small prompt changes produce large output changes, and
  sensitivity is model-specific. Supports treating mechanism choice as model-dependent.

## D. Why alignment metrics improved while accuracy fell (T5, T6)

- **NLG Evaluation Metrics Beyond Correlation Analysis** (arXiv:2305.08566, 2023).
  Task-agnostic metrics (BLEU, BERTScore, perplexity) correlate weakly with human judgement
  and do not reliably discern system quality. This is the published basis for our
  dissociation result: gemma2-2b improved BERTScore on all ten datasets while task
  accuracy fell 6.4 points.
- **Reasoning Efficiently Through Adaptive CoT Compression** (arXiv:2509.14093, 2025).
  "Longer CoT does not always help", and excessive reasoning can cause errors; conversely,
  compression can cost accuracy when it removes necessary steps. Frames the recall-for-
  precision trade (T4) as a length-accuracy trade rather than a quality gain.
- **ConCISE** (arXiv:2505.04881, 2025). Compression applied at the representation level
  preserves accuracy, unlike our prompt-level constraint. Useful contrast for the
  limitation that MGCoT constrains the surface form, not the reasoning process.

## E. Why the ill-posed benchmarks behave differently (T9)

- **Missing Premise exacerbates Overthinking** (arXiv:2504.06514, 2025). For ill-posed
  questions with missing premises, response length increases drastically and thinking
  becomes redundant. QASports (no plausibility question) and SayCan (no action inventory)
  are exactly this case, and are where MGCoT does worst.
- **Revisiting Prompt Sensitivity: The Role of Prompt Underspecification**
  (arXiv:2602.04297, 2026). Much observed prompt sensitivity is attributable to
  underspecified prompts that weakly constrain the output space. Supports reporting those
  two benchmarks separately rather than as reasoning failures.

## F. Why deepseek-r1-8b produced no answers (T8)

- **The Coupling Tax** (arXiv:2605.07686, 2026), as in B: shared budgets truncate answers.
- **Mitigating Overthinking via Manifold Steering** (arXiv:2505.22411, 2025) and
  **Batch Prompting Suppresses Overthinking** (arXiv:2511.04108, 2025). Reasoning models
  "enter recursive self-doubt loops that exhaust token budgets without producing an
  answer". This is precisely deepseek-r1-8b's 97-98% empty-answer rate, and it reframes
  the observation as a known property of reasoning-distilled models rather than a defect
  specific to this study.

## G. Edge-deployment framing (T2)

- **Characterizing Energy Footprint and Efficiency of SLMs on Edge Devices**
  (arXiv:2511.11624, 2025). Power efficiency of Llama 3.2, Phi-3 Mini and Gemma 2 on
  Raspberry Pi 5 and Jetson boards; overlaps three of our seven models and supports
  reporting generation cost as a deployment-relevant quantity.
- **Token Level Routing Inference System for Edge Devices** (arXiv:2505.*, 2025) and
  **CE-LSLM** (arXiv:2505.14085, 2025). Model and token routing between edge and cloud,
  the practical setting in which a mechanism-selection rule (MGCoT vs SCoT per model) is
  actionable.

## Notes on source quality

The Crossref query results are largely unusable for these topics: several entries are
table or figure fragments (DOIs ending in `/table-8`, `/fig-3`) and several are unrelated
summarization papers. The arXiv set is on topic but consists of preprints, which
Reviewer 1 criticised. Before submission, the selected preprints should be checked for
published versions (Crossref `match` by title), and the ones that remain preprints should
be a minority of the added citations.
