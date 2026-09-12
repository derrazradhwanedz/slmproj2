# Trends and patterns to argue in the Discussion

Every number below comes from `results/analysis/patterns.json` and
`results/analysis/supplementary_tests.xlsx` (paired Wilcoxon or exact McNemar,
Benjamini-Hochberg corrected, with bootstrap CIs and rank-biserial effects).

## T1. MGCoT's effect on generation cost is model-dependent, not universal

| Model | Generation time | Output length | Budget exhausted (SCoT -> MGCoT) |
|---|---|---|---|
| gemma2-2b | −49.1% | −52% | 29% -> 13% |
| openchat-7b | −40.8% | −41% | 14% -> 9% |
| qwen2-1.5b | −11.3% | −14% | 12% -> 12% |
| phi3-mini | −0.8% (ns) | −14% | 27% -> 56% |
| llama3.2-1b | +14.1% | −11% | 26% -> 84% |
| mistral-7b | +29.3% | +6% | 19% -> 70% |

Argument: an instruction that specifies *how the answer should look* shortens
generation for models that comply, and lengthens it for models that respond to
the instruction itself instead of the task.

## T2. Two hardware effects hold for all seven models

- System RAM lower under MGCoT (−0.2% to −2.1%), significant on 9-10 of 10 datasets per model.
- GPU utilisation higher under MGCoT (+2.1% to +125.9%).
- GPU memory unchanged (resident model weights dominate).

Caveat to state explicitly: the profiler samples once per streamed token, so
reported time grows with the number of generated tokens; time is a
length-sensitive cost measure, not a pure latency measurement.

## T3. Reference alignment improves for the compliant models

Aggregate changes, significant unless marked: openchat-7b RLF1 +83.2%, TP +41.3%,
TF1 +32.5%, NED +85.3%, CF1 +12.1%, BERT-F1 +2.7% (8/10 datasets);
gemma2-2b RLF1 +129.4%, NED +129.6%, TF1 +24.7%, BERT-F1 +3.7% (10/10 datasets);
qwen2-1.5b TF1 +10.9%, NED +18.7%; phi3-mini NED +56.6%, CF1 +8.0%, BERT-F1 +1.1%.
Degradation for llama3.2-1b (TF1 −62.4%, SS −58.4%) and mistral-7b (TF1 −13.5%, SS −15.8%).

## T4. MGCoT trades recall for precision in every model

Token precision rises where the model complies (+11% to +44%), while token
recall falls for all six valid models (−3.4% to −70.9%). Shorter, more focused
answers cover fewer reference tokens but contain fewer unrelated ones.

## T5. Task accuracy does not follow the alignment metrics

| Model | SCoT | MGCoT | Change [95% CI] | McNemar |
|---|---|---|---|---|
| phi3-mini | 46.2% | 38.0% | −8.2 [−12.8, −3.6] | significant |
| openchat-7b | 43.8% | 44.8% | +1.0 [−3.2, +5.4] | ns |
| gemma2-2b | 40.6% | 34.2% | −6.4 [−10.2, −2.6] | significant |
| mistral-7b | 38.0% | 20.6% | −17.4 [−22.4, −12.8] | significant |
| qwen2-1.5b | 36.4% | 36.6% | +0.2 [−3.4, +3.8] | ns |
| llama3.2-1b | 35.4% | 2.8% | −32.6 [−37.2, −28.2] | significant |

MGCoT is significantly more accurate in 1 of 60 model x dataset conditions
(qwen2-1.5b on CLUTRR, 28% -> 48%) and significantly less accurate in 14.

## T6. Dissociation between surface/semantic alignment and correctness

gemma2-2b is the clearest case: BERTScore up on all ten datasets and ROUGE-L
more than doubled, yet accuracy fell 6.4 points. Answers that look more like the
reference are not more often right. This is the empirical core of the
"fluency is not correctness" argument and it also limits what the eleven
engineered quality metrics can certify.

## T7. The benefit depends on instruction-following capacity

Ordering the models by outcome (openchat-7b, qwen2-1.5b > gemma2-2b > phi3-mini >
mistral-7b > llama3.2-1b) does not follow parameter count (7B, 1.5B, 2B, 3.8B,
7B, 1B). What separates them is whether the model answers the question while
respecting eleven numeric constraints, or spends its budget on the constraints.
llama3.2-1b is the extreme case: accuracy 2.8%, budget exhausted in 84% of runs.

## T8. Reasoning-distilled models are budget-incompatible

deepseek-r1-8b produced no visible answer in 97-98% of runs under either
mechanism: its distilled reasoning trace consumes the 300-token budget. This is
a deployment constraint on reasoning-distilled SLMs under fixed budgets, not
evidence about MGCoT.

## T9. Dataset dependence

Gains concentrate on Date, CLUTRR and GSM8K (and StrategyQA for semantic
similarity); SayCan and QASports are unfavourable for most models. Those two
benchmarks reach the model without task framing (no plausibility question, no
action inventory), so they measure prompt formatting more than reasoning.

## T10. Why the targets push towards brevity

The predictor was trained on question/answer pairs whose reference answers are
short labels or numbers, so the predicted profile of an "adequate answer" is
itself short (median target Length 0.001, Coherence 0.055). MGCoT therefore
inherits a brevity bias from the training targets, which explains both the
efficiency gain and the recall loss.

## Literature needed to argue these points

1. Instruction-following and constraint-following capacity of small/instruction-tuned models.
2. Controlled or constrained text generation with explicit attribute targets.
3. Prompt sensitivity and brittleness of small language models.
4. Weak correlation between overlap/embedding metrics (ROUGE, BERTScore) and task correctness.
5. Reasoning length, token budgets, overthinking and efficient reasoning in LLMs.
6. Reasoning-distilled models (DeepSeek-R1 family) and their token consumption.
7. Chain-of-thought length versus accuracy; concise CoT variants.
8. SLM efficiency and deployment on edge hardware.
