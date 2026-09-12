# 3. Results

This section compares Standard Chain-of-Thought (SCoT) and Metric-Guided Chain-of-Thought (MGCoT) on seven small language models (SLMs) across ten reasoning benchmarks. Each model answered the same 50 questions per dataset under both mechanisms, so every comparison is paired at the question level. Tables 1–14 report, for each model, the mean ± standard deviation of each metric per dataset, with the SCoT and MGCoT rows of a dataset placed next to each other; the final pair of rows aggregates all datasets (500 questions per mechanism). In each metric column, the best value across all dataset × mechanism rows is shown in bold and the worst in italics (dark and light grey shading in the accompanying spreadsheets); lower is better for hardware metrics and higher is better for quality metrics.

A difference is described as *significant* when a paired Wilcoxon signed-rank test over the questions of that dataset (or over all 500 questions for the aggregate) remains below 0.05 after Benjamini–Hochberg false-discovery-rate correction within the table. The statements "MGCoT was better/worse on *k* of 10 datasets" follow the two-decimal means shown in the tables.

Two properties of the measurements frame the interpretation. First, the reported generation time grows with the number of generated tokens (the profiler samples resources once per streamed token), so time differences primarily reflect response length. Second, all runs used a 300-token generation budget, well above the length of every reference answer; deepseek-r1-8b spent this budget on its internal reasoning trace in 97–98% of runs and returned an empty visible answer, so its quality scores are close to zero under both mechanisms and are reported for completeness rather than interpreted as answer quality.

## 3.1 Hardware Efficiency

Tables 1–7 report the hardware efficiency metrics defined in the Hardware Efficiency Evaluation Metrics section: Time (min), CPU (%), RAM (GB), RAM (%), PGPU (%) and MGPU (GB).

### 3.1.1 Mistral-7b

MGCoT increased generation time by 29.3% (1.74 ± 0.56 → 2.25 ± 0.70 min), and the increase was significant on all ten datasets, most pronounced on SayCan (+1.10 min) and CLUTRR (+0.99 min). CPU utilisation (+3.9%; 24.34 → 25.28%) and GPU utilisation (+9.9%; 10.90 → 11.98%) were also higher under MGCoT on all ten datasets. The only resource that improved was RAM, which was lower on all ten datasets (12.71 → 12.62 GB, −0.7%; significant on eight). GPU memory was unchanged (3.79 GB), as it is dominated by the resident model weights. The shortest runs were SCoT on ASDiv (1.22 min) and the longest MGCoT on AQUA (2.56 min). Aggregate paired comparisons over all 500 questions (Wilcoxon signed-rank, BH-corrected): Time (min) p < 0.001; CPU (%) p < 0.001; RAM (GB) p < 0.001; RAM (%) p < 0.001; PGPU (%) p < 0.001; MGPU (GB) p = 0.193 (Table 1).

**Table 1.** Hardware efficiency of Mistral-7b under SCoT and MGCoT (mean ± std).

| Dataset | Mechanism | Time (min) | CPU (%) | RAM (GB) | RAM (%) | PGPU (%) | MGPU (GB) |
|:---|:---|---:|---:|---:|---:|---:|---:|
| AQUA | SCoT | 2.46 ± 0.39 | 25.42 ± 4.35 | 13.23 ± 0.73 | 83.53 ± 4.64 | 11.62 ± 1.66 | **3.55 ± 0.22** |
|  | MGCoT | *2.56 ± 0.23* | 26.23 ± 2.74 | 13.16 ± 0.73 | 83.06 ± 4.62 | 11.92 ± 0.44 | 3.58 ± 0.11 |
| ASDiv | SCoT | **1.22 ± 0.36** | 23.35 ± 1.28 | *13.59 ± 0.16* | *85.77 ± 0.98* | 10.82 ± 0.28 | 3.81 ± 0.03 |
|  | MGCoT | 2.00 ± 0.82 | 24.47 ± 1.00 | 13.49 ± 0.14 | 85.17 ± 0.87 | 12.07 ± 1.51 | 3.81 ± 0.00 |
| CLUTRR | SCoT | 1.35 ± 0.31 | 29.56 ± 4.05 | 13.36 ± 0.57 | 84.32 ± 3.59 | 11.55 ± 0.50 | 3.81 ± 0.00 |
|  | MGCoT | 2.34 ± 0.60 | *31.09 ± 4.57* | 13.29 ± 0.63 | 83.90 ± 3.97 | 12.11 ± 0.67 | 3.81 ± 0.00 |
| Date | SCoT | 1.63 ± 0.51 | 26.32 ± 4.39 | 11.77 ± 1.57 | 74.28 ± 9.89 | 10.35 ± 0.45 | 3.81 ± 0.00 |
|  | MGCoT | 2.01 ± 0.82 | 26.46 ± 3.96 | 11.65 ± 1.57 | 73.53 ± 9.91 | *12.51 ± 3.39* | *3.81 ± 0.00* |
| GSM8K | SCoT | 2.22 ± 0.36 | 22.15 ± 0.78 | 11.56 ± 0.72 | 72.95 ± 4.55 | 11.20 ± 0.13 | 3.81 ± 0.00 |
|  | MGCoT | 2.50 ± 0.35 | 23.47 ± 0.80 | **11.50 ± 0.72** | **72.61 ± 4.58** | 11.70 ± 0.16 | 3.81 ± 0.00 |
| MultiArith | SCoT | 1.60 ± 0.35 | 21.97 ± 0.60 | 12.91 ± 0.15 | 81.52 ± 0.97 | 11.06 ± 0.15 | 3.81 ± 0.00 |
|  | MGCoT | 1.91 ± 0.94 | 22.83 ± 0.94 | 12.84 ± 0.15 | 81.06 ± 0.96 | 11.93 ± 0.56 | 3.81 ± 0.00 |
| QASports | SCoT | 1.83 ± 0.55 | 22.26 ± 0.72 | 12.95 ± 0.26 | 81.76 ± 1.62 | 10.52 ± 0.47 | 3.81 ± 0.00 |
|  | MGCoT | 2.16 ± 0.79 | 23.20 ± 1.21 | 12.84 ± 0.27 | 81.07 ± 1.72 | 11.89 ± 1.37 | 3.81 ± 0.00 |
| SayCan | SCoT | 1.42 ± 0.42 | 22.84 ± 2.67 | 12.53 ± 0.74 | 79.10 ± 4.68 | **10.18 ± 0.44** | 3.81 ± 0.00 |
|  | MGCoT | 2.52 ± 0.16 | 24.01 ± 2.40 | 12.41 ± 0.73 | 78.30 ± 4.58 | 11.65 ± 0.30 | 3.81 ± 0.00 |
| StrategyQA | SCoT | 1.98 ± 0.42 | **20.99 ± 0.51** | 11.86 ± 0.27 | 74.86 ± 1.70 | 10.51 ± 0.40 | 3.81 ± 0.00 |
|  | MGCoT | 2.49 ± 0.40 | 22.18 ± 0.88 | 11.79 ± 0.27 | 74.41 ± 1.73 | 11.60 ± 0.21 | 3.81 ± 0.00 |
| SVAMP | SCoT | 1.71 ± 0.52 | 28.54 ± 9.22 | 13.31 ± 0.74 | 84.01 ± 4.67 | 11.23 ± 0.33 | 3.81 ± 0.00 |
|  | MGCoT | 2.02 ± 0.86 | 28.87 ± 6.88 | 13.26 ± 0.74 | 83.71 ± 4.66 | 12.45 ± 2.82 | 3.81 ± 0.00 |
| **All datasets** | SCoT | 1.74 ± 0.56 | 24.34 ± 4.76 | 12.71 ± 1.00 | 80.21 ± 6.29 | 10.90 ± 0.79 | 3.79 ± 0.11 |
|  | MGCoT | 2.25 ± 0.70 | 25.28 ± 4.18 | 12.62 ± 1.00 | 79.68 ± 6.33 | 11.98 ± 1.59 | 3.79 ± 0.08 |

### 3.1.2 Phi3-mini

Aggregate generation time was equivalent under the two mechanisms (1.89 vs 1.88 min, −0.8%, not significant), but the balance differed by dataset: MGCoT was significantly faster on Date (−1.11 min, the fastest cell at 0.51 min) and significantly slower on ASDiv, CLUTRR, SayCan and StrategyQA (the slowest cell, 2.47 min). MGCoT times were more variable on every dataset (higher standard deviation on 10 of 10), indicating that response length depended more on the question. MGCoT produced the largest RAM reduction of all models (13.33 → 13.05 GB, −2.1%, significant on all ten datasets), at the cost of higher CPU (+13.2%; 12.69 → 14.37%) and GPU utilisation (+44.8%; 5.60 → 8.11%), both higher on all ten datasets. Aggregate paired comparisons over all 500 questions (Wilcoxon signed-rank, BH-corrected): Time (min) p = 0.511; CPU (%) p < 0.001; RAM (GB) p < 0.001; RAM (%) p < 0.001; PGPU (%) p < 0.001; MGPU (GB) p = 0.099 (Table 2).

**Table 2.** Hardware efficiency of Phi3-mini under SCoT and MGCoT (mean ± std).

| Dataset | Mechanism | Time (min) | CPU (%) | RAM (GB) | RAM (%) | PGPU (%) | MGPU (GB) |
|:---|:---|---:|---:|---:|---:|---:|---:|
| AQUA | SCoT | 2.40 ± 0.32 | 15.60 ± 4.66 | 12.96 ± 1.30 | 81.78 ± 8.21 | 6.37 ± 0.75 | **3.59 ± 0.15** |
|  | MGCoT | 2.06 ± 0.76 | 18.44 ± 6.47 | **12.64 ± 1.29** | **79.77 ± 8.16** | 9.62 ± 7.41 | 3.61 ± 0.00 |
| ASDiv | SCoT | 1.46 ± 0.37 | 10.98 ± 0.61 | 13.33 ± 0.14 | 84.17 ± 0.91 | 5.42 ± 0.23 | 3.82 ± 0.03 |
|  | MGCoT | 1.84 ± 0.86 | 12.36 ± 0.46 | 13.07 ± 0.14 | 82.50 ± 0.91 | 7.19 ± 0.92 | 3.82 ± 0.00 |
| CLUTRR | SCoT | 1.44 ± 0.33 | 10.88 ± 0.43 | 13.56 ± 0.27 | 85.60 ± 1.68 | 5.54 ± 0.17 | 3.82 ± 0.00 |
|  | MGCoT | 2.14 ± 0.73 | 12.57 ± 0.56 | 13.30 ± 0.25 | 83.98 ± 1.57 | 6.98 ± 0.43 | 3.82 ± 0.00 |
| Date | SCoT | 1.62 ± 0.39 | 10.97 ± 0.41 | 12.97 ± 0.11 | 81.85 ± 0.67 | **5.09 ± 0.17** | 3.82 ± 0.00 |
|  | MGCoT | **0.51 ± 0.53** | 12.40 ± 1.22 | 12.78 ± 0.10 | 80.66 ± 0.63 | *13.47 ± 6.98* | 3.82 ± 0.00 |
| GSM8K | SCoT | 2.23 ± 0.32 | 12.19 ± 2.25 | 13.39 ± 0.51 | 84.51 ± 3.21 | 5.63 ± 0.15 | 3.82 ± 0.00 |
|  | MGCoT | 1.99 ± 0.69 | 13.34 ± 0.95 | 13.09 ± 0.48 | 82.61 ± 3.02 | 7.09 ± 0.36 | 3.82 ± 0.00 |
| MultiArith | SCoT | 1.58 ± 0.20 | 12.11 ± 0.93 | 13.50 ± 0.19 | 85.20 ± 1.23 | 5.51 ± 0.13 | 3.82 ± 0.00 |
|  | MGCoT | 1.48 ± 0.96 | 13.29 ± 1.31 | 13.25 ± 0.17 | 83.64 ± 1.08 | 7.62 ± 0.85 | 3.82 ± 0.00 |
| QASports | SCoT | 2.06 ± 0.44 | 14.74 ± 4.94 | 13.64 ± 0.32 | 86.13 ± 2.02 | 5.65 ± 0.73 | 3.82 ± 0.00 |
|  | MGCoT | 2.16 ± 0.66 | 16.28 ± 5.88 | 13.38 ± 0.32 | 84.43 ± 2.01 | 7.17 ± 0.75 | 3.82 ± 0.00 |
| SayCan | SCoT | 1.98 ± 0.49 | 17.52 ± 3.73 | 13.12 ± 0.93 | 82.83 ± 5.84 | 5.87 ± 0.59 | 3.82 ± 0.00 |
|  | MGCoT | 2.37 ± 0.54 | *19.79 ± 3.99* | 12.71 ± 1.00 | 80.26 ± 6.32 | 7.97 ± 1.22 | 3.82 ± 0.00 |
| StrategyQA | SCoT | 2.40 ± 0.23 | 11.05 ± 0.53 | 12.97 ± 1.30 | 81.86 ± 8.19 | 5.40 ± 0.19 | 3.82 ± 0.00 |
|  | MGCoT | *2.47 ± 0.36* | 12.65 ± 0.80 | 12.70 ± 1.27 | 80.16 ± 8.02 | 6.65 ± 0.26 | 3.82 ± 0.00 |
| SVAMP | SCoT | 1.73 ± 0.48 | **10.82 ± 0.34** | *13.84 ± 0.24* | *87.37 ± 1.54* | 5.50 ± 0.17 | 3.82 ± 0.00 |
|  | MGCoT | 1.73 ± 0.89 | 12.53 ± 0.53 | 13.58 ± 0.20 | 85.71 ± 1.28 | 7.31 ± 0.88 | 3.82 ± 0.00 |
| **All datasets** | SCoT | 1.89 ± 0.51 | 12.69 ± 3.43 | 13.33 ± 0.75 | 84.13 ± 4.72 | 5.60 ± 0.52 | 3.80 ± 0.08 |
|  | MGCoT | 1.87 ± 0.89 | 14.36 ± 4.07 | 13.05 ± 0.75 | 82.37 ± 4.75 | 8.11 ± 3.80 | 3.80 ± 0.06 |

### 3.1.3 Llama3.2-1b

MGCoT increased generation time by 14.1% (1.92 → 2.19 min), significantly on seven datasets, with the largest increases on ASDiv (+1.12 min) and CLUTRR (+0.98 min). The exception was QASports, where MGCoT was significantly faster (−1.41 min) and produced the fastest cell of the table (0.76 min). CPU utilisation was unchanged (5.11 vs 5.12%, not significant), and RAM was slightly lower on nine datasets (9.72 → 9.69 GB, −0.3%). GPU utilisation rose on aggregate (+43.4%; 2.47 → 3.55%) but not uniformly: it was significantly lower on ASDiv, CLUTRR and MultiArith and higher on the remaining datasets, most on QASports (+4.9 points). GPU memory was constant at 3.05 GB. Aggregate paired comparisons over all 500 questions (Wilcoxon signed-rank, BH-corrected): Time (min) p < 0.001; CPU (%) p = 0.917; RAM (GB) p < 0.001; RAM (%) p < 0.001; PGPU (%) p = 0.018; MGPU (GB) p = 0.825 (Table 3).

**Table 3.** Hardware efficiency of Llama3.2-1b under SCoT and MGCoT (mean ± std).

| Dataset | Mechanism | Time (min) | CPU (%) | RAM (GB) | RAM (%) | PGPU (%) | MGPU (GB) |
|:---|:---|---:|---:|---:|---:|---:|---:|
| AQUA | SCoT | 2.38 ± 0.27 | 4.18 ± 0.51 | 9.94 ± 0.32 | 62.76 ± 2.05 | 2.40 ± 0.14 | **3.05 ± 0.00** |
|  | MGCoT | 2.44 ± 0.42 | 4.18 ± 0.42 | 9.93 ± 0.32 | 62.70 ± 2.01 | 2.65 ± 0.85 | 3.05 ± 0.00 |
| ASDiv | SCoT | 1.41 ± 0.27 | 4.22 ± 0.47 | 10.25 ± 0.50 | 64.70 ± 3.17 | 2.65 ± 0.20 | 3.05 ± 0.00 |
|  | MGCoT | 2.53 ± 0.00 | 4.18 ± 0.60 | 10.21 ± 0.50 | 64.44 ± 3.16 | 2.43 ± 0.06 | 3.05 ± 0.00 |
| CLUTRR | SCoT | 1.46 ± 0.45 | 4.10 ± 0.21 | 9.53 ± 0.09 | 60.16 ± 0.55 | 2.73 ± 0.28 | 3.05 ± 0.00 |
|  | MGCoT | 2.44 ± 0.45 | 4.14 ± 0.25 | 9.51 ± 0.08 | 60.05 ± 0.52 | 2.69 ± 1.15 | 3.05 ± 0.00 |
| Date | SCoT | 1.65 ± 0.47 | *9.83 ± 4.84* | *12.84 ± 0.67* | *81.03 ± 4.21* | 2.53 ± 0.25 | 3.05 ± 0.00 |
|  | MGCoT | 2.47 ± 0.33 | 9.59 ± 3.87 | 12.77 ± 0.58 | 80.64 ± 3.69 | 2.59 ± 1.30 | *3.05 ± 0.00* |
| GSM8K | SCoT | 2.12 ± 0.38 | 7.63 ± 4.26 | 10.41 ± 2.63 | 65.72 ± 16.58 | 2.43 ± 0.15 | 3.05 ± 0.00 |
|  | MGCoT | 2.53 ± 0.00 | 7.79 ± 4.18 | 10.32 ± 2.63 | 65.17 ± 16.58 | 2.47 ± 0.06 | 3.05 ± 0.00 |
| MultiArith | SCoT | 1.59 ± 0.33 | 4.13 ± 0.36 | 8.53 ± 0.10 | 53.82 ± 0.62 | 2.58 ± 0.18 | 3.05 ± 0.00 |
|  | MGCoT | 2.53 ± 0.00 | **4.09 ± 0.34** | **8.51 ± 0.10** | **53.70 ± 0.63** | 2.42 ± 0.05 | 3.05 ± 0.00 |
| QASports | SCoT | 2.18 ± 0.40 | 4.15 ± 0.20 | 8.73 ± 0.05 | 55.12 ± 0.30 | 2.27 ± 0.14 | 3.05 ± 0.00 |
|  | MGCoT | **0.76 ± 0.95** | 4.38 ± 0.95 | 8.73 ± 0.06 | 55.12 ± 0.35 | *7.12 ± 3.22* | 3.05 ± 0.00 |
| SayCan | SCoT | 2.08 ± 0.44 | 4.37 ± 0.82 | 8.93 ± 0.14 | 56.36 ± 0.90 | 2.37 ± 0.19 | 3.05 ± 0.00 |
|  | MGCoT | 1.72 ± 1.12 | 4.55 ± 1.50 | 8.91 ± 0.15 | 56.24 ± 0.92 | 6.29 ± 6.91 | 3.05 ± 0.00 |
| StrategyQA | SCoT | 2.47 ± 0.17 | 4.30 ± 0.47 | 9.05 ± 0.15 | 57.14 ± 0.92 | **2.26 ± 0.09** | 3.05 ± 0.00 |
|  | MGCoT | 1.94 ± 1.01 | 4.20 ± 0.44 | 9.03 ± 0.15 | 57.02 ± 0.93 | 4.35 ± 3.97 | 3.05 ± 0.00 |
| SVAMP | SCoT | 1.86 ± 0.47 | 4.17 ± 0.37 | 9.03 ± 0.15 | 56.97 ± 0.95 | 2.51 ± 0.21 | 3.05 ± 0.00 |
|  | MGCoT | *2.53 ± 0.00* | 4.11 ± 0.28 | 9.01 ± 0.14 | 56.84 ± 0.89 | 2.44 ± 0.06 | 3.05 ± 0.00 |
| **All datasets** | SCoT | 1.92 ± 0.52 | 5.11 ± 2.79 | 9.72 ± 1.49 | 61.38 ± 9.40 | 2.47 ± 0.24 | 3.05 ± 0.00 |
|  | MGCoT | 2.19 ± 0.81 | 5.12 ± 2.64 | 9.69 ± 1.47 | 61.19 ± 9.29 | 3.54 ± 3.23 | 3.05 ± 0.00 |

### 3.1.4 Gemma2-2b

Gemma2-2b showed the largest efficiency gain of the study. MGCoT halved generation time (1.98 ± 0.48 → 1.01 ± 0.84 min, −49.1%), with faster generation on all ten datasets (significant on nine); the largest reductions were on QASports (−1.75 min) and SayCan (−1.59 min), and the fastest cell was MGCoT on CLUTRR (0.21 min) against the slowest SCoT on StrategyQA (2.45 min). RAM was lower on all ten datasets (11.74 → 11.68 GB). In exchange, GPU utilisation more than doubled (3.88 → 8.76%, +125.9%, significant on all datasets; +14.4 points on CLUTRR) and CPU rose slightly (+4.0%; 5.21 → 5.42%). MGCoT time was more variable (standard deviation 0.84 vs 0.48 min), consistent with response lengths that adapt to the question rather than a uniformly long reasoning chain. Aggregate paired comparisons over all 500 questions (Wilcoxon signed-rank, BH-corrected): Time (min) p < 0.001; CPU (%) p < 0.001; RAM (GB) p < 0.001; RAM (%) p < 0.001; PGPU (%) p < 0.001; MGPU (GB) p = 0.566 (Table 4).

**Table 4.** Hardware efficiency of Gemma2-2b under SCoT and MGCoT (mean ± std).

| Dataset | Mechanism | Time (min) | CPU (%) | RAM (GB) | RAM (%) | PGPU (%) | MGPU (GB) |
|:---|:---|---:|---:|---:|---:|---:|---:|
| AQUA | SCoT | 2.40 ± 0.29 | 10.66 ± 2.27 | 13.12 ± 0.51 | 82.81 ± 3.24 | 3.82 ± 0.15 | **3.41 ± 0.15** |
|  | MGCoT | 2.18 ± 0.52 | *11.00 ± 2.29* | 13.03 ± 0.43 | 82.24 ± 2.72 | 4.27 ± 0.47 | 3.43 ± 0.00 |
| ASDiv | SCoT | 1.34 ± 0.28 | 6.18 ± 3.60 | 9.58 ± 1.47 | 60.47 ± 9.28 | 4.18 ± 0.21 | 3.64 ± 0.03 |
|  | MGCoT | 0.84 ± 0.75 | 6.51 ± 5.02 | 9.52 ± 1.49 | 60.12 ± 9.44 | 8.64 ± 5.52 | 3.64 ± 0.00 |
| CLUTRR | SCoT | 1.58 ± 0.32 | 4.30 ± 0.35 | 9.24 ± 0.15 | 58.35 ± 0.92 | 4.05 ± 0.20 | 3.64 ± 0.00 |
|  | MGCoT | **0.21 ± 0.20** | 4.84 ± 1.05 | **9.21 ± 0.14** | **58.14 ± 0.90** | *18.45 ± 5.82* | 3.64 ± 0.00 |
| Date | SCoT | 1.75 ± 0.21 | **4.24 ± 0.23** | 9.64 ± 0.07 | 60.88 ± 0.41 | 3.91 ± 0.14 | 3.64 ± 0.00 |
|  | MGCoT | 0.34 ± 0.36 | 4.53 ± 0.44 | 9.61 ± 0.07 | 60.66 ± 0.47 | 13.84 ± 5.44 | 3.64 ± 0.00 |
| GSM8K | SCoT | 2.31 ± 0.34 | 4.32 ± 0.23 | 10.99 ± 0.71 | 69.40 ± 4.50 | 3.78 ± 0.14 | 3.64 ± 0.00 |
|  | MGCoT | 1.52 ± 0.57 | 4.39 ± 0.54 | 10.95 ± 0.71 | 69.12 ± 4.50 | 4.79 ± 0.80 | 3.64 ± 0.00 |
| MultiArith | SCoT | 1.61 ± 0.24 | 4.38 ± 0.30 | 12.50 ± 0.15 | 78.87 ± 0.93 | 4.02 ± 0.18 | *3.64 ± 0.00* |
|  | MGCoT | 0.92 ± 0.68 | 4.38 ± 0.45 | 12.45 ± 0.15 | 78.59 ± 0.95 | 6.24 ± 1.39 | 3.64 ± 0.00 |
| QASports | SCoT | 2.27 ± 0.37 | 4.60 ± 1.12 | *13.42 ± 0.35* | *84.71 ± 2.22* | 3.71 ± 0.18 | 3.64 ± 0.00 |
|  | MGCoT | 0.52 ± 0.63 | 4.68 ± 1.42 | 13.38 ± 0.35 | 84.48 ± 2.22 | 12.15 ± 5.38 | 3.64 ± 0.00 |
| SayCan | SCoT | 2.20 ± 0.38 | 4.45 ± 0.52 | 13.33 ± 0.38 | 84.16 ± 2.40 | 3.73 ± 0.16 | 3.64 ± 0.00 |
|  | MGCoT | 0.61 ± 0.49 | 4.51 ± 0.58 | 13.28 ± 0.37 | 83.85 ± 2.35 | 8.61 ± 3.77 | 3.64 ± 0.00 |
| StrategyQA | SCoT | *2.45 ± 0.19* | 4.70 ± 1.86 | 12.56 ± 0.83 | 79.28 ± 5.26 | **3.66 ± 0.09** | 3.64 ± 0.00 |
|  | MGCoT | 1.83 ± 0.70 | 4.83 ± 2.35 | 12.48 ± 0.86 | 78.80 ± 5.44 | 4.61 ± 0.72 | 3.64 ± 0.00 |
| SVAMP | SCoT | 1.85 ± 0.39 | 4.25 ± 0.38 | 12.96 ± 0.24 | 81.81 ± 1.54 | 3.93 ± 0.19 | 3.64 ± 0.00 |
|  | MGCoT | 1.08 ± 0.61 | 4.51 ± 1.04 | 12.91 ± 0.28 | 81.48 ± 1.79 | 5.99 ± 2.47 | 3.64 ± 0.00 |
| **All datasets** | SCoT | 1.98 ± 0.48 | 5.21 ± 2.43 | 11.73 ± 1.73 | 74.08 ± 10.90 | 3.88 ± 0.23 | 3.62 ± 0.09 |
|  | MGCoT | 1.01 ± 0.84 | 5.42 ± 2.81 | 11.68 ± 1.72 | 73.75 ± 10.87 | 8.76 ± 5.86 | 3.62 ± 0.06 |

### 3.1.5 Qwen2-1.5b

MGCoT reduced generation time by 11.3% (1.13 → 1.00 min), with shorter times on eight datasets, significantly on ASDiv (−0.38 min) and Date (−0.28 min); QASports was the only dataset with a significant increase (+0.31 min). Qwen2-1.5b was the only model whose CPU utilisation decreased under MGCoT (5.24 → 5.05%, −3.5%; lower on seven datasets, significantly on QASports and SayCan). RAM was lower on all datasets (9.33 → 9.31 GB), whereas GPU utilisation increased (+54.7%; 4.42 → 6.84%), significantly on nine datasets. Qwen2-1.5b also had the smallest GPU memory footprint of all models (2.66 GB). Aggregate paired comparisons over all 500 questions (Wilcoxon signed-rank, BH-corrected): Time (min) p = 0.004; CPU (%) p = 0.007; RAM (GB) p < 0.001; RAM (%) p < 0.001; PGPU (%) p < 0.001; MGPU (GB) p = 0.384 (Table 5).

**Table 5.** Hardware efficiency of Qwen2-1.5b under SCoT and MGCoT (mean ± std).

| Dataset | Mechanism | Time (min) | CPU (%) | RAM (GB) | RAM (%) | PGPU (%) | MGPU (GB) |
|:---|:---|---:|---:|---:|---:|---:|---:|
| AQUA | SCoT | *2.06 ± 0.63* | *10.20 ± 3.62* | *12.28 ± 1.98* | *77.50 ± 12.52* | 2.53 ± 0.70 | **2.45 ± 0.15** |
|  | MGCoT | 1.83 ± 0.77 | 9.90 ± 3.33 | 12.19 ± 1.96 | 76.94 ± 12.35 | 3.64 ± 3.13 | 2.47 ± 0.01 |
| ASDiv | SCoT | 0.84 ± 0.47 | 4.82 ± 1.92 | 8.41 ± 0.10 | 53.09 ± 0.61 | 3.74 ± 2.21 | 2.68 ± 0.03 |
|  | MGCoT | 0.45 ± 0.41 | 4.65 ± 0.76 | **8.40 ± 0.12** | **53.06 ± 0.77** | 9.92 ± 6.96 | 2.68 ± 0.00 |
| CLUTRR | SCoT | 0.26 ± 0.29 | 5.50 ± 2.27 | 8.54 ± 0.11 | 53.89 ± 0.73 | 10.57 ± 5.18 | 2.68 ± 0.00 |
|  | MGCoT | **0.14 ± 0.05** | 4.87 ± 1.05 | 8.53 ± 0.12 | 53.84 ± 0.77 | *16.88 ± 4.68* | 2.68 ± 0.00 |
| Date | SCoT | 0.89 ± 0.56 | 4.40 ± 0.50 | 8.45 ± 0.05 | 53.33 ± 0.32 | 3.69 ± 2.24 | 2.68 ± 0.00 |
|  | MGCoT | 0.60 ± 0.40 | 4.61 ± 0.98 | 8.43 ± 0.05 | 53.23 ± 0.33 | 6.70 ± 5.01 | 2.68 ± 0.00 |
| GSM8K | SCoT | 1.86 ± 0.55 | 4.25 ± 0.36 | 8.74 ± 0.15 | 55.16 ± 0.92 | **2.47 ± 0.33** | 2.68 ± 0.00 |
|  | MGCoT | 1.85 ± 0.61 | **4.20 ± 0.34** | 8.73 ± 0.14 | 55.09 ± 0.89 | 2.88 ± 0.74 | 2.68 ± 0.00 |
| MultiArith | SCoT | 1.04 ± 0.22 | 4.31 ± 0.35 | 9.02 ± 0.04 | 56.94 ± 0.26 | 2.85 ± 0.22 | 2.68 ± 0.00 |
|  | MGCoT | 0.89 ± 0.31 | 4.26 ± 0.52 | 9.01 ± 0.04 | 56.85 ± 0.28 | 4.08 ± 2.22 | 2.68 ± 0.00 |
| QASports | SCoT | 0.53 ± 0.63 | 4.85 ± 0.90 | 9.10 ± 0.06 | 57.46 ± 0.40 | 6.05 ± 3.25 | 2.68 ± 0.00 |
|  | MGCoT | 0.84 ± 0.78 | 4.58 ± 1.22 | 9.09 ± 0.06 | 57.40 ± 0.40 | 5.38 ± 2.98 | 2.68 ± 0.00 |
| SayCan | SCoT | 1.01 ± 1.00 | 5.43 ± 1.94 | 9.41 ± 0.12 | 59.38 ± 0.77 | 6.38 ± 4.57 | 2.68 ± 0.00 |
|  | MGCoT | 0.77 ± 0.93 | 4.70 ± 1.20 | 9.38 ± 0.11 | 59.24 ± 0.71 | 10.57 ± 8.67 | 2.68 ± 0.00 |
| StrategyQA | SCoT | 1.59 ± 0.76 | 4.33 ± 0.36 | 9.58 ± 0.26 | 60.47 ± 1.62 | 3.19 ± 2.56 | 2.68 ± 0.00 |
|  | MGCoT | 1.60 ± 0.79 | 4.39 ± 0.77 | 9.57 ± 0.22 | 60.39 ± 1.39 | 3.48 ± 2.58 | 2.68 ± 0.00 |
| SVAMP | SCoT | 1.24 ± 0.52 | 4.26 ± 0.43 | 9.79 ± 0.07 | 61.78 ± 0.42 | 2.76 ± 0.41 | 2.68 ± 0.00 |
|  | MGCoT | 1.05 ± 0.61 | 4.40 ± 0.45 | 9.79 ± 0.08 | 61.77 ± 0.50 | 4.89 ± 3.97 | 2.68 ± 0.00 |
| **All datasets** | SCoT | 1.13 ± 0.80 | 5.24 ± 2.37 | 9.33 ± 1.25 | 58.90 ± 7.92 | 4.42 ± 3.66 | 2.66 ± 0.09 |
|  | MGCoT | 1.00 ± 0.83 | 5.05 ± 2.10 | 9.31 ± 1.23 | 58.78 ± 7.77 | 6.84 ± 6.23 | 2.66 ± 0.06 |

### 3.1.6 Openchat-7b

MGCoT reduced generation time by 40.8% (1.76 → 1.05 min), and the reduction was significant on all ten datasets, largest on Date (−1.33 min) and CLUTRR (−0.99 min); the fastest cell was MGCoT on CLUTRR (0.24 min) and the slowest SCoT on AQUA (2.34 min). RAM was significantly lower on all ten datasets (12.42 → 12.34 GB, −0.6%). CPU utilisation increased on aggregate (+4.1%; 21.81 → 22.70%) but decreased significantly on CLUTRR and Date, the two datasets with the largest time savings. GPU utilisation was higher on all datasets (9.95 → 13.41%, +34.8%). Aggregate paired comparisons over all 500 questions (Wilcoxon signed-rank, BH-corrected): Time (min) p < 0.001; CPU (%) p < 0.001; RAM (GB) p < 0.001; RAM (%) p < 0.001; PGPU (%) p < 0.001; MGPU (GB) p = 0.672 (Table 6).

**Table 6.** Hardware efficiency of Openchat-7b under SCoT and MGCoT (mean ± std).

| Dataset | Mechanism | Time (min) | CPU (%) | RAM (GB) | RAM (%) | PGPU (%) | MGPU (GB) |
|:---|:---|---:|---:|---:|---:|---:|---:|
| AQUA | SCoT | *2.34 ± 0.38* | 28.26 ± 3.96 | 13.07 ± 1.52 | 82.53 ± 9.58 | 12.13 ± 1.13 | **3.58 ± 0.15** |
|  | MGCoT | 2.07 ± 0.60 | 29.64 ± 4.23 | 12.91 ± 1.53 | 81.52 ± 9.68 | 13.12 ± 0.38 | 3.60 ± 0.00 |
| ASDiv | SCoT | 1.47 ± 0.30 | 19.94 ± 0.59 | 10.86 ± 0.17 | 68.58 ± 1.08 | 9.71 ± 0.31 | 3.81 ± 0.03 |
|  | MGCoT | 0.85 ± 0.58 | 20.87 ± 1.66 | **10.79 ± 0.17** | **68.12 ± 1.06** | 13.76 ± 3.93 | 3.81 ± 0.00 |
| CLUTRR | SCoT | 1.23 ± 0.29 | 19.83 ± 0.60 | 11.11 ± 0.10 | 70.12 ± 0.65 | 9.90 ± 0.40 | 3.81 ± 0.00 |
|  | MGCoT | **0.24 ± 0.08** | **18.84 ± 0.91** | 11.05 ± 0.11 | 69.73 ± 0.68 | *18.25 ± 4.87* | 3.81 ± 0.00 |
| Date | SCoT | 1.75 ± 0.40 | 20.09 ± 0.52 | 11.14 ± 0.08 | 70.35 ± 0.53 | 9.23 ± 0.45 | 3.81 ± 0.00 |
|  | MGCoT | 0.42 ± 0.47 | 19.72 ± 1.08 | 11.10 ± 0.09 | 70.06 ± 0.60 | 12.87 ± 3.68 | 3.81 ± 0.00 |
| GSM8K | SCoT | 2.26 ± 0.32 | 20.27 ± 0.41 | 12.32 ± 0.61 | 77.78 ± 3.83 | 9.96 ± 0.27 | 3.81 ± 0.00 |
|  | MGCoT | 1.93 ± 0.56 | 22.54 ± 0.61 | 12.23 ± 0.59 | 77.22 ± 3.75 | 12.84 ± 0.34 | 3.81 ± 0.00 |
| MultiArith | SCoT | 1.64 ± 0.20 | 20.00 ± 0.45 | 12.95 ± 0.11 | 81.76 ± 0.67 | 9.76 ± 0.25 | 3.81 ± 0.00 |
|  | MGCoT | 0.86 ± 0.36 | 21.20 ± 1.03 | 12.89 ± 0.12 | 81.38 ± 0.73 | 12.29 ± 0.60 | 3.81 ± 0.00 |
| QASports | SCoT | 1.61 ± 0.38 | 20.17 ± 0.42 | 13.00 ± 0.13 | 82.06 ± 0.79 | 9.02 ± 0.42 | 3.81 ± 0.00 |
|  | MGCoT | 0.64 ± 0.23 | 20.94 ± 1.30 | 12.93 ± 0.14 | 81.62 ± 0.89 | 12.27 ± 1.47 | 3.81 ± 0.00 |
| SayCan | SCoT | 1.46 ± 0.44 | 20.42 ± 1.76 | 12.97 ± 0.17 | 81.84 ± 1.05 | **8.97 ± 0.66** | 3.81 ± 0.00 |
|  | MGCoT | 0.98 ± 0.81 | 21.01 ± 1.75 | 12.88 ± 0.17 | 81.32 ± 1.06 | 13.11 ± 2.12 | 3.81 ± 0.00 |
| StrategyQA | SCoT | 2.16 ± 0.38 | 20.33 ± 0.62 | 13.20 ± 0.14 | 83.35 ± 0.89 | 9.52 ± 0.47 | 3.81 ± 0.00 |
|  | MGCoT | 1.32 ± 0.57 | 21.94 ± 0.85 | 13.13 ± 0.15 | 82.86 ± 0.97 | 12.45 ± 0.45 | 3.81 ± 0.00 |
| SVAMP | SCoT | 1.73 ± 0.35 | 28.83 ± 8.10 | *13.52 ± 0.42* | *85.35 ± 2.67* | 11.27 ± 1.25 | 3.81 ± 0.00 |
|  | MGCoT | 1.14 ± 0.57 | *30.29 ± 8.16* | 13.45 ± 0.35 | 84.90 ± 2.24 | 13.11 ± 1.36 | *3.81 ± 0.00* |
| **All datasets** | SCoT | 1.76 ± 0.49 | 21.81 ± 4.46 | 12.42 ± 1.09 | 78.37 ± 6.88 | 9.95 ± 1.15 | 3.79 ± 0.08 |
|  | MGCoT | 1.04 ± 0.77 | 22.70 ± 4.86 | 12.34 ± 1.08 | 77.87 ± 6.82 | 13.41 ± 2.98 | 3.79 ± 0.06 |

### 3.1.7 Deepseek-r1-8b

Generation time was nearly constant under both mechanisms (2.56 ± 0.04 vs 2.58 ± 0.07 min) because almost every run reached the 300-token limit; the statistically significant MGCoT increase on nine datasets amounts to at most 0.04 min. Deepseek-r1-8b had the highest CPU utilisation (30.12 → 31.74%, +5.4%) and RAM usage (13.73 → 13.65 GB) of all models. As for the other models, MGCoT slightly lowered RAM (all ten datasets) and slightly raised GPU utilisation (+2.1%). Aggregate paired comparisons over all 500 questions (Wilcoxon signed-rank, BH-corrected): Time (min) p < 0.001; CPU (%) p < 0.001; RAM (GB) p < 0.001; RAM (%) p < 0.001; PGPU (%) p < 0.001; MGPU (GB) p = 0.051 (Table 7).

**Table 7.** Hardware efficiency of Deepseek-r1-8b under SCoT and MGCoT (mean ± std).

| Dataset | Mechanism | Time (min) | CPU (%) | RAM (GB) | RAM (%) | PGPU (%) | MGPU (GB) |
|:---|:---|---:|---:|---:|---:|---:|---:|
| AQUA | SCoT | 2.58 ± 0.11 | 37.33 ± 5.12 | 13.70 ± 1.02 | 86.48 ± 6.46 | 11.77 ± 0.28 | 3.80 ± 0.00 |
|  | MGCoT | *2.61 ± 0.02* | 39.04 ± 6.41 | 13.60 ± 1.02 | 85.86 ± 6.43 | 12.16 ± 0.34 | 3.80 ± 0.00 |
| ASDiv | SCoT | 2.55 ± 0.00 | 26.72 ± 0.40 | 12.99 ± 0.75 | 82.03 ± 4.70 | 11.70 ± 0.14 | 3.82 ± 0.00 |
|  | MGCoT | 2.55 ± 0.16 | 28.03 ± 0.48 | 12.91 ± 0.74 | 81.49 ± 4.67 | 11.84 ± 0.18 | 3.82 ± 0.00 |
| CLUTRR | SCoT | 2.56 ± 0.00 | 26.75 ± 0.41 | 13.55 ± 0.48 | 85.55 ± 3.03 | 11.69 ± 0.10 | 3.82 ± 0.00 |
|  | MGCoT | 2.57 ± 0.10 | 28.16 ± 1.02 | 13.49 ± 0.54 | 85.16 ± 3.41 | 11.90 ± 0.12 | 3.82 ± 0.00 |
| Date | SCoT | **2.54 ± 0.01** | 26.79 ± 0.68 | 13.63 ± 0.17 | 86.03 ± 1.08 | **11.28 ± 0.18** | 3.82 ± 0.00 |
|  | MGCoT | 2.56 ± 0.11 | 28.06 ± 0.66 | 13.57 ± 0.20 | 85.63 ± 1.28 | 11.78 ± 0.24 | 3.82 ± 0.00 |
| GSM8K | SCoT | 2.56 ± 0.01 | 26.92 ± 1.19 | *14.53 ± 0.45* | *91.72 ± 2.83* | 11.72 ± 0.11 | 3.82 ± 0.00 |
|  | MGCoT | 2.60 ± 0.01 | 28.55 ± 2.57 | 14.44 ± 0.43 | 91.15 ± 2.74 | 11.88 ± 0.16 | 3.82 ± 0.00 |
| MultiArith | SCoT | 2.56 ± 0.02 | 38.77 ± 5.57 | 14.21 ± 0.47 | 89.71 ± 2.96 | 11.87 ± 0.28 | 3.21 ± 0.48 |
|  | MGCoT | 2.60 ± 0.01 | *42.03 ± 6.75* | 14.15 ± 0.46 | 89.31 ± 2.92 | 12.21 ± 0.30 | 3.23 ± 0.47 |
| QASports | SCoT | 2.56 ± 0.02 | 36.57 ± 7.48 | 13.35 ± 1.41 | 84.28 ± 8.89 | 13.42 ± 3.46 | 2.69 ± 0.27 |
|  | MGCoT | 2.59 ± 0.02 | 37.78 ± 8.28 | 13.25 ± 1.45 | 83.64 ± 9.18 | *13.65 ± 3.42* | 2.69 ± 0.28 |
| SayCan | SCoT | 2.55 ± 0.00 | **26.67 ± 0.48** | 12.92 ± 0.61 | 81.53 ± 3.85 | 11.68 ± 0.12 | 2.66 ± 0.00 |
|  | MGCoT | 2.58 ± 0.01 | 27.98 ± 0.65 | **12.84 ± 0.59** | **81.07 ± 3.73** | 11.85 ± 0.16 | 2.66 ± 0.00 |
| StrategyQA | SCoT | 2.55 ± 0.00 | 27.88 ± 4.74 | 14.17 ± 0.48 | 89.43 ± 3.04 | 11.66 ± 0.12 | 2.78 ± 0.20 |
|  | MGCoT | 2.59 ± 0.00 | 29.32 ± 5.43 | 14.05 ± 0.50 | 88.68 ± 3.16 | 11.90 ± 0.17 | 2.78 ± 0.20 |
| SVAMP | SCoT | 2.56 ± 0.00 | 26.84 ± 0.55 | 14.24 ± 0.56 | 89.88 ± 3.54 | 11.75 ± 0.13 | 3.94 ± 0.03 |
|  | MGCoT | 2.59 ± 0.01 | 28.41 ± 0.99 | 14.16 ± 0.51 | 89.35 ± 3.24 | 11.85 ± 0.14 | *3.94 ± 0.00* |
| **All datasets** | SCoT | 2.56 ± 0.04 | 30.12 ± 6.14 | 13.73 ± 0.88 | 86.66 ± 5.58 | 11.85 ± 1.22 | 3.44 ± 0.54 |
|  | MGCoT | 2.58 ± 0.07 | 31.74 ± 6.84 | 13.65 ± 0.89 | 86.14 ± 5.61 | 12.10 ± 1.21 | 3.44 ± 0.54 |

### 3.1.8 Summary

Three hardware patterns hold across models. (i) Generation time depends on the model: MGCoT shortened generation for gemma2-2b (−49.1%), openchat-7b (−40.8%) and qwen2-1.5b (−11.3%), lengthened it for mistral-7b (+29.3%) and llama3.2-1b (+14.1%), and left it unchanged for phi3-mini and deepseek-r1-8b. (ii) RAM was lower under MGCoT for all seven models (−0.2% to −2.1%), consistently on nine or ten of ten datasets. (iii) GPU utilisation was higher under MGCoT for all seven models (+2.1% to +125.9%), and CPU utilisation was higher for five, unchanged for llama3.2-1b and lower for qwen2-1.5b. GPU memory did not differ between mechanisms for any model.

## 3.2 Response Quality

Tables 8–14 report the response quality metrics defined in the Response Quality Metrics section: RLF1, TP, TR, TF1, NED, CF1, BERT-F1, SS and task accuracy (ACC), the proportion of questions whose extracted final answer matches the reference. Absolute overlap values are low for every model, because reference answers are short labels or numbers while generated answers contain the reasoning, so the comparison rests on the differences between mechanisms.

### 3.2.1 Mistral-7b

MGCoT did not improve answer quality for mistral-7b. Token F1 decreased by 13.5% (0.067 → 0.058), with lower values on six datasets and significantly lower values on five, alongside token recall (−18.3%; 0.38 → 0.31) and token precision (−12.2%); ROUGE-L (−2.4%), edit similarity (+3.3%) and character F1 (+0.1%) were unchanged on aggregate. Semantic similarity fell by 15.8% (0.281 → 0.237), significantly on six datasets, and BERTScore was marginally lower (0.782 → 0.777, −0.6%; significant on CLUTRR, SayCan and StrategyQA). The losses were concentrated on GSM8K (TF1 −0.065, SS −0.125) and CLUTRR (SS −0.110), while Date was the only dataset that gained under MGCoT (RLF1 +0.055, NED +0.050, CF1 +0.051) and StrategyQA the only one with a significant SS gain (+0.028). The highest values of the table were all reached by SCoT on GSM8K (TF1 0.40, CF1 0.83, BERT-F1 0.84, SS 0.78). Task accuracy fell from 38.0% to 20.6% (−17.4 points, 95% CI [−22.4, −12.8]), significantly on five datasets (ASDiv, CLUTRR, GSM8K, MultiArith and SVAMP) and on no dataset in favour of MGCoT. Aggregate paired comparisons over all 500 questions (Wilcoxon signed-rank, BH-corrected): RLF1 p < 0.001; TP p < 0.001; TR p < 0.001; TF1 p < 0.001; NED p < 0.001; CF1 p = 0.012; BERT-F1 p < 0.001; SS p < 0.001; ACC p < 0.001 (Table 8).

**Table 8.** Response quality of Mistral-7b under SCoT and MGCoT (mean ± std).

| Dataset | Mechanism | RLF1 | TP | TR | TF1 | NED | CF1 | BERT-F1 | SS | ACC |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AQUA | SCoT | 0.01 ± 0.01 | 0.01 ± 0.01 | 0.50 ± 0.51 | 0.01 ± 0.01 | 0.00 ± 0.00 | 0.04 ± 0.00 | 0.76 ± 0.01 | 0.14 ± 0.06 | 0.24 ± 0.43 |
|  | MGCoT | 0.01 ± 0.01 | 0.01 ± 0.01 | 0.48 ± 0.50 | 0.01 ± 0.01 | 0.00 ± 0.00 | 0.05 ± 0.00 | 0.76 ± 0.01 | 0.09 ± 0.05 | 0.12 ± 0.33 |
| ASDiv | SCoT | 0.03 ± 0.01 | 0.02 ± 0.01 | **0.96 ± 0.20** | 0.04 ± 0.01 | 0.00 ± 0.00 | 0.07 ± 0.03 | 0.78 ± 0.01 | 0.29 ± 0.07 | **0.86 ± 0.35** |
|  | MGCoT | 0.03 ± 0.04 | 0.02 ± 0.02 | 0.82 ± 0.39 | 0.03 ± 0.03 | 0.00 ± 0.01 | 0.07 ± 0.03 | 0.78 ± 0.01 | 0.21 ± 0.08 | 0.22 ± 0.42 |
| CLUTRR | SCoT | 0.04 ± 0.02 | 0.02 ± 0.01 | 0.25 ± 0.15 | 0.03 ± 0.02 | 0.05 ± 0.01 | 0.51 ± 0.03 | 0.81 ± 0.01 | 0.37 ± 0.06 | 0.34 ± 0.48 |
|  | MGCoT | 0.02 ± 0.02 | 0.01 ± 0.01 | 0.19 ± 0.17 | 0.01 ± 0.02 | 0.04 ± 0.04 | 0.46 ± 0.06 | 0.80 ± 0.01 | 0.26 ± 0.06 | 0.16 ± 0.37 |
| Date | SCoT | 0.04 ± 0.02 | 0.00 ± 0.01 | 0.08 ± 0.19 | 0.01 ± 0.02 | 0.02 ± 0.01 | 0.25 ± 0.04 | 0.79 ± 0.02 | 0.37 ± 0.10 | 0.16 ± 0.37 |
|  | MGCoT | 0.09 ± 0.18 | 0.03 ± 0.14 | 0.11 ± 0.21 | 0.02 ± 0.10 | 0.07 ± 0.16 | 0.30 ± 0.18 | 0.79 ± 0.05 | 0.37 ± 0.16 | 0.32 ± 0.47 |
| GSM8K | SCoT | **0.24 ± 0.06** | **0.32 ± 0.09** | 0.58 ± 0.09 | **0.40 ± 0.08** | **0.20 ± 0.06** | **0.83 ± 0.06** | **0.84 ± 0.01** | **0.78 ± 0.06** | 0.44 ± 0.50 |
|  | MGCoT | 0.20 ± 0.09 | 0.27 ± 0.13 | 0.51 ± 0.15 | 0.34 ± 0.14 | 0.19 ± 0.08 | 0.80 ± 0.08 | 0.83 ± 0.03 | 0.65 ± 0.23 | 0.12 ± 0.33 |
| MultiArith | SCoT | 0.02 ± 0.01 | 0.02 ± 0.01 | 0.44 ± 0.16 | 0.03 ± 0.01 | 0.00 ± 0.00 | 0.11 ± 0.02 | 0.76 ± 0.01 | 0.17 ± 0.06 | 0.78 ± 0.42 |
|  | MGCoT | 0.02 ± 0.03 | 0.01 ± 0.01 | 0.33 ± 0.24 | 0.03 ± 0.03 | 0.01 ± 0.01 | 0.11 ± 0.03 | 0.76 ± 0.02 | 0.14 ± 0.05 | 0.34 ± 0.48 |
| QASports | SCoT | 0.01 ± 0.01 | 0.01 ± 0.01 | 0.44 ± 0.50 | 0.01 ± 0.01 | *0.00 ± 0.00* | *0.03 ± 0.03* | 0.77 ± 0.02 | 0.09 ± 0.04 | 0.16 ± 0.37 |
|  | MGCoT | 0.01 ± 0.01 | 0.00 ± 0.00 | 0.16 ± 0.37 | 0.00 ± 0.01 | 0.00 ± 0.00 | 0.04 ± 0.02 | 0.76 ± 0.02 | 0.10 ± 0.04 | 0.18 ± 0.39 |
| SayCan | SCoT | 0.07 ± 0.05 | 0.08 ± 0.03 | 0.20 ± 0.07 | 0.11 ± 0.05 | 0.15 ± 0.05 | 0.67 ± 0.05 | 0.79 ± 0.02 | 0.42 ± 0.13 | 0.00 ± 0.00 |
|  | MGCoT | 0.04 ± 0.04 | 0.06 ± 0.02 | 0.27 ± 0.09 | 0.10 ± 0.03 | 0.12 ± 0.05 | 0.70 ± 0.06 | 0.77 ± 0.01 | 0.37 ± 0.14 | 0.00 ± 0.00 |
| StrategyQA | SCoT | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.21 ± 0.02 | 0.78 ± 0.01 | *0.04 ± 0.06* | 0.18 ± 0.39 |
|  | MGCoT | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.19 ± 0.03 | 0.77 ± 0.01 | 0.07 ± 0.06 | 0.36 ± 0.48 |
| SVAMP | SCoT | 0.01 ± 0.01 | 0.01 ± 0.01 | 0.38 ± 0.22 | 0.03 ± 0.02 | 0.00 ± 0.00 | 0.11 ± 0.03 | *0.76 ± 0.01* | 0.14 ± 0.06 | 0.64 ± 0.48 |
|  | MGCoT | 0.02 ± 0.06 | 0.01 ± 0.03 | 0.25 ± 0.25 | 0.02 ± 0.04 | 0.01 ± 0.02 | 0.12 ± 0.05 | 0.76 ± 0.02 | 0.12 ± 0.06 | 0.24 ± 0.43 |
| **All datasets** | SCoT | 0.05 ± 0.07 | 0.05 ± 0.10 | 0.38 ± 0.37 | 0.07 ± 0.12 | 0.04 ± 0.07 | 0.28 ± 0.27 | 0.78 ± 0.03 | 0.28 ± 0.22 | 0.38 ± 0.49 |
|  | MGCoT | 0.05 ± 0.09 | 0.04 ± 0.10 | 0.31 ± 0.35 | 0.06 ± 0.11 | 0.05 ± 0.09 | 0.28 ± 0.27 | 0.78 ± 0.03 | 0.24 ± 0.21 | 0.21 ± 0.40 |

### 3.2.2 Phi3-mini

MGCoT raised the metrics that reward concise, well-aligned answers for phi3-mini: edit similarity increased by 56.6% (0.043 → 0.068) and character F1 by 8.0% (0.289 → 0.312), both significant, while ROUGE-L (+64.9%), token precision (+29.2%) and token F1 (+9.5%) increased without reaching significance on aggregate. BERTScore increased by 1.1% (0.781 → 0.790, significant) and semantic similarity by 4.9% (0.263 → 0.276, not significant). The gains were concentrated on Date (RLF1 +0.34, NED +0.24, CF1 +0.23, BERT-F1 +0.091, SS +0.289; the best cells of the table at RLF1 0.38 and BERT-F1 0.88) and MultiArith, both significant. Token recall moved in the opposite direction (−19.8%; 0.42 → 0.34), driven by QASports (−0.60) and ASDiv (−0.26), and CLUTRR and SayCan were significantly lower on both overlap and meaning (SS −0.126 on SayCan). Task accuracy fell from 46.2% to 38.0% (−8.2 points, 95% CI [−12.8, −3.6]), significantly on ASDiv and MultiArith, although it was nominally higher on AQUA, QASports and StrategyQA. Aggregate paired comparisons over all 500 questions (Wilcoxon signed-rank, BH-corrected): RLF1 p = 0.263; TP p = 0.481; TR p < 0.001; TF1 p = 0.209; NED p = 0.021; CF1 p = 0.011; BERT-F1 p < 0.001; SS p = 0.481; ACC p = 0.003 (Table 9).

**Table 9.** Response quality of Phi3-mini under SCoT and MGCoT (mean ± std).

| Dataset | Mechanism | RLF1 | TP | TR | TF1 | NED | CF1 | BERT-F1 | SS | ACC |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AQUA | SCoT | 0.01 ± 0.01 | 0.01 ± 0.01 | 0.52 ± 0.50 | 0.02 ± 0.02 | 0.00 ± 0.00 | 0.04 ± 0.01 | 0.76 ± 0.01 | 0.13 ± 0.06 | 0.30 ± 0.46 |
|  | MGCoT | 0.01 ± 0.01 | 0.01 ± 0.01 | 0.58 ± 0.50 | 0.02 ± 0.02 | 0.00 ± 0.00 | 0.04 ± 0.01 | *0.75 ± 0.11* | 0.14 ± 0.09 | 0.38 ± 0.49 |
| ASDiv | SCoT | 0.02 ± 0.01 | 0.02 ± 0.01 | **0.96 ± 0.20** | 0.04 ± 0.01 | 0.00 ± 0.00 | 0.07 ± 0.03 | 0.78 ± 0.01 | 0.27 ± 0.07 | 0.82 ± 0.39 |
|  | MGCoT | 0.02 ± 0.02 | 0.02 ± 0.01 | 0.70 ± 0.46 | 0.03 ± 0.03 | 0.00 ± 0.00 | 0.07 ± 0.03 | 0.78 ± 0.01 | 0.21 ± 0.09 | 0.38 ± 0.49 |
| CLUTRR | SCoT | 0.05 ± 0.02 | 0.02 ± 0.01 | 0.29 ± 0.12 | 0.03 ± 0.01 | 0.05 ± 0.01 | 0.51 ± 0.02 | 0.81 ± 0.01 | 0.35 ± 0.06 | 0.54 ± 0.50 |
|  | MGCoT | 0.03 ± 0.02 | 0.01 ± 0.01 | 0.24 ± 0.15 | 0.02 ± 0.02 | 0.04 ± 0.03 | 0.49 ± 0.07 | 0.80 ± 0.01 | 0.30 ± 0.10 | 0.44 ± 0.50 |
| Date | SCoT | 0.05 ± 0.03 | 0.01 ± 0.01 | 0.15 ± 0.23 | 0.01 ± 0.02 | 0.02 ± 0.01 | 0.27 ± 0.04 | 0.79 ± 0.02 | 0.30 ± 0.08 | 0.38 ± 0.49 |
|  | MGCoT | **0.38 ± 0.34** | 0.17 ± 0.37 | 0.20 ± 0.25 | 0.13 ± 0.24 | **0.26 ± 0.22** | 0.50 ± 0.20 | **0.88 ± 0.06** | 0.59 ± 0.21 | 0.38 ± 0.49 |
| GSM8K | SCoT | 0.27 ± 0.06 | **0.35 ± 0.09** | 0.58 ± 0.08 | **0.43 ± 0.09** | 0.21 ± 0.06 | 0.83 ± 0.06 | 0.85 ± 0.01 | **0.79 ± 0.06** | 0.62 ± 0.49 |
|  | MGCoT | 0.29 ± 0.09 | 0.34 ± 0.14 | 0.56 ± 0.09 | 0.41 ± 0.11 | 0.23 ± 0.06 | **0.83 ± 0.06** | 0.85 ± 0.02 | 0.79 ± 0.10 | 0.62 ± 0.49 |
| MultiArith | SCoT | 0.02 ± 0.00 | 0.02 ± 0.00 | 0.50 ± 0.00 | 0.04 ± 0.00 | 0.00 ± 0.00 | 0.12 ± 0.02 | 0.76 ± 0.01 | 0.16 ± 0.05 | **0.94 ± 0.24** |
|  | MGCoT | 0.04 ± 0.03 | 0.02 ± 0.01 | 0.48 ± 0.10 | 0.04 ± 0.02 | 0.01 ± 0.01 | 0.13 ± 0.03 | 0.77 ± 0.01 | 0.15 ± 0.06 | 0.70 ± 0.46 |
| QASports | SCoT | 0.01 ± 0.01 | 0.01 ± 0.01 | 0.60 ± 0.49 | 0.02 ± 0.01 | 0.00 ± 0.00 | 0.03 ± 0.03 | 0.77 ± 0.02 | 0.09 ± 0.05 | 0.10 ± 0.30 |
|  | MGCoT | 0.00 ± 0.01 | *0.00 ± 0.00* | *0.00 ± 0.00* | *0.00 ± 0.00* | *0.00 ± 0.00* | *0.03 ± 0.02* | 0.77 ± 0.02 | 0.11 ± 0.06 | 0.12 ± 0.33 |
| SayCan | SCoT | 0.05 ± 0.05 | 0.05 ± 0.03 | 0.17 ± 0.07 | 0.08 ± 0.04 | 0.13 ± 0.05 | 0.68 ± 0.04 | 0.78 ± 0.01 | 0.39 ± 0.13 | 0.00 ± 0.00 |
|  | MGCoT | 0.02 ± 0.02 | 0.05 ± 0.02 | 0.18 ± 0.09 | 0.07 ± 0.03 | 0.12 ± 0.05 | 0.69 ± 0.07 | 0.77 ± 0.01 | 0.26 ± 0.11 | 0.00 ± 0.00 |
| StrategyQA | SCoT | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.02 ± 0.14 | 0.00 ± 0.01 | 0.00 ± 0.00 | 0.21 ± 0.02 | 0.77 ± 0.01 | *0.02 ± 0.06* | 0.16 ± 0.37 |
|  | MGCoT | *0.00 ± 0.00* | 0.00 ± 0.00 | 0.02 ± 0.14 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.20 ± 0.03 | 0.77 ± 0.01 | 0.07 ± 0.06 | 0.18 ± 0.39 |
| SVAMP | SCoT | 0.02 ± 0.01 | 0.02 ± 0.01 | 0.45 ± 0.15 | 0.03 ± 0.01 | 0.00 ± 0.00 | 0.12 ± 0.03 | 0.76 ± 0.01 | 0.13 ± 0.06 | 0.76 ± 0.43 |
|  | MGCoT | 0.02 ± 0.02 | 0.02 ± 0.01 | 0.44 ± 0.16 | 0.04 ± 0.02 | 0.01 ± 0.01 | 0.13 ± 0.04 | 0.76 ± 0.01 | 0.13 ± 0.05 | 0.60 ± 0.49 |
| **All datasets** | SCoT | 0.05 ± 0.08 | 0.05 ± 0.11 | 0.42 ± 0.37 | 0.07 ± 0.13 | 0.04 ± 0.07 | 0.29 ± 0.27 | 0.78 ± 0.03 | 0.26 ± 0.22 | 0.46 ± 0.50 |
|  | MGCoT | 0.08 ± 0.17 | 0.06 ± 0.16 | 0.34 ± 0.34 | 0.08 ± 0.15 | 0.07 ± 0.12 | 0.31 ± 0.29 | 0.79 ± 0.06 | 0.28 ± 0.24 | 0.38 ± 0.49 |

### 3.2.3 Llama3.2-1b

Llama3.2-1b degraded on every quality dimension under MGCoT. ROUGE-L fell by 70.0% (0.042 → 0.013), token F1 by 62.4% (0.065 → 0.024), token precision by 60.4%, token recall by 70.9% (0.35 → 0.10), character F1 by 7.4% and edit similarity by 15.4%, with significantly lower values on eight of ten datasets for the overlap metrics and on all ten for recall. Meaning followed: semantic similarity dropped by 58.4% (0.266 → 0.110) and BERTScore by 1.8% (0.775 → 0.761), both significantly lower on eight datasets. The largest losses were on GSM8K (TF1 −0.259, SS −0.631) and ASDiv (TR −0.88). Only StrategyQA improved (SS +0.052, NED +0.009) together with QASports on BERTScore (+0.030), and every best cell of the table belongs to SCoT. Task accuracy collapsed from 35.4% to 2.8% (−32.6 points, 95% CI [−37.2, −28.2]), significantly lower on six datasets and higher on none, confirming that the model stopped producing usable final answers under MGCoT. Aggregate paired comparisons over all 500 questions (Wilcoxon signed-rank, BH-corrected): RLF1 p < 0.001; TP p < 0.001; TR p < 0.001; TF1 p < 0.001; NED p < 0.001; CF1 p < 0.001; BERT-F1 p < 0.001; SS p < 0.001; ACC p < 0.001 (Table 10).

**Table 10.** Response quality of Llama3.2-1b under SCoT and MGCoT (mean ± std).

| Dataset | Mechanism | RLF1 | TP | TR | TF1 | NED | CF1 | BERT-F1 | SS | ACC |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AQUA | SCoT | 0.00 ± 0.00 | 0.00 ± 0.01 | 0.30 ± 0.46 | 0.01 ± 0.01 | 0.00 ± 0.00 | 0.05 ± 0.00 | 0.76 ± 0.01 | 0.11 ± 0.06 | 0.04 ± 0.20 |
|  | MGCoT | 0.00 ± 0.01 | 0.00 ± 0.01 | 0.24 ± 0.43 | 0.01 ± 0.01 | 0.00 ± 0.00 | 0.05 ± 0.01 | 0.75 ± 0.01 | 0.07 ± 0.03 | 0.04 ± 0.20 |
| ASDiv | SCoT | 0.02 ± 0.00 | 0.02 ± 0.00 | **0.98 ± 0.14** | 0.04 ± 0.01 | 0.00 ± 0.00 | 0.07 ± 0.02 | 0.77 ± 0.01 | 0.30 ± 0.06 | **0.82 ± 0.39** |
|  | MGCoT | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.10 ± 0.30 | 0.00 ± 0.01 | 0.00 ± 0.00 | 0.05 ± 0.03 | 0.76 ± 0.01 | 0.10 ± 0.05 | 0.02 ± 0.14 |
| CLUTRR | SCoT | 0.03 ± 0.02 | 0.01 ± 0.01 | 0.23 ± 0.15 | 0.03 ± 0.02 | 0.04 ± 0.01 | 0.50 ± 0.02 | 0.80 ± 0.01 | 0.34 ± 0.06 | 0.46 ± 0.50 |
|  | MGCoT | 0.00 ± 0.01 | 0.00 ± 0.00 | 0.01 ± 0.07 | 0.00 ± 0.00 | 0.03 ± 0.02 | 0.45 ± 0.04 | 0.79 ± 0.01 | 0.09 ± 0.04 | 0.06 ± 0.24 |
| Date | SCoT | 0.03 ± 0.02 | 0.00 ± 0.01 | 0.11 ± 0.21 | 0.01 ± 0.02 | 0.02 ± 0.01 | 0.26 ± 0.04 | 0.78 ± 0.01 | 0.30 ± 0.08 | 0.18 ± 0.39 |
|  | MGCoT | 0.01 ± 0.01 | 0.00 ± 0.00 | 0.03 ± 0.12 | 0.00 ± 0.01 | 0.01 ± 0.00 | 0.23 ± 0.04 | *0.74 ± 0.02* | 0.10 ± 0.07 | 0.04 ± 0.20 |
| GSM8K | SCoT | **0.24 ± 0.08** | **0.32 ± 0.09** | 0.56 ± 0.09 | **0.40 ± 0.09** | **0.19 ± 0.06** | **0.82 ± 0.06** | **0.84 ± 0.02** | **0.78 ± 0.07** | 0.32 ± 0.47 |
|  | MGCoT | 0.08 ± 0.05 | 0.10 ± 0.06 | 0.25 ± 0.10 | 0.14 ± 0.07 | 0.14 ± 0.05 | 0.76 ± 0.06 | 0.77 ± 0.01 | 0.15 ± 0.14 | 0.02 ± 0.14 |
| MultiArith | SCoT | 0.01 ± 0.01 | 0.01 ± 0.01 | 0.43 ± 0.18 | 0.03 ± 0.01 | 0.00 ± 0.00 | 0.11 ± 0.02 | 0.75 ± 0.01 | 0.16 ± 0.06 | 0.78 ± 0.42 |
|  | MGCoT | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.04 ± 0.14 | 0.00 ± 0.01 | 0.00 ± 0.00 | 0.10 ± 0.03 | 0.74 ± 0.01 | 0.10 ± 0.03 | 0.02 ± 0.14 |
| QASports | SCoT | 0.01 ± 0.01 | 0.00 ± 0.01 | 0.26 ± 0.44 | 0.01 ± 0.01 | 0.00 ± 0.00 | 0.03 ± 0.03 | 0.75 ± 0.01 | 0.09 ± 0.05 | 0.36 ± 0.48 |
|  | MGCoT | 0.00 ± 0.01 | 0.00 ± 0.00 | 0.04 ± 0.20 | 0.00 ± 0.00 | *0.00 ± 0.00* | *0.01 ± 0.02* | 0.78 ± 0.02 | 0.09 ± 0.06 | 0.00 ± 0.00 |
| SayCan | SCoT | 0.06 ± 0.04 | 0.07 ± 0.03 | 0.24 ± 0.09 | 0.11 ± 0.05 | 0.12 ± 0.05 | 0.70 ± 0.04 | 0.78 ± 0.01 | 0.43 ± 0.13 | 0.00 ± 0.00 |
|  | MGCoT | 0.03 ± 0.03 | 0.07 ± 0.05 | 0.17 ± 0.09 | 0.08 ± 0.04 | 0.13 ± 0.05 | 0.66 ± 0.09 | 0.77 ± 0.02 | 0.24 ± 0.12 | 0.00 ± 0.00 |
| StrategyQA | SCoT | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.02 ± 0.14 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.21 ± 0.02 | 0.77 ± 0.01 | *0.02 ± 0.06* | 0.14 ± 0.35 |
|  | MGCoT | *0.00 ± 0.00* | *0.00 ± 0.00* | *0.00 ± 0.00* | *0.00 ± 0.00* | 0.01 ± 0.02 | 0.22 ± 0.06 | 0.77 ± 0.02 | 0.07 ± 0.05 | 0.06 ± 0.24 |
| SVAMP | SCoT | 0.01 ± 0.01 | 0.01 ± 0.01 | 0.35 ± 0.23 | 0.02 ± 0.02 | 0.00 ± 0.00 | 0.11 ± 0.03 | 0.75 ± 0.01 | 0.13 ± 0.06 | 0.44 ± 0.50 |
|  | MGCoT | 0.00 ± 0.01 | 0.00 ± 0.00 | 0.13 ± 0.22 | 0.01 ± 0.01 | 0.00 ± 0.00 | 0.11 ± 0.03 | 0.74 ± 0.01 | 0.09 ± 0.03 | 0.02 ± 0.14 |
| **All datasets** | SCoT | 0.04 ± 0.07 | 0.05 ± 0.10 | 0.35 ± 0.36 | 0.06 ± 0.12 | 0.04 ± 0.07 | 0.29 ± 0.27 | 0.78 ± 0.03 | 0.27 ± 0.22 | 0.35 ± 0.48 |
|  | MGCoT | 0.01 ± 0.03 | 0.02 ± 0.04 | 0.10 ± 0.22 | 0.02 ± 0.05 | 0.03 ± 0.06 | 0.26 ± 0.26 | 0.76 ± 0.02 | 0.11 ± 0.09 | 0.03 ± 0.17 |

### 3.2.4 Gemma2-2b

Gemma2-2b improved on nearly all quality metrics under MGCoT. ROUGE-L more than doubled (0.044 → 0.101, +129.4%), edit similarity rose by 129.6% (0.038 → 0.087), token precision by 44.4%, token F1 by 24.7% (0.067 → 0.083) and character F1 by 9.1% (0.287 → 0.314), all significant; ROUGE-L and character F1 were significantly higher on six and seven datasets respectively, and edit similarity on eight. BERTScore rose by 3.7% (0.771 → 0.799) and was significantly higher on all ten datasets — the most consistent semantic gain of the study — while semantic similarity rose by 7.6% on aggregate (not significant after correction) with significant gains on CLUTRR, Date (+0.281) and GSM8K. The largest gains were on Date (RLF1 +0.27, BERT-F1 +0.084) and CLUTRR (RLF1 +0.16), and the best cells of RLF1, TF1, CF1 and SS were MGCoT on GSM8K (0.32, 0.44, 0.84, 0.80). The exceptions were token recall (−28.6%; 0.39 → 0.28, lower on nine datasets) and the two tasks presented without task framing, QASports and SayCan (SS −0.128). Task accuracy, however, fell from 40.6% to 34.2% (−6.4 points, 95% CI [−10.2, −2.6]; significant on MultiArith), so the closer surface and semantic alignment did not translate into more correct answers. Aggregate paired comparisons over all 500 questions (Wilcoxon signed-rank, BH-corrected): RLF1 p < 0.001; TP p < 0.001; TR p < 0.001; TF1 p < 0.001; NED p < 0.001; CF1 p < 0.001; BERT-F1 p < 0.001; SS p = 0.096; ACC p = 0.003 (Table 11).

**Table 11.** Response quality of Gemma2-2b under SCoT and MGCoT (mean ± std).

| Dataset | Mechanism | RLF1 | TP | TR | TF1 | NED | CF1 | BERT-F1 | SS | ACC |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AQUA | SCoT | 0.01 ± 0.01 | 0.01 ± 0.01 | 0.48 ± 0.50 | 0.01 ± 0.01 | 0.00 ± 0.00 | 0.05 ± 0.00 | 0.75 ± 0.01 | 0.11 ± 0.06 | 0.26 ± 0.44 |
|  | MGCoT | 0.01 ± 0.01 | 0.01 ± 0.01 | 0.42 ± 0.50 | 0.01 ± 0.02 | 0.00 ± 0.00 | 0.05 ± 0.01 | 0.76 ± 0.01 | 0.11 ± 0.06 | 0.22 ± 0.42 |
| ASDiv | SCoT | 0.02 ± 0.01 | 0.02 ± 0.01 | **0.88 ± 0.33** | 0.04 ± 0.01 | 0.00 ± 0.00 | 0.07 ± 0.03 | 0.77 ± 0.01 | 0.26 ± 0.07 | 0.84 ± 0.37 |
|  | MGCoT | 0.06 ± 0.07 | 0.03 ± 0.03 | 0.76 ± 0.43 | 0.06 ± 0.06 | 0.01 ± 0.01 | 0.08 ± 0.05 | 0.80 ± 0.02 | 0.28 ± 0.09 | 0.68 ± 0.47 |
| CLUTRR | SCoT | 0.03 ± 0.02 | 0.02 ± 0.01 | 0.33 ± 0.21 | 0.03 ± 0.02 | 0.04 ± 0.01 | 0.50 ± 0.03 | 0.80 ± 0.01 | 0.33 ± 0.06 | 0.28 ± 0.45 |
|  | MGCoT | 0.19 ± 0.08 | 0.06 ± 0.07 | 0.17 ± 0.17 | 0.09 ± 0.09 | 0.21 ± 0.10 | 0.62 ± 0.06 | 0.84 ± 0.01 | 0.38 ± 0.05 | 0.42 ± 0.50 |
| Date | SCoT | 0.05 ± 0.02 | 0.01 ± 0.01 | 0.21 ± 0.25 | 0.02 ± 0.02 | 0.02 ± 0.00 | 0.27 ± 0.04 | 0.78 ± 0.01 | 0.30 ± 0.09 | 0.40 ± 0.49 |
|  | MGCoT | 0.32 ± 0.20 | 0.02 ± 0.06 | 0.08 ± 0.19 | 0.04 ± 0.09 | 0.22 ± 0.13 | 0.43 ± 0.16 | **0.86 ± 0.05** | 0.59 ± 0.14 | 0.26 ± 0.44 |
| GSM8K | SCoT | 0.23 ± 0.05 | 0.31 ± 0.08 | 0.53 ± 0.10 | 0.38 ± 0.07 | 0.19 ± 0.05 | 0.82 ± 0.06 | 0.83 ± 0.01 | 0.76 ± 0.07 | 0.42 ± 0.50 |
|  | MGCoT | **0.32 ± 0.09** | **0.41 ± 0.13** | 0.51 ± 0.11 | **0.44 ± 0.10** | **0.25 ± 0.06** | **0.84 ± 0.04** | 0.85 ± 0.02 | **0.80 ± 0.11** | 0.48 ± 0.50 |
| MultiArith | SCoT | 0.02 ± 0.00 | 0.02 ± 0.00 | 0.49 ± 0.07 | 0.03 ± 0.01 | 0.00 ± 0.00 | 0.11 ± 0.02 | 0.75 ± 0.01 | 0.16 ± 0.05 | **0.98 ± 0.14** |
|  | MGCoT | 0.05 ± 0.03 | 0.03 ± 0.01 | 0.45 ± 0.15 | 0.05 ± 0.03 | 0.01 ± 0.01 | 0.13 ± 0.03 | 0.78 ± 0.01 | 0.15 ± 0.05 | 0.78 ± 0.42 |
| QASports | SCoT | 0.01 ± 0.01 | 0.00 ± 0.01 | 0.36 ± 0.48 | 0.01 ± 0.01 | 0.00 ± 0.00 | 0.03 ± 0.02 | 0.75 ± 0.02 | 0.10 ± 0.04 | 0.10 ± 0.30 |
|  | MGCoT | 0.00 ± 0.00 | *0.00 ± 0.00* | *0.00 ± 0.00* | *0.00 ± 0.00* | *0.00 ± 0.00* | *0.00 ± 0.01* | 0.79 ± 0.02 | 0.08 ± 0.05 | 0.04 ± 0.20 |
| SayCan | SCoT | 0.06 ± 0.04 | 0.08 ± 0.03 | 0.26 ± 0.08 | 0.12 ± 0.04 | 0.12 ± 0.05 | 0.69 ± 0.05 | 0.77 ± 0.01 | 0.43 ± 0.12 | 0.00 ± 0.00 |
|  | MGCoT | 0.04 ± 0.04 | 0.10 ± 0.09 | 0.14 ± 0.08 | 0.11 ± 0.07 | 0.15 ± 0.05 | 0.62 ± 0.05 | 0.78 ± 0.02 | 0.31 ± 0.13 | 0.00 ± 0.00 |
| StrategyQA | SCoT | *0.00 ± 0.00* | 0.00 ± 0.00 | 0.02 ± 0.14 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.21 ± 0.02 | 0.76 ± 0.01 | 0.03 ± 0.06 | 0.18 ± 0.39 |
|  | MGCoT | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.02 ± 0.14 | 0.00 ± 0.00 | 0.01 ± 0.00 | 0.23 ± 0.03 | 0.78 ± 0.01 | *0.02 ± 0.06* | 0.10 ± 0.30 |
| SVAMP | SCoT | 0.01 ± 0.01 | 0.01 ± 0.01 | 0.36 ± 0.23 | 0.02 ± 0.02 | 0.00 ± 0.00 | 0.12 ± 0.03 | *0.75 ± 0.01* | 0.13 ± 0.05 | 0.60 ± 0.49 |
|  | MGCoT | 0.02 ± 0.03 | 0.01 ± 0.02 | 0.25 ± 0.25 | 0.03 ± 0.03 | 0.01 ± 0.01 | 0.13 ± 0.05 | 0.77 ± 0.01 | 0.12 ± 0.07 | 0.44 ± 0.50 |
| **All datasets** | SCoT | 0.04 ± 0.07 | 0.05 ± 0.09 | 0.39 ± 0.35 | 0.07 ± 0.11 | 0.04 ± 0.06 | 0.29 ± 0.27 | 0.77 ± 0.03 | 0.26 ± 0.22 | 0.41 ± 0.49 |
|  | MGCoT | 0.10 ± 0.14 | 0.07 ± 0.13 | 0.28 ± 0.34 | 0.08 ± 0.14 | 0.09 ± 0.12 | 0.31 ± 0.29 | 0.80 ± 0.04 | 0.28 ± 0.25 | 0.34 ± 0.47 |

### 3.2.5 Qwen2-1.5b

MGCoT produced moderate gains for qwen2-1.5b: token F1 increased by 10.9% (0.073 → 0.081) and edit similarity by 18.7% (0.062 → 0.074), both significant, while ROUGE-L (+21.9%), token precision (+11.2%) and character F1 (+1.8%) increased without reaching significance. Semantic quality was preserved rather than improved: BERTScore (0.792 → 0.795) and semantic similarity (0.251 → 0.254) did not differ significantly, and qwen2-1.5b retained the highest SCoT BERTScore baseline of all models (0.792). The gains were significant mainly on ASDiv (RLF1 +0.066, TF1 +0.049, BERT-F1 +0.015) with smaller gains on CLUTRR and Date, and StrategyQA was significantly higher on SS. Token recall was slightly lower (−3.4%, not significant; significant only on SVAMP) and QASports lower on SS. The best overlap cells remained with SCoT on GSM8K (RLF1 0.28, TF1 0.43), whereas the best SS cell was MGCoT on GSM8K (0.80). Task accuracy was unchanged overall (36.4% vs 36.6%, 95% CI [−3.4, +3.8]) and significantly higher on CLUTRR (28% to 48%), the only condition in the study where MGCoT significantly improved accuracy. Aggregate paired comparisons over all 500 questions (Wilcoxon signed-rank, BH-corrected): RLF1 p = 0.092; TP p = 0.019; TR p = 0.211; TF1 p = 0.025; NED p < 0.001; CF1 p = 0.097; BERT-F1 p = 0.127; SS p = 1.000; ACC p = 1.000 (Table 12).

**Table 12.** Response quality of Qwen2-1.5b under SCoT and MGCoT (mean ± std).

| Dataset | Mechanism | RLF1 | TP | TR | TF1 | NED | CF1 | BERT-F1 | SS | ACC |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AQUA | SCoT | 0.01 ± 0.01 | 0.01 ± 0.01 | 0.46 ± 0.50 | 0.01 ± 0.02 | 0.00 ± 0.00 | 0.05 ± 0.01 | *0.76 ± 0.01* | 0.13 ± 0.06 | 0.24 ± 0.43 |
|  | MGCoT | 0.02 ± 0.06 | 0.01 ± 0.02 | 0.54 ± 0.50 | 0.02 ± 0.04 | 0.00 ± 0.01 | 0.05 ± 0.01 | 0.76 ± 0.02 | 0.13 ± 0.06 | 0.36 ± 0.48 |
| ASDiv | SCoT | 0.04 ± 0.05 | 0.03 ± 0.02 | 0.88 ± 0.33 | 0.06 ± 0.04 | 0.01 ± 0.01 | 0.08 ± 0.04 | 0.79 ± 0.02 | 0.25 ± 0.08 | 0.84 ± 0.37 |
|  | MGCoT | 0.11 ± 0.10 | 0.06 ± 0.04 | **0.90 ± 0.30** | 0.11 ± 0.08 | 0.02 ± 0.02 | 0.11 ± 0.05 | 0.81 ± 0.02 | 0.26 ± 0.09 | 0.86 ± 0.35 |
| CLUTRR | SCoT | 0.12 ± 0.09 | 0.03 ± 0.05 | 0.13 ± 0.16 | 0.05 ± 0.08 | 0.19 ± 0.09 | 0.64 ± 0.08 | 0.84 ± 0.02 | 0.32 ± 0.07 | 0.28 ± 0.45 |
|  | MGCoT | 0.15 ± 0.09 | 0.05 ± 0.06 | 0.16 ± 0.17 | 0.08 ± 0.09 | **0.25 ± 0.07** | 0.67 ± 0.06 | 0.84 ± 0.01 | 0.33 ± 0.06 | 0.48 ± 0.50 |
| Date | SCoT | 0.09 ± 0.11 | 0.00 ± 0.01 | 0.04 ± 0.14 | 0.01 ± 0.02 | 0.06 ± 0.09 | 0.30 ± 0.08 | 0.81 ± 0.04 | 0.37 ± 0.13 | 0.16 ± 0.37 |
|  | MGCoT | 0.12 ± 0.15 | 0.00 ± 0.01 | 0.01 ± 0.07 | 0.00 ± 0.01 | 0.10 ± 0.12 | 0.32 ± 0.11 | 0.82 ± 0.04 | 0.39 ± 0.15 | 0.10 ± 0.30 |
| GSM8K | SCoT | **0.28 ± 0.06** | **0.36 ± 0.10** | 0.56 ± 0.10 | **0.43 ± 0.08** | 0.22 ± 0.06 | **0.83 ± 0.06** | **0.85 ± 0.02** | 0.80 ± 0.07 | 0.36 ± 0.48 |
|  | MGCoT | 0.27 ± 0.07 | 0.35 ± 0.10 | 0.53 ± 0.09 | 0.42 ± 0.09 | 0.22 ± 0.07 | 0.82 ± 0.06 | 0.85 ± 0.02 | **0.80 ± 0.06** | 0.38 ± 0.49 |
| MultiArith | SCoT | 0.02 ± 0.01 | 0.02 ± 0.01 | 0.47 ± 0.12 | 0.04 ± 0.01 | 0.01 ± 0.00 | 0.12 ± 0.02 | 0.77 ± 0.01 | 0.14 ± 0.06 | **0.90 ± 0.30** |
|  | MGCoT | 0.03 ± 0.01 | 0.02 ± 0.01 | 0.45 ± 0.15 | 0.04 ± 0.02 | 0.01 ± 0.01 | 0.13 ± 0.02 | 0.77 ± 0.01 | 0.14 ± 0.06 | 0.84 ± 0.37 |
| QASports | SCoT | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.02 ± 0.14 | 0.00 ± 0.00 | *0.00 ± 0.00* | *0.00 ± 0.01* | 0.78 ± 0.02 | 0.09 ± 0.05 | 0.04 ± 0.20 |
|  | MGCoT | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.01 ± 0.02 | 0.77 ± 0.01 | 0.06 ± 0.04 | 0.04 ± 0.20 |
| SayCan | SCoT | 0.03 ± 0.04 | 0.10 ± 0.06 | 0.13 ± 0.07 | 0.10 ± 0.04 | 0.12 ± 0.05 | 0.62 ± 0.10 | 0.78 ± 0.02 | 0.27 ± 0.16 | 0.00 ± 0.00 |
|  | MGCoT | 0.03 ± 0.04 | 0.12 ± 0.09 | 0.13 ± 0.08 | 0.10 ± 0.05 | 0.12 ± 0.05 | 0.60 ± 0.10 | 0.78 ± 0.02 | 0.25 ± 0.14 | 0.00 ± 0.00 |
| StrategyQA | SCoT | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.02 ± 0.14 | 0.00 ± 0.01 | 0.01 ± 0.01 | 0.24 ± 0.04 | 0.78 ± 0.01 | *0.02 ± 0.06* | 0.20 ± 0.40 |
|  | MGCoT | *0.00 ± 0.00* | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.01 ± 0.01 | 0.24 ± 0.05 | 0.78 ± 0.01 | 0.05 ± 0.06 | 0.10 ± 0.30 |
| SVAMP | SCoT | 0.02 ± 0.01 | 0.02 ± 0.01 | 0.41 ± 0.19 | 0.04 ± 0.02 | 0.01 ± 0.00 | 0.13 ± 0.04 | 0.76 ± 0.02 | 0.12 ± 0.06 | 0.62 ± 0.49 |
|  | MGCoT | 0.02 ± 0.04 | 0.02 ± 0.02 | 0.30 ± 0.25 | 0.04 ± 0.04 | 0.01 ± 0.01 | 0.14 ± 0.05 | 0.77 ± 0.02 | 0.11 ± 0.06 | 0.50 ± 0.51 |
| **All datasets** | SCoT | 0.06 ± 0.10 | 0.06 ± 0.11 | 0.31 ± 0.35 | 0.07 ± 0.13 | 0.06 ± 0.09 | 0.30 ± 0.28 | 0.79 ± 0.03 | 0.25 ± 0.23 | 0.36 ± 0.48 |
|  | MGCoT | 0.08 ± 0.11 | 0.06 ± 0.11 | 0.30 ± 0.36 | 0.08 ± 0.13 | 0.07 ± 0.10 | 0.31 ± 0.28 | 0.79 ± 0.04 | 0.25 ± 0.23 | 0.37 ± 0.48 |

### 3.2.6 Openchat-7b

Openchat-7b showed the broadest improvement of the study. Under MGCoT, ROUGE-L increased by 83.2% (0.048 → 0.089), edit similarity by 85.3% (0.044 → 0.082), token precision by 41.3%, token F1 by 32.5% (0.067 → 0.089) and character F1 by 12.1% (0.289 → 0.324), all significant, with significantly higher token F1 on seven datasets and edit similarity and character F1 on seven. BERTScore increased by 2.7% (0.778 → 0.799), significantly on eight datasets and lower on none. The largest gains were on Date (RLF1 +0.19, BERT-F1 +0.066, SS +0.205) and CLUTRR (NED +0.18, TF1 +0.08), and the best cells of TF1 (0.47), CF1 (0.85), BERT-F1 (0.86) and SS (0.82) were all MGCoT on GSM8K. Aggregate semantic similarity was unchanged (0.279 → 0.276) because dataset effects cancelled: significant gains on Date and GSM8K against significant losses on AQUA, ASDiv, MultiArith, SVAMP, QASports and SayCan (−0.135). As for gemma2-2b, token recall decreased (−15.1%). Task accuracy was preserved (43.8% vs 44.8%, 95% CI [−3.2, +5.4], not significant), with nominal gains on five datasets, so openchat-7b obtained the alignment and efficiency gains at no cost in correctness. Aggregate paired comparisons over all 500 questions (Wilcoxon signed-rank, BH-corrected): RLF1 p < 0.001; TP p < 0.001; TR p < 0.001; TF1 p < 0.001; NED p < 0.001; CF1 p < 0.001; BERT-F1 p < 0.001; SS p = 0.003; ACC p = 0.786 (Table 13).

**Table 13.** Response quality of Openchat-7b under SCoT and MGCoT (mean ± std).

| Dataset | Mechanism | RLF1 | TP | TR | TF1 | NED | CF1 | BERT-F1 | SS | ACC |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AQUA | SCoT | 0.01 ± 0.01 | 0.01 ± 0.01 | 0.48 ± 0.50 | 0.01 ± 0.01 | 0.00 ± 0.00 | 0.05 ± 0.00 | 0.76 ± 0.01 | 0.15 ± 0.05 | 0.30 ± 0.46 |
|  | MGCoT | 0.01 ± 0.01 | 0.01 ± 0.01 | 0.62 ± 0.49 | 0.02 ± 0.02 | 0.00 ± 0.00 | 0.05 ± 0.01 | 0.76 ± 0.01 | 0.11 ± 0.06 | 0.44 ± 0.50 |
| ASDiv | SCoT | 0.02 ± 0.00 | 0.02 ± 0.00 | **0.98 ± 0.14** | 0.04 ± 0.01 | 0.00 ± 0.00 | 0.07 ± 0.02 | 0.77 ± 0.01 | 0.28 ± 0.07 | 0.86 ± 0.35 |
|  | MGCoT | 0.05 ± 0.06 | 0.03 ± 0.03 | 0.82 ± 0.39 | 0.06 ± 0.05 | 0.01 ± 0.01 | 0.08 ± 0.04 | 0.79 ± 0.02 | 0.26 ± 0.08 | 0.74 ± 0.44 |
| CLUTRR | SCoT | 0.04 ± 0.03 | 0.01 ± 0.01 | 0.23 ± 0.16 | 0.03 ± 0.02 | 0.06 ± 0.01 | 0.51 ± 0.02 | 0.81 ± 0.01 | 0.34 ± 0.06 | 0.36 ± 0.48 |
|  | MGCoT | 0.18 ± 0.09 | 0.07 ± 0.06 | 0.22 ± 0.16 | 0.11 ± 0.08 | 0.24 ± 0.08 | 0.67 ± 0.06 | 0.84 ± 0.01 | 0.36 ± 0.07 | 0.52 ± 0.50 |
| Date | SCoT | 0.03 ± 0.02 | 0.00 ± 0.01 | 0.04 ± 0.14 | 0.00 ± 0.01 | 0.02 ± 0.01 | 0.25 ± 0.05 | 0.78 ± 0.02 | 0.34 ± 0.09 | 0.26 ± 0.44 |
|  | MGCoT | 0.23 ± 0.16 | 0.02 ± 0.05 | 0.09 ± 0.19 | 0.03 ± 0.08 | 0.17 ± 0.09 | 0.38 ± 0.10 | 0.84 ± 0.04 | 0.55 ± 0.10 | 0.36 ± 0.48 |
| GSM8K | SCoT | 0.26 ± 0.07 | 0.33 ± 0.09 | 0.57 ± 0.11 | 0.41 ± 0.09 | 0.21 ± 0.06 | 0.83 ± 0.06 | 0.84 ± 0.02 | 0.78 ± 0.06 | 0.50 ± 0.51 |
|  | MGCoT | **0.30 ± 0.08** | **0.40 ± 0.11** | 0.58 ± 0.09 | **0.47 ± 0.10** | **0.24 ± 0.07** | **0.85 ± 0.06** | **0.86 ± 0.02** | **0.82 ± 0.07** | 0.52 ± 0.50 |
| MultiArith | SCoT | 0.02 ± 0.00 | 0.02 ± 0.00 | 0.49 ± 0.07 | 0.03 ± 0.01 | 0.00 ± 0.00 | 0.12 ± 0.02 | *0.75 ± 0.01* | 0.18 ± 0.05 | **0.90 ± 0.30** |
|  | MGCoT | 0.04 ± 0.02 | 0.03 ± 0.01 | 0.47 ± 0.12 | 0.05 ± 0.02 | 0.01 ± 0.01 | 0.14 ± 0.03 | 0.77 ± 0.01 | 0.15 ± 0.06 | 0.84 ± 0.37 |
| QASports | SCoT | 0.01 ± 0.01 | 0.01 ± 0.01 | 0.60 ± 0.49 | 0.02 ± 0.01 | 0.00 ± 0.00 | 0.03 ± 0.03 | 0.77 ± 0.02 | 0.10 ± 0.05 | 0.14 ± 0.35 |
|  | MGCoT | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | *0.00 ± 0.00* | *0.00 ± 0.01* | 0.78 ± 0.01 | 0.07 ± 0.04 | 0.02 ± 0.14 |
| SayCan | SCoT | 0.07 ± 0.04 | 0.07 ± 0.03 | 0.18 ± 0.07 | 0.10 ± 0.04 | 0.15 ± 0.05 | 0.70 ± 0.04 | 0.79 ± 0.01 | 0.44 ± 0.12 | 0.00 ± 0.00 |
|  | MGCoT | 0.04 ± 0.04 | 0.09 ± 0.08 | 0.15 ± 0.08 | 0.10 ± 0.06 | 0.14 ± 0.05 | 0.65 ± 0.05 | 0.79 ± 0.02 | 0.30 ± 0.14 | 0.00 ± 0.00 |
| StrategyQA | SCoT | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.21 ± 0.02 | 0.77 ± 0.01 | 0.03 ± 0.06 | 0.34 ± 0.48 |
|  | MGCoT | 0.00 ± 0.01 | 0.00 ± 0.01 | 0.04 ± 0.20 | 0.00 ± 0.01 | 0.01 ± 0.00 | 0.27 ± 0.04 | 0.79 ± 0.01 | *0.03 ± 0.06* | 0.28 ± 0.45 |
| SVAMP | SCoT | 0.01 ± 0.01 | 0.02 ± 0.01 | 0.42 ± 0.19 | 0.03 ± 0.01 | 0.00 ± 0.00 | 0.12 ± 0.03 | 0.75 ± 0.01 | 0.14 ± 0.06 | 0.72 ± 0.45 |
|  | MGCoT | 0.03 ± 0.03 | 0.02 ± 0.02 | 0.40 ± 0.20 | 0.05 ± 0.03 | 0.01 ± 0.01 | 0.14 ± 0.04 | 0.77 ± 0.01 | 0.12 ± 0.06 | 0.76 ± 0.43 |
| **All datasets** | SCoT | 0.05 ± 0.08 | 0.05 ± 0.10 | 0.40 ± 0.37 | 0.07 ± 0.12 | 0.04 ± 0.07 | 0.29 ± 0.27 | 0.78 ± 0.03 | 0.28 ± 0.22 | 0.44 ± 0.50 |
|  | MGCoT | 0.09 ± 0.12 | 0.07 ± 0.13 | 0.34 ± 0.35 | 0.09 ± 0.14 | 0.08 ± 0.11 | 0.32 ± 0.29 | 0.80 ± 0.04 | 0.28 ± 0.25 | 0.45 ± 0.50 |

### 3.2.7 Deepseek-r1-8b

All quality scores were close to zero under both mechanisms, with every aggregate mean at or below 0.002 under SCoT and 0.024 under MGCoT, because the model returned no visible answer in 97–98% of runs. The apparent relative changes (e.g., ROUGE-L +1,466%, SS +156%) come from the roughly 3% of runs that did return an answer, mostly on Date and CLUTRR, and none of the differences was significant. These values measure budget compliance rather than answer quality. Task accuracy was 0.2% under SCoT and 2.4% under MGCoT, both at floor level, as almost no run produced a visible answer. Aggregate paired comparisons over all 500 questions (Wilcoxon signed-rank, BH-corrected): RLF1 p = 0.145; TP p = 0.145; TR p = 0.145; TF1 p = 0.145; NED p = 0.145; CF1 p = 0.145; BERT-F1 p = 0.364; SS p = 0.458; ACC p = 0.145 (Table 14).

**Table 14.** Response quality of Deepseek-r1-8b under SCoT and MGCoT (mean ± std).

| Dataset | Mechanism | RLF1 | TP | TR | TF1 | NED | CF1 | BERT-F1 | SS | ACC |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AQUA | SCoT | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
|  | MGCoT | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| ASDiv | SCoT | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | **0.13 ± 0.30** | 0.04 ± 0.08 | 0.00 ± 0.00 |
|  | MGCoT | 0.00 ± 0.02 | 0.00 ± 0.01 | 0.04 ± 0.20 | 0.00 ± 0.02 | 0.00 ± 0.00 | 0.00 ± 0.02 | 0.07 ± 0.22 | 0.02 ± 0.07 | 0.06 ± 0.24 |
| CLUTRR | SCoT | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
|  | MGCoT | 0.02 ± 0.07 | 0.01 ± 0.04 | 0.03 ± 0.10 | 0.02 ± 0.06 | 0.03 ± 0.09 | **0.07 ± 0.21** | 0.09 ± 0.26 | 0.03 ± 0.09 | **0.10 ± 0.30** |
| Date | SCoT | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
|  | MGCoT | **0.08 ± 0.27** | **0.08 ± 0.27** | 0.04 ± 0.14 | **0.05 ± 0.18** | **0.05 ± 0.18** | 0.07 ± 0.23 | 0.08 ± 0.26 | **0.07 ± 0.23** | 0.08 ± 0.27 |
| GSM8K | SCoT | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
|  | MGCoT | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| MultiArith | SCoT | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.01 | 0.00 ± 0.03 | 0.02 ± 0.11 | 0.00 ± 0.02 | 0.00 ± 0.00 |
|  | MGCoT | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| QASports | SCoT | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
|  | MGCoT | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| SayCan | SCoT | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
|  | MGCoT | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| StrategyQA | SCoT | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
|  | MGCoT | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.01 ± 0.04 | 0.02 ± 0.11 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| SVAMP | SCoT | 0.01 ± 0.05 | 0.00 ± 0.02 | 0.01 ± 0.07 | 0.00 ± 0.03 | 0.00 ± 0.01 | 0.01 ± 0.03 | 0.03 ± 0.16 | 0.01 ± 0.05 | 0.02 ± 0.14 |
|  | MGCoT | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| **All datasets** | SCoT | 0.00 ± 0.01 | 0.00 ± 0.01 | 0.00 ± 0.02 | 0.00 ± 0.01 | 0.00 ± 0.00 | 0.00 ± 0.01 | 0.02 ± 0.12 | 0.00 ± 0.03 | 0.00 ± 0.04 |
|  | MGCoT | 0.01 ± 0.09 | 0.01 ± 0.09 | 0.01 ± 0.08 | 0.01 ± 0.06 | 0.01 ± 0.07 | 0.01 ± 0.10 | 0.02 ± 0.14 | 0.01 ± 0.08 | 0.02 ± 0.15 |

### 3.2.8 Summary

MGCoT improved answer quality for openchat-7b and gemma2-2b on nearly all metrics and datasets, moderately for qwen2-1.5b, and selectively for phi3-mini (Date and MultiArith); it reduced quality strongly for llama3.2-1b and mildly for mistral-7b. Three regularities hold across models. First, the gains concentrate on Date and CLUTRR and, for gemma2-2b and openchat-7b, on GSM8K, whereas SayCan and QASports — the two benchmarks presented without task framing — are lower under MGCoT for most models. Second, MGCoT shifts overlap from recall towards precision: token precision, ROUGE-L, edit similarity and token F1 increase while token recall decreases for every model (−3% to −71%), in line with shorter and more focused responses that contain fewer unrelated words but cover fewer reference words. Third, the two semantic metrics diverge: BERTScore, which compares token embeddings and rewards concise answers, improved for gemma2-2b (ten of ten datasets), openchat-7b (eight of ten) and phi3-mini, was preserved for qwen2-1.5b and fell for llama3.2-1b and mistral-7b, whereas sentence-level similarity moved by dataset rather than by model, with large gains on Date (+0.20 to +0.29 for phi3-mini, gemma2-2b and openchat-7b), significant gains on StrategyQA for four models, and significant losses on SayCan for five of the six models with valid answers. Task accuracy, however, did not follow the alignment metrics: it was preserved for openchat-7b (43.8% vs 44.8%) and qwen2-1.5b (36.4% vs 36.6%) and significantly lower for phi3-mini (−8.2 points), gemma2-2b (−6.4), mistral-7b (−17.4) and llama3.2-1b (−32.6). Across the 60 model x dataset conditions MGCoT was significantly more accurate in one (qwen2-1.5b on CLUTRR, 28% to 48%) and significantly less accurate in 14, so the gains reported above concern generation cost and reference alignment rather than correctness.

Across the three dimensions, the results separate into two scenarios.

In the first scenario MGCoT is advantageous. For openchat-7b and qwen2-1.5b it reduced generation cost (−40.8% and −11.3% generation time) and improved reference alignment (openchat-7b: +83.2% ROUGE-L F1, +32.5% token F1, +2.7% BERTScore F1, significant on six to eight datasets) while preserving task accuracy, and qwen2-1.5b gained the only significant accuracy improvement of the study (CLUTRR, 28% to 48%). Gemma2-2b belongs to this group for cost and alignment (−49.1% generation time, +129.4% ROUGE-L F1, +3.7% BERTScore F1 on all ten datasets) but not for correctness, as its accuracy fell by 6.4 points. By dataset, the advantage was strongest on Date, CLUTRR and GSM8K.

For gemma2-2b and openchat-7b, MGCoT delivered simultaneous efficiency and alignment gains: 41–49% shorter generation, lower RAM on every dataset, significantly higher precision-type lexical overlap on six to eight datasets, and significantly higher BERT-F1 on eight to ten datasets. Qwen2-1.5b showed the same direction with smaller effects (11% shorter generation, lower CPU, moderate syntactic gains, semantic parity). By dataset, the advantage was strongest and most consistent on Date and CLUTRR, followed by GSM8K and StrategyQA.

For llama3.2-1b and mistral-7b, MGCoT lengthened generation (+14% and +29%) and reduced both lexical overlap and semantic similarity on most datasets, most strongly for llama3.2-1b (SS −58%, TF1 −62%), and it reduced task accuracy by 32.6 and 17.4 points respectively. Phi3-mini fell between the two scenarios, with gains on Date and MultiArith and losses on CLUTRR, SayCan and recall-type metrics. By dataset, SayCan and QASports — the two benchmarks without explicit task framing — were the least favourable to MGCoT across models, and recall-type metrics decreased even for models that otherwise benefited.

Irrespective of the scenario, MGCoT lowered system RAM (all seven models) and raised GPU utilisation (all seven models), while GPU memory was unaffected. Deepseek-r1-8b could not be evaluated for answer quality under either mechanism because its reasoning trace exhausted the 300-token budget, which identifies budget compliance as a prerequisite for deploying reasoning-distilled SLMs in constrained settings.
