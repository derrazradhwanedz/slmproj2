"""Evaluation entrypoint: runs SCoT and MGCoT on every dataset and saves results.

Run from the repository root as: python main.py
"""

import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import pandas as pd
import torch
from langchain_ollama import OllamaLLM

from logger import get_logger
from metrics import character as character_metrics
from metrics import lexical as lexical_metrics
from metrics import semantic as semantic_metrics
from metrics import token as token_metrics
from metrics.hardware import SystemProfiler
from nn import QualityProfilePredictor
from prompts.mgcot import mgcot_template
from prompts.scot import scot_template
from utils import load_yaml

logger = get_logger(__name__)

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_CONFIG = load_yaml("app.yaml")
TEST_FILENAME = "test.jsonl"


def _build_comparison_metrics() -> Dict[str, Any]:
    """Instantiate every comparison metric used to score a generated answer.

    Returns:
        A dict mapping output column name to a callable metric instance,
        each taking (prediction, reference) and returning a score.
    """
    return {
        "rouge1_f1": lexical_metrics.Rouge1(),
        "rouge2_f1": lexical_metrics.Rouge2(),
        "rougeL_f1": lexical_metrics.RougeL(),
        "bleu": lexical_metrics.Bleu(),
        "exact_match": lexical_metrics.ExactMatch(),
        "token_precision": token_metrics.TokenPrecision(),
        "token_recall": token_metrics.TokenRecall(),
        "token_f1": token_metrics.TokenF1(),
        "token_overlap_count": token_metrics.TokenOverlapCount(),
        "normalized_edit_distance": character_metrics.NormalizedEditDistance(),
        "char_f1": character_metrics.CharF1(),
        "bert_score_f1": semantic_metrics.BertScoreF1(),
        "semantic_similarity": semantic_metrics.SemanticSimilarity(),
    }


def _render_scot(question: str) -> str:
    """Render the SCoT prompt for a question.

    scot_template is a module-level singleton (its system instruction is
    fixed at import time); its user buffer is reset here so repeated calls
    across a dataset loop don't keep accumulating prior questions.

    Args:
        question: The raw question text.

    Returns:
        The formatted SCoT prompt.
    """
    scot_template.user = ""
    return scot_template.format(question)


def _render_mgcot(question: str, target_metrics: Dict[str, float]) -> str:
    """Render the MGCoT prompt for a question, injecting predicted target metrics.

    mgcot_template's system text contains a literal "{metrics_str}"
    placeholder set at import time. str.format() does not resolve braces
    inside an already-substituted value, so the placeholder is filled here
    via a plain string replace before building the final prompt.

    Args:
        question: The raw question text.
        target_metrics: Predicted target quality profile for the answer,
            as returned by QualityProfilePredictor.predict().

    Returns:
        The formatted MGCoT prompt.
    """
    metrics_str = "\n".join(f"- **{name}**: {value:.3f}" for name, value in target_metrics.items())
    system_with_metrics = mgcot_template.system.replace("{metrics_str}", metrics_str)

    mgcot_template.user = ""
    mgcot_template.add_text(question, mode="user")
    return mgcot_template.template.format(
        system_instruction=system_with_metrics,
        question=mgcot_template.user,
    )


class ModelEvaluator:
    """Evaluates one LLM on one dataset under both SCoT and MGCoT."""

    def __init__(self, model_name: str, dataset_path: str, dataset_name: str) -> None:
        """Initialize the LLM, quality predictor, and comparison metrics.

        Args:
            model_name: Ollama model tag (e.g. "phi3:mini").
            dataset_path: Directory holding this dataset's JSON/JSONL files.
            dataset_name: Human-readable dataset name, used in output rows.

        Raises:
            FileNotFoundError: If the trained QAPredictorNN model or its
                fitted scalers are missing (run src/nn/build_dataset.py
                then src/nn/train.py first).
        """
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name

        llm_cfg = _CONFIG["llm"]
        self.llm = OllamaLLM(
            model=model_name,
            num_gpu=llm_cfg["num_gpu"],
            num_predict=llm_cfg["num_predict"],
            temperature=llm_cfg["temperature"],
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        nnet_cfg = _CONFIG["nnet"]
        self.quality_predictor = QualityProfilePredictor(
            model_path=os.path.join(_APP_DIR, nnet_cfg["model_path"]),
            q_scaler_path=os.path.join(_APP_DIR, nnet_cfg["q_scaler_path"]),
            a_scaler_path=os.path.join(_APP_DIR, nnet_cfg["a_scaler_path"]),
            device=device,
        )

        self.sample_interval = _CONFIG["sample_interval"]
        self.comparison_metrics = _build_comparison_metrics()

        logger.info(f"Evaluator initialized for '{dataset_name}' using '{model_name}' on {device}")

    def _load_dataset(self) -> List[Dict[str, str]]:
        """Load this dataset's test-split records.

        Only TEST_FILENAME within dataset_path is read, so folders that
        also bundle train/dev splits alongside it (e.g. GSM8K, StrategyQA)
        don't get silently merged into the evaluation set.

        Returns:
            A list of question/answer record dicts. Empty list on failure.
        """
        test_file_path = os.path.join(self.dataset_path, TEST_FILENAME)
        if not os.path.isfile(test_file_path):
            logger.error(f"No {TEST_FILENAME} found in '{self.dataset_name}' ({self.dataset_path})")
            return []

        try:
            with open(test_file_path, "r", encoding="utf-8") as infile:
                content = infile.read().strip()
            data = [json.loads(line) for line in content.split("\n") if line.strip()]
        except (OSError, json.JSONDecodeError) as exc:
            logger.error(f"Failed to load dataset '{self.dataset_name}': {exc}")
            return []

        logger.info(f"Loaded {len(data)} records from '{self.dataset_name}'")
        return data

    def _run_mechanism(self, prompt: str, stream: bool = False) -> Dict[str, Any]:
        """Send a prompt to the LLM, profiling resources while it streams.

        Args:
            prompt: The fully rendered prompt to send.
            stream: If True, print each chunk of the SLM's response to the
                terminal as it arrives.

        Returns:
            A dict with the final answer text, elapsed time, and average
            resource usage sampled during generation.
        """
        profiler = SystemProfiler(sample_interval=self.sample_interval)
        start_time = time.time()
        final_answer = ""
        for chunk in profiler.wrap(self.llm.stream(prompt)):
            final_answer += chunk
            if stream:
                print(chunk, end="", flush=True)
        if stream:
            print()
        delta_time = time.time() - start_time

        return {
            "final_answer": final_answer,
            "delta_time_seconds": round(delta_time, 4),
            **profiler.get_averages(),
        }

    def _score_answer(self, final_answer: str, gold_answer: str) -> Dict[str, float]:
        """Score a generated answer against the gold answer.

        Args:
            final_answer: The model's generated answer.
            gold_answer: The reference answer.

        Returns:
            A dict of every comparison metric name to its value.
        """
        return {
            name: metric(final_answer, gold_answer)
            for name, metric in self.comparison_metrics.items()
        }

    def evaluate(self, max_records: Optional[int] = None, stream: bool = False) -> pd.DataFrame:
        """Run SCoT and MGCoT on every record, one row per (record, mechanism).

        Args:
            max_records: Cap on records to process. None processes all.
            stream: If True, print each chunk of the SLM's response to the
                terminal as it arrives.

        Returns:
            A DataFrame with two rows per input record (one per mechanism),
            sharing the same record_id so they can be compared directly.
        """
        data = self._load_dataset()
        if not data:
            return pd.DataFrame()
        if max_records is not None:
            data = data[:max_records]

        results = []
        for i, record in enumerate(data):
            question = str(record.get("question", ""))
            gold_answer = str(record.get("answer", record.get("explanation", "")))
            logger.info(f"[{self.dataset_name}] record {i + 1}/{len(data)}")

            base_row = {
                "record_id": i + 1,
                "dataset": self.dataset_name,
                "question": question,
                "gold_answer": gold_answer,
            }

            scot_prompt = _render_scot(question)
            scot_run = self._run_mechanism(scot_prompt, stream=stream)
            results.append({
                **base_row,
                "mechanism": "SCoT",
                **scot_run,
                **self._score_answer(scot_run["final_answer"], gold_answer),
            })

            target_metrics = self.quality_predictor.predict(question)
            mgcot_prompt = _render_mgcot(question, target_metrics)
            mgcot_run = self._run_mechanism(mgcot_prompt, stream=stream)
            results.append({
                **base_row,
                "mechanism": "MGCoT",
                **mgcot_run,
                **self._score_answer(mgcot_run["final_answer"], gold_answer),
                **{f"target_{name}": value for name, value in target_metrics.items()},
            })

        return pd.DataFrame(results)


class BenchmarkRunner:
    """Runs the SCoT/MGCoT evaluation sweep across one or many models."""

    def __init__(self) -> None:
        """Resolve dataset directories and output directory from app.yaml.

        Raises:
            FileNotFoundError: If dataset_dir does not exist.
            RuntimeError: If dataset_dir contains no dataset subfolders.
        """
        self.dataset_root = os.path.normpath(os.path.join(_APP_DIR, _CONFIG["dataset_dir"]))
        self.output_dir = os.path.normpath(os.path.join(_APP_DIR, _CONFIG["output_dir"]))
        self.dataset_dirs = self._discover_datasets()
        os.makedirs(self.output_dir, exist_ok=True)

    def _discover_datasets(self) -> List[str]:
        """Find every dataset subfolder under dataset_root.

        Returns:
            Paths to every dataset subfolder.

        Raises:
            FileNotFoundError: If dataset_root does not exist.
            RuntimeError: If dataset_root contains no subfolders.
        """
        if not os.path.isdir(self.dataset_root):
            raise FileNotFoundError(f"dataset_dir not found: {self.dataset_root}")

        dataset_dirs = [
            os.path.join(self.dataset_root, d)
            for d in os.listdir(self.dataset_root)
            if os.path.isdir(os.path.join(self.dataset_root, d))
        ]
        if not dataset_dirs:
            raise RuntimeError(f"No datasets found in {self.dataset_root}")

        logger.info(f"Found {len(dataset_dirs)} datasets: {[os.path.basename(d) for d in dataset_dirs]}")
        return dataset_dirs

    def _filter_datasets(self, datasets: Optional[List[str]]) -> List[str]:
        """Filter self.dataset_dirs down to the requested dataset names.

        Args:
            datasets: Dataset folder names to include (case-insensitive),
                e.g. ["aqua", "gsm8k"]. None includes every dataset found.

        Returns:
            Paths to the matching dataset subfolders.

        Raises:
            ValueError: If any requested name matches no discovered dataset.
        """
        if datasets is None:
            return self.dataset_dirs

        wanted = {name.lower() for name in datasets}
        matched = [d for d in self.dataset_dirs if os.path.basename(d).lower() in wanted]

        found_names = {os.path.basename(d).lower() for d in matched}
        missing = wanted - found_names
        if missing:
            raise ValueError(f"Unknown dataset name(s): {sorted(missing)}")

        return matched

    def process_one(
        self,
        model_name: str,
        max_records: Optional[int] = None,
        stream: bool = False,
        datasets: Optional[List[str]] = None,
    ) -> None:
        """Evaluate one model on every dataset and save per-dataset + combined CSVs.

        Args:
            model_name: Ollama model tag (e.g. "phi3:mini").
            max_records: Cap on records per dataset. Defaults to app.yaml's
                max_records if not given.
            stream: If True, print each chunk of the SLM's response to the
                terminal as it arrives.
            datasets: Dataset folder names to evaluate (case-insensitive),
                e.g. ["aqua", "gsm8k"]. None evaluates every dataset found.
        """
        if max_records is None:
            max_records = _CONFIG.get("max_records")

        safe_model_name = model_name.replace(":", "_")
        all_dfs = []

        for dataset_path in self._filter_datasets(datasets):
            dataset_name = os.path.basename(os.path.normpath(dataset_path))
            logger.info(f"Evaluating dataset: {dataset_name}")

            evaluator = ModelEvaluator(
                model_name=model_name,
                dataset_path=dataset_path,
                dataset_name=dataset_name,
            )
            df = evaluator.evaluate(max_records=max_records, stream=stream)
            if df.empty:
                logger.warning(f"No results for '{dataset_name}'")
                continue

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(self.output_dir, f"{timestamp}-{safe_model_name}-{dataset_name}.csv")
            df.to_csv(filepath, index=False)
            logger.info(f"Saved: {filepath}")
            all_dfs.append(df)

        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            combined_path = os.path.join(self.output_dir, f"combined_{timestamp}-{safe_model_name}-ALL.csv")
            combined_df.to_csv(combined_path, index=False)
            logger.info(f"Combined results saved: {combined_path}")

    def process_many(
        self,
        model_names: List[str],
        max_records: Optional[int] = None,
        stream: bool = False,
        datasets: Optional[List[str]] = None,
    ) -> None:
        """Evaluate multiple models, each on every dataset.

        Args:
            model_names: Ollama model tags to evaluate in sequence.
            max_records: Cap on records per dataset. Defaults to app.yaml's
                max_records if not given.
            stream: If True, print each chunk of the SLM's response to the
                terminal as it arrives.
            datasets: Dataset folder names to evaluate (case-insensitive),
                e.g. ["aqua", "gsm8k"]. None evaluates every dataset found.
        """
        for model_name in model_names:
            logger.info(f"Evaluating model: {model_name}")
            self.process_one(model_name, max_records=max_records, stream=stream, datasets=datasets)


def main() -> None:
    """CLI entrypoint: evaluate every model in model_names on every dataset."""
    # run all models in sequence, each on every dataset, saving per-dataset and combined CSVs
    runner = BenchmarkRunner()
    runner.process_many(
        model_names=["phi3:mini", 
                     "llama3.2:1b", 
                     "gemma2:2b", 
                     "qwen2:1.5b", 
                     "mistral:7b", 
                     "openchat:7b", 
                     "deepseek-r1:8b"],
        max_records=50,
        datasets=["aqua", 
                  "asdiv", 
                  "clutrr", 
                  "date", 
                  "gsm8k", 
                  "MultiArith", 
                  "QASports", 
                  "saycan", 
                  "StrategyQA", 
                  "SVAMP"],
    )
    
    # run a single model on every dataset, saving per-dataset and combined CSVs
    # runner.process_one(model_name="phi3:mini", max_records=2, stream=True)
    
    
if __name__ == "__main__":
    main()
