"""
AITrustEval - Ragas Evaluation

Uses the centralized Model Manager so the LLM judge is not hard-coded.
Designed for offline/local Ollama execution.
"""

from typing import Dict, Optional

import pandas as pd

from evaluation.dataset.schema_detector import validate_dataset
from evaluation.llm.model_manager import (
    select_llm_model,
    select_embedding_model,
    get_model_summary,
)

from evaluation.llm.ollama_client import (
    is_ollama_available,
)

# Ragas
try:
    from ragas import EvaluationDataset, evaluate
    from ragas.metrics.collections import (
        Faithfulness,
        AnswerRelevancy,
        AnswerCorrectness,
        ContextPrecision,
        ContextRecall,
    )

    RAGAS_AVAILABLE = True

except Exception:
    RAGAS_AVAILABLE = False


# ============================================================
# DEFAULTS
# ============================================================

DEFAULT_MODEL = "mistral"


# ============================================================
# ENVIRONMENT CHECK
# ============================================================

def check_ragas_environment(
    model: Optional[str] = None,
) -> Dict:
    """
    Check whether the local environment is ready for Ragas.

    Model selection is delegated to the Model Manager.
    """

    selected_model = model or select_llm_model(DEFAULT_MODEL)

    ollama_available = is_ollama_available()

    model_available = False

    if ollama_available and selected_model:
        from evaluation.llm.ollama_client import is_model_available

        try:
            model_available = is_model_available(selected_model)
        except Exception:
            model_available = False

    return {
        "ragas_installed": RAGAS_AVAILABLE,
        "ollama_available": ollama_available,
        "model": selected_model,
        "model_available": model_available,
        "embedding_model": select_embedding_model(),
        "ready": (
            RAGAS_AVAILABLE
            and ollama_available
            and model_available
        ),
    }


# ============================================================
# VALUE CLEANING
# ============================================================

def clean_value(value):
    """
    Convert dataset values into formats accepted by Ragas.
    """

    if value is None:
        return ""

    if isinstance(value, float) and pd.isna(value):
        return ""

    return value


def prepare_context(value):
    """
    Convert context values into a list of strings.

    Handles:
    - list
    - tuple
    - string
    - string representation of a list
    """

    value = clean_value(value)

    if isinstance(value, list):
        return [str(item) for item in value]

    if isinstance(value, tuple):
        return [str(item) for item in value]

    if isinstance(value, str):

        text = value.strip()

        # Try parsing Python-style list strings.
        if text.startswith("[") and text.endswith("]"):
            try:
                import ast

                parsed = ast.literal_eval(text)

                if isinstance(parsed, list):
                    return [str(item) for item in parsed]

            except Exception:
                pass

        return [text]

    return [str(value)]


# ============================================================
# COLUMN MAPPING
# ============================================================

def get_ragas_column_mapping(df: pd.DataFrame) -> Dict:
    """
    Automatically detect the columns required for Ragas.
    """

    validation = validate_dataset(df)

    mapping = validation.get("mapping", {})

    return {
        "question": mapping.get("question"),
        "answer": mapping.get("answer"),
        "ground_truth": mapping.get("ground_truth"),
        "context": mapping.get("context"),
    }


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_ragas_dataset(
    df: pd.DataFrame,
    mapping: Dict,
) -> EvaluationDataset:
    """
    Convert a normal dataframe into a Ragas EvaluationDataset.
    """

    required = [
        "question",
        "answer",
        "ground_truth",
        "context",
    ]

    missing = [
        field
        for field in required
        if not mapping.get(field)
    ]

    if missing:
        raise ValueError(
            "Missing required Ragas fields: "
            + ", ".join(missing)
        )

    samples = []

    for _, row in df.iterrows():

        sample = {
            "user_input": str(
                clean_value(
                    row[mapping["question"]]
                )
            ),

            "response": str(
                clean_value(
                    row[mapping["answer"]]
                )
            ),

            "reference": str(
                clean_value(
                    row[mapping["ground_truth"]]
                )
            ),

            "retrieved_contexts": prepare_context(
                row[mapping["context"]]
            ),
        }

        samples.append(sample)

    return EvaluationDataset.from_list(samples)


# ============================================================
# LOCAL LLM CREATION
# ============================================================

def create_local_llm(
    model: Optional[str] = None,
):
    """
    Create a local Ragas LLM using Ollama.

    No cloud API is used.
    """

    selected_model = select_llm_model(
        model or DEFAULT_MODEL
    )

    if not selected_model:
        raise RuntimeError(
            "No local LLM model is available."
        )

    try:
        from openai import OpenAI
        from ragas.llms import llm_factory

    except ImportError as exc:
        raise RuntimeError(
            "Required Ragas/OpenAI components are unavailable."
        ) from exc

    client = OpenAI(
        api_key="ollama",
        base_url="http://localhost:11434/v1",
    )

    return llm_factory(
        selected_model,
        provider="openai",
        client=client,
    )


# ============================================================
# RAGAS EVALUATION
# ============================================================

def run_ragas_evaluation(
    df: pd.DataFrame,
    model: Optional[str] = None,
    selected_metrics: Optional[list] = None,
) -> Dict:
    """
    Run Ragas evaluation using the centralized Model Manager.

    Parameters
    ----------
    df:
        Input dataset.

    model:
        Optional preferred local LLM.

    selected_metrics:
        Optional list of metric names.

    Returns
    -------
    dict
        Standardized evaluation response.
    """

    environment = check_ragas_environment(model)

    if not environment["ragas_installed"]:
        return {
            "status": "unavailable",
            "message": "Ragas is not installed.",
            "environment": environment,
        }

    if not environment["ollama_available"]:
        return {
            "status": "unavailable",
            "message": (
                "Ollama is unavailable. "
                "This is expected on the development laptop "
                "if Ollama is not installed."
            ),
            "environment": environment,
        }

    if not environment["model_available"]:
        return {
            "status": "unavailable",
            "message": (
                f"Selected local model "
                f"'{environment['model']}' is unavailable."
            ),
            "environment": environment,
        }

    mapping = get_ragas_column_mapping(df)

    try:
        evaluation_dataset = prepare_ragas_dataset(
            df,
            mapping,
        )

        llm = create_local_llm(
            environment["model"]
        )

        # ----------------------------------------------------
        # Metric selection
        # ----------------------------------------------------

        metric_objects = {
            "faithfulness": Faithfulness(),
            "answer_relevancy": AnswerRelevancy(),
            "answer_correctness": AnswerCorrectness(),
            "context_precision": ContextPrecision(),
            "context_recall": ContextRecall(),
        }

        if selected_metrics:

            metrics = [
                metric_objects[name]
                for name in selected_metrics
                if name in metric_objects
            ]

        else:

            metrics = list(
                metric_objects.values()
            )

        # ----------------------------------------------------
        # Run Ragas
        # ----------------------------------------------------

        result = evaluate(
            evaluation_dataset,
            metrics=metrics,
            llm=llm,
        )

        # Convert result to dataframe when possible.
        try:
            result_df = result.to_pandas()

            records = result_df.to_dict(
                orient="records"
            )

        except Exception:

            result_df = None
            records = []

        return {
            "status": "success",
            "message": "Ragas evaluation completed.",
            "model": environment["model"],
            "embedding_model": environment[
                "embedding_model"
            ],
            "mapping": mapping,
            "metrics": records,
            "raw_result": result,
            "environment": environment,
        }

    except Exception as exc:

        return {
            "status": "error",
            "message": str(exc),
            "model": environment["model"],
            "embedding_model": environment[
                "embedding_model"
            ],
            "mapping": mapping,
            "environment": environment,
        }


# ============================================================
# CLI TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("AITrustEval - Ragas Environment")
    print("=" * 60)

    environment = check_ragas_environment()

    print(
        f"\nRagas installed: "
        f"{environment['ragas_installed']}"
    )

    print(
        f"Ollama available: "
        f"{environment['ollama_available']}"
    )

    print(
        f"Selected LLM: "
        f"{environment['model']}"
    )

    print(
        f"Selected embedding model: "
        f"{environment['embedding_model']}"
    )

    print(
        f"LLM available: "
        f"{environment['model_available']}"
    )

    print(
        f"Environment ready: "
        f"{environment['ready']}"
    )

    print("\nModel summary:")

    summary = get_model_summary()

    print(
        f"  Configured/Detected LLMs: "
        f"{summary['llm_count']}"
    )

    print(
        f"  Configured/Detected Embeddings: "
        f"{summary['embedding_count']}"
    )

    print(
        f"  Configured/Detected Vision: "
        f"{summary['vision_count']}"
    )

    print("\n" + "=" * 60)