# ============================================================
# AITrustEval - Ragas Evaluation
# ============================================================
#
# Purpose:
#   Evaluate RAG/LLM answer quality using Ragas.
#
# Supported metrics:
#   - Faithfulness
#   - Answer Relevancy
#   - Answer Correctness
#   - Context Precision
#   - Context Recall
#
# Designed for:
#   - Offline execution
#   - Local Ollama
#   - Local Mistral / approved LLM models
#   - AITrustEval 5-field dataset
#
# Dataset fields:
#   question
#   context
#   answer
#   metadata
#   ground_truth
# ============================================================

from typing import Dict, Optional
import ast
import json

import pandas as pd

from evaluation.llm.model_manager import (
    select_llm_model,
    select_embedding_model,
    get_model_summary,
)

from evaluation.llm.ollama_client import (
    is_ollama_available,
    is_model_available,
)


# ============================================================
# RAGAS IMPORT
# ============================================================

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
    Check whether the local Ragas environment is ready.

    No cloud API is used.
    """

    selected_model = (
        model
        or select_llm_model(
            DEFAULT_MODEL
        )
    )

    ollama_available = (
        is_ollama_available()
    )

    model_available = False

    if (
        ollama_available
        and selected_model
    ):

        try:

            model_available = (
                is_model_available(
                    selected_model
                )
            )

        except Exception:

            model_available = False

    return {

        "ragas_installed":
            RAGAS_AVAILABLE,

        "ollama_available":
            ollama_available,

        "model":
            selected_model,

        "model_available":
            model_available,

        "embedding_model":
            select_embedding_model(),

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
    Convert missing values into empty strings.
    """

    if value is None:
        return ""

    if isinstance(
        value,
        float
    ) and pd.isna(value):

        return ""

    return value


# ============================================================
# CONTEXT PARSING
# ============================================================

def prepare_context(value):
    """
    Convert context into a list of strings.

    Supports:

        list
        tuple
        JSON list
        Python list string
        dictionary
        normal string

    Example:

        '["chunk 1", "chunk 2"]'

    becomes:

        ["chunk 1", "chunk 2"]
    """

    value = clean_value(
        value
    )

    # --------------------------------------------------------
    # Already a list
    # --------------------------------------------------------

    if isinstance(
        value,
        list
    ):

        return [
            str(item)
            for item in value
        ]

    # --------------------------------------------------------
    # Tuple
    # --------------------------------------------------------

    if isinstance(
        value,
        tuple
    ):

        return [
            str(item)
            for item in value
        ]

    # --------------------------------------------------------
    # Dictionary
    # --------------------------------------------------------

    if isinstance(
        value,
        dict
    ):

        # Common context keys
        for key in [
            "context",
            "contexts",
            "documents",
            "retrieved_contexts",
            "chunks",
        ]:

            if key in value:

                nested = value[key]

                return prepare_context(
                    nested
                )

        return [
            json.dumps(
                value,
                ensure_ascii=False
            )
        ]

    # --------------------------------------------------------
    # String
    # --------------------------------------------------------

    if isinstance(
        value,
        str
    ):

        text = value.strip()

        if not text:

            return []

        # ----------------------------------------------------
        # Try JSON
        # ----------------------------------------------------

        if (
            text.startswith("[")
            and text.endswith("]")
        ):

            try:

                parsed = json.loads(
                    text
                )

                if isinstance(
                    parsed,
                    list
                ):

                    return [
                        str(item)
                        for item in parsed
                    ]

            except Exception:

                pass

        # ----------------------------------------------------
        # Try Python literal
        # ----------------------------------------------------

        if (
            text.startswith("[")
            and text.endswith("]")
        ):

            try:

                parsed = ast.literal_eval(
                    text
                )

                if isinstance(
                    parsed,
                    list
                ):

                    return [
                        str(item)
                        for item in parsed
                    ]

            except Exception:

                pass

        # ----------------------------------------------------
        # Normal text
        # ----------------------------------------------------

        return [text]

    # --------------------------------------------------------
    # Other types
    # --------------------------------------------------------

    return [
        str(value)
    ]


# ============================================================
# GROUND TRUTH PREPARATION
# ============================================================

def prepare_ground_truth(value):
    """
    Convert ground truth into a single string.

    Handles:

        string
        list
        tuple
        dictionary
        JSON/Python list strings
    """

    value = clean_value(
        value
    )

    if isinstance(
        value,
        list
    ):

        return " ".join(
            str(item)
            for item in value
        )

    if isinstance(
        value,
        tuple
    ):

        return " ".join(
            str(item)
            for item in value
        )

    if isinstance(
        value,
        dict
    ):

        return json.dumps(
            value,
            ensure_ascii=False
        )

    if isinstance(
        value,
        str
    ):

        text = value.strip()

        if (
            text.startswith("[")
            and text.endswith("]")
        ):

            # Try JSON
            try:

                parsed = json.loads(
                    text
                )

                if isinstance(
                    parsed,
                    list
                ):

                    return " ".join(
                        str(item)
                        for item in parsed
                    )

            except Exception:

                pass

            # Try Python literal
            try:

                parsed = ast.literal_eval(
                    text
                )

                if isinstance(
                    parsed,
                    list
                ):

                    return " ".join(
                        str(item)
                        for item in parsed
                    )

            except Exception:

                pass

        return text

    return str(value)


# ============================================================
# COLUMN MAPPING
# ============================================================

def get_ragas_column_mapping(
    df: pd.DataFrame
) -> Dict:
    """
    Detect the five-field AITrustEval schema.

    Returns actual dataframe column names.
    """

    aliases = {

        "question": [
            "question",
            "query",
            "user_question",
            "prompt",
        ],

        "context": [
            "context",
            "contexts",
            "retrieved_context",
            "retrieved_contexts",
        ],

        "answer": [
            "answer",
            "response",
            "model_answer",
            "generated_answer",
            "output",
        ],

        "ground_truth": [
            "ground_truth",
            "groundtruth",
            "reference",
            "reference_answer",
            "expected_answer",
        ],

    }

    normalized = {

        str(column)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_"):
        column

        for column in df.columns
    }

    mapping = {}

    for field, field_aliases in aliases.items():

        mapping[field] = None

        for alias in field_aliases:

            alias_normalized = (
                alias
                .strip()
                .lower()
                .replace(" ", "_")
                .replace("-", "_")
            )

            if (
                alias_normalized
                in normalized
            ):

                mapping[field] = (
                    normalized[
                        alias_normalized
                    ]
                )

                break

    return mapping


# ============================================================
# VALIDATE RAGAS MAPPING
# ============================================================

def validate_ragas_mapping(
    mapping: Dict
):
    """
    Validate that all fields required by Ragas
    are available.
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

    return (
        len(missing) == 0,
        missing
    )


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_ragas_dataset(
    df: pd.DataFrame,
    mapping: Dict,
) -> EvaluationDataset:
    """
    Convert dataframe into Ragas EvaluationDataset.
    """

    valid, missing = (
        validate_ragas_mapping(
            mapping
        )
    )

    if not valid:

        raise ValueError(
            "Missing required Ragas fields: "
            + ", ".join(missing)
        )

    samples = []

    for _, row in df.iterrows():

        sample = {

            "user_input": str(
                clean_value(
                    row[
                        mapping["question"]
                    ]
                )
            ),

            "response": str(
                clean_value(
                    row[
                        mapping["answer"]
                    ]
                )
            ),

            "reference": (
                prepare_ground_truth(
                    row[
                        mapping["ground_truth"]
                    ]
                )
            ),

            "retrieved_contexts": (
                prepare_context(
                    row[
                        mapping["context"]
                    ]
                )
            ),
        }

        samples.append(
            sample
        )

    return (
        EvaluationDataset.from_list(
            samples
        )
    )


# ============================================================
# LOCAL LLM CREATION
# ============================================================

def create_local_llm(
    model: Optional[str] = None,
):
    """
    Create the local Ragas LLM through Ollama.

    No cloud API is used.
    """

    selected_model = (
        select_llm_model(
            model
            or DEFAULT_MODEL
        )
    )

    if not selected_model:

        raise RuntimeError(
            "No local LLM model is available."
        )

    try:

        from openai import OpenAI
        from ragas.llms import (
            llm_factory
        )

    except ImportError as exc:

        raise RuntimeError(
            "Required Ragas/OpenAI "
            "components are unavailable."
        ) from exc

    client = OpenAI(

        api_key="ollama",

        base_url=(
            "http://localhost:11434/v1"
        ),
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
    Run Ragas evaluation.

    Returns a standardized dictionary.
    """

    # --------------------------------------------------------
    # Environment
    # --------------------------------------------------------

    environment = (
        check_ragas_environment(
            model
        )
    )

    # --------------------------------------------------------
    # Ragas installed?
    # --------------------------------------------------------

    if not environment[
        "ragas_installed"
    ]:

        return {

            "status":
                "unavailable",

            "message":
                "Ragas is not installed.",

            "environment":
                environment,
        }

    # --------------------------------------------------------
    # Ollama available?
    # --------------------------------------------------------

    if not environment[
        "ollama_available"
    ]:

        return {

            "status":
                "unavailable",

            "message": (
                "Ollama is unavailable. "
                "This is expected on the "
                "development laptop when "
                "Ollama is not installed "
                "or running."
            ),

            "environment":
                environment,
        }

    # --------------------------------------------------------
    # Model available?
    # --------------------------------------------------------

    if not environment[
        "model_available"
    ]:

        return {

            "status":
                "unavailable",

            "message": (
                f"Selected local model "
                f"'{environment['model']}' "
                f"is unavailable."
            ),

            "environment":
                environment,
        }

    # --------------------------------------------------------
    # Column mapping
    # --------------------------------------------------------

    mapping = (
        get_ragas_column_mapping(
            df
        )
    )

    # --------------------------------------------------------
    # Validate mapping
    # --------------------------------------------------------

    valid, missing = (
        validate_ragas_mapping(
            mapping
        )
    )

    if not valid:

        return {

            "status":
                "unavailable",

            "message": (
                "Missing required Ragas "
                "fields: "
                + ", ".join(missing)
            ),

            "mapping":
                mapping,

            "environment":
                environment,
        }

    # --------------------------------------------------------
    # Prepare and evaluate
    # --------------------------------------------------------

    try:

        evaluation_dataset = (
            prepare_ragas_dataset(
                df,
                mapping
            )
        )

        llm = create_local_llm(
            environment["model"]
        )

        # ----------------------------------------------------
        # Metric objects
        # ----------------------------------------------------

        metric_objects = {

            "faithfulness":
                Faithfulness(),

            "answer_relevancy":
                AnswerRelevancy(),

            "answer_correctness":
                AnswerCorrectness(),

            "context_precision":
                ContextPrecision(),

            "context_recall":
                ContextRecall(),
        }

        # ----------------------------------------------------
        # Select metrics
        # ----------------------------------------------------

        if selected_metrics:

            metrics = [

                metric_objects[name]

                for name
                in selected_metrics

                if name
                in metric_objects
            ]

        else:

            metrics = list(
                metric_objects.values()
            )

        if not metrics:

            return {

                "status":
                    "error",

                "message":
                    "No valid Ragas metrics were selected.",

                "mapping":
                    mapping,

                "environment":
                    environment,
            }

        # ----------------------------------------------------
        # Run Ragas
        # ----------------------------------------------------

        result = evaluate(

            evaluation_dataset,

            metrics=metrics,

            llm=llm,
        )

        # ----------------------------------------------------
        # Convert results
        # ----------------------------------------------------

        try:

            result_df = (
                result.to_pandas()
            )

            records = (
                result_df.to_dict(
                    orient="records"
                )
            )

        except Exception:

            result_df = None
            records = []

        # ----------------------------------------------------
        # Calculate summary
        # ----------------------------------------------------

        summary = {}

        if result_df is not None:

            for metric_name in [
                "faithfulness",
                "answer_relevancy",
                "answer_correctness",
                "context_precision",
                "context_recall",
            ]:

                if (
                    metric_name
                    in result_df.columns
                ):

                    numeric_values = pd.to_numeric(
                        result_df[
                            metric_name
                        ],
                        errors="coerce"
                    )

                    if numeric_values.notna().any():

                        summary[
                            metric_name
                        ] = float(
                            numeric_values.mean()
                        )

        return {

            "status":
                "success",

            "message":
                "Ragas evaluation completed.",

            "model":
                environment["model"],

            "embedding_model":
                environment[
                    "embedding_model"
                ],

            "mapping":
                mapping,

            "metrics":
                records,

            "summary":
                summary,

            "row_count":
                len(df),

            "raw_result":
                result,

            "environment":
                environment,
        }

    except Exception as exc:

        return {

            "status":
                "error",

            "message":
                str(exc),

            "model":
                environment["model"],

            "embedding_model":
                environment[
                    "embedding_model"
                ],

            "mapping":
                mapping,

            "environment":
                environment,
        }


# ============================================================
# CLI TEST
# ============================================================

if __name__ == "__main__":

    DATASET_PATH = (
        "data/sample/"
        "synthetic_5field_dataset.csv"
    )

    print(
        "=" * 65
    )

    print(
        "AITrustEval - Ragas Evaluation Test"
    )

    print(
        "=" * 65
    )

    # --------------------------------------------------------
    # Environment
    # --------------------------------------------------------

    print(
        "\nChecking Ragas environment..."
    )

    environment = (
        check_ragas_environment()
    )

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

    # --------------------------------------------------------
    # Model summary
    # --------------------------------------------------------

    print(
        "\nModel summary:"
    )

    summary = (
        get_model_summary()
    )

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

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print(
        "\nLoading dataset..."
    )

    df = pd.read_csv(
        DATASET_PATH
    )

    print(
        f"Rows: {len(df)}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    # --------------------------------------------------------
    # Mapping
    # --------------------------------------------------------

    mapping = (
        get_ragas_column_mapping(
            df
        )
    )

    print(
        "\n========== COLUMN MAPPING =========="
    )

    for field, column in mapping.items():

        print(
            f"{field:<15} -> {column}"
        )

    # --------------------------------------------------------
    # Run
    # --------------------------------------------------------

    print(
        "\nRunning Ragas evaluation..."
    )

    results = (
        run_ragas_evaluation(
            df
        )
    )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    print(
        "\n========== RESULT =========="
    )

    print(
        f"Status: "
        f"{results.get('status')}"
    )

    print(
        f"Message: "
        f"{results.get('message')}"
    )

    if results.get(
        "summary"
    ):

        print(
            "\n========== SUMMARY =========="
        )

        for metric, score in (
            results[
                "summary"
            ].items()
        ):

            print(
                f"{metric:<25} -> "
                f"{score:.4f}"
            )

    print(
        "\n" + "=" * 65
    )

    print(
        "RAGAS TEST COMPLETE"
    )

    print(
        "=" * 65
    )