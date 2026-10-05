"""
AITrustEval - Main Evaluation Engine

Central orchestration layer for:
    1. Dataset schema detection
    2. Dataset validation
    3. Metric availability
    4. Semantic retrieval evaluation
    5. Ragas / LLM evaluation
    6. Security evaluation
    7. Reliability evaluation

The engine is designed for:
    - Offline execution
    - Local Ollama models
    - DRDO Ubuntu environment
    - Windows development environment
    - 5-field dataset:
        question
        context
        answer
        metadata
        ground_truth
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, Optional

import pandas as pd


# ============================================================
# PROJECT IMPORTS
# ============================================================

from evaluation.dataset.metric_availability import (
    build_metric_availability,
)

from evaluation.retrieval.evaluator import (
    evaluate_semantic_retrieval,
)

from evaluation.ragas.ragas_evaluator import (
    run_ragas_evaluation,
)


# ============================================================
# OPTIONAL SECURITY IMPORT
# ============================================================

try:
    from evaluation.security.security_evaluator import (
        evaluate_security,
        summarize_security,
    )

    SECURITY_AVAILABLE = True

except Exception as security_import_error:

    SECURITY_AVAILABLE = False
    SECURITY_IMPORT_ERROR = str(security_import_error)

    evaluate_security = None
    summarize_security = None


# ============================================================
# OPTIONAL RELIABILITY IMPORT
# ============================================================

try:
    from evaluation.reliability.reliability_evaluator import (
        evaluate_reliability,
    )

    RELIABILITY_AVAILABLE = True

except Exception as reliability_import_error:

    RELIABILITY_AVAILABLE = False
    RELIABILITY_IMPORT_ERROR = str(reliability_import_error)

    evaluate_reliability = None


# ============================================================
# FIELD ALIASES
# ============================================================

FIELD_ALIASES = {

    "question": [
        "question",
        "query",
        "user_query",
        "user_question",
        "prompt",
        "input",
    ],

    "context": [
        "context",
        "contexts",
        "retrieved_context",
        "retrieved_contexts",
        "retrieved_documents",
        "documents",
        "retrieval_context",
    ],

    "answer": [
        "answer",
        "response",
        "generated_answer",
        "model_answer",
        "output",
        "assistant_response",
    ],

    "ground_truth": [
        "ground_truth",
        "groundtruth",
        "ground_truths",
        "reference",
        "reference_answer",
        "expected_answer",
        "target",
    ],

    "metadata": [
        "metadata",
        "meta",
        "information",
        "additional_metadata",
    ],

    "latency": [
        "latency",
        "response_time",
        "response_time_ms",
        "latency_ms",
        "time_taken",
        "duration",
        "duration_ms",
    ],

    "status": [
        "status",
        "success",
        "success_status",
        "result",
        "outcome",
        "error",
        "failure",
    ],
}


# ============================================================
# NORMALIZE COLUMN NAME
# ============================================================

def normalize_column_name(value: Any) -> str:

    text = str(value).strip().lower()

    replacements = [
        (" ", "_"),
        ("-", "_"),
        ("/", "_"),
        (".", "_"),
    ]

    for old, new in replacements:
        text = text.replace(old, new)

    while "__" in text:
        text = text.replace("__", "_")

    return text


# ============================================================
# DETECT SCHEMA
# ============================================================

def detect_schema(df: pd.DataFrame) -> Dict[str, Optional[str]]:

    normalized_columns = {
        normalize_column_name(column): column
        for column in df.columns
    }

    schema = {}

    for logical_field, aliases in FIELD_ALIASES.items():

        detected_column = None

        for alias in aliases:

            normalized_alias = normalize_column_name(alias)

            if normalized_alias in normalized_columns:

                detected_column = normalized_columns[
                    normalized_alias
                ]

                break

        schema[logical_field] = detected_column

    return schema


# ============================================================
# VALIDATE DATASET
# ============================================================

def validate_dataset(
    df: pd.DataFrame,
    schema: Dict[str, Optional[str]],
) -> Dict[str, Any]:

    required_fields = [
        "question",
        "context",
        "answer",
        "ground_truth",
    ]

    missing_fields = [
        field
        for field in required_fields
        if not schema.get(field)
    ]

    warnings = []

    if not schema.get("latency"):

        warnings.append(
            "Latency field was not detected. "
            "Latency metrics cannot be calculated directly "
            "unless latency information exists inside metadata."
        )

    if not schema.get("status"):

        warnings.append(
            "Status field was not detected. "
            "Success/failure metrics may be unavailable."
        )

    if df.empty:

        return {
            "valid": False,
            "missing_fields": missing_fields,
            "warnings": warnings,
            "message": "Dataset is empty.",
        }

    if missing_fields:

        return {
            "valid": False,
            "missing_fields": missing_fields,
            "warnings": warnings,
            "message": (
                "Required dataset fields are missing: "
                + ", ".join(missing_fields)
            ),
        }

    return {
        "valid": True,
        "missing_fields": [],
        "warnings": warnings,
        "message": "All required dataset fields were detected.",
    }


# ============================================================
# CHECK REPEATED QUESTIONS
# ============================================================

def has_repeated_questions(
    df: pd.DataFrame,
    schema: Dict[str, Optional[str]],
) -> bool:

    question_column = schema.get("question")

    if not question_column:
        return False

    try:

        questions = (
            df[question_column]
            .dropna()
            .astype(str)
            .str.strip()
        )

        if questions.empty:
            return False

        return bool(questions.duplicated().any())

    except Exception:
        return False


# ============================================================
# METRIC AVAILABILITY
# ============================================================

def get_metric_availability(
    df: pd.DataFrame,
    schema: Dict[str, Optional[str]],
) -> Dict[str, bool]:

    try:

        availability = build_metric_availability(
            df,
            schema,
        )

    except TypeError:

        try:

            availability = build_metric_availability(
                {
                    "detected_columns": schema
                }
            )

        except Exception:

            availability = {}

    except Exception:

        availability = {}

    if not isinstance(availability, dict):

        availability = {}

    # --------------------------------------------------------
    # Ensure expected fields exist
    # --------------------------------------------------------

    has_question = bool(schema.get("question"))
    has_context = bool(schema.get("context"))
    has_answer = bool(schema.get("answer"))
    has_ground_truth = bool(schema.get("ground_truth"))
    has_metadata = bool(schema.get("metadata"))
    has_latency = bool(schema.get("latency"))
    has_status = bool(schema.get("status"))

    semantic_retrieval_ready = (
        has_context
        and has_ground_truth
    )

    # --------------------------------------------------------
    # Baseline answer quality
    # --------------------------------------------------------

    availability["answer_relevancy"] = (
        has_question
        and has_answer
    )

    availability["answer_correctness"] = (
        has_answer
        and has_ground_truth
    )

    availability["context_answer_similarity"] = (
        has_context
        and has_answer
    )

    # --------------------------------------------------------
    # RAGAS
    # --------------------------------------------------------

    availability["faithfulness"] = (
        has_context
        and has_answer
    )

    availability["ragas_answer_relevancy"] = (
        has_question
        and has_answer
    )

    availability["ragas_answer_correctness"] = (
        has_answer
        and has_ground_truth
    )

    availability["context_precision"] = (
        has_context
        and has_ground_truth
    )

    availability["context_recall"] = (
        has_context
        and has_ground_truth
    )

    # --------------------------------------------------------
    # Semantic retrieval
    # --------------------------------------------------------

    availability["semantic_retrieval"] = (
        semantic_retrieval_ready
    )

    availability["precision_at_k"] = (
        semantic_retrieval_ready
    )

    availability["recall_at_k"] = (
        semantic_retrieval_ready
    )

    availability["mrr"] = (
        semantic_retrieval_ready
    )

    availability["ndcg"] = (
        semantic_retrieval_ready
    )

    availability["hit_rate"] = (
        semantic_retrieval_ready
    )

    # --------------------------------------------------------
    # Reliability
    # --------------------------------------------------------

    availability["latency"] = has_latency

    availability["success_failure"] = has_status

    # Consistency requires repeated questions.
    availability["consistency"] = has_repeated_questions(
        df,
        schema,
    )

    # Robustness requires prompt variations / perturbations.
    robustness_columns = [
        "prompt_variation",
        "prompt_variant",
        "perturbation",
        "variation",
        "adversarial_prompt",
    ]

    normalized_columns = {
        normalize_column_name(column)
        for column in df.columns
    }

    availability["robustness"] = any(
        normalize_column_name(column)
        in normalized_columns
        for column in robustness_columns
    )

    availability["metadata"] = has_metadata

    return availability


# ============================================================
# RETRIEVAL EVALUATION
# ============================================================

def run_retrieval_evaluation(
    df: pd.DataFrame,
    schema: Dict[str, Optional[str]],
) -> Dict[str, Any]:

    if not schema.get("context"):

        return {
            "status": "unavailable",
            "message": (
                "Context field was not detected."
            ),
        }

    if not schema.get("ground_truth"):

        return {
            "status": "unavailable",
            "message": (
                "Ground-truth field was not detected."
            ),
        }

    try:

        result = evaluate_semantic_retrieval(
            df
        )

        if isinstance(result, dict):

            return result

        return {
            "status": "error",
            "message": (
                "Retrieval evaluator returned "
                "an unexpected result type."
            ),
        }

    except Exception as exc:

        return {
            "status": "error",
            "message": str(exc),
        }


# ============================================================
# RAGAS EVALUATION
# ============================================================

def run_ragas_layer(
    df: pd.DataFrame,
    schema: Dict[str, Optional[str]],
) -> Dict[str, Any]:

    required_fields = [
        "question",
        "context",
        "answer",
        "ground_truth",
    ]

    missing_fields = [
        field
        for field in required_fields
        if not schema.get(field)
    ]

    if missing_fields:

        return {
            "status": "unavailable",
            "message": (
                "Ragas requires: "
                + ", ".join(required_fields)
                + ". Missing: "
                + ", ".join(missing_fields)
            ),
        }

    try:

        result = run_ragas_evaluation(
            df
        )

        if isinstance(result, dict):

            return result

        return {
            "status": "error",
            "message": (
                "Ragas evaluator returned "
                "an unexpected result type."
            ),
        }

    except Exception as exc:

        return {
            "status": "error",
            "message": str(exc),
        }


# ============================================================
# SECURITY EVALUATION
# ============================================================

def run_security_evaluation(
    df: pd.DataFrame,
    schema: Dict[str, Optional[str]],
) -> Dict[str, Any]:

    question_column = schema.get("question")
    answer_column = schema.get("answer")

    # --------------------------------------------------------
    # Required fields
    # --------------------------------------------------------

    if not question_column or not answer_column:

        return {
            "status": "unavailable",
            "message": (
                "Security evaluation requires "
                "question and answer fields."
            ),
            "summary": {},
            "results": pd.DataFrame(),
        }

    # --------------------------------------------------------
    # Security module unavailable
    # --------------------------------------------------------

    if not SECURITY_AVAILABLE:

        return {
            "status": "error",
            "message": (
                "Security evaluator could not be imported: "
                + SECURITY_IMPORT_ERROR
            ),
            "summary": {},
            "results": pd.DataFrame(),
        }

    # --------------------------------------------------------
    # Run security evaluation
    # --------------------------------------------------------

    try:

        results = evaluate_security(
            df,
            question_column=question_column,
            answer_column=answer_column,
            use_mistral=True,
            model="mistral",
        )

        # ----------------------------------------------------
        # Summary
        # ----------------------------------------------------

        try:

            summary = summarize_security(
                results
            )

        except Exception:

            summary = {}

        # ----------------------------------------------------
        # Convert summary if necessary
        # ----------------------------------------------------

        if not isinstance(summary, dict):

            summary = {}

        # ----------------------------------------------------
        # Determine Mistral status
        # ----------------------------------------------------

        mistral_status = "unavailable"

        if isinstance(results, pd.DataFrame):

            if "mistral_status" in results.columns:

                statuses = (
                    results["mistral_status"]
                    .astype(str)
                    .str.lower()
                    .tolist()
                )

                if "success" in statuses:

                    mistral_status = "success"

                elif "unavailable" in statuses:

                    mistral_status = "unavailable"

                elif statuses:

                    mistral_status = statuses[0]

        # ----------------------------------------------------
        # Rule-based security can still complete without
        # Mistral.
        # ----------------------------------------------------

        return {
            "status": "completed",
            "message": (
                "Security evaluation completed."
            ),
            "summary": summary,
            "results": results,
            "mistral_status": mistral_status,
            "methodology": {
                "rule_based": True,
                "local_mistral": (
                    mistral_status == "success"
                ),
                "checks": [
                    "prompt_injection",
                    "jailbreak",
                    "toxicity",
                    "data_leakage",
                ],
            },
        }

    except Exception as exc:

        return {
            "status": "error",
            "message": str(exc),
            "summary": {},
            "results": pd.DataFrame(),
        }


# ============================================================
# RELIABILITY EVALUATION
# ============================================================

def run_reliability_evaluation(
    df: pd.DataFrame,
) -> Dict[str, Any]:

    if not RELIABILITY_AVAILABLE:

        return {
            "status": "error",
            "message": (
                "Reliability evaluator could not be imported: "
                + RELIABILITY_IMPORT_ERROR
            ),
        }

    try:

        result = evaluate_reliability(
            df
        )

        # The reliability evaluator already returns
        # availability information and reasons.
        if isinstance(result, dict):

            result.setdefault(
                "status",
                "completed",
            )

            result.setdefault(
                "message",
                "Reliability evaluation completed.",
            )

            return result

        return {
            "status": "completed",
            "message": (
                "Reliability evaluator returned "
                "an unexpected result type."
            ),
            "result": result,
        }

    except Exception as exc:

        return {
            "status": "error",
            "message": str(exc),
        }


# ============================================================
# MAIN EVALUATION
# ============================================================

def run_evaluation(
    df: pd.DataFrame,
) -> Dict[str, Any]:

    if not isinstance(df, pd.DataFrame):

        raise TypeError(
            "run_evaluation() expects a pandas DataFrame."
        )

    # ========================================================
    # SCHEMA
    # ========================================================

    schema = detect_schema(
        df
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    validation = validate_dataset(
        df,
        schema,
    )

    # ========================================================
    # METRIC AVAILABILITY
    # ========================================================

    metric_availability = get_metric_availability(
        df,
        schema,
    )

    # ========================================================
    # STOP EARLY IF INVALID
    # ========================================================

    if not validation["valid"]:

        return {
            "status": "invalid",
            "schema": schema,
            "validation": validation,
            "metric_availability": metric_availability,
            "retrieval": {
                "status": "unavailable",
                "message": (
                    "Dataset validation failed."
                ),
            },
            "ragas": {
                "status": "unavailable",
                "message": (
                    "Dataset validation failed."
                ),
            },
            "security": {
                "status": "unavailable",
                "message": (
                    "Dataset validation failed."
                ),
            },
            "reliability": {
                "status": "unavailable",
                "message": (
                    "Dataset validation failed."
                ),
            },
        }

    # ========================================================
    # RETRIEVAL
    # ========================================================

    retrieval_result = run_retrieval_evaluation(
        df,
        schema,
    )

    # ========================================================
    # RAGAS
    # ========================================================

    ragas_result = run_ragas_layer(
        df,
        schema,
    )

    # ========================================================
    # SECURITY
    # ========================================================

    security_result = run_security_evaluation(
        df,
        schema,
    )

    # ========================================================
    # RELIABILITY
    # ========================================================

    reliability_result = run_reliability_evaluation(
        df
    )

    # ========================================================
    # FINAL STATUS
    # ========================================================

    statuses = [
        retrieval_result.get(
            "status",
            "unknown",
        ),
        ragas_result.get(
            "status",
            "unknown",
        ),
        security_result.get(
            "status",
            "unknown",
        ),
        reliability_result.get(
            "status",
            "unknown",
        ),
    ]

    if "error" in statuses:

        overall_status = "completed_with_errors"

    elif "completed" in statuses:

        overall_status = "completed"

    else:

        overall_status = "completed"

    # ========================================================
    # RETURN CENTRAL RESULT
    # ========================================================

    return {
        "status": overall_status,

        "dataset": {
            "rows": int(len(df)),
            "columns": list(df.columns),
            "column_count": int(len(df.columns)),
            "missing_values": int(
                df.isna().sum().sum()
            ),
            "duplicate_rows": int(
                df.duplicated().sum()
            ),
        },

        "schema": schema,

        "validation": validation,

        "metric_availability": metric_availability,

        "retrieval": retrieval_result,

        "ragas": ragas_result,

        "security": security_result,

        "reliability": reliability_result,
    }


# ============================================================
# STATUS SUMMARY
# ============================================================

def get_evaluation_status_summary(
    evaluation_result: Dict[str, Any],
) -> Dict[str, str]:

    return {
        "retrieval": evaluation_result
        .get("retrieval", {})
        .get("status", "unknown"),

        "ragas": evaluation_result
        .get("ragas", {})
        .get("status", "unknown"),

        "security": evaluation_result
        .get("security", {})
        .get("status", "unknown"),

        "reliability": evaluation_result
        .get("reliability", {})
        .get("status", "unknown"),
    }


# ============================================================
# SAFE VALUE FORMATTER
# ============================================================

def print_value(
    value: Any,
) -> str:

    if value is None:

        return "None"

    if isinstance(value, float):

        return f"{value:.4f}"

    if isinstance(value, bool):

        return str(value)

    if isinstance(value, (dict, list)):

        try:

            return json.dumps(
                value,
                indent=2,
                default=str,
            )

        except Exception:

            return str(value)

    return str(value)


# ============================================================
# PRINT SECURITY SUMMARY
# ============================================================

def print_security_summary(
    security_result: Dict[str, Any],
) -> None:

    print(
        "\n========== SECURITY =========="
    )

    print(
        "Status:",
        security_result.get(
            "status",
            "unknown",
        ),
    )

    print(
        "Message:",
        security_result.get(
            "message",
            "",
        ),
    )

    summary = security_result.get(
        "summary",
        {},
    )

    if summary:

        print(
            "\nSecurity Summary:"
        )

        preferred_keys = [
            "total_samples",
            "high_risk",
            "medium_risk",
            "low_risk",
            "prompt_injection",
            "jailbreak",
            "toxicity",
            "data_leakage",
        ]

        printed = set()

        for key in preferred_keys:

            if key in summary:

                print(
                    f"{key:25} -> "
                    f"{print_value(summary[key])}"
                )

                printed.add(key)

        for key, value in summary.items():

            if key not in printed:

                if isinstance(
                    value,
                    (
                        str,
                        int,
                        float,
                        bool,
                    ),
                ):

                    print(
                        f"{key:25} -> "
                        f"{print_value(value)}"
                    )

    print(
        "\nMistral semantic evaluation:",
        security_result.get(
            "mistral_status",
            "unknown",
        ),
    )


# ============================================================
# PRINT RELIABILITY SUMMARY
# ============================================================

def print_reliability_summary(
    reliability_result: Dict[str, Any],
) -> None:

    print(
        "\n========== RELIABILITY =========="
    )

    print(
        "Status:",
        reliability_result.get(
            "status",
            "unknown",
        ),
    )

    print(
        "Message:",
        reliability_result.get(
            "message",
            "",
        ),
    )

    available_metrics = reliability_result.get(
        "available_metrics",
        [],
    )

    unavailable_metrics = reliability_result.get(
        "unavailable_metrics",
        [],
    )

    print(
        "\nAvailable metrics:"
    )

    if available_metrics:

        for metric in available_metrics:

            print(
                f"  - {metric}"
            )

    else:

        print(
            "  None"
        )

    print(
        "\nUnavailable metrics:"
    )

    if unavailable_metrics:

        for metric in unavailable_metrics:

            if isinstance(metric, dict):

                name = metric.get(
                    "metric",
                    "Unknown",
                )

                reason = metric.get(
                    "reason",
                    "No reason provided.",
                )

                print(
                    f"  - {name}: {reason}"
                )

            else:

                print(
                    f"  - {metric}"
                )

    else:

        print(
            "  None"
        )

    # --------------------------------------------------------
    # Print individual metric groups
    # --------------------------------------------------------

    preferred_groups = [
        "latency",
        "success_failure",
        "consistency",
        "robustness",
    ]

    for group in preferred_groups:

        if group in reliability_result:

            value = reliability_result[
                group
            ]

            print(
                f"\n{group}:"
            )

            if isinstance(
                value,
                dict,
            ):

                for key, item in value.items():

                    if isinstance(
                        item,
                        (
                            str,
                            int,
                            float,
                            bool,
                        ),
                    ):

                        print(
                            f"  {key}: "
                            f"{print_value(item)}"
                        )

            else:

                print(
                    f"  {print_value(value)}"
                )


# ============================================================
# CLI TEST
# ============================================================

def main() -> None:

    print(
        "=" * 70
    )

    print(
        "AITrustEval - Main Evaluation Engine Test"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset_path = (
        "data/sample/"
        "synthetic_5field_dataset.csv"
    )

    print(
        "\nLoading dataset..."
    )

    try:

        df = pd.read_csv(
            dataset_path
        )

    except Exception as exc:

        print(
            "\nERROR: Unable to load dataset."
        )

        print(
            str(exc)
        )

        sys.exit(1)

    print(
        f"Rows: {len(df)}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    print(
        "\nColumns detected:"
    )

    for column in df.columns:

        print(
            f"  - {column}"
        )

    # --------------------------------------------------------
    # Run engine
    # --------------------------------------------------------

    print(
        "\nRunning main evaluation engine..."
    )

    try:

        result = run_evaluation(
            df
        )

    except Exception as exc:

        print(
            "\nENGINE ERROR:"
        )

        print(
            str(exc)
        )

        raise

    # ========================================================
    # SCHEMA
    # ========================================================

    print(
        "\n========== SCHEMA =========="
    )

    schema = result.get(
        "schema",
        {},
    )

    for field in [
        "question",
        "context",
        "answer",
        "ground_truth",
        "metadata",
        "latency",
        "status",
    ]:

        value = schema.get(
            field
        )

        if value:

            print(
                f"{field:18} -> {value}"
            )

        else:

            print(
                f"{field:18} -> NOT DETECTED"
            )

    # ========================================================
    # VALIDATION
    # ========================================================

    print(
        "\n========== VALIDATION =========="
    )

    validation = result.get(
        "validation",
        {},
    )

    print(
        "Valid:",
        validation.get(
            "valid",
            False,
        ),
    )

    warnings = validation.get(
        "warnings",
        [],
    )

    if warnings:

        print(
            "\nWarnings:"
        )

        for warning in warnings:

            print(
                f"  - {warning}"
            )

    print(
        "\nMessage:"
    )

    print(
        validation.get(
            "message",
            "",
        )
    )

    # ========================================================
    # METRIC AVAILABILITY
    # ========================================================

    print(
        "\n========== METRIC AVAILABILITY =========="
    )

    availability = result.get(
        "metric_availability",
        {},
    )

    for metric, available in availability.items():

        print(
            f"{metric:30} -> "
            f"{available}"
        )

    # ========================================================
    # RETRIEVAL
    # ========================================================

    print(
        "\n========== RETRIEVAL =========="
    )

    retrieval = result.get(
        "retrieval",
        {},
    )

    preferred_retrieval_keys = [
        "total_rows",
        "successful_rows",
        "failed_rows",
        "unavailable_rows",
        "k",
        "similarity_threshold",
        "embedding_model",
        "precision@5",
        "recall@5",
        "mrr",
        "ndcg@5",
        "hit_rate@5",
    ]

    printed_retrieval = set()

    for key in preferred_retrieval_keys:

        if key in retrieval:

            print(
                f"{key:25} -> "
                f"{print_value(retrieval[key])}"
            )

            printed_retrieval.add(key)

    print(
        "\nStatus:",
        retrieval.get(
            "status",
            "unknown",
        ),
    )

    print(
        "Message:",
        retrieval.get(
            "message",
            "",
        ),
    )

    # ========================================================
    # RAGAS
    # ========================================================

    print(
        "\n========== RAGAS =========="
    )

    ragas = result.get(
        "ragas",
        {},
    )

    print(
        "Status:",
        ragas.get(
            "status",
            "unknown",
        ),
    )

    print(
        "Message:",
        ragas.get(
            "message",
            "",
        ),
    )

    # ========================================================
    # SECURITY
    # ========================================================

    print_security_summary(
        result.get(
            "security",
            {},
        )
    )

    # ========================================================
    # RELIABILITY
    # ========================================================

    print_reliability_summary(
        result.get(
            "reliability",
            {},
        )
    )

    # ========================================================
    # FINAL STATUS SUMMARY
    # ========================================================

    print(
        "\n========== STATUS SUMMARY =========="
    )

    status_summary = get_evaluation_status_summary(
        result
    )

    for category, status in status_summary.items():

        print(
            f"{category:15} -> {status}"
        )

    # ========================================================
    # FINAL
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "MAIN ENGINE TEST COMPLETE"
    )

    print(
        "=" * 70
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()