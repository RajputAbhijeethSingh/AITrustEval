"""
AITrustEval - Reliability & Performance Evaluator

Evaluates reliability/performance metrics only when the
required dataset fields are available.

Metrics:
- Latency: mean, median, min, max, P95, P99
- Success rate
- Failure rate
- Consistency / response stability
- Robustness readiness

Metrics are marked unavailable when their required
information does not exist. No values are fabricated.
"""

import ast
from typing import Dict, Optional

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# COLUMN ALIASES
# ============================================================

LATENCY_ALIASES = [
    "latency",
    "latency_ms",
    "response_time",
    "response_time_ms",
    "response_latency",
    "response_latency_ms",
    "duration_ms",
]

STATUS_ALIASES = [
    "status",
    "response_status",
    "request_status",
    "success",
    "is_success",
    "error",
    "failed",
]

QUESTION_ALIASES = [
    "question",
    "query",
    "user_query",
    "user_question",
    "prompt",
    "input",
]

ANSWER_ALIASES = [
    "answer",
    "response",
    "generated_answer",
    "model_answer",
    "output",
]

VARIATION_ALIASES = [
    "variation",
    "variant",
    "perturbation",
    "test_variant",
    "prompt_variant",
]


# ============================================================
# HELPERS
# ============================================================

def _normalize_column_name(value: str) -> str:
    return str(value).strip().lower()


def find_column(
    df: pd.DataFrame,
    aliases,
) -> Optional[str]:
    """
    Find the first dataframe column matching one of the aliases.
    """

    normalized_columns = {
        _normalize_column_name(column): column
        for column in df.columns
    }

    for alias in aliases:

        normalized_alias = _normalize_column_name(alias)

        if normalized_alias in normalized_columns:
            return normalized_columns[normalized_alias]

    return None


def clean_text(value) -> str:
    """
    Convert values into clean strings.

    Handles list-like strings such as:
    ['answer text']
    """

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    if isinstance(value, (list, tuple)):

        return " ".join(
            str(item)
            for item in value
        )

    if isinstance(value, str):

        value = value.strip()

        if value.startswith("[") and value.endswith("]"):

            try:

                parsed = ast.literal_eval(value)

                if isinstance(parsed, (list, tuple)):

                    return " ".join(
                        str(item)
                        for item in parsed
                    )

            except Exception:
                pass

        return value

    return str(value)


def unavailable(reason: str) -> Dict:
    """
    Standard response for an unavailable metric.
    """

    return {
        "available": False,
        "reason": reason,
    }


# ============================================================
# LATENCY
# ============================================================

def evaluate_latency(
    df: pd.DataFrame,
    latency_column: Optional[str] = None,
) -> Dict:
    """
    Calculate latency statistics.

    Assumes latency is expressed in milliseconds when the
    dataset uses a latency_ms/response_time_ms-style column.

    For a generic 'latency' column, the source unit should be
    documented by the dataset owner.
    """

    latency_column = (
        latency_column
        or find_column(
            df,
            LATENCY_ALIASES,
        )
    )

    if not latency_column:

        return unavailable(
            "No latency or response-time column was detected."
        )

    values = pd.to_numeric(
        df[latency_column],
        errors="coerce",
    ).dropna()

    # Negative latency values are invalid.
    values = values[
        values >= 0
    ]

    if values.empty:

        return unavailable(
            "The detected latency column contains no valid numeric values."
        )

    return {
        "available": True,
        "column": latency_column,
        "count": int(len(values)),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "min": float(values.min()),
        "max": float(values.max()),
        "p95": float(
            np.percentile(
                values,
                95,
            )
        ),
        "p99": float(
            np.percentile(
                values,
                99,
            )
        ),
    }


# ============================================================
# SUCCESS / FAILURE RATE
# ============================================================

def _status_to_success(
    value,
    column_name: str,
):
    """
    Convert common status values into:
        True  -> success
        False -> failure
        None  -> unknown
    """

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    column = _normalize_column_name(
        column_name
    )

    # Boolean values
    if isinstance(value, (bool, np.bool_)):

        if column in ["error", "failed"]:
            return not bool(value)

        return bool(value)

    text = str(value).strip().lower()

    success_values = {
        "success",
        "successful",
        "ok",
        "passed",
        "pass",
        "true",
        "1",
        "200",
        "completed",
    }

    failure_values = {
        "failure",
        "failed",
        "error",
        "false",
        "0",
        "timeout",
        "exception",
    }

    # Special handling for error/failed columns.
    if column in ["error", "failed"]:

        if text in {
            "false",
            "0",
            "no",
            "none",
            "",
        }:
            return True

        if text in {
            "true",
            "1",
            "yes",
        }:
            return False

    if text in success_values:
        return True

    if text in failure_values:
        return False

    # HTTP-like status codes
    try:

        numeric = int(float(text))

        if 200 <= numeric < 400:
            return True

        if numeric >= 400:
            return False

    except Exception:
        pass

    return None


def evaluate_success_failure(
    df: pd.DataFrame,
    status_column: Optional[str] = None,
) -> Dict:
    """
    Calculate success and failure rates when a usable
    status/success/error field exists.
    """

    status_column = (
        status_column
        or find_column(
            df,
            STATUS_ALIASES,
        )
    )

    if not status_column:

        return unavailable(
            "No status, success, error, or failure column was detected."
        )

    interpreted = []

    for value in df[status_column]:

        status = _status_to_success(
            value,
            status_column,
        )

        if status is not None:
            interpreted.append(status)

    if not interpreted:

        return unavailable(
            "The detected status column could not be interpreted reliably."
        )

    total = len(interpreted)

    successes = sum(
        1
        for value in interpreted
        if value
    )

    failures = total - successes

    return {
        "available": True,
        "column": status_column,
        "evaluated_samples": total,
        "success_count": successes,
        "failure_count": failures,
        "success_rate": successes / total,
        "failure_rate": failures / total,
    }


# ============================================================
# CONSISTENCY
# ============================================================

def _pairwise_similarity(
    texts,
) -> Optional[float]:
    """
    Calculate average pairwise TF-IDF cosine similarity.
    """

    cleaned = [
        clean_text(text)
        for text in texts
    ]

    cleaned = [
        text
        for text in cleaned
        if text
    ]

    if len(cleaned) < 2:
        return None

    try:

        vectorizer = TfidfVectorizer()

        vectors = vectorizer.fit_transform(
            cleaned
        )

        matrix = cosine_similarity(
            vectors
        )

        similarities = []

        for i in range(len(cleaned)):

            for j in range(
                i + 1,
                len(cleaned),
            ):

                similarities.append(
                    float(matrix[i, j])
                )

        if not similarities:
            return None

        return float(
            np.mean(similarities)
        )

    except ValueError:
        return None


def evaluate_consistency(
    df: pd.DataFrame,
    question_column: Optional[str] = None,
    answer_column: Optional[str] = None,
) -> Dict:
    """
    Measure response consistency only when the dataset contains
    repeated questions with multiple answers.

    This is a baseline lexical consistency metric using TF-IDF
    cosine similarity. It is not a semantic LLM-based stability
    metric.
    """

    question_column = (
        question_column
        or find_column(
            df,
            QUESTION_ALIASES,
        )
    )

    answer_column = (
        answer_column
        or find_column(
            df,
            ANSWER_ALIASES,
        )
    )

    if not question_column or not answer_column:

        return unavailable(
            "Question and answer columns are required for consistency evaluation."
        )

    working = df[
        [
            question_column,
            answer_column,
        ]
    ].copy()

    working["_question_clean"] = (
        working[
            question_column
        ].apply(clean_text)
    )

    repeated = working[
        working.duplicated(
            "_question_clean",
            keep=False,
        )
    ]

    repeated = repeated[
        repeated["_question_clean"] != ""
    ]

    if repeated.empty:

        return unavailable(
            "No repeated questions were found. Multiple responses to the same input are required for consistency evaluation."
        )

    group_scores = []

    group_details = []

    for question, group in repeated.groupby(
        "_question_clean"
    ):

        if len(group) < 2:
            continue

        score = _pairwise_similarity(
            group[
                answer_column
            ].tolist()
        )

        if score is None:
            continue

        group_scores.append(score)

        group_details.append(
            {
                "question": question,
                "response_count": int(
                    len(group)
                ),
                "consistency_score": score,
            }
        )

    if not group_scores:

        return unavailable(
            "Repeated questions were detected, but consistency could not be calculated."
        )

    return {
        "available": True,
        "method": (
            "Average pairwise TF-IDF cosine similarity "
            "between responses to identical questions."
        ),
        "repeated_question_groups": len(
            group_scores
        ),
        "consistency_score": float(
            np.mean(group_scores)
        ),
        "groups": group_details,
    }


# ============================================================
# ROBUSTNESS READINESS
# ============================================================

def evaluate_robustness_readiness(
    df: pd.DataFrame,
) -> Dict:
    """
    Determine whether the dataset contains explicit variation
    information suitable for robustness evaluation.

    This function intentionally does not invent a robustness
    score from ordinary QA rows.
    """

    variation_column = find_column(
        df,
        VARIATION_ALIASES,
    )

    if not variation_column:

        return unavailable(
            "No prompt variation or perturbation field was detected. Robustness requires controlled variants or adversarial/perturbed inputs."
        )

    non_null = df[
        variation_column
    ].dropna()

    if non_null.empty:

        return unavailable(
            "A variation column exists but contains no usable values."
        )

    return {
        "available": True,
        "column": variation_column,
        "variation_samples": int(
            len(non_null)
        ),
        "message": (
            "Variation data is available. "
            "A robustness score can be computed once the "
            "original-to-variant relationship and evaluation "
            "criterion are defined."
        ),
    }


# ============================================================
# FULL RELIABILITY EVALUATION
# ============================================================

def evaluate_reliability(
    df: pd.DataFrame,
    latency_column: Optional[str] = None,
    status_column: Optional[str] = None,
    question_column: Optional[str] = None,
    answer_column: Optional[str] = None,
) -> Dict:
    """
    Run the complete dataset-aware reliability evaluation.
    """

    latency = evaluate_latency(
        df,
        latency_column,
    )

    success_failure = (
        evaluate_success_failure(
            df,
            status_column,
        )
    )

    consistency = evaluate_consistency(
        df,
        question_column,
        answer_column,
    )

    robustness = (
        evaluate_robustness_readiness(
            df
        )
    )

    available_metrics = []
    unavailable_metrics = []

    metric_results = {
        "latency": latency,
        "success_failure": success_failure,
        "consistency": consistency,
        "robustness": robustness,
    }

    for name, result in metric_results.items():

        if result.get(
            "available",
            False,
        ):
            available_metrics.append(name)

        else:
            unavailable_metrics.append(
                {
                    "metric": name,
                    "reason": result.get(
                        "reason",
                        "Required data is unavailable.",
                    ),
                }
            )

    return {
        "total_samples": int(
            len(df)
        ),
        "available_metrics": available_metrics,
        "unavailable_metrics": unavailable_metrics,
        "latency": latency,
        "success_failure": success_failure,
        "consistency": consistency,
        "robustness": robustness,
    }


# ============================================================
# CLI TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 65)
    print("AITrustEval - Reliability & Performance Evaluation")
    print("=" * 65)

    # --------------------------------------------------------
    # Test dataset with performance information
    # --------------------------------------------------------

    test_data = pd.DataFrame(
        {
            "question": [
                "What is AI?",
                "What is AI?",
                "What is Python?",
                "What is Python?",
            ],

            "answer": [
                "AI is artificial intelligence.",
                "Artificial intelligence is known as AI.",
                "Python is a programming language.",
                "Python is a high-level programming language.",
            ],

            "latency_ms": [
                420,
                510,
                390,
                620,
            ],

            "status": [
                "success",
                "success",
                "success",
                "failed",
            ],
        }
    )

    results = evaluate_reliability(
        test_data
    )

    print(
        f"\nTotal samples: "
        f"{results['total_samples']}"
    )

    print("\nAvailable metrics:")

    for metric in results[
        "available_metrics"
    ]:
        print(f"  ✓ {metric}")

    print("\nUnavailable metrics:")

    for item in results[
        "unavailable_metrics"
    ]:
        print(
            f"  ✗ {item['metric']}: "
            f"{item['reason']}"
        )

    print("\n========== LATENCY ==========")

    latency = results["latency"]

    if latency["available"]:

        print(
            f"Mean: {latency['mean']:.2f}"
        )

        print(
            f"Median: {latency['median']:.2f}"
        )

        print(
            f"P95: {latency['p95']:.2f}"
        )

        print(
            f"P99: {latency['p99']:.2f}"
        )

    else:

        print(latency["reason"])

    print(
        "\n========== SUCCESS / FAILURE =========="
    )

    reliability = results[
        "success_failure"
    ]

    if reliability["available"]:

        print(
            f"Success rate: "
            f"{reliability['success_rate']:.2%}"
        )

        print(
            f"Failure rate: "
            f"{reliability['failure_rate']:.2%}"
        )

    else:

        print(
            reliability["reason"]
        )

    print(
        "\n========== CONSISTENCY =========="
    )

    consistency = results[
        "consistency"
    ]

    if consistency["available"]:

        print(
            f"Consistency score: "
            f"{consistency['consistency_score']:.4f}"
        )

        print(
            f"Repeated question groups: "
            f"{consistency['repeated_question_groups']}"
        )

    else:

        print(
            consistency["reason"]
        )

    print(
        "\n========== ROBUSTNESS =========="
    )

    robustness = results[
        "robustness"
    ]

    if robustness["available"]:

        print(
            robustness["message"]
        )

    else:

        print(
            robustness["reason"]
        )

    print("\n" + "=" * 65)