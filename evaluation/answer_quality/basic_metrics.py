"""
AITrustEval - Baseline Answer Quality Metrics

Lightweight offline evaluation layer.

Metrics:
- Answer Relevancy
- Answer Correctness
- Context-Answer Similarity

Important:
These are baseline lexical/reference-similarity measurements.
They are NOT replacements for RAGAS or an LLM-as-a-judge evaluator.

The implementation is fully offline and does not require:
- Ollama
- Mistral
- Internet
- Cloud APIs
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_LOW_THRESHOLD = 0.50
DEFAULT_WARNING_THRESHOLD = 0.75


# ============================================================
# TEXT UTILITIES
# ============================================================

def clean_text(value: Any) -> str:
    """
    Convert a dataset value into normalized text.
    """

    if value is None:
        return ""

    try:

        if pd.isna(value):
            return ""

    except Exception:

        pass

    return str(value).strip()


def parse_serialized_value(
    value: Any,
) -> Any:
    """
    Parse JSON/Python-literal serialized values when possible.
    """

    if value is None:
        return None

    if isinstance(
        value,
        (list, dict),
    ):

        return value

    text = clean_text(value)

    if not text:
        return None

    try:

        return json.loads(text)

    except Exception:

        pass

    try:

        return ast.literal_eval(text)

    except Exception:

        return None


def extract_context_text(
    value: Any,
) -> str:
    """
    Convert context field into a single text representation.

    Supports:
    - plain strings
    - JSON list of chunks
    - Python list of chunks
    - dictionaries
    """

    parsed = parse_serialized_value(
        value
    )

    if parsed is None:

        return clean_text(value)

    if isinstance(
        parsed,
        str,
    ):

        return parsed

    if isinstance(
        parsed,
        dict,
    ):

        if "context" in parsed:

            return clean_text(
                parsed["context"]
            )

        return " ".join(
            clean_text(item)
            for item in parsed.values()
        )

    if isinstance(
        parsed,
        list,
    ):

        chunks = []


        for item in parsed:

            if isinstance(
                item,
                dict,
            ):

                if "context" in item:

                    chunks.append(
                        clean_text(
                            item["context"]
                        )
                    )

                elif "text" in item:

                    chunks.append(
                        clean_text(
                            item["text"]
                        )
                    )

                else:

                    chunks.append(
                        " ".join(
                            clean_text(v)
                            for v in item.values()
                        )
                    )

            else:

                chunks.append(
                    clean_text(item)
                )


        return " ".join(
            chunk
            for chunk in chunks
            if chunk
        )

    return clean_text(parsed)


def tokenize(
    text: str,
) -> List[str]:
    """
    Basic word tokenization used only for
    fallback similarity.
    """

    return re.findall(
        r"\b[a-zA-Z0-9]+\b",
        text.lower(),
    )


# ============================================================
# SIMILARITY
# ============================================================

def tfidf_similarity(
    text_a: str,
    text_b: str,
) -> float:
    """
    Calculate cosine similarity using TF-IDF.

    Returns a score between 0 and 1.
    """

    text_a = clean_text(text_a)
    text_b = clean_text(text_b)

    if not text_a or not text_b:

        return 0.0

    if text_a.lower() == text_b.lower():

        return 1.0

    try:

        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
        )

        matrix = vectorizer.fit_transform(
            [
                text_a,
                text_b,
            ]
        )

        score = cosine_similarity(
            matrix[0:1],
            matrix[1:2],
        )[0][0]

        return float(
            np.clip(
                score,
                0.0,
                1.0,
            )
        )

    except Exception:

        return lexical_overlap(
            text_a,
            text_b,
        )


def lexical_overlap(
    text_a: str,
    text_b: str,
) -> float:
    """
    Fallback token overlap score.
    """

    tokens_a = set(
        tokenize(text_a)
    )

    tokens_b = set(
        tokenize(text_b)
    )

    if not tokens_a or not tokens_b:

        return 0.0

    intersection = (
        tokens_a
        & tokens_b
    )

    union = (
        tokens_a
        | tokens_b
    )

    if not union:

        return 0.0

    return float(
        len(intersection)
        / len(union)
    )


# ============================================================
# INDIVIDUAL METRICS
# ============================================================

def calculate_answer_relevancy(
    question: Any,
    answer: Any,
) -> float:
    """
    Measure alignment between question and generated answer.
    """

    return tfidf_similarity(
        clean_text(question),
        clean_text(answer),
    )


def calculate_answer_correctness(
    answer: Any,
    ground_truth: Any,
) -> float:
    """
    Measure reference-answer similarity.

    This is a lightweight baseline and should not be
    interpreted as semantic factual verification.
    """

    return tfidf_similarity(
        clean_text(answer),
        clean_text(ground_truth),
    )


def calculate_context_answer_similarity(
    answer: Any,
    context: Any,
) -> float:
    """
    Measure similarity between generated answer and
    retrieved context.
    """

    context_text = extract_context_text(
        context
    )

    return tfidf_similarity(
        clean_text(answer),
        context_text,
    )


# ============================================================
# SCORE CLASSIFICATION
# ============================================================

def classify_score(
    score: Optional[float],
    low_threshold: float = DEFAULT_LOW_THRESHOLD,
    warning_threshold: float = DEFAULT_WARNING_THRESHOLD,
) -> str:
    """
    Classify an evaluation score.
    """

    if score is None:

        return "Unavailable"

    try:

        score = float(score)

    except Exception:

        return "Unavailable"

    if score < low_threshold:

        return "Poor"

    if score < warning_threshold:

        return "Warning"

    return "Good"


# ============================================================
# SINGLE ROW
# ============================================================

def evaluate_row(
    row: pd.Series,
    question_column: str,
    answer_column: str,
    ground_truth_column: str,
    context_column: str,
) -> Dict[str, Any]:
    """
    Evaluate one dataset row.
    """

    question = row.get(
        question_column,
        "",
    )

    answer = row.get(
        answer_column,
        "",
    )

    ground_truth = row.get(
        ground_truth_column,
        "",
    )

    context = row.get(
        context_column,
        "",
    )


    answer_relevancy = (
        calculate_answer_relevancy(
            question,
            answer,
        )
    )


    answer_correctness = (
        calculate_answer_correctness(
            answer,
            ground_truth,
        )
    )


    context_answer_similarity = (
        calculate_context_answer_similarity(
            answer,
            context,
        )
    )


    return {
        "answer_relevancy":
            answer_relevancy,

        "answer_correctness":
            answer_correctness,

        "context_answer_similarity":
            context_answer_similarity,
    }


# ============================================================
# DATASET EVALUATION
# ============================================================

def evaluate_basic_answer_quality(
    df: pd.DataFrame,
    schema: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Evaluate baseline answer quality over the complete dataset.
    """

    if df is None:

        return {
            "status": "error",
            "message": "Dataset is None.",
            "averages": {},
            "sample_results": [],
        }


    if df.empty:

        return {
            "status": "unavailable",
            "message": "Dataset is empty.",
            "averages": {},
            "sample_results": [],
        }


    schema = (
        schema
        if isinstance(
            schema,
            dict,
        )
        else {}
    )


    question_column = schema.get(
        "question"
    )

    answer_column = schema.get(
        "answer"
    )

    ground_truth_column = schema.get(
        "ground_truth"
    )

    context_column = schema.get(
        "context"
    )


    required = {
        "question":
            question_column,

        "answer":
            answer_column,

        "ground_truth":
            ground_truth_column,

        "context":
            context_column,
    }


    missing = [
        field
        for field, column
        in required.items()
        if not column
        or column not in df.columns
    ]


    if missing:

        return {
            "status": "unavailable",

            "message":
                "Baseline answer-quality evaluation "
                "requires question, answer, ground_truth "
                "and context fields. Missing: "
                + ", ".join(missing),

            "averages": {},

            "sample_results": [],

            "methodology": {
                "type":
                    "baseline_lexical_reference_similarity",

                "limitation":
                    "Baseline lexical similarity is not "
                    "a replacement for an LLM-based "
                    "evaluation methodology.",
            },
        }


    results = []


    for index, row in df.iterrows():

        sample_id = index + 1


        try:

            scores = evaluate_row(
                row=row,

                question_column=
                    question_column,

                answer_column=
                    answer_column,

                ground_truth_column=
                    ground_truth_column,

                context_column=
                    context_column,
            )


            results.append(
                {
                    "id":
                        sample_id,

                    "answer_relevancy":
                        scores[
                            "answer_relevancy"
                        ],

                    "answer_correctness":
                        scores[
                            "answer_correctness"
                        ],

                    "context_answer_similarity":
                        scores[
                            "context_answer_similarity"
                        ],
                }
            )


        except Exception as error:

            results.append(
                {
                    "id":
                        sample_id,

                    "answer_relevancy":
                        np.nan,

                    "answer_correctness":
                        np.nan,

                    "context_answer_similarity":
                        np.nan,

                    "error":
                        str(error),
                }
            )


    results_df = pd.DataFrame(
        results
    )


    metric_columns = [
        "answer_relevancy",
        "answer_correctness",
        "context_answer_similarity",
    ]


    averages = {}


    for metric in metric_columns:

        if metric not in results_df.columns:

            continue


        values = pd.to_numeric(
            results_df[
                metric
            ],
            errors="coerce",
        )


        if values.notna().any():

            averages[
                metric
            ] = float(
                values.mean()
            )


    return {
        "status":
            "completed",

        "message":
            "Baseline answer-quality evaluation completed.",

        "total_samples":
            len(df),

        "successful_samples":
            len(results_df),

        "averages":
            averages,

        "average_scores":
            averages,

        "sample_results":
            results_df.to_dict(
                orient="records"
            ),

        "results":
            results_df.to_dict(
                orient="records"
            ),

        "methodology": {
            "type":
                "baseline_lexical_reference_similarity",

            "metrics": [
                "answer_relevancy",
                "answer_correctness",
                "context_answer_similarity",
            ],

            "description":
                "Offline TF-IDF cosine similarity is used "
                "as a lightweight baseline.",

            "limitation":
                "These baseline scores are not equivalent "
                "to RAGAS or LLM-as-a-judge evaluation.",
        },
    }


# ============================================================
# CONVERT TO DATAFRAME
# ============================================================

def results_to_dataframe(
    result: Dict[str, Any],
) -> pd.DataFrame:
    """
    Convert baseline result into DataFrame.
    """

    records = result.get(
        "sample_results",
        [],
    )


    if not records:

        return pd.DataFrame()


    return pd.DataFrame(
        records
    )


# ============================================================
# CLI TEST
# ============================================================

if __name__ == "__main__":

    from pathlib import Path


    candidates = [
        Path(
            "data/sample/"
            "synthetic_5field_dataset.csv"
        ),

        Path(
            "data/processed/"
            "normalized_ragas_eval.csv"
        ),

        Path(
            "data/input/"
            "uploaded_dataset.csv"
        ),
    ]


    dataset_path = None


    for candidate in candidates:

        if candidate.exists():

            dataset_path = candidate

            break


    if dataset_path is None:

        print(
            "No test dataset found."
        )

        raise SystemExit(1)


    df = pd.read_csv(
        dataset_path
    )


    schema = {
        "question":
            "question",

        "context":
            "context",

        "answer":
            "answer",

        "ground_truth":
            "ground_truth",

        "metadata":
            "metadata",
    }


    result = (
        evaluate_basic_answer_quality(
            df,
            schema,
        )
    )


    print(
        "=" * 70
    )

    print(
        "AITrustEval - Baseline Answer Quality"
    )

    print(
        "=" * 70
    )

    print(
        f"Dataset rows: {len(df)}"
    )

    print(
        f"Status: {result['status']}"
    )


    print(
        "\nAverage Scores:"
    )


    for metric, score in (
        result.get(
            "averages",
            {},
        ).items()
    ):

        print(
            f"{metric}: "
            f"{score:.4f}"
        )


    print(
        "\nBaseline evaluation complete."
    )