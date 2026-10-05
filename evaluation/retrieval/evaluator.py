# ============================================================
# AITrustEval - Semantic Retrieval Evaluator
# ============================================================
#
# Purpose:
# Evaluate retrieval quality from a dataset containing:
#
#   question
#   context
#   answer
#   metadata
#   ground_truth
#
# Traditional retrieval systems normally provide ranked
# retrieved documents + gold relevance labels.
#
# Our DRDO dataset does not necessarily contain explicit
# relevance labels, so this evaluator uses LOCAL EMBEDDINGS
# to estimate semantic relevance between:
#
#       Retrieved Context Chunk
#               vs
#          Ground Truth
#
# This is therefore a:
#
#       SEMANTIC / PROXY RETRIEVAL EVALUATION
#
# and must be clearly described as such in the final report.
#
# Metrics:
#
#   Precision@K
#   Recall@K
#   MRR
#   NDCG@K
#   Hit Rate@K
#
# Embeddings:
#
#   Ollama local embedding model
#
# Default:
#
#   qwen3-embedding:0.6b
#
# ============================================================

import ast
import json
import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests

from evaluation.retrieval.metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    ndcg,
    hit_rate,
)


# ============================================================
# CONFIGURATION
# ============================================================

OLLAMA_BASE_URL = (
    "http://localhost:11434"
)

DEFAULT_EMBEDDING_MODEL = (
    "qwen3-embedding:0.6b"
)

DEFAULT_SIMILARITY_THRESHOLD = 0.60

DEFAULT_K = 5

REQUEST_TIMEOUT = 30


# ============================================================
# OLLAMA AVAILABILITY
# ============================================================

def is_ollama_available() -> bool:
    """
    Check whether Ollama is running locally.
    """

    try:

        response = requests.get(
            f"{OLLAMA_BASE_URL}/api/tags",
            timeout=5,
        )

        return response.status_code == 200

    except requests.RequestException:

        return False


# ============================================================
# AVAILABLE OLLAMA MODELS
# ============================================================

def get_available_models() -> List[str]:
    """
    Return locally available Ollama models.
    """

    try:

        response = requests.get(
            f"{OLLAMA_BASE_URL}/api/tags",
            timeout=5,
        )

        if response.status_code != 200:

            return []

        data = response.json()

        models = []

        for model in data.get(
            "models",
            [],
        ):

            name = model.get(
                "name"
            )

            if name:

                models.append(name)

        return models

    except (
        requests.RequestException,
        ValueError,
    ):

        return []


# ============================================================
# GENERIC VALUE PARSER
# ============================================================

def parse_value(
    value: Any,
) -> Any:
    """
    Convert strings containing JSON/Python structures
    into Python objects.

    Examples:

        '[1, 2, 3]' -> [1, 2, 3]

        '{"text": "hello"}'
            -> {"text": "hello"}

        'hello' -> 'hello'
    """

    if value is None:

        return None

    if isinstance(
        value,
        (list, tuple, dict),
    ):

        return value

    if isinstance(
        value,
        float,
    ) and math.isnan(value):

        return None

    if not isinstance(
        value,
        str,
    ):

        return value

    value = value.strip()

    if not value:

        return None

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    try:

        return json.loads(value)

    except (
        json.JSONDecodeError,
        TypeError,
    ):

        pass

    # --------------------------------------------------------
    # Python literal
    # --------------------------------------------------------

    try:

        return ast.literal_eval(value)

    except (
        ValueError,
        SyntaxError,
        TypeError,
    ):

        pass

    # --------------------------------------------------------
    # Plain string
    # --------------------------------------------------------

    return value


# ============================================================
# CONTEXT PARSER
# ============================================================

def parse_context(
    value: Any,
) -> List[str]:
    """
    Convert the context field into a list of ranked
    context chunks.

    Supported examples:

        [
            "chunk 1",
            "chunk 2",
            "chunk 3"
        ]

    or:

        [
            {"text": "chunk 1"},
            {"text": "chunk 2"}
        ]

    or a single plain string.
    """

    parsed = parse_value(
        value
    )

    if parsed is None:

        return []

    # --------------------------------------------------------
    # List
    # --------------------------------------------------------

    if isinstance(
        parsed,
        list,
    ):

        contexts = []

        for item in parsed:

            # String chunk
            if isinstance(
                item,
                str,
            ):

                text = item.strip()

                if text:

                    contexts.append(
                        text
                    )

                continue

            # Dictionary chunk
            if isinstance(
                item,
                dict,
            ):

                text = (
                    item.get("text")
                    or item.get("content")
                    or item.get("context")
                    or item.get("document")
                    or item.get("page_content")
                )

                if text is not None:

                    text = str(
                        text
                    ).strip()

                    if text:

                        contexts.append(
                            text
                        )

                continue

            # Other object
            text = str(
                item
            ).strip()

            if text:

                contexts.append(
                    text
                )

        return contexts

    # --------------------------------------------------------
    # Dictionary
    # --------------------------------------------------------

    if isinstance(
        parsed,
        dict,
    ):

        text = (
            parsed.get("text")
            or parsed.get("content")
            or parsed.get("context")
            or parsed.get("document")
            or parsed.get("page_content")
        )

        if text is not None:

            return [
                str(text).strip()
            ]

        return []

    # --------------------------------------------------------
    # Plain string
    # --------------------------------------------------------

    text = str(
        parsed
    ).strip()

    if text:

        return [text]

    return []


# ============================================================
# GROUND TRUTH PARSER
# ============================================================

def parse_ground_truth(
    value: Any,
) -> List[str]:
    """
    Convert ground_truth into a list of reference texts.
    """

    parsed = parse_value(
        value
    )

    if parsed is None:

        return []

    # --------------------------------------------------------
    # List
    # --------------------------------------------------------

    if isinstance(
        parsed,
        list,
    ):

        results = []

        for item in parsed:

            if isinstance(
                item,
                dict,
            ):

                text = (
                    item.get("text")
                    or item.get("content")
                    or item.get("answer")
                    or item.get("reference")
                )

                if text is not None:

                    text = str(
                        text
                    ).strip()

                    if text:

                        results.append(
                            text
                        )

            else:

                text = str(
                    item
                ).strip()

                if text:

                    results.append(
                        text
                    )

        return results

    # --------------------------------------------------------
    # Dictionary
    # --------------------------------------------------------

    if isinstance(
        parsed,
        dict,
    ):

        text = (
            parsed.get("text")
            or parsed.get("content")
            or parsed.get("answer")
            or parsed.get("reference")
        )

        if text is not None:

            text = str(
                text
            ).strip()

            if text:

                return [text]

        return []

    # --------------------------------------------------------
    # Plain string
    # --------------------------------------------------------

    text = str(
        parsed
    ).strip()

    if text:

        return [text]

    return []


# ============================================================
# OLLAMA EMBEDDINGS
# ============================================================

def get_ollama_embeddings(
    texts: List[str],
    model: str = DEFAULT_EMBEDDING_MODEL,
) -> List[List[float]]:
    """
    Generate embeddings using the local Ollama embedding API.

    Endpoint:

        POST /api/embed
    """

    if not texts:

        return []

    payload = {
        "model": model,
        "input": texts,
    }

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/embed",
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    data = response.json()

    # --------------------------------------------------------
    # Current Ollama response
    # --------------------------------------------------------

    embeddings = data.get(
        "embeddings"
    )

    if embeddings is not None:

        return embeddings

    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    embedding = data.get(
        "embedding"
    )

    if embedding is not None:

        return [embedding]

    raise ValueError(
        "Ollama embedding response did not contain "
        "'embeddings' or 'embedding'."
    )


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(
    vector_a: List[float],
    vector_b: List[float],
) -> float:
    """
    Calculate cosine similarity between two vectors.
    """

    a = np.asarray(
        vector_a,
        dtype=float,
    )

    b = np.asarray(
        vector_b,
        dtype=float,
    )

    denominator = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )

    if denominator == 0:

        return 0.0

    similarity = (
        np.dot(a, b)
        / denominator
    )

    return float(
        similarity
    )


# ============================================================
# RELEVANCE SCORE CALCULATION
# ============================================================

def calculate_relevance_scores(
    contexts: List[str],
    ground_truths: List[str],
    model: str = DEFAULT_EMBEDDING_MODEL,
) -> List[float]:
    """
    Calculate semantic relevance scores.

    For each retrieved context chunk:

        relevance =
            maximum cosine similarity
            against all ground-truth references
    """

    if not contexts:

        return []

    if not ground_truths:

        return [0.0] * len(
            contexts
        )

    # --------------------------------------------------------
    # Generate all embeddings in one request
    # --------------------------------------------------------

    texts = (
        contexts
        + ground_truths
    )

    embeddings = get_ollama_embeddings(
        texts,
        model=model,
    )

    context_count = len(
        contexts
    )

    context_embeddings = (
        embeddings[:context_count]
    )

    ground_truth_embeddings = (
        embeddings[context_count:]
    )

    relevance_scores = []

    for context_embedding in (
        context_embeddings
    ):

        similarities = [

            cosine_similarity(
                context_embedding,
                ground_truth_embedding,
            )

            for ground_truth_embedding
            in ground_truth_embeddings
        ]

        if similarities:

            relevance_scores.append(
                max(similarities)
            )

        else:

            relevance_scores.append(
                0.0
            )

    return relevance_scores


# ============================================================
# SEMANTIC RETRIEVAL METRICS
# ============================================================

def calculate_semantic_retrieval_metrics(
    relevance_scores: List[float],
    k: int = DEFAULT_K,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> Dict[str, float]:
    """
    Convert semantic similarity scores into retrieval metrics.

    Binary relevance:

        score >= threshold
            -> relevant

        score < threshold
            -> not relevant

    NDCG uses the original similarity scores as graded
    relevance.
    """

    if not relevance_scores:

        return {

            "precision@k": 0.0,

            "recall@k": 0.0,

            "mrr": 0.0,

            "ndcg@k": 0.0,

            "hit_rate@k": 0.0,
        }

    # --------------------------------------------------------
    # Binary relevance
    # --------------------------------------------------------

    binary_relevance = [

        1 if score >= threshold else 0

        for score in relevance_scores
    ]

    # --------------------------------------------------------
    # Total relevant
    #
    # IMPORTANT:
    # This is semantic/proxy relevance, not a gold-label
    # retrieval denominator.
    # --------------------------------------------------------

    total_relevant = sum(
        binary_relevance
    )

    # --------------------------------------------------------
    # Precision
    # --------------------------------------------------------

    precision = precision_at_k(
        binary_relevance,
        k,
    )

    # --------------------------------------------------------
    # Recall
    # --------------------------------------------------------

    recall = recall_at_k(
        binary_relevance,
        total_relevant,
        k,
    )

    # --------------------------------------------------------
    # Reciprocal Rank
    # --------------------------------------------------------

    rr = reciprocal_rank(
        binary_relevance
    )

    # --------------------------------------------------------
    # NDCG
    #
    # Use raw semantic similarity as graded relevance.
    # --------------------------------------------------------

    ndcg_score = ndcg(
        relevance_scores,
        k,
    )

    # --------------------------------------------------------
    # Hit Rate
    # --------------------------------------------------------

    hit = (
        1.0
        if any(
            binary_relevance[:k]
        )
        else 0.0
    )

    return {

        "precision@k":
            float(precision),

        "recall@k":
            float(recall),

        "mrr":
            float(rr),

        "ndcg@k":
            float(ndcg_score),

        "hit_rate@k":
            float(hit),
    }


# ============================================================
# SINGLE ROW EVALUATION
# ============================================================

def evaluate_semantic_row(
    row: pd.Series,
    context_column: str,
    ground_truth_column: str,
    model: str = DEFAULT_EMBEDDING_MODEL,
    k: int = DEFAULT_K,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> Dict[str, Any]:
    """
    Evaluate one dataset row.
    """

    contexts = parse_context(
        row.get(
            context_column
        )
    )

    ground_truths = parse_ground_truth(
        row.get(
            ground_truth_column
        )
    )

    # --------------------------------------------------------
    # Empty context
    # --------------------------------------------------------

    if not contexts:

        return {

            "status":
                "failed",

            "message":
                "No context chunks were found.",

            "context_count":
                0,

            "ground_truth_count":
                len(ground_truths),

            "ranked_context":
                False,

            "relevance_scores":
                [],

            "binary_relevance":
                [],

            "precision@k":
                None,

            "recall@k":
                None,

            "mrr":
                None,

            "ndcg@k":
                None,

            "hit_rate@k":
                None,
        }

    # --------------------------------------------------------
    # Empty ground truth
    # --------------------------------------------------------

    if not ground_truths:

        return {

            "status":
                "failed",

            "message":
                "No ground truth was found.",

            "context_count":
                len(contexts),

            "ground_truth_count":
                0,

            "ranked_context":
                len(contexts) > 1,

            "relevance_scores":
                [],

            "binary_relevance":
                [],

            "precision@k":
                None,

            "recall@k":
                None,

            "mrr":
                None,

            "ndcg@k":
                None,

            "hit_rate@k":
                None,
        }

    # --------------------------------------------------------
    # Semantic relevance
    # --------------------------------------------------------

    relevance_scores = (
        calculate_relevance_scores(
            contexts=contexts,
            ground_truths=ground_truths,
            model=model,
        )
    )

    # --------------------------------------------------------
    # Binary relevance
    # --------------------------------------------------------

    binary_relevance = [

        1 if score >= threshold else 0

        for score in relevance_scores
    ]

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics = (
        calculate_semantic_retrieval_metrics(
            relevance_scores,
            k=k,
            threshold=threshold,
        )
    )

    return {

        "status":
            "success",

        "message":
            "Semantic retrieval evaluation completed.",

        "context_count":
            len(contexts),

        "ground_truth_count":
            len(ground_truths),

        "ranked_context":
            len(contexts) > 1,

        "contexts":
            contexts,

        "relevance_scores":
            relevance_scores,

        "binary_relevance":
            binary_relevance,

        "precision@k":
            metrics[
                "precision@k"
            ],

        "recall@k":
            metrics[
                "recall@k"
            ],

        "mrr":
            metrics[
                "mrr"
            ],

        "ndcg@k":
            metrics[
                "ndcg@k"
            ],

        "hit_rate@k":
            metrics[
                "hit_rate@k"
            ],
    }


# ============================================================
# FULL DATASET EVALUATION
# ============================================================

def evaluate_semantic_retrieval(
    df: pd.DataFrame,
    k: int = DEFAULT_K,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> Dict[str, Any]:
    """
    Evaluate semantic retrieval for the complete dataset.
    """

    # ========================================================
    # CHECK REQUIRED COLUMNS
    # ========================================================

    required_columns = [
        "context",
        "ground_truth",
    ]

    missing_columns = [

        column

        for column in required_columns

        if column not in df.columns
    ]

    if missing_columns:

        return {

            "status":
                "error",

            "message":
                (
                    "Required retrieval columns are missing: "
                    + ", ".join(
                        missing_columns
                    )
                ),

            "summary": {

                "total_rows":
                    len(df),

                "successful_rows":
                    0,

                "failed_rows":
                    0,

                "unavailable_rows":
                    0,

                "k":
                    k,

                "similarity_threshold":
                    threshold,

                "embedding_model":
                    embedding_model,

                "precision@5":
                    None,

                "recall@5":
                    None,

                "mrr":
                    None,

                "ndcg@5":
                    None,

                "hit_rate@5":
                    None,
            },

            "rows": [],
        }

    # ========================================================
    # CHECK OLLAMA ONCE
    # ========================================================

    ollama_available = (
        is_ollama_available()
    )

    if not ollama_available:

        unavailable_rows = []

        for index in range(
            len(df)
        ):

            row = df.iloc[
                index
            ]

            contexts = parse_context(
                row.get(
                    "context"
                )
            )

            ground_truths = (
                parse_ground_truth(
                    row.get(
                        "ground_truth"
                    )
                )
            )

            unavailable_rows.append({

                "row_index":
                    index,

                "status":
                    "unavailable",

                "message":
                    (
                        "Ollama is not available. "
                        "Semantic retrieval evaluation "
                        "requires the local embedding model."
                    ),

                "context_count":
                    len(contexts),

                "ground_truth_count":
                    len(ground_truths),

                "ranked_context":
                    len(contexts) > 1,

                "relevance_scores":
                    [],

                "binary_relevance":
                    [],

                "precision@k":
                    None,

                "recall@k":
                    None,

                "mrr":
                    None,

                "ndcg@k":
                    None,

                "hit_rate@k":
                    None,
            })

        return {

            "status":
                "unavailable",

            "message":
                (
                    "Ollama is unavailable. "
                    "This is expected on the development "
                    "laptop when Ollama is not installed "
                    "or running."
                ),

            "summary": {

                "total_rows":
                    len(df),

                "successful_rows":
                    0,

                "failed_rows":
                    0,

                "unavailable_rows":
                    len(df),

                "k":
                    k,

                "similarity_threshold":
                    threshold,

                "embedding_model":
                    embedding_model,

                "precision@5":
                    None,

                "recall@5":
                    None,

                "mrr":
                    None,

                "ndcg@5":
                    None,

                "hit_rate@5":
                    None,
            },

            "rows":
                unavailable_rows,
        }

    # ========================================================
    # CHECK EMBEDDING MODEL
    # ========================================================

    available_models = (
        get_available_models()
    )

    if available_models:

        model_exists = any(

            embedding_model == model
            or embedding_model in model
            or model in embedding_model

            for model
            in available_models
        )

    else:

        model_exists = False

    if not model_exists:

        unavailable_rows = []

        for index in range(
            len(df)
        ):

            row = df.iloc[
                index
            ]

            contexts = parse_context(
                row.get(
                    "context"
                )
            )

            ground_truths = (
                parse_ground_truth(
                    row.get(
                        "ground_truth"
                    )
                )
            )

            unavailable_rows.append({

                "row_index":
                    index,

                "status":
                    "unavailable",

                "message":
                    (
                        f"Embedding model "
                        f"'{embedding_model}' "
                        "was not found in Ollama."
                    ),

                "context_count":
                    len(contexts),

                "ground_truth_count":
                    len(ground_truths),

                "ranked_context":
                    len(contexts) > 1,

                "relevance_scores":
                    [],

                "binary_relevance":
                    [],

                "precision@k":
                    None,

                "recall@k":
                    None,

                "mrr":
                    None,

                "ndcg@k":
                    None,

                "hit_rate@k":
                    None,
            })

        return {

            "status":
                "unavailable",

            "message":
                (
                    f"Embedding model "
                    f"'{embedding_model}' "
                    "was not found in Ollama."
                ),

            "summary": {

                "total_rows":
                    len(df),

                "successful_rows":
                    0,

                "failed_rows":
                    0,

                "unavailable_rows":
                    len(df),

                "k":
                    k,

                "similarity_threshold":
                    threshold,

                "embedding_model":
                    embedding_model,

                "precision@5":
                    None,

                "recall@5":
                    None,

                "mrr":
                    None,

                "ndcg@5":
                    None,

                "hit_rate@5":
                    None,
            },

            "rows":
                unavailable_rows,
        }

    # ========================================================
    # EVALUATE ROWS
    # ========================================================

    row_results = []

    successful_rows = 0

    failed_rows = 0

    for index in range(
        len(df)
    ):

        row = df.iloc[
            index
        ]

        try:

            result = (
                evaluate_semantic_row(
                    row=row,
                    context_column="context",
                    ground_truth_column="ground_truth",
                    model=embedding_model,
                    k=k,
                    threshold=threshold,
                )
            )

        except Exception as exc:

            result = {

                "status":
                    "failed",

                "message":
                    str(exc),

                "context_count":
                    0,

                "ground_truth_count":
                    0,

                "ranked_context":
                    False,

                "relevance_scores":
                    [],

                "binary_relevance":
                    [],

                "precision@k":
                    None,

                "recall@k":
                    None,

                "mrr":
                    None,

                "ndcg@k":
                    None,

                "hit_rate@k":
                    None,
            }

        result[
            "row_index"
        ] = index

        row_results.append(
            result
        )

        if result.get(
            "status"
        ) == "success":

            successful_rows += 1

        else:

            failed_rows += 1

    # ========================================================
    # AGGREGATE SUCCESSFUL RESULTS
    # ========================================================

    successful_results = [

        result

        for result
        in row_results

        if result.get(
            "status"
        ) == "success"
    ]

    def average_metric(
        metric_name: str,
    ):

        values = [

            result[
                metric_name
            ]

            for result
            in successful_results

            if result.get(
                metric_name
            ) is not None
        ]

        if not values:

            return None

        return float(
            np.mean(values)
        )

    precision = average_metric(
        "precision@k"
    )

    recall = average_metric(
        "recall@k"
    )

    mrr = average_metric(
        "mrr"
    )

    ndcg_score = average_metric(
        "ndcg@k"
    )

    hit = average_metric(
        "hit_rate@k"
    )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    return {

        "status":
            "completed",

        "message":
            (
                "Semantic retrieval evaluation "
                "completed."
            ),

        "summary": {

            "total_rows":
                len(df),

            "successful_rows":
                successful_rows,

            "failed_rows":
                failed_rows,

            "unavailable_rows":
                0,

            "k":
                k,

            "similarity_threshold":
                threshold,

            "embedding_model":
                embedding_model,

            "precision@5":
                precision,

            "recall@5":
                recall,

            "mrr":
                mrr,

            "ndcg@5":
                ndcg_score,

            "hit_rate@5":
                hit,
        },

        "rows":
            row_results,

        "methodology": {

            "type":
                "semantic_proxy_retrieval",

            "description":
                (
                    "Retrieved context chunks are compared "
                    "with ground-truth text using local "
                    "embedding cosine similarity."
                ),

            "threshold":
                threshold,

            "k":
                k,

            "limitations":
                (
                    "These are semantic/proxy retrieval "
                    "metrics because explicit gold-standard "
                    "document relevance labels are not "
                    "provided by the five-field dataset."
                ),
        },
    }


# ============================================================
# LEGACY / STANDARD RETRIEVAL EVALUATION
# ============================================================

def evaluate_retrieval(
    relevance_lists: List[List[int]],
    k: int = 5,
) -> Dict[str, float]:
    """
    Evaluate retrieval using explicit binary relevance labels.

    This function is retained for testing and for datasets
    that already contain true relevance labels.

    Example:

        [
            [1, 0, 1, 0, 0],
            [0, 1, 1, 0, 0]
        ]
    """

    if not relevance_lists:

        return {

            "precision@5":
                0.0,

            "recall@5":
                0.0,

            "mrr":
                0.0,

            "ndcg@5":
                0.0,

            "hit_rate@5":
                0.0,
        }

    precision_values = []

    recall_values = []

    reciprocal_ranks = []

    ndcg_values = []

    hits = []

    for relevance in relevance_lists:

        # ----------------------------------------------------
        # Total relevant documents
        # ----------------------------------------------------

        total_relevant = sum(
            relevance
        )

        # ----------------------------------------------------
        # Precision
        # ----------------------------------------------------

        precision_values.append(
            precision_at_k(
                relevance,
                k,
            )
        )

        # ----------------------------------------------------
        # Recall
        # ----------------------------------------------------

        recall_values.append(
            recall_at_k(
                relevance,
                total_relevant,
                k,
            )
        )

        # ----------------------------------------------------
        # Reciprocal Rank
        # ----------------------------------------------------

        reciprocal_ranks.append(
            reciprocal_rank(
                relevance
            )
        )

        # ----------------------------------------------------
        # NDCG
        # ----------------------------------------------------

        ndcg_values.append(
            ndcg(
                relevance,
                k,
            )
        )

        # ----------------------------------------------------
        # Hit Rate
        # ----------------------------------------------------

        hits.append(
            1.0
            if any(
                relevance[:k]
            )
            else 0.0
        )

    return {

        "precision@5":
            float(
                np.mean(
                    precision_values
                )
            ),

        "recall@5":
            float(
                np.mean(
                    recall_values
                )
            ),

        "mrr":
            float(
                np.mean(
                    reciprocal_ranks
                )
            ),

        "ndcg@5":
            float(
                np.mean(
                    ndcg_values
                )
            ),

        "hit_rate@5":
            float(
                np.mean(
                    hits
                )
            ),
    }


# ============================================================
# COMMAND-LINE TEST
# ============================================================

if __name__ == "__main__":

    print(
        "=" * 65
    )

    print(
        "AITrustEval - Retrieval Evaluator Test"
    )

    print(
        "=" * 65
    )

    # ========================================================
    # EXISTING EXPLICIT RETRIEVAL TEST
    # ========================================================

    print(
        "\n========== EXISTING RETRIEVAL EVALUATION =========="
    )

    test_relevance = [

        [1, 0, 1, 0, 0],

        [0, 1, 1, 0, 0],

        [1, 1, 0, 0, 0],

        [0, 0, 1, 0, 0],
    ]

    baseline = evaluate_retrieval(
        test_relevance,
        k=5,
    )

    for metric, value in (
        baseline.items()
    ):

        print(
            f"{metric}: {value:.4f}"
        )

    # ========================================================
    # OLLAMA STATUS
    # ========================================================

    print(
        "\n========== OLLAMA STATUS =========="
    )

    ollama_status = (
        is_ollama_available()
    )

    print(
        f"Ollama available: "
        f"{ollama_status}"
    )

    print(
        f"Embedding model: "
        f"{DEFAULT_EMBEDDING_MODEL}"
    )

    if ollama_status:

        models = get_available_models()

        print(
            "\nAvailable Ollama models:"
        )

        if models:

            for model in models:

                print(
                    f"  - {model}"
                )

        else:

            print(
                "  No models detected."
            )

    # ========================================================
    # SEMANTIC RETRIEVAL TEST
    # ========================================================

    print(
        "\n========== SEMANTIC RETRIEVAL TEST =========="
    )

    DATASET_PATH = (
        "data/sample/"
        "synthetic_5field_dataset.csv"
    )

    try:

        df = pd.read_csv(
            DATASET_PATH
        )

        print(
            f"Dataset rows: "
            f"{len(df)}"
        )

        print(
            f"Dataset columns: "
            f"{len(df.columns)}"
        )

        result = (
            evaluate_semantic_retrieval(
                df,
                k=DEFAULT_K,
                threshold=DEFAULT_SIMILARITY_THRESHOLD,
                embedding_model=DEFAULT_EMBEDDING_MODEL,
            )
        )

        # ----------------------------------------------------
        # Summary
        # ----------------------------------------------------

        print(
            "\n========== SUMMARY =========="
        )

        summary = result.get(
            "summary",
            {},
        )

        for key, value in (
            summary.items()
        ):

            print(
                f"{key}: {value}"
            )

        # ----------------------------------------------------
        # Status
        # ----------------------------------------------------

        print(
            "\nStatus:"
        )

        print(
            result.get(
                "status"
            )
        )

        # ----------------------------------------------------
        # Message
        # ----------------------------------------------------

        if result.get(
            "message"
        ):

            print(
                "\nMessage:"
            )

            print(
                result[
                    "message"
                ]
            )

        # ----------------------------------------------------
        # First 5 rows
        # ----------------------------------------------------

        rows = result.get(
            "rows",
            [],
        )

        if rows:

            print(
                "\n========== FIRST 5 ROWS =========="
            )

            for row in rows[:5]:

                print(
                    f"\nRow "
                    f"{row.get('row_index')}"
                )

                print(
                    f"Status: "
                    f"{row.get('status')}"
                )

                print(
                    f"Context count: "
                    f"{row.get('context_count')}"
                )

                print(
                    f"Ground truth count: "
                    f"{row.get('ground_truth_count')}"
                )

                print(
                    f"Ranked context: "
                    f"{row.get('ranked_context')}"
                )

                print(
                    f"Message: "
                    f"{row.get('message')}"
                )

        # ----------------------------------------------------
        # Methodology
        # ----------------------------------------------------

        methodology = result.get(
            "methodology"
        )

        if methodology:

            print(
                "\n========== METHODOLOGY =========="
            )

            print(
                f"Type: "
                f"{methodology.get('type')}"
            )

            print(
                f"Description: "
                f"{methodology.get('description')}"
            )

            print(
                f"Threshold: "
                f"{methodology.get('threshold')}"
            )

            print(
                f"K: "
                f"{methodology.get('k')}"
            )

            print(
                "\nLimitation:"
            )

            print(
                methodology.get(
                    "limitations"
                )
            )

    except FileNotFoundError:

        print(
            "\nDataset not found:"
        )

        print(
            DATASET_PATH
        )

    except Exception as exc:

        print(
            "\nSemantic retrieval test failed:"
        )

        print(
            str(exc)
        )

    print(
        "\n" + "=" * 65
    )

    print(
        "RETRIEVAL EVALUATOR TEST COMPLETE"
    )

    print(
        "=" * 65
    )