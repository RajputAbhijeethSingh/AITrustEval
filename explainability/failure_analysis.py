# ============================================================
# AITrustEval - Failure Analysis
# ============================================================

from typing import Any, Dict, List, Optional


# ============================================================
# SCORE STATUS
# ============================================================

def classify_score(
    score: Optional[float],
    good_threshold: float = 0.75,
    warning_threshold: float = 0.50,
) -> str:
    """
    Convert a metric score into a human-readable status.
    """

    if score is None:
        return "unavailable"

    if score >= good_threshold:
        return "good"

    if score >= warning_threshold:
        return "warning"

    return "poor"


# ============================================================
# FAILURE TYPE
# ============================================================

def identify_failure_type(
    metric: str,
    score: Optional[float],
) -> Optional[str]:
    """
    Identify the general failure category for a metric.
    """

    if score is None:
        return None

    if score >= 0.75:
        return None

    metric = metric.lower()

    if "faithfulness" in metric:
        return "unsupported_answer"

    if "correctness" in metric:
        return "incorrect_answer"

    if "relevancy" in metric:
        return "irrelevant_answer"

    if "precision" in metric:
        return "irrelevant_retrieval"

    if "recall" in metric:
        return "missing_information"

    if "ndcg" in metric:
        return "poor_ranking"

    if "mrr" in metric:
        return "late_relevant_retrieval"

    if "hit_rate" in metric:
        return "retrieval_miss"

    return "metric_failure"


# ============================================================
# FAILURE DESCRIPTION
# ============================================================

def describe_failure(
    failure_type: Optional[str],
) -> str:
    """
    Explain what a failure type means.
    """

    descriptions = {

        "unsupported_answer":
            (
                "The generated answer contains information "
                "that is not sufficiently supported by the "
                "retrieved context."
            ),

        "incorrect_answer":
            (
                "The generated answer differs from the "
                "expected ground-truth answer."
            ),

        "irrelevant_answer":
            (
                "The generated answer does not sufficiently "
                "address the user's question."
            ),

        "irrelevant_retrieval":
            (
                "The retrieved context contains information "
                "that is not sufficiently relevant to the "
                "ground truth."
            ),

        "missing_information":
            (
                "Relevant information required to answer "
                "the question was not sufficiently retrieved."
            ),

        "poor_ranking":
            (
                "Relevant information exists but is not ranked "
                "in an optimal position."
            ),

        "late_relevant_retrieval":
            (
                "The first relevant context appears at a "
                "relatively late retrieval position."
            ),

        "retrieval_miss":
            (
                "No sufficiently relevant context was found "
                "within the evaluated top-K results."
            ),

        "metric_failure":
            (
                "The metric score is below the expected "
                "quality threshold."
            ),
    }

    return descriptions.get(
        failure_type,
        "The metric did not meet the expected quality level.",
    )


# ============================================================
# FAILURE ANALYSIS FOR ONE RESULT
# ============================================================

def analyze_metric(
    metric: str,
    score: Optional[float],
) -> Dict[str, Any]:
    """
    Analyze a single metric result.
    """

    status = classify_score(
        score
    )

    failure_type = identify_failure_type(
        metric,
        score,
    )

    return {

        "metric": metric,

        "score": score,

        "status": status,

        "failure_type": failure_type,

        "description":
            describe_failure(
                failure_type
            )
            if failure_type
            else "Metric is performing within the expected range.",
    }


# ============================================================
# ANALYZE METRIC DICTIONARY
# ============================================================

def analyze_metrics(
    metrics: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Analyze a dictionary of metric scores.
    """

    results = []

    for metric, score in metrics.items():

        # Ignore non-numeric values
        if score is None:
            results.append(
                analyze_metric(
                    metric,
                    None,
                )
            )
            continue

        try:

            numeric_score = float(
                score
            )

        except (
            TypeError,
            ValueError,
        ):

            continue

        results.append(
            analyze_metric(
                metric,
                numeric_score,
            )
        )

    return results


# ============================================================
# FAILED METRICS
# ============================================================

def get_failed_metrics(
    analyses: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Return metrics classified as warning or poor.
    """

    return [

        result

        for result in analyses

        if result.get(
            "status"
        )
        in (
            "warning",
            "poor",
        )
    ]


# ============================================================
# SUMMARY
# ============================================================

def summarize_failures(
    analyses: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Generate an overall failure summary.
    """

    total = len(
        analyses
    )

    good = sum(

        1

        for result in analyses

        if result.get(
            "status"
        ) == "good"
    )

    warning = sum(

        1

        for result in analyses

        if result.get(
            "status"
        ) == "warning"
    )

    poor = sum(

        1

        for result in analyses

        if result.get(
            "status"
        ) == "poor"
    )

    unavailable = sum(

        1

        for result in analyses

        if result.get(
            "status"
        ) == "unavailable"
    )

    return {

        "total_metrics":
            total,

        "good":
            good,

        "warning":
            warning,

        "poor":
            poor,

        "unavailable":
            unavailable,

        "failed_metrics":
            warning + poor,
    }


# ============================================================
# COMPLETE FAILURE ANALYSIS
# ============================================================

def analyze_failures(
    metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Complete failure analysis pipeline.
    """

    analyses = analyze_metrics(
        metrics
    )

    failed = get_failed_metrics(
        analyses
    )

    summary = summarize_failures(
        analyses
    )

    return {

        "analyses":
            analyses,

        "failed_metrics":
            failed,

        "summary":
            summary,
    }


# ============================================================
# CLI TEST
# ============================================================

if __name__ == "__main__":

    print(
        "=" * 65
    )

    print(
        "AITrustEval - Failure Analysis Test"
    )

    print(
        "=" * 65
    )

    test_metrics = {

        "faithfulness":
            0.42,

        "answer_correctness":
            0.81,

        "answer_relevancy":
            0.63,

        "precision@5":
            0.45,

        "recall@5":
            0.90,

        "mrr":
            0.38,

        "ndcg@5":
            0.72,

        "hit_rate@5":
            0.95,
    }

    results = analyze_failures(
        test_metrics
    )

    print(
        "\n========== METRIC ANALYSIS =========="
    )

    for result in results[
        "analyses"
    ]:

        print(
            f"\nMetric: "
            f"{result['metric']}"
        )

        print(
            f"Score: "
            f"{result['score']}"
        )

        print(
            f"Status: "
            f"{result['status']}"
        )

        print(
            f"Failure type: "
            f"{result['failure_type']}"
        )

        print(
            f"Description: "
            f"{result['description']}"
        )

    print(
        "\n========== SUMMARY =========="
    )

    for key, value in (
        results[
            "summary"
        ].items()
    ):

        print(
            f"{key}: {value}"
        )

    print(
        "\n========== FAILED METRICS =========="
    )

    for result in (
        results[
            "failed_metrics"
        ]
    ):

        print(
            f"{result['metric']}: "
            f"{result['score']} "
            f"-> "
            f"{result['failure_type']}"
        )

    print(
        "\n" + "=" * 65
    )

    print(
        "FAILURE ANALYSIS TEST COMPLETE"
    )

    print(
        "=" * 65
    )