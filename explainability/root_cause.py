"""
AITrustEval - Root Cause Analysis

Converts detected evaluation failures into:
- probable root causes
- possible contributing factors
- recommendations

This module is intentionally conservative.
Root causes are preliminary analytical explanations,
not definitive causal claims.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

import pandas as pd


# ============================================================
# ROOT-CAUSE KNOWLEDGE BASE
# ============================================================

ROOT_CAUSES = {

    "unsupported_answer": {
        "root_cause":
            "The generated answer may contain claims that are not sufficiently supported by the retrieved context.",

        "possible_sources": [
            "Insufficient retrieved context",
            "Irrelevant retrieved documents",
            "Incomplete context coverage",
            "Model generated unsupported information",
        ],

        "recommendation":
            "Improve retrieval relevance and context coverage, then verify answer claims against retrieved evidence.",
    },


    "incorrect_answer": {
        "root_cause":
            "The generated answer may not sufficiently match the expected reference answer.",

        "possible_sources": [
            "Incorrect model reasoning",
            "Incomplete context",
            "Ambiguous question",
            "Incorrect or incomplete reference information",
        ],

        "recommendation":
            "Inspect the retrieved evidence and improve answer generation and grounding.",
    },


    "irrelevant_answer": {
        "root_cause":
            "The generated answer may not adequately address the user's question.",

        "possible_sources": [
            "Question-answer semantic mismatch",
            "Weak prompt understanding",
            "Insufficient question context",
            "Model response drift",
        ],

        "recommendation":
            "Improve question understanding and constrain answer generation to the requested information.",
    },


    "irrelevant_retrieval": {
        "root_cause":
            "Retrieved context may contain information that is not sufficiently relevant to the question.",

        "possible_sources": [
            "Weak retrieval query",
            "Poor embedding representation",
            "Incorrect ranking",
            "Noisy knowledge base",
        ],

        "recommendation":
            "Improve embedding quality, retrieval ranking and document filtering.",
    },


    "missing_information": {
        "root_cause":
            "Relevant information may not have been retrieved or may be missing from the available context.",

        "possible_sources": [
            "Retrieval miss",
            "Incomplete knowledge base",
            "Low recall",
            "Insufficient retrieval depth",
        ],

        "recommendation":
            "Increase retrieval coverage and investigate missing or incorrectly indexed documents.",
    },


    "late_relevant_retrieval": {
        "root_cause":
            "Relevant information may appear too low in the retrieval ranking.",

        "possible_sources": [
            "Poor ranking model",
            "Embedding mismatch",
            "Weak query-document similarity",
            "Noisy top-ranked results",
        ],

        "recommendation":
            "Improve ranking quality and evaluate retrieval at multiple K values.",
    },


    "poor_ranking": {
        "root_cause":
            "Relevant information may be present but incorrectly ordered in the retrieved results.",

        "possible_sources": [
            "Ranking model weakness",
            "Embedding similarity limitations",
            "Duplicate or noisy documents",
            "Incorrect relevance estimation",
        ],

        "recommendation":
            "Tune retrieval ranking and evaluate the ordering of relevant documents.",
    },


    "retrieval_miss": {
        "root_cause":
            "No sufficiently relevant retrieved item was detected within the evaluated retrieval depth.",

        "possible_sources": [
            "Low retrieval recall",
            "Missing document",
            "Poor query representation",
            "Embedding mismatch",
        ],

        "recommendation":
            "Improve retrieval recall and verify that the required information exists in the knowledge base.",
    },


    "metric_failure": {
        "root_cause":
            "The evaluated metric indicates a potential quality issue that requires further investigation.",

        "possible_sources": [
            "Data quality issue",
            "Model behavior",
            "Evaluation methodology",
            "Insufficient evaluation evidence",
        ],

        "recommendation":
            "Inspect the individual sample, supporting evidence and evaluation methodology before making a final conclusion.",
    },
}


# ============================================================
# METRIC → FAILURE TYPE
# ============================================================

METRIC_FAILURE_MAP = {

    "faithfulness":
        "unsupported_answer",

    "answer_correctness":
        "incorrect_answer",

    "correctness":
        "incorrect_answer",

    "answer_relevancy":
        "irrelevant_answer",

    "relevancy":
        "irrelevant_answer",

    "context_precision":
        "irrelevant_retrieval",

    "precision":
        "irrelevant_retrieval",

    "precision@5":
        "irrelevant_retrieval",

    "context_recall":
        "missing_information",

    "recall":
        "missing_information",

    "recall@5":
        "missing_information",

    "mrr":
        "late_relevant_retrieval",

    "ndcg":
        "poor_ranking",

    "ndcg@5":
        "poor_ranking",

    "hit_rate":
        "retrieval_miss",

    "hit_rate@5":
        "retrieval_miss",

    "context_answer_similarity":
        "unsupported_answer",
}


# ============================================================
# HELPERS
# ============================================================

def _normalize_metric_name(
    metric: Any,
) -> str:

    return (
        str(metric)
        .strip()
        .lower()
        .replace(" ", "_")
    )


def _as_list(
    value: Any,
) -> List[Any]:

    if value is None:

        return []

    if isinstance(
        value,
        list,
    ):

        return value

    if isinstance(
        value,
        tuple,
    ):

        return list(value)

    if isinstance(
        value,
        set,
    ):

        return list(value)

    if isinstance(
        value,
        str,
    ):

        if not value.strip():

            return []

        return [
            value
        ]

    return [
        value
    ]


def _safe_score(
    value: Any,
):

    try:

        number = float(
            value
        )

        if pd.isna(
            number
        ):

            return None

        return number

    except Exception:

        return None


# ============================================================
# SCORE → FAILURE TYPE
# ============================================================

def identify_failure_type(
    metric: Any,
    score: Any = None,
) -> str:

    normalized = (
        _normalize_metric_name(
            metric
        )
    )


    if normalized in METRIC_FAILURE_MAP:

        return METRIC_FAILURE_MAP[
            normalized
        ]


    # Handle variations such as:
    # answer_correctness_score
    # precision_at_5
    # recall_at_5

    if (
        "correctness"
        in normalized
    ):

        return "incorrect_answer"


    if (
        "relevancy"
        in normalized
        or "relevance"
        in normalized
    ):

        return "irrelevant_answer"


    if (
        "faithfulness"
        in normalized
    ):

        return "unsupported_answer"


    if (
        "precision"
        in normalized
    ):

        return "irrelevant_retrieval"


    if (
        "recall"
        in normalized
    ):

        return "missing_information"


    if (
        "mrr"
        in normalized
    ):

        return "late_relevant_retrieval"


    if (
        "ndcg"
        in normalized
    ):

        return "poor_ranking"


    if (
        "hit"
        in normalized
    ):

        return "retrieval_miss"


    return "metric_failure"


# ============================================================
# ROOT CAUSE LOOKUP
# ============================================================

def get_root_cause(
    failure_type: str,
) -> Dict[str, Any]:

    return ROOT_CAUSES.get(
        failure_type,
        ROOT_CAUSES[
            "metric_failure"
        ],
    )


def analyze_root_cause(
    failure_type: str,
    metric: Any = None,
    score: Any = None,
) -> Dict[str, Any]:

    failure_type = (
        failure_type
        or "metric_failure"
    )


    information = get_root_cause(
        failure_type
    )


    result = {

        "failure_type":
            failure_type,

        "metric":
            metric,

        "score":
            score,

        "root_cause":
            information[
                "root_cause"
            ],

        "possible_sources":
            information[
                "possible_sources"
            ],

        "recommendation":
            information[
                "recommendation"
            ],
    }


    return result


# ============================================================
# MULTIPLE ROOT CAUSES
# ============================================================

def analyze_root_causes(
    failures: Iterable[Any],
) -> List[Dict[str, Any]]:

    results = []


    for failure in failures:

        if isinstance(
            failure,
            dict,
        ):

            metric = failure.get(
                "metric"
            )

            score = failure.get(
                "score"
            )

            failure_type = failure.get(
                "failure_type"
            )


        else:

            metric = str(
                failure
            )

            score = None

            failure_type = None


        if not failure_type:

            failure_type = (
                identify_failure_type(
                    metric,
                    score,
                )
            )


        results.append(
            analyze_root_cause(
                failure_type=
                    failure_type,

                metric=
                    metric,

                score=
                    score,
            )
        )


    return results


# ============================================================
# ROOT CAUSE SUMMARY
# ============================================================

def summarize_root_causes(
    analyses: List[Dict[str, Any]],
) -> Dict[str, Any]:

    if not analyses:

        return {
            "total":
                0,

            "root_causes":
                {},

            "recommendations":
                [],
        }


    counts = {}

    recommendations = []


    for analysis in analyses:

        failure_type = (
            analysis.get(
                "failure_type",
                "metric_failure",
            )
        )


        counts[
            failure_type
        ] = (
            counts.get(
                failure_type,
                0,
            )
            + 1
        )


        recommendation = (
            analysis.get(
                "recommendation"
            )
        )


        if (
            recommendation
            and recommendation
            not in recommendations
        ):

            recommendations.append(
                recommendation
            )


    return {

        "total":
            len(analyses),

        "root_causes":
            counts,

        "recommendations":
            recommendations,
    }


# ============================================================
# FAILURE DATAFRAME NORMALIZATION
# ============================================================

def _extract_failed_metrics(
    row: pd.Series,
) -> List[str]:

    # Preferred format:
    # failed_metrics = [...]
    if "failed_metrics" in row.index:

        metrics = _as_list(
            row.get(
                "failed_metrics"
            )
        )


        if metrics:

            return [
                str(metric)
                for metric in metrics
            ]


    # Alternative format:
    # failure_metrics = [...]
    if "failure_metrics" in row.index:

        metrics = _as_list(
            row.get(
                "failure_metrics"
            )
        )


        if metrics:

            return [
                str(metric)
                for metric in metrics
            ]


    # Single metric column.
    if "metric" in row.index:

        metric = row.get(
            "metric"
        )


        if metric is not None:

            return [
                str(metric)
            ]


    # Some failure-analysis implementations
    # may store failed metric names under
    # "failures".

    if "failures" in row.index:

        metrics = _as_list(
            row.get(
                "failures"
            )
        )


        if metrics:

            return [
                str(metric)
                for metric in metrics
            ]


    return []


# ============================================================
# GENERATE ROOT-CAUSE REPORT
# ============================================================

def generate_root_cause_report(
    failures_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert failure-analysis output into a
    dashboard-ready root-cause report.

    Expected output columns:

    - id
    - severity
    - failed_metrics
    - failure_count
    - root_cause
    - recommendation
    """

    if failures_df is None:

        return pd.DataFrame(
            columns=[
                "id",
                "severity",
                "failed_metrics",
                "failure_count",
                "root_cause",
                "recommendation",
            ]
        )


    if not isinstance(
        failures_df,
        pd.DataFrame,
    ):

        try:

            failures_df = pd.DataFrame(
                failures_df
            )

        except Exception:

            return pd.DataFrame()


    if failures_df.empty:

        return pd.DataFrame(
            columns=[
                "id",
                "severity",
                "failed_metrics",
                "failure_count",
                "root_cause",
                "recommendation",
            ]
        )


    report_rows = []


    for index, row in (
        failures_df.iterrows()
    ):

        # ----------------------------------------------------
        # ID
        # ----------------------------------------------------

        sample_id = row.get(
            "id",
            index + 1,
        )


        # ----------------------------------------------------
        # SEVERITY
        # ----------------------------------------------------

        severity = row.get(
            "severity",
            "Low",
        )


        if severity is None:

            severity = "Low"


        severity = str(
            severity
        ).title()


        # ----------------------------------------------------
        # FAILED METRICS
        # ----------------------------------------------------

        failed_metrics = (
            _extract_failed_metrics(
                row
            )
        )


        # ----------------------------------------------------
        # FALLBACK
        # ----------------------------------------------------

        if not failed_metrics:

            # Look for known metric columns
            # containing numeric low scores.

            known_metrics = [
                "answer_relevancy",
                "answer_correctness",
                "context_answer_similarity",
                "faithfulness",
                "context_precision",
                "context_recall",
                "precision@5",
                "recall@5",
                "mrr",
                "ndcg@5",
                "hit_rate@5",
            ]


            for metric in (
                known_metrics
            ):

                if metric not in row.index:

                    continue


                score = _safe_score(
                    row.get(
                        metric
                    )
                )


                if (
                    score is not None
                    and score < 0.50
                ):

                    failed_metrics.append(
                        metric
                    )


        # ----------------------------------------------------
        # ROOT CAUSE
        # ----------------------------------------------------

        root_causes = []


        recommendations_for_row = []


        for metric in (
            failed_metrics
        ):

            failure_type = (
                identify_failure_type(
                    metric
                )
            )


            analysis = (
                analyze_root_cause(
                    failure_type=
                        failure_type,

                    metric=
                        metric,

                    score=
                        row.get(
                            metric
                        )
                        if metric
                        in row.index
                        else None,
                )
            )


            root_causes.append(
                analysis[
                    "root_cause"
                ]
            )


            recommendation = (
                analysis[
                    "recommendation"
                ]
            )


            if (
                recommendation
                and recommendation
                not in recommendations_for_row
            ):

                recommendations_for_row.append(
                    recommendation
                )


        # ----------------------------------------------------
        # NO FAILURE INFORMATION
        # ----------------------------------------------------

        if not root_causes:

            root_causes.append(
                "The available failure record does not contain enough metric-level information to determine a specific root cause."
            )


        if not recommendations_for_row:

            recommendations_for_row.append(
                "Inspect the sample-level evidence and evaluation criteria before making a final conclusion."
            )


        report_rows.append(
            {
                "id":
                    sample_id,

                "severity":
                    severity,

                "failed_metrics":
                    ", ".join(
                        failed_metrics
                    )
                    if failed_metrics
                    else "Not Available",

                "failure_count":
                    len(
                        failed_metrics
                    ),

                "root_cause":
                    " | ".join(
                        dict.fromkeys(
                            root_causes
                        )
                    ),

                "recommendation":
                    " | ".join(
                        dict.fromkeys(
                            recommendations_for_row
                        )
                    ),
            }
        )


    return pd.DataFrame(
        report_rows
    )


# ============================================================
# COMPLETE ROOT-CAUSE ANALYSIS
# ============================================================

def generate_root_cause_analysis(
    failures_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Generate a structured root-cause analysis payload.
    """

    report_df = (
        generate_root_cause_report(
            failures_df
        )
    )


    if report_df.empty:

        return {
            "status":
                "completed",

            "total_samples":
                0,

            "report":
                report_df,

            "summary": {
                "total":
                    0,

                "root_causes":
                    {},

                "recommendations":
                    [],
            },
        }


    analyses = []


    for _, row in (
        report_df.iterrows()
    ):

        metrics = _as_list(
            row.get(
                "failed_metrics"
            )
        )


        if isinstance(
            row.get(
                "failed_metrics"
            ),
            str,
        ):

            metrics = [
                metric.strip()
                for metric in
                row[
                    "failed_metrics"
                ].split(",")
                if metric.strip()
            ]


        for metric in metrics:

            failure_type = (
                identify_failure_type(
                    metric
                )
            )


            analyses.append(
                analyze_root_cause(
                    failure_type=
                        failure_type,

                    metric=
                        metric,
                )
            )


    summary = (
        summarize_root_causes(
            analyses
        )
    )


    return {

        "status":
            "completed",

        "total_samples":
            len(report_df),

        "report":
            report_df,

        "analyses":
            analyses,

        "summary":
            summary,
    }


# ============================================================
# CLI TEST
# ============================================================

if __name__ == "__main__":

    print(
        "=" * 70
    )

    print(
        "AITrustEval - Root Cause Analysis Test"
    )

    print(
        "=" * 70
    )


    sample_failures = pd.DataFrame(
        [
            {
                "id":
                    1,

                "severity":
                    "High",

                "failed_metrics":
                    [
                        "answer_correctness",
                        "answer_relevancy",
                    ],
            },

            {
                "id":
                    2,

                "severity":
                    "Medium",

                "failed_metrics":
                    [
                        "precision@5",
                        "mrr",
                    ],
            },

            {
                "id":
                    3,

                "severity":
                    "Low",

                "failed_metrics":
                    [
                        "ndcg@5",
                    ],
            },
        ]
    )


    report = (
        generate_root_cause_report(
            sample_failures
        )
    )


    print(
        "\nRoot-Cause Report:"
    )


    print(
        report.to_string(
            index=False
        )
    )


    print(
        "\n"
        + "=" * 70
    )

    print(
        "Root Cause Analysis Test Complete"
    )

    print(
        "=" * 70
    )