"""
AITrustEval - Explainability Engine

Provides:

    - Failure analysis
    - Root-cause analysis
    - Evidence analysis
    - Recommendations
    - Sample-level investigation
    - Overall explainability summary

Important architecture:

    The evaluation engine should run first.

    Then:

        run_explainability(
            df,
            evaluation_result=evaluation_result
        )

    This prevents the evaluation engine from being executed twice.

The module also supports:

    run_explainability(df)

for standalone testing/backward compatibility.
"""

from __future__ import annotations

import ast
import json
from typing import Any, Dict, List, Optional

import pandas as pd


# ============================================================
# PROJECT IMPORTS
# ============================================================

from evaluation.engine.evaluator import (
    run_evaluation,
)

from explainability.failure_analysis import (
    analyze_metrics,
    summarize_failures,
    get_failed_metrics,
)

from explainability.root_cause import (
    analyze_root_causes,
    summarize_root_causes,
)

from explainability.evidence import (
    analyze_context_evidence,
    analyze_ground_truth_evidence,
    analyze_retrieval_evidence,
    identify_evidence_gap,
)


# ============================================================
# GENERAL HELPERS
# ============================================================

def safe_dict(
    value: Any,
) -> Dict[str, Any]:

    if isinstance(
        value,
        dict,
    ):

        return value

    return {}


def safe_list(
    value: Any,
) -> List[Any]:

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

    return []


def clean_value(
    value: Any,
) -> Any:

    if value is None:
        return None

    if isinstance(
        value,
        float,
    ):

        if pd.isna(value):
            return None

        return float(value)

    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
        ),
    ):

        return value

    if isinstance(
        value,
        pd.Series,
    ):

        return value.to_dict()

    if isinstance(
        value,
        pd.DataFrame,
    ):

        return value.to_dict(
            orient="records"
        )

    if isinstance(
        value,
        dict,
    ):

        return {
            str(key):
                clean_value(item)
            for key, item in value.items()
        }

    if isinstance(
        value,
        list,
    ):

        return [
            clean_value(item)
            for item in value
        ]

    return str(value)


# ============================================================
# VALUE PARSING
# ============================================================

def parse_serialized_value(
    value: Any,
) -> Any:

    """
    Parse values stored as:

        JSON
        Python literal
        plain strings
        lists
        dictionaries

    This is especially useful for the five-field dataset where
    context and metadata are commonly serialized strings.
    """

    if value is None:
        return None

    if isinstance(
        value,
        (
            list,
            dict,
        ),
    ):

        return value

    if not isinstance(
        value,
        str,
    ):

        return value

    text = value.strip()

    if not text:
        return ""

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    try:

        return json.loads(
            text
        )

    except Exception:
        pass

    # --------------------------------------------------------
    # Python literal
    # --------------------------------------------------------

    try:

        return ast.literal_eval(
            text
        )

    except Exception:
        pass

    return value


# ============================================================
# CONTEXT NORMALIZATION
# ============================================================

def normalize_context(
    context: Any,
) -> List[Dict[str, Any]]:

    """
    Convert different context representations into:

        [
            {
                "rank": 1,
                "context": "..."
            },
            ...
        ]

    Supported:

        - list of dictionaries
        - list of strings
        - dictionary
        - serialized JSON list
        - serialized Python list
        - plain text
    """

    parsed = parse_serialized_value(
        context
    )

    # --------------------------------------------------------
    # List
    # --------------------------------------------------------

    if isinstance(
        parsed,
        list,
    ):

        normalized = []

        for index, item in enumerate(
            parsed,
            start=1,
        ):

            if isinstance(
                item,
                dict,
            ):

                text = (
                    item.get(
                        "context"
                    )
                    or item.get(
                        "text"
                    )
                    or item.get(
                        "content"
                    )
                    or item.get(
                        "document"
                    )
                    or ""
                )

                rank = (
                    item.get(
                        "rank"
                    )
                    or item.get(
                        "position"
                    )
                    or index
                )

                normalized.append(
                    {
                        "rank": rank,
                        "context": str(
                            text
                        ),
                    }
                )

            else:

                normalized.append(
                    {
                        "rank": index,
                        "context": str(
                            item
                        ),
                    }
                )

        return normalized

    # --------------------------------------------------------
    # Dictionary
    # --------------------------------------------------------

    if isinstance(
        parsed,
        dict,
    ):

        if (
            "context"
            in parsed
        ):

            text = parsed.get(
                "context"
            )

            if isinstance(
                text,
                list,
            ):

                return normalize_context(
                    text
                )

            return [
                {
                    "rank": 1,
                    "context": str(
                        text
                    ),
                }
            ]

        if (
            "chunks"
            in parsed
        ):

            return normalize_context(
                parsed.get(
                    "chunks"
                )
            )

        if (
            "documents"
            in parsed
        ):

            return normalize_context(
                parsed.get(
                    "documents"
                )
            )

        # Generic dictionary:
        # use values as context items.

        normalized = []

        for index, (
            key,
            value,
        ) in enumerate(
            parsed.items(),
            start=1,
        ):

            normalized.append(
                {
                    "rank": index,
                    "context": (
                        str(value)
                    ),
                }
            )

        return normalized

    # --------------------------------------------------------
    # Plain text
    # --------------------------------------------------------

    if isinstance(
        parsed,
        str,
    ):

        return [
            {
                "rank": 1,
                "context": parsed,
            }
        ]

    return []


# ============================================================
# GROUND TRUTH NORMALIZATION
# ============================================================

def normalize_ground_truth(
    ground_truth: Any,
) -> str:

    parsed = parse_serialized_value(
        ground_truth
    )

    if parsed is None:

        return ""

    if isinstance(
        parsed,
        list,
    ):

        return " ".join(
            str(item)
            for item in parsed
        )

    if isinstance(
        parsed,
        dict,
    ):

        for key in [
            "ground_truth",
            "reference",
            "answer",
            "text",
        ]:

            if key in parsed:

                return str(
                    parsed[key]
                )

        return json.dumps(
            parsed,
            ensure_ascii=False,
        )

    return str(
        parsed
    )


# ============================================================
# METRIC EXTRACTION
# ============================================================

def extract_metric_scores(
    evaluation_result: Dict[str, Any],
) -> Dict[str, Any]:

    """
    Extract all available aggregate metric scores.

    This function deliberately ignores unavailable values.
    """

    evaluation_result = safe_dict(
        evaluation_result
    )

    scores = {}

    # --------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------

    retrieval = safe_dict(
        evaluation_result.get(
            "retrieval"
        )
    )

    retrieval_mapping = {
        "precision@5":
            retrieval.get(
                "precision@5"
            ),

        "recall@5":
            retrieval.get(
                "recall@5"
            ),

        "mrr":
            retrieval.get(
                "mrr"
            ),

        "ndcg@5":
            retrieval.get(
                "ndcg@5"
            ),

        "hit_rate@5":
            retrieval.get(
                "hit_rate@5"
            ),
    }

    for metric, value in (
        retrieval_mapping.items()
    ):

        if value is not None:

            try:

                scores[
                    metric
                ] = float(value)

            except Exception:
                pass

    # --------------------------------------------------------
    # RAGAS
    # --------------------------------------------------------

    ragas = safe_dict(
        evaluation_result.get(
            "ragas"
        )
    )

    ragas_results = (
        ragas.get(
            "results"
        )
        or ragas.get(
            "metrics"
        )
    )

    if isinstance(
        ragas_results,
        pd.DataFrame,
    ):

        for column in ragas_results.columns:

            numeric = pd.to_numeric(
                ragas_results[
                    column
                ],
                errors="coerce",
            )

            if numeric.notna().any():

                scores[
                    column
                ] = float(
                    numeric.mean()
                )

    elif isinstance(
        ragas_results,
        list,
    ):

        results_df = pd.DataFrame(
            ragas_results
        )

        for column in results_df.columns:

            numeric = pd.to_numeric(
                results_df[
                    column
                ],
                errors="coerce",
            )

            if numeric.notna().any():

                scores[
                    column
                ] = float(
                    numeric.mean()
                )

    elif isinstance(
        ragas_results,
        dict,
    ):

        for metric, value in (
            ragas_results.items()
        ):

            if isinstance(
                value,
                dict,
            ):

                value = (
                    value.get(
                        "mean"
                    )
                    or value.get(
                        "average"
                    )
                    or value.get(
                        "score"
                    )
                )

            try:

                if value is not None:

                    scores[
                        metric
                    ] = float(
                        value
                    )

            except Exception:
                pass

    # --------------------------------------------------------
    # Other possible aggregate scores
    # --------------------------------------------------------

    for section_name in [
        "baseline",
        "answer_quality",
        "answer_evaluation",
    ]:

        section = safe_dict(
            evaluation_result.get(
                section_name
            )
        )

        for metric, value in (
            section.items()
        ):

            try:

                if (
                    isinstance(
                        value,
                        (
                            int,
                            float,
                        ),
                    )
                    and not pd.isna(value)
                ):

                    scores[
                        metric
                    ] = float(
                        value
                    )

            except Exception:
                pass

    return scores


# ============================================================
# SAMPLE-LEVEL METRIC EXTRACTION
# ============================================================

def extract_sample_results(
    evaluation_result: Dict[str, Any],
) -> pd.DataFrame:

    """
    Extract sample-level baseline/RAGAS results if available.
    """

    evaluation_result = safe_dict(
        evaluation_result
    )

    possible_keys = [
        "sample_results",
        "results",
        "ragas_results",
        "baseline_results",
    ]

    for key in possible_keys:

        value = evaluation_result.get(
            key
        )

        if isinstance(
            value,
            pd.DataFrame,
        ):

            if not value.empty:

                return value.copy()

        if isinstance(
            value,
            list,
        ):

            try:

                result_df = pd.DataFrame(
                    value
                )

                if not result_df.empty:

                    return result_df

            except Exception:
                pass

    # --------------------------------------------------------
    # Look inside RAGAS
    # --------------------------------------------------------

    ragas = safe_dict(
        evaluation_result.get(
            "ragas"
        )
    )

    for key in [
        "results",
        "sample_results",
        "data",
    ]:

        value = ragas.get(
            key
        )

        if isinstance(
            value,
            pd.DataFrame,
        ):

            if not value.empty:

                return value.copy()

        if isinstance(
            value,
            list,
        ):

            try:

                result_df = pd.DataFrame(
                    value
                )

                if not result_df.empty:

                    return result_df

            except Exception:
                pass

    return pd.DataFrame()


# ============================================================
# OVERALL QUALITY
# ============================================================

def calculate_overall_quality(
    scores: Dict[str, Any],
) -> Dict[str, Any]:

    numeric_scores = []

    for value in scores.values():

        try:

            number = float(
                value
            )

            if not pd.isna(
                number
            ):

                numeric_scores.append(
                    number
                )

        except Exception:
            pass

    if not numeric_scores:

        return {
            "score": None,
            "classification": "unavailable",
            "metrics_analyzed": 0,
        }

    score = sum(
        numeric_scores
    ) / len(
        numeric_scores
    )

    if score >= 0.75:

        classification = "good"

    elif score >= 0.50:

        classification = "warning"

    else:

        classification = "poor"

    return {
        "score": score,
        "classification": classification,
        "metrics_analyzed": len(
            numeric_scores
        ),
    }


# ============================================================
# SAMPLE EXPLAINABILITY
# ============================================================

def analyze_sample(
    row: pd.Series,
    sample_id: int,
    evaluation_result: Optional[
        Dict[str, Any]
    ] = None,
) -> Dict[str, Any]:

    """
    Generate explainability information for one dataset row.
    """

    evaluation_result = (
        safe_dict(
            evaluation_result
        )
    )

    # --------------------------------------------------------
    # Field names
    # --------------------------------------------------------

    schema = safe_dict(
        evaluation_result.get(
            "schema"
        )
    )

    question_column = (
        schema.get(
            "question"
        )
        or "question"
    )

    context_column = (
        schema.get(
            "context"
        )
        or "context"
    )

    answer_column = (
        schema.get(
            "answer"
        )
        or "answer"
    )

    ground_truth_column = (
        schema.get(
            "ground_truth"
        )
        or "ground_truth"
    )

    # --------------------------------------------------------
    # Values
    # --------------------------------------------------------

    question = row.get(
        question_column,
        "",
    )

    context = row.get(
        context_column,
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

    normalized_context = (
        normalize_context(
            context
        )
    )

    normalized_ground_truth = (
        normalize_ground_truth(
            ground_truth
        )
    )

    # --------------------------------------------------------
    # Evidence
    # --------------------------------------------------------

    try:

        context_evidence = (
            analyze_context_evidence(
                answer,
                normalized_context,
            )
        )

    except Exception as exc:

        context_evidence = {
            "status": "error",
            "message": str(exc),
        }

    try:

        ground_truth_evidence = (
            analyze_ground_truth_evidence(
                normalized_ground_truth,
                normalized_context,
            )
        )

    except Exception as exc:

        ground_truth_evidence = {
            "status": "error",
            "message": str(exc),
        }

    try:

        retrieval_evidence = (
            analyze_retrieval_evidence(
                normalized_ground_truth,
                normalized_context,
            )
        )

    except Exception as exc:

        retrieval_evidence = {
            "status": "error",
            "message": str(exc),
        }

    try:

        evidence_gap = (
            identify_evidence_gap(
                answer,
                normalized_context,
                normalized_ground_truth,
            )
        )

    except Exception as exc:

        evidence_gap = {
            "status": "error",
            "message": str(exc),
        }

    return {
        "id": sample_id,

        "question": clean_value(
            question
        ),

        "answer": clean_value(
            answer
        ),

        "ground_truth": clean_value(
            ground_truth
        ),

        "context_evidence":
            clean_value(
                context_evidence
            ),

        "ground_truth_evidence":
            clean_value(
                ground_truth_evidence
            ),

        "retrieval_evidence":
            clean_value(
                retrieval_evidence
            ),

        "evidence_gap":
            clean_value(
                evidence_gap
            ),
    }


# ============================================================
# MAIN EXPLAINABILITY FUNCTION
# ============================================================

def run_explainability(
    df: pd.DataFrame,
    evaluation_result: Optional[
        Dict[str, Any]
    ] = None,
) -> Dict[str, Any]:

    """
    Run the complete explainability pipeline.

    Preferred usage:

        evaluation = run_evaluation(df)

        explainability = run_explainability(
            df,
            evaluation_result=evaluation
        )

    This avoids executing run_evaluation twice.

    Backward-compatible usage:

        explainability = run_explainability(df)

    will execute run_evaluation internally.
    """

    if not isinstance(
        df,
        pd.DataFrame,
    ):

        raise TypeError(
            "df must be a pandas DataFrame."
        )

    # --------------------------------------------------------
    # Evaluation result
    # --------------------------------------------------------

    if evaluation_result is None:

        evaluation_result = (
            run_evaluation(
                df
            )
        )

    evaluation_result = safe_dict(
        evaluation_result
    )

    # --------------------------------------------------------
    # Aggregate scores
    # --------------------------------------------------------

    aggregate_scores = (
        extract_metric_scores(
            evaluation_result
        )
    )

    overall_quality = (
        calculate_overall_quality(
            aggregate_scores
        )
    )

    # --------------------------------------------------------
    # Failure analysis
    # --------------------------------------------------------

    metric_analysis = []

    if aggregate_scores:

        try:

            metric_analysis = (
                analyze_metrics(
                    aggregate_scores
                )
            )

        except Exception:

            metric_analysis = []

    failure_summary = {}

    if metric_analysis:

        try:

            failure_summary = (
                summarize_failures(
                    metric_analysis
                )
            )

        except Exception:

            failure_summary = {}

    failed_metrics = []

    if metric_analysis:

        try:

            failed_metrics = (
                get_failed_metrics(
                    metric_analysis
                )
            )

        except Exception:

            failed_metrics = []

    # --------------------------------------------------------
    # Root cause analysis
    # --------------------------------------------------------

    root_cause_results = []

    if metric_analysis:

        try:

            root_cause_results = (
                analyze_root_causes(
                    metric_analysis
                )
            )

        except Exception:

            root_cause_results = []

    root_cause_summary = {}

    if root_cause_results:

        try:

            root_cause_summary = (
                summarize_root_causes(
                    root_cause_results
                )
            )

        except Exception:

            root_cause_summary = {}

    # --------------------------------------------------------
    # Sample-level evaluation results
    # --------------------------------------------------------

    sample_results = (
        extract_sample_results(
            evaluation_result
        )
    )

    # --------------------------------------------------------
    # Sample-level explainability
    # --------------------------------------------------------

    sample_analysis = []

    for index, row in df.iterrows():

        sample_id = (
            index + 1
        )

        try:

            result = analyze_sample(
                row,
                sample_id,
                evaluation_result,
            )

        except Exception as exc:

            result = {
                "id": sample_id,
                "status": "error",
                "message": str(exc),
            }

        # ----------------------------------------------------
        # Attach sample-level evaluation scores when available
        # ----------------------------------------------------

        if not sample_results.empty:

            matching_row = None

            if "id" in sample_results.columns:

                matches = sample_results[
                    sample_results["id"]
                    == sample_id
                ]

                if not matches.empty:

                    matching_row = (
                        matches.iloc[0]
                    )

            elif (
                len(sample_results)
                >= sample_id
            ):

                matching_row = (
                    sample_results.iloc[
                        sample_id - 1
                    ]
                )

            if matching_row is not None:

                sample_metrics = {}

                for (
                    key,
                    value,
                ) in matching_row.items():

                    if key == "id":
                        continue

                    cleaned = clean_value(
                        value
                    )

                    if isinstance(
                        cleaned,
                        (
                            int,
                            float,
                        ),
                    ):

                        sample_metrics[
                            key
                        ] = cleaned

                result[
                    "metrics"
                ] = sample_metrics

                # ------------------------------------------------
                # Failure analysis for this sample
                # ------------------------------------------------

                if sample_metrics:

                    try:

                        sample_metric_analysis = (
                            analyze_metrics(
                                sample_metrics
                            )
                        )

                    except Exception:

                        sample_metric_analysis = []

                    result[
                        "metric_analysis"
                    ] = clean_value(
                        sample_metric_analysis
                    )

                    try:

                        result[
                            "failed_metrics"
                        ] = clean_value(
                            get_failed_metrics(
                                sample_metric_analysis
                            )
                        )

                    except Exception:

                        result[
                            "failed_metrics"
                        ] = []

                    try:

                        sample_root_causes = (
                            analyze_root_causes(
                                sample_metric_analysis
                            )
                        )

                    except Exception:

                        sample_root_causes = []

                    result[
                        "root_causes"
                    ] = clean_value(
                        sample_root_causes
                    )

        sample_analysis.append(
            result
        )

    # --------------------------------------------------------
    # Recommendation generation
    # --------------------------------------------------------

    recommendations = []

    if root_cause_results:

        for item in root_cause_results:

            if not isinstance(
                item,
                dict,
            ):

                continue

            recommendation = (
                item.get(
                    "recommendation"
                )
            )

            if recommendation:

                recommendation = str(
                    recommendation
                )

                if (
                    recommendation
                    not in recommendations
                ):

                    recommendations.append(
                        recommendation
                    )

    # --------------------------------------------------------
    # Status
    # --------------------------------------------------------

    if aggregate_scores:

        status = "completed"

        message = (
            "Explainability analysis completed "
            "using available evaluation scores."
        )

    else:

        status = "completed"

        message = (
            "Explainability analysis completed. "
            "No aggregate evaluation scores are currently "
            "available, so metric-based failure analysis is "
            "limited. Sample-level evidence analysis remains "
            "available."
        )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    result = {
        "status": status,

        "message": message,

        "overall_quality":
            overall_quality,

        "aggregate_metrics":
            clean_value(
                aggregate_scores
            ),

        "metric_analysis":
            clean_value(
                metric_analysis
            ),

        "failed_metrics":
            clean_value(
                failed_metrics
            ),

        "failure_summary":
            clean_value(
                failure_summary
            ),

        "root_cause_analysis":
            clean_value(
                root_cause_results
            ),

        "root_cause_summary":
            clean_value(
                root_cause_summary
            ),

        "recommendations":
            clean_value(
                recommendations
            ),

        "sample_results":
            clean_value(
                sample_results
            ),

        "samples":
            clean_value(
                sample_analysis
            ),

        "samples_analyzed":
            len(
                sample_analysis
            ),
    }

    return result


# ============================================================
# CONVENIENCE FUNCTION
# ============================================================

def run_explainability_with_evaluation(
    df: pd.DataFrame,
    evaluation_result: Dict[str, Any],
) -> Dict[str, Any]:

    """
    Explicit convenience wrapper for dashboard usage.
    """

    return run_explainability(
        df,
        evaluation_result=evaluation_result,
    )


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    print()
    print(
        "=" * 70
    )

    print(
        "AITrustEval - Explainability Engine Test"
    )

    print(
        "=" * 70
    )

    print()

    dataset_path = (
        "data/sample/"
        "synthetic_5field_dataset.csv"
    )

    print(
        "Loading dataset..."
    )

    try:

        df = pd.read_csv(
            dataset_path
        )

    except Exception as exc:

        print(
            f"Unable to load dataset: {exc}"
        )

        raise SystemExit(1)

    print(
        f"Rows: {len(df)}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    print()

    print(
        "Running centralized evaluation..."
    )

    evaluation = run_evaluation(
        df
    )

    print(
        f"Evaluation status: "
        f"{evaluation.get('status')}"
    )

    print()

    print(
        "Running explainability..."
    )

    result = run_explainability(
        df,
        evaluation_result=evaluation,
    )

    print()

    print(
        "========== EXPLAINABILITY SUMMARY =========="
    )

    print(
        f"Status: "
        f"{result.get('status')}"
    )

    print(
        f"Message: "
        f"{result.get('message')}"
    )

    print(
        f"Samples analyzed: "
        f"{result.get('samples_analyzed')}"
    )

    overall_quality = result.get(
        "overall_quality",
        {},
    )

    print()

    print(
        "Overall Quality:"
    )

    print(
        f"  Score: "
        f"{overall_quality.get('score')}"
    )

    print(
        f"  Classification: "
        f"{overall_quality.get('classification')}"
    )

    print(
        f"  Metrics analyzed: "
        f"{overall_quality.get('metrics_analyzed')}"
    )

    print()

    print(
        "Aggregate Metrics:"
    )

    aggregate_metrics = result.get(
        "aggregate_metrics",
        {},
    )

    if aggregate_metrics:

        for (
            metric,
            value,
        ) in aggregate_metrics.items():

            print(
                f"  {metric}: "
                f"{value}"
            )

    else:

        print(
            "  None"
        )

    print()

    print(
        "Failed Metrics:"
    )

    failed_metrics = result.get(
        "failed_metrics",
        [],
    )

    if failed_metrics:

        for metric in failed_metrics:

            print(
                f"  - {metric}"
            )

    else:

        print(
            "  None"
        )

    print()

    print(
        "Failure Summary:"
    )

    failure_summary = result.get(
        "failure_summary",
        {},
    )

    if failure_summary:

        for (
            key,
            value,
        ) in failure_summary.items():

            print(
                f"  {key}: "
                f"{value}"
            )

    else:

        print(
            "  None"
        )

    print()

    print(
        "Root Cause Summary:"
    )

    root_cause_summary = result.get(
        "root_cause_summary",
        {},
    )

    if root_cause_summary:

        for (
            key,
            value,
        ) in root_cause_summary.items():

            print(
                f"  {key}: "
                f"{value}"
            )

    else:

        print(
            "  None"
        )

    print()

    print(
        "Recommendations:"
    )

    recommendations = result.get(
        "recommendations",
        [],
    )

    if recommendations:

        for recommendation in recommendations:

            print(
                f"  - {recommendation}"
            )

    else:

        print(
            "  None"
        )

    print()

    print(
        "========== FIRST SAMPLE =========="
    )

    samples = result.get(
        "samples",
        [],
    )

    if samples:

        first_sample = samples[0]

        print(
            f"Sample ID: "
            f"{first_sample.get('id')}"
        )

        print(
            f"Question: "
            f"{first_sample.get('question')}"
        )

        print()

        print(
            "Evidence Gap:"
        )

        print(
            first_sample.get(
                "evidence_gap",
                {},
            )
        )

        print()

        context_evidence = (
            first_sample.get(
                "context_evidence",
                {},
            )
        )

        print(
            "Context Evidence:"
        )

        if isinstance(
            context_evidence,
            dict,
        ):

            chunks = (
                context_evidence.get(
                    "chunks",
                    [],
                )
            )

            if chunks:

                for chunk in chunks:

                    print(
                        f"  Rank "
                        f"{chunk.get('rank')}"
                        f" | Overlap "
                        f"{chunk.get('overlap')}"
                    )

            else:

                print(
                    context_evidence
                )

        else:

            print(
                context_evidence
            )

    else:

        print(
            "No sample results."
        )

    print()

    print(
        "=" * 70
    )

    print(
        "EXPLAINABILITY ENGINE TEST COMPLETE"
    )

    print(
        "=" * 70
    )