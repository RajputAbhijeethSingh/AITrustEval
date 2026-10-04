from datetime import datetime


def create_evaluation_result(
    dataset_info,
    detected_columns,
    metric_availability,
    metric_results,
    failures=None,
    warnings=None,
):
    """
    Create a standardized AITrustEval evaluation result.

    All evaluation modules will eventually return their
    results through this common structure.
    """

    if failures is None:
        failures = []

    if warnings is None:
        warnings = []

    return {
        "project": {
            "name": "AITrustEval",
            "version": "0.1",
        },

        "evaluation": {
            "timestamp": datetime.now().isoformat(),
        },

        "dataset": dataset_info,

        "schema": detected_columns,

        "metric_availability": metric_availability,

        "metrics": metric_results,

        "failures": failures,

        "warnings": warnings,
    }


def count_available_metrics(metric_availability):
    """
    Count how many metrics are available and unavailable.
    """

    available = 0
    unavailable = 0

    for group in metric_availability.values():

        for metric in group.values():

            if metric["available"]:
                available += 1
            else:
                unavailable += 1

    return {
        "available": available,
        "unavailable": unavailable,
        "total": available + unavailable,
    }


def print_result_summary(result):
    """
    Print a compact summary of an evaluation result.
    """

    print("\n========== AITrustEval RESULT SUMMARY ==========")

    print(
        f"Project: {result['project']['name']}"
    )

    print(
        f"Version: {result['project']['version']}"
    )

    print(
        f"Rows evaluated: "
        f"{result['dataset']['rows']}"
    )

    metric_counts = count_available_metrics(
        result["metric_availability"]
    )

    print(
        f"Available metrics: "
        f"{metric_counts['available']}"
    )

    print(
        f"Unavailable metrics: "
        f"{metric_counts['unavailable']}"
    )

    print("\nMetric Results:")

    for metric, score in result["metrics"].items():

        print(
            f"  {metric}: {score:.4f}"
        )

    print(
        "\n========== END RESULT SUMMARY =========="
    )


if __name__ == "__main__":

    # Simple test

    dataset_info = {
        "rows": 30,
        "columns": 4,
    }

    detected_columns = {
        "question": "question",
        "answer": "answer",
        "ground_truth": "ground_truths",
        "context": "contexts",
    }

    metric_availability = {
        "Answer Quality": {
            "answer_relevancy": {
                "available": True
            },
            "answer_correctness": {
                "available": True
            },
            "faithfulness": {
                "available": True
            },
        },
        "Retrieval": {
            "precision_at_k": {
                "available": False
            },
            "recall_at_k": {
                "available": False
            },
        },
    }

    metric_results = {
        "answer_relevancy": 0.4113,
        "answer_correctness": 0.2913,
        "context_answer_similarity": 0.3653,
    }

    result = create_evaluation_result(
        dataset_info=dataset_info,
        detected_columns=detected_columns,
        metric_availability=metric_availability,
        metric_results=metric_results,
    )

    print_result_summary(result)