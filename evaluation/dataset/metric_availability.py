def build_metric_availability(validation_result):
    """
    Convert dataset validation results into a structured
    metric availability report.
    """

    capabilities = validation_result["capabilities"]

    metric_groups = {
        "Answer Quality": [
            "answer_relevancy",
            "answer_correctness",
            "faithfulness",
            "context_metrics",
        ],
        "Retrieval": [
            "precision_at_k",
            "recall_at_k",
            "mrr",
            "ndcg",
            "hit_rate",
        ],
        "Performance": [
            "latency",
        ],
    }

    availability = {}

    for group, metrics in metric_groups.items():

        availability[group] = {}

        for metric in metrics:

            available = capabilities.get(metric, False)

            availability[group][metric] = {
                "available": available
            }

    return availability


def print_metric_availability(availability):
    """Display metric availability in a readable format."""

    print("\n========== METRIC AVAILABILITY ==========")

    for group, metrics in availability.items():

        print(f"\n[{group}]")

        for metric, information in metrics.items():

            if information["available"]:
                status = "AVAILABLE"
            else:
                status = "NOT AVAILABLE"

            print(f"  {metric}: {status}")

    print("\n========== END AVAILABILITY ==========")


if __name__ == "__main__":

    import pandas as pd

    from schema_detector import validate_dataset

    input_file = "data/sample/ragas_eval_30.csv"

    df = pd.read_csv(input_file)

    validation_result = validate_dataset(df)

    availability = build_metric_availability(validation_result)

    print_metric_availability(availability)