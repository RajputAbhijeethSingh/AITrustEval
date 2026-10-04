import pandas as pd

from evaluation.dataset.schema_detector import validate_dataset
from evaluation.dataset.metric_availability import build_metric_availability
from evaluation.results.result_schema import create_evaluation_result
from evaluation.answer_quality.basic_metrics import similarity, clean_text


INPUT_FILE = "data/sample/ragas_eval_30.csv"
OUTPUT_FILE = "outputs/results/unified_evaluation_results.csv"


def run_evaluation(df):
    """
    Main AITrustEval evaluation pipeline.

    Returns a standardized AITrustEval result.
    """

    # --------------------------------------------------
    # 1. Validate dataset
    # --------------------------------------------------

    validation = validate_dataset(df)

    if not validation["valid"]:
        raise ValueError(
            "Dataset validation failed: "
            + "; ".join(validation["errors"])
        )

    # --------------------------------------------------
    # 2. Determine metric availability
    # --------------------------------------------------

    availability = build_metric_availability(validation)

    columns = validation["detected_columns"]

    # --------------------------------------------------
    # 3. Run applicable baseline metrics
    # --------------------------------------------------

    sample_results = []

    for index, row in df.iterrows():

        question = clean_text(
            row[columns["question"]]
        )

        answer = clean_text(
            row[columns["answer"]]
        )

        ground_truth = ""

        if "ground_truth" in columns:
            ground_truth = clean_text(
                row[columns["ground_truth"]]
            )

        context = ""

        if "context" in columns:
            context = clean_text(
                row[columns["context"]]
            )

        result = {
            "id": index + 1
        }

        # Answer Relevancy
        if availability["Answer Quality"][
            "answer_relevancy"
        ]["available"]:

            result["answer_relevancy"] = round(
                similarity(
                    question,
                    answer
                ),
                4,
            )

        # Answer Correctness
        if availability["Answer Quality"][
            "answer_correctness"
        ]["available"]:

            result["answer_correctness"] = round(
                similarity(
                    answer,
                    ground_truth
                ),
                4,
            )

        # Context-Answer Similarity
        #
        # This is a BASELINE metric.
        # It is not the official Ragas Faithfulness metric.
        if availability["Answer Quality"][
            "faithfulness"
        ]["available"]:

            result["context_answer_similarity"] = round(
                similarity(
                    context,
                    answer
                ),
                4,
            )

        sample_results.append(result)

    results_df = pd.DataFrame(
        sample_results
    )

    # --------------------------------------------------
    # 4. Save sample-level results
    # --------------------------------------------------

    results_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # --------------------------------------------------
    # 5. Calculate aggregate metrics
    # --------------------------------------------------

    averages = {}

    for column in results_df.columns:

        if column == "id":
            continue

        averages[column] = round(
            float(results_df[column].mean()),
            4,
        )

    # --------------------------------------------------
    # 6. Dataset information
    # --------------------------------------------------

    dataset_info = {
        "rows": validation["rows"],
        "columns": len(validation["columns"]),
        "column_names": validation["columns"],
    }

    # --------------------------------------------------
    # 7. Create standardized result
    # --------------------------------------------------

    standardized_result = create_evaluation_result(
        dataset_info=dataset_info,
        detected_columns=validation[
            "detected_columns"
        ],
        metric_availability=availability,
        metric_results=averages,
        failures=[],
        warnings=validation["warnings"],
    )

    # Add sample-level results separately.
    standardized_result[
        "sample_results"
    ] = results_df

    return standardized_result


if __name__ == "__main__":

    print(
        "========== AITrustEval EVALUATION ENGINE =========="
    )

    # --------------------------------------------------
    # Load dataset
    # --------------------------------------------------

    df = pd.read_csv(
        INPUT_FILE
    )

    # --------------------------------------------------
    # Run evaluation
    # --------------------------------------------------

    evaluation = run_evaluation(df)

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    print("\nDataset:")

    print(
        f"Rows: "
        f"{evaluation['dataset']['rows']}"
    )

    print(
        f"Columns: "
        f"{evaluation['dataset']['columns']}"
    )

    # --------------------------------------------------
    # Schema
    # --------------------------------------------------

    print("\nDetected Schema:")

    for field, column in evaluation[
        "schema"
    ].items():

        print(
            f"  {field} -> {column}"
        )

    # --------------------------------------------------
    # Metric Availability
    # --------------------------------------------------

    print("\nMetric Availability:")

    for group, metrics in evaluation[
        "metric_availability"
    ].items():

        print(
            f"\n[{group}]"
        )

        for metric, information in metrics.items():

            status = (
                "AVAILABLE"
                if information["available"]
                else "NOT AVAILABLE"
            )

            print(
                f"  {metric}: {status}"
            )

    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    print(
        "\n========== BASELINE RESULTS =========="
    )

    for metric, score in evaluation[
        "metrics"
    ].items():

        print(
            f"{metric}: {score:.4f}"
        )

    # --------------------------------------------------
    # Sample Results
    # --------------------------------------------------

    print(
        "\n========== FIRST 5 SAMPLE RESULTS =========="
    )

    print(
        evaluation[
            "sample_results"
        ]
        .head()
        .to_string(index=False)
    )

    print(
        f"\nResults saved to: "
        f"{OUTPUT_FILE}"
    )

    print(
        "\n========== EVALUATION COMPLETE =========="
    )