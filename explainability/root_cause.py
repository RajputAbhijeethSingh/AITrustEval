import pandas as pd


def determine_root_causes(row):
    """
    Determine likely root causes from failed evaluation metrics.

    This is a rule-based preliminary analysis.
    It does not claim to establish the actual cause.
    """

    causes = []
    recommendations = []

    failed_metrics = str(
        row.get("failed_metrics", "")
    )

    failure_count = int(
        row.get("failure_count", 0)
    )

    # --------------------------------------------------
    # Answer Relevancy
    # --------------------------------------------------

    if "answer_relevancy" in failed_metrics:

        causes.append(
            "Generated answer may not directly address "
            "the user's question."
        )

        recommendations.append(
            "Review the generated answer for question alignment."
        )

    # --------------------------------------------------
    # Answer Correctness
    # --------------------------------------------------

    if "answer_correctness" in failed_metrics:

        causes.append(
            "Generated answer may differ from the "
            "reference answer."
        )

        recommendations.append(
            "Compare the generated answer with the reference answer."
        )

    # --------------------------------------------------
    # Context Alignment
    # --------------------------------------------------

    if "context_answer_similarity" in failed_metrics:

        causes.append(
            "Generated answer may have weak alignment "
            "with the available context."
        )

        recommendations.append(
            "Check whether the retrieved context contains "
            "the information required to answer the question."
        )

    # --------------------------------------------------
    # Multiple failures
    # --------------------------------------------------

    if failure_count >= 3:

        causes.append(
            "Multiple quality indicators failed simultaneously."
        )

        recommendations.append(
            "Perform detailed sample-level investigation "
            "before drawing conclusions."
        )

    elif failure_count >= 2:

        causes.append(
            "More than one quality indicator requires investigation."
        )

    # --------------------------------------------------
    # No failure
    # --------------------------------------------------

    if not causes:

        causes.append(
            "No rule-based failure cause identified."
        )

    if not recommendations:

        recommendations.append(
            "No immediate investigation recommendation."
        )

    return {
        "root_cause": " ".join(causes),
        "recommendation": " ".join(
            recommendations
        ),
    }


def generate_root_cause_report(failure_df):
    """
    Add root-cause and recommendation fields
    to the failure analysis dataframe.
    """

    if failure_df.empty:
        return failure_df.copy()

    output = failure_df.copy()

    root_causes = []
    recommendations = []

    for _, row in output.iterrows():

        analysis = determine_root_causes(row)

        root_causes.append(
            analysis["root_cause"]
        )

        recommendations.append(
            analysis["recommendation"]
        )

    output["root_cause"] = root_causes
    output["recommendation"] = recommendations

    return output


if __name__ == "__main__":

    print(
        "========== ROOT CAUSE ANALYSIS TEST =========="
    )

    input_file = (
        "outputs/results/"
        "unified_evaluation_results.csv"
    )

    df = pd.read_csv(input_file)

    # Import failure analysis
    from explainability.failure_analysis import (
        analyze_failures,
    )

    failures = analyze_failures(df)

    report = generate_root_cause_report(
        failures
    )

    print(
        f"\nSamples requiring investigation: "
        f"{len(report)}"
    )

    print(
        "\n========== ROOT CAUSE FINDINGS =========="
    )

    if report.empty:

        print(
            "No failures identified."
        )

    else:

        # Show a compact version first
        display_columns = [
            "id",
            "failed_metrics",
            "severity",
            "root_cause",
            "recommendation",
        ]

        print(
            report[
                display_columns
            ].to_string(index=False)
        )

    print(
        "\n========== TEST COMPLETE =========="
    )