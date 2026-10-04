import plotly.express as px
import pandas as pd


def create_metric_bar_chart(averages):
    """
    Create a bar chart showing average evaluation scores.
    """

    if not averages:
        return None

    data = pd.DataFrame(
        {
            "Metric": [
                metric.replace("_", " ").title()
                for metric in averages.keys()
            ],
            "Score": list(averages.values()),
        }
    )

    figure = px.bar(
        data,
        x="Metric",
        y="Score",
        text="Score",
        title="Average Evaluation Scores",
    )

    figure.update_yaxes(
        range=[0, 1],
        title="Score",
    )

    figure.update_xaxes(
        title="Metric",
    )

    figure.update_traces(
        texttemplate="%{text:.3f}",
        textposition="outside",
    )

    return figure


def create_sample_score_chart(results_df):
    """
    Create a line chart showing evaluation scores
    across individual samples.
    """

    score_columns = [
        column
        for column in results_df.columns
        if column != "id"
    ]

    if not score_columns:
        return None

    data = results_df[
        ["id"] + score_columns
    ].melt(
        id_vars="id",
        var_name="Metric",
        value_name="Score",
    )

    data["Metric"] = data["Metric"].str.replace(
        "_",
        " ",
    ).str.title()

    figure = px.line(
        data,
        x="id",
        y="Score",
        color="Metric",
        markers=True,
        title="Evaluation Scores by Sample",
    )

    figure.update_yaxes(
        range=[0, 1],
        title="Score",
    )

    figure.update_xaxes(
        title="Sample ID",
    )

    return figure