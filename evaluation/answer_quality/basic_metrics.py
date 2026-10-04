import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


INPUT_FILE = "data/processed/normalized_ragas_eval.csv"
OUTPUT_FILE = "outputs/results/basic_evaluation_results.csv"


def similarity(text_a, text_b):
    """Calculate TF-IDF cosine similarity between two texts."""

    if not text_a or not text_b:
        return 0.0

    vectorizer = TfidfVectorizer(stop_words="english")

    try:
        vectors = vectorizer.fit_transform([text_a, text_b])
        score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        return float(score)
    except ValueError:
        return 0.0


def clean_text(value):
    """
    Convert normal text or list-like dataset values
    into clean text for evaluation.
    """

    if value is None:
        return ""

    if isinstance(value, float) and pd.isna(value):
        return ""

    # Handle actual Python lists/tuples
    if isinstance(value, (list, tuple)):
        return " ".join(str(item) for item in value)

    # Handle lists stored as strings, e.g.
    # "['answer one', 'answer two']"
    if isinstance(value, str):

        text = value.strip()

        if text.startswith("[") and text.endswith("]"):

            try:
                import ast

                parsed = ast.literal_eval(text)

                if isinstance(parsed, (list, tuple)):
                    return " ".join(
                        str(item) for item in parsed
                    )

            except (ValueError, SyntaxError):
                pass

        return text

    return str(value)


def evaluate_dataset():
    print("========== AITrustEval BASIC EVALUATION ==========")

    df = pd.read_csv(INPUT_FILE)

    results = []

    for _, row in df.iterrows():

        question = clean_text(row["question"])
        context = clean_text(row["context"])
        answer = clean_text(row["answer"])
        ground_truth = clean_text(row["ground_truth"])

        # Question → Answer similarity
        answer_relevancy = similarity(question, answer)

        # Answer → Ground Truth similarity
        answer_correctness = similarity(answer, ground_truth)

        # Context → Answer similarity
        context_answer_similarity = similarity(context, answer)

        results.append({
            "id": row["id"],
            "question": question,
            "answer_relevancy": round(answer_relevancy, 4),
            "answer_correctness": round(answer_correctness, 4),
            "context_answer_similarity": round(
                context_answer_similarity, 4
            )
        })

    results_df = pd.DataFrame(results)

    results_df.to_csv(OUTPUT_FILE, index=False)

    print(f"Evaluated samples: {len(results_df)}")
    print(f"Results saved to: {OUTPUT_FILE}")

    print("\n========== AVERAGE SCORES ==========")

    print(
        f"Answer Relevancy: "
        f"{results_df['answer_relevancy'].mean():.4f}"
    )

    print(
        f"Answer Correctness: "
        f"{results_df['answer_correctness'].mean():.4f}"
    )

    print(
        f"Context-Answer Similarity: "
        f"{results_df['context_answer_similarity'].mean():.4f}"
    )

    print("\n========== FIRST 5 RESULTS ==========")
    print(results_df.head().to_string(index=False))

    print("\n========== EVALUATION COMPLETE ==========")


if __name__ == "__main__":
    evaluate_dataset()