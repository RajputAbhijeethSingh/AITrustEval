import json
import os
import pandas as pd


OUTPUT_PATH = "data/sample/synthetic_5field_dataset.csv"


DATA = [
    {
        "question": "What is machine learning?",
        "context": [
            "Machine learning is a branch of artificial intelligence that enables systems to learn patterns from data.",
            "Deep learning uses neural networks with multiple layers to learn complex representations.",
            "Supervised learning uses labeled training data to learn a mapping between inputs and outputs.",
            "Data preprocessing includes cleaning, transformation, and normalization of data.",
            "Reinforcement learning learns actions through rewards and penalties."
        ],
        "answer": "Machine learning is a branch of artificial intelligence that enables systems to learn patterns from data.",
        "metadata": {"category": "AI", "difficulty": "easy", "source": "synthetic"},
        "ground_truth": "Machine learning is a branch of artificial intelligence that enables systems to learn patterns from data."
    },

    {
        "question": "What is supervised learning?",
        "context": [
            "Supervised learning uses labeled training data to learn a mapping between inputs and outputs.",
            "Unsupervised learning discovers patterns in data without labeled outputs.",
            "Reinforcement learning learns through interaction with an environment using rewards.",
            "Classification predicts discrete categories from input data.",
            "Regression predicts continuous numerical values."
        ],
        "answer": "Supervised learning uses labeled training data to learn the relationship between inputs and outputs.",
        "metadata": {"category": "Machine Learning", "difficulty": "easy", "source": "synthetic"},
        "ground_truth": "Supervised learning uses labeled training data to learn a mapping between inputs and outputs."
    },

    {
        "question": "What is unsupervised learning?",
        "context": [
            "Unsupervised learning discovers patterns or structures in data without labeled outputs.",
            "Supervised learning requires labeled examples during training.",
            "Clustering groups similar data points together.",
            "Regression predicts continuous numerical values.",
            "Classification assigns inputs to predefined categories."
        ],
        "answer": "Unsupervised learning finds patterns in data without using labeled outputs.",
        "metadata": {"category": "Machine Learning", "difficulty": "easy", "source": "synthetic"},
        "ground_truth": "Unsupervised learning discovers patterns or structures in data without labeled outputs."
    },

    {
        "question": "What is reinforcement learning?",
        "context": [
            "Reinforcement learning trains an agent through interactions with an environment using rewards and penalties.",
            "Supervised learning uses labeled training examples.",
            "An agent selects actions based on the current state.",
            "A reward provides feedback about the quality of an action.",
            "Unsupervised learning does not require labeled outputs."
        ],
        "answer": "Reinforcement learning trains an agent through interaction with an environment using rewards and penalties.",
        "metadata": {"category": "Machine Learning", "difficulty": "easy", "source": "synthetic"},
        "ground_truth": "Reinforcement learning trains an agent through interactions with an environment using rewards and penalties."
    },

    {
        "question": "What is deep learning?",
        "context": [
            "Deep learning uses neural networks with multiple layers to learn complex representations from data.",
            "Machine learning includes methods that learn patterns from data.",
            "Convolutional neural networks are commonly used for image processing.",
            "Neural networks consist of interconnected computational units.",
            "Transfer learning reuses knowledge learned from a previous task."
        ],
        "answer": "Deep learning uses multi-layer neural networks to learn complex representations from data.",
        "metadata": {"category": "Deep Learning", "difficulty": "easy", "source": "synthetic"},
        "ground_truth": "Deep learning uses neural networks with multiple layers to learn complex representations from data."
    },

    {
        "question": "What is a convolutional neural network?",
        "context": [
            "A convolutional neural network is a neural network architecture commonly used for image and visual data.",
            "Pooling reduces the spatial dimensions of feature maps.",
            "Recurrent neural networks are designed for sequential data.",
            "Convolutional layers learn spatial features from input images.",
            "Neural networks contain interconnected computational units."
        ],
        "answer": "A convolutional neural network is a neural network architecture commonly used for image and visual data.",
        "metadata": {"category": "Deep Learning", "difficulty": "medium", "source": "synthetic"},
        "ground_truth": "A convolutional neural network is a neural network architecture commonly used for image and visual data."
    },

    {
        "question": "What is overfitting?",
        "context": [
            "Overfitting occurs when a model learns the training data too closely and performs poorly on unseen data.",
            "Underfitting occurs when a model is too simple to capture important patterns.",
            "Regularization can reduce overfitting by penalizing model complexity.",
            "Cross-validation estimates model performance on unseen data.",
            "Training accuracy measures performance on the training dataset."
        ],
        "answer": "Overfitting occurs when a model performs well on training data but poorly on unseen data.",
        "metadata": {"category": "Machine Learning", "difficulty": "medium", "source": "synthetic"},
        "ground_truth": "Overfitting occurs when a model learns the training data too closely and performs poorly on unseen data."
    },

    {
        "question": "What is underfitting?",
        "context": [
            "Underfitting occurs when a model is too simple to capture important patterns in the data.",
            "Overfitting occurs when a model memorizes training data too closely.",
            "Increasing model complexity can sometimes reduce underfitting.",
            "Regularization is commonly used to control model complexity.",
            "Training data is used to fit machine learning models."
        ],
        "answer": "Underfitting occurs when a model is too simple to capture important patterns.",
        "metadata": {"category": "Machine Learning", "difficulty": "medium", "source": "synthetic"},
        "ground_truth": "Underfitting occurs when a model is too simple to capture important patterns in the data."
    },

    {
        "question": "What is cross-validation?",
        "context": [
            "Cross-validation evaluates a model by repeatedly training and validating it on different subsets of data.",
            "K-fold cross-validation divides data into k subsets.",
            "A validation set can be used to tune model parameters.",
            "Test data should normally be kept separate for final evaluation.",
            "Training data is used to fit the model."
        ],
        "answer": "Cross-validation evaluates a model using different subsets of the available data.",
        "metadata": {"category": "Model Evaluation", "difficulty": "medium", "source": "synthetic"},
        "ground_truth": "Cross-validation evaluates a model by repeatedly training and validating it on different subsets of data."
    },

    {
        "question": "What is precision in classification?",
        "context": [
            "Precision is the proportion of predicted positive cases that are actually positive.",
            "Recall is the proportion of actual positive cases correctly identified.",
            "Accuracy is the proportion of all predictions that are correct.",
            "F1 score combines precision and recall using their harmonic mean.",
            "A confusion matrix summarizes classification predictions."
        ],
        "answer": "Precision is the proportion of predicted positive cases that are actually positive.",
        "metadata": {"category": "Evaluation Metrics", "difficulty": "medium", "source": "synthetic"},
        "ground_truth": "Precision is the proportion of predicted positive cases that are actually positive."
    },

    {
        "question": "What is recall in classification?",
        "context": [
            "Recall is the proportion of actual positive cases that are correctly identified.",
            "Precision measures how many predicted positives are actually positive.",
            "Accuracy measures the fraction of all predictions that are correct.",
            "F1 score combines precision and recall.",
            "A confusion matrix contains true positives and false negatives."
        ],
        "answer": "Recall measures the proportion of actual positive cases correctly identified.",
        "metadata": {"category": "Evaluation Metrics", "difficulty": "medium", "source": "synthetic"},
        "ground_truth": "Recall is the proportion of actual positive cases that are correctly identified."
    },

    {
        "question": "What is the F1 score?",
        "context": [
            "The F1 score is the harmonic mean of precision and recall.",
            "Precision measures the correctness of positive predictions.",
            "Recall measures how many actual positives were detected.",
            "Accuracy measures overall prediction correctness.",
            "The confusion matrix provides counts used to calculate classification metrics."
        ],
        "answer": "The F1 score is the harmonic mean of precision and recall.",
        "metadata": {"category": "Evaluation Metrics", "difficulty": "medium", "source": "synthetic"},
        "ground_truth": "The F1 score is the harmonic mean of precision and recall."
    },

    {
        "question": "What is a confusion matrix?",
        "context": [
            "A confusion matrix summarizes the predicted and actual classes of a classification model.",
            "True positives are positive examples correctly predicted as positive.",
            "False positives are negative examples incorrectly predicted as positive.",
            "Accuracy can be calculated from the values in a confusion matrix.",
            "Precision and recall are also derived from confusion matrix values."
        ],
        "answer": "A confusion matrix summarizes predicted and actual classes in a classification problem.",
        "metadata": {"category": "Evaluation Metrics", "difficulty": "easy", "source": "synthetic"},
        "ground_truth": "A confusion matrix summarizes the predicted and actual classes of a classification model."
    },

    {
        "question": "What is natural language processing?",
        "context": [
            "Natural language processing enables computers to process and understand human language.",
            "Text classification assigns documents to categories.",
            "Tokenization divides text into smaller units such as words or subwords.",
            "Named entity recognition identifies entities such as people and organizations.",
            "Computer vision processes visual information."
        ],
        "answer": "Natural language processing enables computers to process and understand human language.",
        "metadata": {"category": "NLP", "difficulty": "easy", "source": "synthetic"},
        "ground_truth": "Natural language processing enables computers to process and understand human language."
    },

    {
        "question": "What is tokenization?",
        "context": [
            "Tokenization divides text into smaller units called tokens.",
            "Stemming reduces words to their root forms.",
            "Named entity recognition identifies entities in text.",
            "Text embeddings represent text as numerical vectors.",
            "Natural language processing works with human language."
        ],
        "answer": "Tokenization divides text into smaller units called tokens.",
        "metadata": {"category": "NLP", "difficulty": "easy", "source": "synthetic"},
        "ground_truth": "Tokenization divides text into smaller units called tokens."
    },

    {
        "question": "What are word embeddings?",
        "context": [
            "Word embeddings represent words or text as numerical vectors that capture semantic relationships.",
            "Tokenization converts text into tokens.",
            "Cosine similarity can compare embedding vectors.",
            "TF-IDF represents text using weighted word frequencies.",
            "Neural language models learn representations from text."
        ],
        "answer": "Word embeddings represent words as numerical vectors that capture semantic relationships.",
        "metadata": {"category": "NLP", "difficulty": "medium", "source": "synthetic"},
        "ground_truth": "Word embeddings represent words or text as numerical vectors that capture semantic relationships."
    },

    {
        "question": "What is sentiment analysis?",
        "context": [
            "Sentiment analysis determines the emotional or opinion polarity expressed in text.",
            "Text classification can assign text to predefined categories.",
            "Natural language processing processes human language.",
            "Named entity recognition identifies entities.",
            "Topic modeling discovers themes in documents."
        ],
        "answer": "Sentiment analysis determines the emotional or opinion polarity expressed in text.",
        "metadata": {"category": "NLP", "difficulty": "easy", "source": "synthetic"},
        "ground_truth": "Sentiment analysis determines the emotional or opinion polarity expressed in text."
    },

    {
        "question": "What is retrieval augmented generation?",
        "context": [
            "Retrieval augmented generation combines information retrieval with language generation.",
            "A retriever searches a knowledge source for relevant information.",
            "A generator uses retrieved information to produce an answer.",
            "RAG can help reduce unsupported responses by providing relevant context.",
            "Embeddings can be used to retrieve semantically similar documents."
        ],
        "answer": "Retrieval augmented generation combines information retrieval with language generation.",
        "metadata": {"category": "RAG", "difficulty": "medium", "source": "synthetic"},
        "ground_truth": "Retrieval augmented generation combines information retrieval with language generation."
    },

    {
        "question": "What is a vector database?",
        "context": [
            "A vector database stores and searches numerical vector representations of data.",
            "Vector similarity search retrieves items with similar embeddings.",
            "Embeddings represent text or other data as vectors.",
            "Traditional relational databases store structured records in tables.",
            "RAG systems often use vector databases for semantic retrieval."
        ],
        "answer": "A vector database stores and searches numerical vector representations of data.",
        "metadata": {"category": "RAG", "difficulty": "medium", "source": "synthetic"},
        "ground_truth": "A vector database stores and searches numerical vector representations of data."
    },

    {
        "question": "What is semantic search?",
        "context": [
            "Semantic search retrieves information based on meaning rather than exact keyword matching.",
            "Embeddings represent the semantic meaning of text as vectors.",
            "Cosine similarity can measure similarity between embeddings.",
            "Keyword search matches terms appearing in documents.",
            "Vector databases can support semantic search."
        ],
        "answer": "Semantic search retrieves information based on meaning rather than exact keyword matching.",
        "metadata": {"category": "Information Retrieval", "difficulty": "medium", "source": "synthetic"},
        "ground_truth": "Semantic search retrieves information based on meaning rather than exact keyword matching."
    },

    {
        "question": "What is cosine similarity?",
        "context": [
            "Cosine similarity measures the similarity between two vectors based on the cosine of the angle between them.",
            "Embeddings can be compared using cosine similarity.",
            "A value closer to one generally indicates greater directional similarity.",
            "Euclidean distance measures straight-line distance between vectors.",
            "Vector databases can use similarity measures for retrieval."
        ],
        "answer": "Cosine similarity measures similarity between two vectors using the cosine of the angle between them.",
        "metadata": {"category": "Embeddings", "difficulty": "medium", "source": "synthetic"},
        "ground_truth": "Cosine similarity measures the similarity between two vectors based on the cosine of the angle between them."
    },

    {
        "question": "What is hallucination in a language model?",
        "context": [
            "A hallucination occurs when a language model generates information that is unsupported or factually incorrect.",
            "Grounding an answer in retrieved evidence can reduce unsupported responses.",
            "Faithfulness measures whether an answer is supported by its context.",
            "Answer correctness compares an answer with a reference answer.",
            "Retrieval quality affects the evidence available to a language model."
        ],
        "answer": "A hallucination occurs when a language model generates unsupported or factually incorrect information.",
        "metadata": {"category": "LLM Evaluation", "difficulty": "medium", "source": "synthetic"},
        "ground_truth": "A hallucination occurs when a language model generates information that is unsupported or factually incorrect."
    },

    {
        "question": "What is faithfulness in RAG evaluation?",
        "context": [
            "Faithfulness measures whether a generated answer is supported by the retrieved context.",
            "An answer can be relevant to a question but still contain unsupported claims.",
            "RAG systems provide retrieved context to the language model.",
            "Hallucination can occur when generated claims are unsupported.",
            "Ground truth can be used to evaluate answer correctness."
        ],
        "answer": "Faithfulness measures whether a generated answer is supported by the retrieved context.",
        "metadata": {"category": "LLM Evaluation", "difficulty": "medium", "source": "synthetic"},
        "ground_truth": "Faithfulness measures whether a generated answer is supported by the retrieved context."
    },

    {
        "question": "What is answer relevancy?",
        "context": [
            "Answer relevancy measures how directly an answer addresses the user's question.",
            "Answer correctness compares the answer against a reference.",
            "Faithfulness measures support from retrieved context.",
            "A relevant answer should address the information requested by the question.",
            "Retrieval quality influences the information available for generating an answer."
        ],
        "answer": "Answer relevancy measures how directly an answer addresses the user's question.",
        "metadata": {"category": "LLM Evaluation", "difficulty": "medium", "source": "synthetic"},
        "ground_truth": "Answer relevancy measures how directly an answer addresses the user's question."
    },

    {
        "question": "What is prompt injection?",
        "context": [
            "Prompt injection is an attack that attempts to manipulate an AI system through specially crafted input instructions.",
            "Attackers may try to override system instructions using malicious prompts.",
            "Input validation can help reduce some prompt-based attacks.",
            "Jailbreak attacks attempt to bypass model safety restrictions.",
            "Data leakage occurs when sensitive information is exposed."
        ],
        "answer": "Prompt injection is an attack that attempts to manipulate an AI system through crafted input instructions.",
        "metadata": {"category": "AI Security", "difficulty": "medium", "source": "synthetic"},
        "ground_truth": "Prompt injection is an attack that attempts to manipulate an AI system through specially crafted input instructions."
    },

    {
        "question": "What is a jailbreak attack?",
        "context": [
            "A jailbreak attack attempts to bypass the safety or behavioral restrictions of an AI model.",
            "Prompt injection manipulates model instructions through crafted inputs.",
            "Safety policies define behaviors a model should avoid.",
            "Input filtering can detect some malicious prompts.",
            "Data leakage involves unauthorized exposure of sensitive information."
        ],
        "answer": "A jailbreak attack attempts to bypass the safety or behavioral restrictions of an AI model.",
        "metadata": {"category": "AI Security", "difficulty": "medium", "source": "synthetic"},
        "ground_truth": "A jailbreak attack attempts to bypass the safety or behavioral restrictions of an AI model."
    },

    {
        "question": "What is data leakage in AI systems?",
        "context": [
            "Data leakage occurs when sensitive or confidential information is exposed to unauthorized users or systems.",
            "Access controls restrict who can access sensitive information.",
            "Prompt injection can sometimes be used to attempt information disclosure.",
            "Privacy protection reduces the risk of exposing personal information.",
            "Security testing can identify potential information disclosure vulnerabilities."
        ],
        "answer": "Data leakage occurs when sensitive or confidential information is exposed to unauthorized users or systems.",
        "metadata": {"category": "AI Security", "difficulty": "medium", "source": "synthetic"},
        "ground_truth": "Data leakage occurs when sensitive or confidential information is exposed to unauthorized users or systems."
    },

    {
        "question": "What is robustness in machine learning?",
        "context": [
            "Robustness is the ability of a machine learning system to maintain reliable performance under changes, noise, or challenging inputs.",
            "Adversarial examples can intentionally perturb inputs to affect model predictions.",
            "Data augmentation can improve performance under some input variations.",
            "Accuracy measures performance on evaluated examples.",
            "Stress testing can reveal weaknesses under unusual conditions."
        ],
        "answer": "Robustness is the ability of a machine learning system to maintain reliable performance under challenging inputs or changes.",
        "metadata": {"category": "Trustworthy AI", "difficulty": "medium", "source": "synthetic"},
        "ground_truth": "Robustness is the ability of a machine learning system to maintain reliable performance under changes, noise, or challenging inputs."
    },

    {
        "question": "What is reliability in an AI system?",
        "context": [
            "Reliability is the ability of an AI system to perform consistently and correctly under expected operating conditions.",
            "Consistency measures whether similar inputs produce stable results.",
            "Latency measures the time taken to produce a response.",
            "Robustness evaluates behavior under challenging or changed conditions.",
            "Monitoring can help identify system failures."
        ],
        "answer": "Reliability is the ability of an AI system to perform consistently and correctly under expected conditions.",
        "metadata": {"category": "Trustworthy AI", "difficulty": "medium", "source": "synthetic"},
        "ground_truth": "Reliability is the ability of an AI system to perform consistently and correctly under expected operating conditions."
    },

    {
        "question": "What is NDCG in information retrieval?",
        "context": [
            "NDCG measures the quality of a ranked retrieval list using graded relevance and position discounts.",
            "Higher ranked relevant results contribute more strongly to NDCG.",
            "Precision measures the proportion of retrieved results that are relevant.",
            "MRR focuses on the rank of the first relevant result.",
            "Hit rate measures whether at least one relevant result was retrieved."
        ],
        "answer": "NDCG measures the quality of a ranked retrieval list using graded relevance and position discounts.",
        "metadata": {"category": "Information Retrieval", "difficulty": "hard", "source": "synthetic"},
        "ground_truth": "NDCG measures the quality of a ranked retrieval list using graded relevance and position discounts."
    },

   

    
]


def main():
    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True
    )

    rows = []

    for item in DATA:

        rows.append({
            "question": item["question"],

            # JSON string so CSV preserves the ranked
            # list of retrieved context chunks.
            "context": json.dumps(
                item["context"],
                ensure_ascii=False
            ),

            "answer": item["answer"],

            "metadata": json.dumps(
                item["metadata"],
                ensure_ascii=False
            ),

            "ground_truth": item["ground_truth"],
        })

    df = pd.DataFrame(rows)

    df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print("=" * 60)
    print("AITrustEval - Synthetic 5-Field Dataset")
    print("=" * 60)

    print(f"\nOutput:")
    print(f"  {OUTPUT_PATH}")

    print(f"\nRows:")
    print(f"  {len(df)}")

    print("\nColumns:")
    for column in df.columns:
        print(f"  - {column}")

    print("\nContext chunks per row:")
    print("  5 ranked chunks")

    print("\nDataset shape:")
    print(f"  {df.shape}")

    print("\nDataset created successfully.")


if __name__ == "__main__":
    main()