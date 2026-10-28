"""Evaluate the chatbot on a manually labelled test set and tune the threshold.

Run from the project root:
    python evaluate.py

data/test_set.csv has three columns:
    question           - a user-style question
    expected_question  - the FAQ question it should match, or UNKNOWN if the
                         chatbot should refuse (out-of-scope question)
    split              - "dev" (used to CHOOSE the threshold) or "test"
                         (used ONLY to report the final result)

Metrics for each threshold:
    accuracy  = (correct answers + correct refusals) / all questions
    precision = correct answers / answers given        (how trustworthy answers are)
    recall    = correct answers / in-scope questions   (how many we managed to answer)
"""

import sys
from pathlib import Path

import pandas as pd

from src.chatbot import FAQChatbot, FAQDataError

TEST_SET_PATH = Path(__file__).resolve().parent / "data" / "test_set.csv"
THRESHOLDS = [round(0.05 * step, 2) for step in range(1, 15)]  # 0.05 ... 0.70


def predict(bot, questions, threshold):
    """Return the matched FAQ question, or UNKNOWN, for each test question."""
    predictions = []
    for question in questions:
        index, scores = bot.best_match(question)
        if index is None or scores[index] < threshold:
            predictions.append("UNKNOWN")
        else:
            predictions.append(bot.faqs.at[index, "question"])
    return predictions


def same_answer(bot, question_a, question_b):
    """Two FAQ questions count as the same if they share an answer (paraphrases)."""
    if question_a == question_b:
        return True
    if "UNKNOWN" in (question_a, question_b):
        return False
    answers = dict(zip(bot.faqs["question"], bot.faqs["answer"]))
    return answers.get(question_a) == answers.get(question_b)


def score_split(bot, data, threshold):
    predictions = predict(bot, data["question"], threshold)
    expected = list(data["expected_question"])
    correct_answers = answered = in_scope = correct_refusals = 0
    for pred, exp in zip(predictions, expected):
        if exp != "UNKNOWN":
            in_scope += 1
        if pred != "UNKNOWN":
            answered += 1
            if exp != "UNKNOWN" and same_answer(bot, pred, exp):
                correct_answers += 1
        elif exp == "UNKNOWN":
            correct_refusals += 1
    total = len(data)
    return {
        "threshold": threshold,
        "accuracy": (correct_answers + correct_refusals) / total,
        "precision": correct_answers / answered if answered else 0.0,
        "recall": correct_answers / in_scope if in_scope else 0.0,
    }


def main():
    try:
        bot = FAQChatbot()
        data = pd.read_csv(TEST_SET_PATH, dtype=str).fillna("")
    except (FAQDataError, FileNotFoundError, pd.errors.EmptyDataError) as error:
        print(f"Error: {error}")
        sys.exit(1)

    dev = data[data["split"] == "dev"]
    test = data[data["split"] == "test"]
    print(f"{len(dev)} dev questions, {len(test)} test questions\n")

    print("Dev set results (used to choose the threshold):")
    print(f"{'threshold':>9} {'accuracy':>9} {'precision':>10} {'recall':>7}")
    rows = [score_split(bot, dev, t) for t in THRESHOLDS]
    for row in rows:
        print(f"{row['threshold']:>9.2f} {row['accuracy']:>9.2f} "
              f"{row['precision']:>10.2f} {row['recall']:>7.2f}")

    # Pick the highest dev accuracy. Several thresholds often tie, so take the
    # middle of the tied range: it is the most stable choice, not an extreme.
    top_accuracy = max(row["accuracy"] for row in rows)
    tied = [row for row in rows if row["accuracy"] == top_accuracy]
    best = tied[len(tied) // 2]
    print(f"\nBest threshold on dev set: {best['threshold']:.2f} "
          f"({len(tied)} thresholds tied on accuracy)")

    final = score_split(bot, test, best["threshold"])
    print("\nHeld-out test set with that threshold:")
    print(f"  accuracy={final['accuracy']:.2f}  precision={final['precision']:.2f}  "
          f"recall={final['recall']:.2f}")
    print("\nSet DEFAULT_THRESHOLD in src/chatbot.py to the value you choose.")
    print("Note: a small test set gives rough estimates; add more labelled questions.")


if __name__ == "__main__":
    main()
