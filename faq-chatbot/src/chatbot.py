"""Retrieval-based FAQ chatbot using TF-IDF and cosine similarity.

How it works:
    1. Load FAQ questions/answers from a CSV file.
    2. Preprocess every stored question and turn it into a TF-IDF vector.
    3. For a user question: preprocess it, turn it into a vector with the SAME
       fitted vectorizer, and measure cosine similarity to every FAQ question.
    4. If the best score reaches the threshold, return that FAQ's answer;
       otherwise return a friendly fallback message.

This is information retrieval, not text generation, and it does not
"understand" language: it only measures word overlap (weighted by TF-IDF).

Run a terminal chat from the project root:
    python -m src.chatbot
"""

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .preprocess import preprocess_text

# Path is built from this file's location, so it works from any machine.
DEFAULT_FAQ_PATH = Path(__file__).resolve().parent.parent / "data" / "faqs.csv"

DEFAULT_THRESHOLD = 0.25   # selected using the development split in evaluate.py
AMBIGUITY_MARGIN = 0.05    # two different answers this close => ask the user

UNKNOWN_MESSAGE = (
    "Sorry, I couldn't find a suitable answer. Please try rephrasing your question."
)
EMPTY_MESSAGE = "Please type a question and I'll try to help."


class FAQDataError(Exception):
    """Raised when the FAQ file is missing, unreadable, or badly formatted."""


@dataclass
class ChatResponse:
    """Everything the chatbot returns for one user question."""

    status: str                      # "answer", "ambiguous", "unknown" or "empty"
    message: str                     # text to show the user
    score: float = 0.0               # similarity of the best FAQ (0 to 1)
    matched_question: str = ""       # best FAQ question (useful for debugging)
    alternatives: list = field(default_factory=list)  # [(question, score), ...]


def load_faqs(csv_path) -> pd.DataFrame:
    """Read and validate the FAQ CSV. Returns a DataFrame with question/answer."""
    path = Path(csv_path)
    if not path.is_file():
        raise FAQDataError(f"FAQ file not found: {path}")
    try:
        # utf-8-sig also reads files saved by Excel, which add an invisible BOM.
        df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    except pd.errors.EmptyDataError:
        raise FAQDataError(f"FAQ file is empty: {path}") from None
    except UnicodeDecodeError:
        raise FAQDataError(
            "FAQ file is not UTF-8. In Excel use 'Save As > CSV UTF-8'."
        ) from None
    except pd.errors.ParserError as error:
        raise FAQDataError(f"FAQ file is badly formatted ({error}).") from error
    except OSError as error:  # e.g. file locked by Excel, no permission
        raise FAQDataError(
            f"FAQ file could not be opened ({error}). Close it in other programs and retry."
        ) from error

    df.columns = [str(column).strip().lower() for column in df.columns]
    missing = {"question", "answer"} - set(df.columns)
    if missing:
        raise FAQDataError(
            f"FAQ file must have 'question' and 'answer' columns; missing: {sorted(missing)}"
        )

    # Keep optional alias metadata while preserving compatibility with two-column CSVs.
    keep_columns = ["question", "answer"] + (["aliases"] if "aliases" in df.columns else [])
    df = df[keep_columns].dropna(subset=["question", "answer"]).copy()
    df["question"] = df["question"].str.strip()
    df["answer"] = df["answer"].str.strip()
    df = df[(df["question"] != "") & (df["answer"] != "")].reset_index(drop=True)
    if df.empty:
        raise FAQDataError("FAQ file has no usable question/answer rows.")
    return df


class FAQChatbot:
    """Loads FAQs once, fits TF-IDF once, then answers many questions."""

    def __init__(self, csv_path=DEFAULT_FAQ_PATH, threshold=DEFAULT_THRESHOLD,
                 ambiguity_margin=AMBIGUITY_MARGIN):
        self.threshold = threshold
        self.ambiguity_margin = ambiguity_margin
        self.faqs = load_faqs(csv_path)
        self._fit()

    def _fit(self):
        """Preprocess stored questions and fit TF-IDF on them (only here)."""
        # Each FAQ can include semicolon-separated paraphrases in an optional
        # CSV column named `aliases`. Train on these variants but keep one result
        # per original FAQ. This improves matching when users phrase questions
        # differently, without duplicating answers in the app.
        self._variant_owner = []
        processed = []
        for faq_index, row in self.faqs.iterrows():
            variants = [row["question"]]
            alias_text = row.get("aliases", "")
            if isinstance(alias_text, str) and alias_text.strip():
                variants.extend(alias.strip() for alias in alias_text.split(";") if alias.strip())
            for variant in variants:
                cleaned = preprocess_text(variant)
                if cleaned:
                    processed.append(cleaned)
                    self._variant_owner.append(faq_index)

        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        try:
            self.variant_matrix = self.vectorizer.fit_transform(processed)
        except ValueError as error:
            raise FAQDataError(
                "FAQ questions contain no usable words after preprocessing."
            ) from error

    def best_match(self, user_question):
        """Return (best_index, scores) without applying the threshold.

        Returns (None, None) if the question has no usable words. Used by
        get_answer() and by evaluate.py (to test many thresholds quickly).
        """
        processed = preprocess_text(user_question)
        if not processed:
            return None, None
        user_vector = self.vectorizer.transform([processed])  # NOT re-fitted
        variant_scores = cosine_similarity(user_vector, self.variant_matrix).ravel()
        # Use the strongest variant score for each original FAQ.
        scores = [0.0] * len(self.faqs)
        for variant_index, score in enumerate(variant_scores):
            faq_index = self._variant_owner[variant_index]
            if score > scores[faq_index]:
                scores[faq_index] = float(score)
        scores = __import__("numpy").asarray(scores)
        return int(scores.argmax()), scores

    def get_answer(self, user_question) -> ChatResponse:
        """Answer one user question."""
        if not isinstance(user_question, str) or not user_question.strip():
            return ChatResponse(status="empty", message=EMPTY_MESSAGE)

        best_index, scores = self.best_match(user_question)
        if best_index is None:  # e.g. only punctuation or only stop words
            return ChatResponse(status="unknown", message=UNKNOWN_MESSAGE)

        # Stable descending sort: on equal scores the earlier FAQ comes first,
        # and ranked[0] is always the best match used below.
        ranked = (-scores).argsort(kind="stable")
        best_index = int(ranked[0])
        best_score = float(scores[best_index])
        best_question = self.faqs.at[best_index, "question"]
        alternatives = [(self.faqs.at[i, "question"], float(scores[i])) for i in ranked[:3]]

        if best_score < self.threshold:
            return ChatResponse("unknown", UNKNOWN_MESSAGE, best_score,
                                best_question, alternatives)

        best_answer = self.faqs.at[best_index, "answer"]

        # Ambiguity: a runner-up with a DIFFERENT answer and a nearly equal score.
        for other in ranked[1:4]:
            other_score = float(scores[other])
            if other_score < self.threshold:
                break
            if (best_score - other_score) <= self.ambiguity_margin and \
                    self.faqs.at[other, "answer"] != best_answer:
                other_question = self.faqs.at[other, "question"]
                message = (
                    "Your question matches more than one FAQ. Did you mean:\n\n"
                    f"1. {best_question}\n"
                    f"2. {other_question}\n\n"
                    "Please ask again using one of these questions."
                )
                return ChatResponse("ambiguous", message, best_score,
                                    best_question, alternatives)

        return ChatResponse("answer", best_answer, best_score, best_question, alternatives)


def main():
    """Simple terminal chat for quick testing."""
    try:
        bot = FAQChatbot()
    except FAQDataError as error:
        print(f"Error: {error}")
        return
    print("FAQ Chatbot (type 'quit' to exit)")
    while True:
        try:
            question = input("You: ")
        except (EOFError, KeyboardInterrupt):
            break
        if question.strip().lower() in {"quit", "exit"}:
            break
        response = bot.get_answer(question)
        print(f"Bot: {response.message}")
        print(f"     [score={response.score:.2f}, best match: {response.matched_question}]\n")


if __name__ == "__main__":
    main()
