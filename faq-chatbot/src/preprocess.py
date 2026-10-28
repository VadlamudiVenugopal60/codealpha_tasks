"""Text preprocessing for the FAQ chatbot.

Pipeline (each step is a small function so you can test and explain it):

    raw text -> clean_text -> tokenize -> remove_stop_words -> stem_tokens -> "joined string"

Example:
    "How do I PAY my tuition fees?!"
    clean_text        -> "how do i pay my tuition fees"
    tokenize          -> ["how", "do", "i", "pay", "my", "tuition", "fees"]
    remove_stop_words -> ["pay", "tuition", "fees"]
    stem_tokens       -> ["pai", "tuition", "fee"]
    final string      -> "pai tuition fee"

The same function is used for the stored FAQ questions and for the user's
question. Using identical processing on both sides is what makes the
comparison fair.
"""

import re
from functools import lru_cache

from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer

# Negations change meaning ("can" vs "cannot"), so they are NEVER removed.
NEGATIONS = {
    "no", "not", "nor", "never", "none", "neither", "without", "cannot",
    "cant", "dont", "doesnt", "didnt", "wont", "isnt", "arent", "wasnt",
    "werent", "shouldnt", "couldnt", "wouldnt", "havent", "hasnt", "hadnt",
}

# A small hand-written stop-word list. We use our own list instead of NLTK's
# stopwords corpus so the project needs no extra downloads, and so we control
# exactly which words are dropped.
_COMMON_WORDS = {
    "a", "an", "the", "and", "or", "but", "if", "of", "at", "by", "for",
    "with", "about", "to", "from", "in", "on", "up", "out", "as", "is",
    "are", "was", "were", "be", "been", "being", "am", "do", "does", "did",
    "doing", "have", "has", "had", "having", "i", "me", "my", "myself",
    "we", "our", "you", "your", "he", "him", "his", "she", "her", "it",
    "its", "they", "them", "their", "this", "that", "these", "those",
    "what", "which", "who", "whom", "when", "where", "why", "how", "will",
    "would", "can", "could", "should", "may", "might", "shall", "so",
    "than", "too", "very", "s", "t", "just", "there", "any", "some",
    "please", "tell", "us", "get", "want", "know",
}
STOP_WORDS = _COMMON_WORDS - NEGATIONS

# Keep only letters and digits. NLTK's RegexpTokenizer does the splitting.
_tokenizer = RegexpTokenizer(r"[a-z0-9]+")
_stemmer = PorterStemmer()


def clean_text(text: str) -> str:
    """Lowercase, drop apostrophes, replace other punctuation, squeeze spaces.

    "Don't know?!  Where's the   library..." -> "dont know wheres the library"
    """
    text = text.lower()
    text = text.replace("'", "").replace("’", "")  # don't -> dont
    text = re.sub(r"[^a-z0-9\s]", " ", text)             # punctuation -> space
    text = re.sub(r"\s+", " ", text).strip()             # collapse whitespace
    return text


def tokenize(text: str) -> list[str]:
    """Split cleaned text into word tokens using NLTK."""
    return _tokenizer.tokenize(text)


def remove_stop_words(tokens: list[str]) -> list[str]:
    """Drop very common words that carry little meaning (negations are kept)."""
    return [token for token in tokens if token not in STOP_WORDS]


@lru_cache(maxsize=None)
def _stem(word: str) -> str:
    return _stemmer.stem(word)


def stem_tokens(tokens: list[str]) -> list[str]:
    """Reduce words to a rough root: 'fees' -> 'fee', 'payment' -> 'payment'.

    Stemming is crude (it can produce non-words such as 'admiss'), but it
    helps different forms of a word match each other.
    """
    return [_stem(token) for token in tokens]


def preprocess_text(text, use_stop_words: bool = True, use_stemming: bool = True) -> str:
    """Run the whole pipeline and return a single space-separated string.

    Non-string or empty input returns "".
    """
    if not isinstance(text, str):
        return ""
    tokens = tokenize(clean_text(text))
    if use_stop_words:
        tokens = remove_stop_words(tokens)
    if use_stemming:
        tokens = stem_tokens(tokens)
    return " ".join(tokens)


if __name__ == "__main__":
    # Quick demo:  python -m src.preprocess
    samples = [
        "How do I PAY my tuition fees?!",
        "Can't I apply   after the deadline???",
        "Where's the library?",
    ]
    for sample in samples:
        cleaned = clean_text(sample)
        tokens = tokenize(cleaned)
        no_stop = remove_stop_words(tokens)
        stemmed = stem_tokens(no_stop)
        print(f"original   : {sample}")
        print(f"cleaned    : {cleaned}")
        print(f"tokens     : {tokens}")
        print(f"no stopwds : {no_stop}")
        print(f"stemmed    : {stemmed}")
        print()
