# Concepts and Viva Guide

## 1. What is what? (be precise in your viva)

| Part of this project | Is it machine learning? |
|---|---|
| Lowercasing, punctuation cleaning, tokenizing, stop words | No. Rule-based text preprocessing. |
| Stemming | No. Rule-based (the Porter algorithm). |
| TF-IDF | Not learning in the neural-network sense. It is a statistical weighting. `fit` learns the vocabulary and word weights (IDF) from the FAQ data, so it is a very simple "learned from data" step. |
| Cosine similarity + threshold | No. A formula plus a hand-tuned number. This is information retrieval. |

Do **not** say this chatbot trains a neural network, "understands" language,
or is a generative AI. It finds the stored question with the most similar words.

## 2. Concepts with small examples

**AI / ML.** AI is the broad goal of making machines do tasks that need
intelligence. ML is the part where a program learns patterns from data instead
of following only hand-written rules.

**NLP.** Getting computers to work with human language. Here it turns free text
into numbers so questions can be compared.

**Retrieval-based chatbot.** It picks an answer from a fixed set of stored
answers. A generative chatbot writes new text. Retrieval is predictable and
cannot invent facts, which suits FAQs.

**Why CSV.** The data is small, tabular, and easy for non-programmers to edit.
A database would be unnecessary here.

**Tokenization and normalization.** Tokenization splits text into words:
`"pay my fees"` -> `["pay","my","fees"]`. Normalization makes different
spellings of the same thing equal: `"FEES?"` -> `"fees"`.

**Stop words.** Very common words ("the", "is", "how") add little meaning.
Limitation: removing them can destroy meaning, e.g. "not". That is why this
project keeps negations. Question words like "how" can also matter in other
projects.

**Stemming.** Cuts words to a root so `fees` and `fee` match. Limitation:
crude, may give non-words (`admission` -> `admiss`) or miss related words
(`lost` vs `lose`).

**TF-IDF.** TF (term frequency) = how often a word appears in a question.
IDF (inverse document frequency) = how rare the word is across all FAQs. Words
that appear in nearly every FAQ ("college") get a low weight; words that
appear in few ("hostel") get a high weight. So a match on "hostel" counts more
than a match on "college".

**Vector.** A list of numbers, one per vocabulary word. If the vocabulary is
[hostel, fee, cancel], then "hostel fee" is about [1, 1, 0] (before weighting).

**Cosine similarity.** The cosine of the angle between two vectors: 1 = same
direction (same words), 0 = no shared words.
Example: A = [1,1,0] and B = [1,0,1]. Dot product = 1. Lengths = sqrt(2) each.
Cosine = 1 / (sqrt(2) * sqrt(2)) = 0.5. It measures word-overlap similarity,
not true meaning.

**Threshold and fallback.** The best FAQ always has *some* score, even for
nonsense. The threshold (0.30 to start) says "below this, don't answer". It must
be tested: too low gives wrong answers, too high gives needless refusals.

**Training vs test data.** Here the FAQ questions are the "training" data (the
vectorizer is fitted on them). The test set is separate, hand-labelled
questions the bot has never seen. Tune the threshold on the dev part, report
results on the test part.

**Accuracy, precision, recall (as used in evaluate.py).**
- Accuracy: share of all test questions handled correctly (right answer, or a correct refusal).
- Precision: of the answers the bot gave, how many were right. Low precision = confident wrong answers.
- Recall: of the answerable questions, how many it answered correctly. Low recall = too many refusals.
Raising the threshold usually raises precision and lowers recall.

**TF-IDF vs embeddings/transformers.** TF-IDF only sees exact (stemmed) words,
so "bus" and "transport" are unrelated. Embedding models map text to vectors
where similar *meanings* are close, so paraphrases and synonyms match. They need
a larger model but give better results on varied wording.

## 3. Likely viva / interview questions

1. **Why TF-IDF instead of plain word counts?** It down-weights common words and up-weights distinctive ones, so matches on important words count more.
2. **Why fit the vectorizer only on the FAQ data?** The vocabulary and IDF values must be the same for every comparison. Re-fitting per query would give a meaningless vocabulary and incompatible vectors.
3. **Why cosine similarity and not Euclidean distance?** Cosine ignores text length and compares direction (word mix), which suits short questions of different lengths.
4. **How does the bot handle an unrelated question?** If the best similarity is below the threshold it returns a fallback message instead of an answer.
5. **How did you choose the threshold?** Tried values on a labelled dev set, picked the best accuracy, then checked it on a held-out test set.
6. **Why keep negations?** "Can I..." and "I cannot..." have opposite meanings; removing "not" would make them look the same.
7. **Is this machine learning?** Only lightly: TF-IDF learns vocabulary and word weights from data. Matching is similarity-based retrieval, not a trained classifier or neural network.
8. **What are the limitations?** No understanding of synonyms or meaning, sensitive to wording, English only, small test set.
9. **How would you improve it?** Sentence embeddings, synonym expansion, intent classification, logging unanswered questions, larger evaluation set.
10. **What does the ambiguity check do?** If two FAQs with different answers have nearly equal scores, the bot asks the user which one they meant instead of guessing.
11. **How do you add new FAQs?** Add rows to `data/faqs.csv` and restart. No code changes.
12. **Why not an LLM API?** The goal is to learn NLP fundamentals; retrieval is cheap, offline, predictable, and cannot hallucinate.
