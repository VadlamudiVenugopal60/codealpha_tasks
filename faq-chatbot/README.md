# FAQ Chatbot Using NLP (TF-IDF + Cosine Similarity)

A retrieval-based FAQ chatbot. You type a question in natural language; the bot
preprocesses it, compares it with a CSV of FAQs, and returns the answer of the
most similar FAQ. If nothing is similar enough, it says so instead of guessing.

It is **not** a generative AI chatbot: no LLM, no API keys, no paid services.
It measures word overlap; it does not understand meaning.

## Project structure

```
faq-chatbot/
├── data/
│   ├── faqs.csv          # 62 sample college FAQs (question, answer)
│   └── test_set.csv      # labelled questions for evaluation (dev/test split)
├── src/
│   ├── __init__.py
│   ├── preprocess.py     # clean, tokenize (NLTK), stop words, stemming (NLTK)
│   └── chatbot.py        # CSV loading, TF-IDF, cosine similarity, threshold
├── tests/
│   ├── __init__.py
│   └── test_chatbot.py   # 16 unit tests
├── docs/
│   └── CONCEPTS_AND_VIVA.md   # concepts explained, viva/interview Q&A
├── app.py                # Streamlit chat interface
├── evaluate.py           # accuracy / precision / recall + threshold tuning
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup (Windows PowerShell, VS Code)

Open the `faq-chatbot` folder in VS Code (File > Open Folder), open a terminal
(Ctrl + `), then:

```powershell
python --version                      # needs Python 3.9 or newer
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope Process Bypass
pip install -r requirements.txt
```

In VS Code choose the interpreter: Ctrl+Shift+P > "Python: Select Interpreter" > `.venv`.

## Run

Always run commands from the project root (the `faq-chatbot` folder).

```powershell
python -m src.preprocess              # demo of each preprocessing step
python -m src.chatbot                 # chat in the terminal
streamlit run app.py                  # web interface (opens in your browser)
python -m unittest discover -s tests -t . -v   # run the tests
python evaluate.py                    # accuracy, precision, recall, threshold tuning
```

## How it works

```
 faqs.csv ──► preprocess each question ──► TfidfVectorizer.fit_transform ──► FAQ matrix
                                                         │ (vectorizer is fitted ONCE)
 user question ──► same preprocessing ──► vectorizer.transform ──► user vector
                                                         │
                         cosine similarity (user vector vs every FAQ vector)
                                                         │
                  best score >= threshold?  ── yes ──► return that FAQ's answer
                                            └─ no  ──► "Sorry, I couldn't find..."
```

Also handled: empty input, bad or missing CSV (clear error message), and
ambiguity (two FAQs with *different* answers and nearly equal scores: the bot
asks "Did you mean...?").

Preprocessing steps (see `src/preprocess.py`):

| Step | Before | After |
|---|---|---|
| Lowercase + clean punctuation | `How do I PAY my fees?!` | `how do i pay my fees` |
| Tokenize (NLTK) | `how do i pay my fees` | `['how','do','i','pay','my','fees']` |
| Remove stop words (negations kept) | that list | `['pay','fees']` |
| Stem (NLTK Porter) | `['pay','fees']` | `['pai','fee']` |

## Using your own FAQs

1. Open `data/faqs.csv` in Excel or VS Code.
2. Keep the header `question,answer`. One FAQ per row.
3. Add several rows with different wordings of the same question (same answer).
4. Save as UTF-8 CSV, then restart the app. No Python changes needed.
5. Update the welcome text and example questions in `app.py`.
6. Make a new `data/test_set.csv` for your topic and re-run `evaluate.py`.

The sample answers are generic placeholders. Replace them with your
institution's real details.

## Threshold tuning

`DEFAULT_THRESHOLD = 0.30` in `src/chatbot.py` is only a starting guess.
Too low: the bot answers unrelated questions. Too high: it refuses valid
paraphrases. `evaluate.py` picks a threshold using the `dev` rows and reports
the result on the untouched `test` rows. Never tune and report on the same data.
The Streamlit sidebar also has a threshold slider for experimenting.

## Known limitations

- TF-IDF matches words, not meaning. "Does the campus have wifi?" can miss
  "Is there Wi-Fi on campus?", and "transport" does not match "bus".
- Short questions can match the wrong FAQ when they share common words.
- The bundled test set is small (44 questions), so accuracy figures are rough.
- Only English is supported.

## Future improvements

Sentence embeddings (e.g. sentence-transformers) for meaning-based matching;
synonym expansion; intent classification; multilingual support; logging unknown
questions to find missing FAQs; retrieval metrics such as top-3 accuracy.

## Resume description (edit to match what you actually did)

**FAQ Chatbot using NLP** | Python, NLTK, scikit-learn, Pandas, Streamlit
- Built a retrieval-based FAQ chatbot that matches user questions to 60+ stored FAQs using TF-IDF vectors and cosine similarity.
- Implemented text preprocessing (normalisation, NLTK tokenisation, stop-word removal with negation handling, stemming).
- Added a configurable similarity threshold, fallback replies for unknown questions, and ambiguity handling.
- Evaluated on a manually labelled dev/test split (accuracy, precision, recall) and wrote unit tests; built a Streamlit chat UI.

## Short explanation for college evaluation

"This project is an FAQ chatbot based on Natural Language Processing and
information retrieval. FAQ questions are cleaned and converted into TF-IDF
vectors. A user's question goes through the same steps, and cosine similarity
finds the closest stored question. Its answer is returned if the similarity
passes a threshold; otherwise the bot asks the user to rephrase. It is
evaluated on a labelled test set and has a Streamlit interface."

See `docs/CONCEPTS_AND_VIVA.md` for concept explanations and viva questions.

## Accuracy improvements

- Added an optional `aliases` column to `data/faqs.csv`. Put alternate phrasings for an FAQ in this column, separated by semicolons (`;`). The chatbot compares a user question against both the canonical question and its aliases, then returns the answer for the best-scoring FAQ.
- The chatbot remains compatible with CSV files that contain only `question` and `answer`.
- TF-IDF uses `sublinear_tf=True` and the default similarity threshold is `0.25`, selected using the development split in `evaluate.py`.
- On the included 22-question held-out test split, the evaluation script reports accuracy 0.82, precision 0.76, and recall 0.81. The original baseline was 0.68, 0.59, and 0.62 respectively. These are results on a small, hand-labelled test set, not a guarantee of real-world performance; add more unseen test questions for a more reliable estimate.

To add aliases, edit the `aliases` cell for a row, for example:

```csv
question,answer,aliases
How do I pay the tuition fees?,"Tuition fees can be paid online...","Tuition payment methods;Ways to pay college fees;How can I make a fee payment"
```
