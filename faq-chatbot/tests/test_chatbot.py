"""Tests for the FAQ chatbot. Run from the project root with:

    python -m unittest discover -s tests -t . -v

(or `pytest` if you have installed it).
"""

import tempfile
import unittest
from pathlib import Path

from src.chatbot import EMPTY_MESSAGE, UNKNOWN_MESSAGE, FAQChatbot, FAQDataError
from src.preprocess import clean_text, preprocess_text, remove_stop_words, tokenize


class PreprocessTests(unittest.TestCase):
    def test_clean_text_lowercases_and_removes_punctuation(self):
        self.assertEqual(clean_text("  How DO I   pay?!  "), "how do i pay")

    def test_apostrophes_are_joined(self):
        self.assertEqual(clean_text("Don't"), "dont")

    def test_negations_are_kept(self):
        tokens = remove_stop_words(tokenize("i cannot pay and i dont know"))
        self.assertIn("cannot", tokens)
        self.assertIn("dont", tokens)

    def test_non_string_input_gives_empty_string(self):
        self.assertEqual(preprocess_text(None), "")
        self.assertEqual(preprocess_text(123), "")


class MatchingTests(unittest.TestCase):
    """Uses the real dataset in data/faqs.csv."""

    @classmethod
    def setUpClass(cls):
        cls.bot = FAQChatbot()

    def _answer_for(self, question):
        return self.bot.faqs.loc[self.bot.faqs["question"] == question, "answer"].iloc[0]

    def test_exact_match(self):
        response = self.bot.get_answer("How do I apply for admission?")
        self.assertEqual(response.status, "answer")
        self.assertEqual(response.message, self._answer_for("How do I apply for admission?"))
        self.assertGreater(response.score, 0.9)

    def test_paraphrase(self):
        response = self.bot.get_answer("what is the process to apply for admission")
        self.assertEqual(response.matched_question, "How do I apply for admission?")

    def test_punctuation_and_case_do_not_matter(self):
        plain = self.bot.get_answer("what are the library timings")
        noisy = self.bot.get_answer("WHAT ARE THE LIBRARY TIMINGS???!!!")
        self.assertEqual(plain.matched_question, noisy.matched_question)

    def test_empty_questions(self):
        for text in ["", "   ", None, "?!?!"]:
            response = self.bot.get_answer(text)
            self.assertIn(response.status, {"empty", "unknown"})
        self.assertEqual(self.bot.get_answer("").message, EMPTY_MESSAGE)

    def test_unrelated_question_gets_fallback(self):
        response = self.bot.get_answer("What is the capital of France?")
        self.assertEqual(response.status, "unknown")
        self.assertEqual(response.message, UNKNOWN_MESSAGE)

    def test_similar_wording_different_meaning(self):
        apply_hostel = self.bot.get_answer("How do I apply for hostel accommodation?")
        cancel_hostel = self.bot.get_answer("How do I cancel my hostel allotment?")
        self.assertNotEqual(apply_hostel.message, cancel_hostel.message)
        self.assertEqual(apply_hostel.matched_question, "How do I apply for hostel accommodation?")
        self.assertEqual(cancel_hostel.matched_question, "How do I cancel my hostel allotment?")

    def test_threshold_is_configurable(self):
        # Cosine similarity never exceeds 1, so a threshold above 1 rejects
        # everything, while the default threshold accepts an exact match.
        question = "How do I apply for admission?"
        self.assertEqual(FAQChatbot(threshold=1.01).get_answer(question).status, "unknown")
        self.assertEqual(self.bot.get_answer(question).status, "answer")


class DataErrorTests(unittest.TestCase):
    def _write(self, directory, name, text):
        path = Path(directory) / name
        path.write_text(text, encoding="utf-8")
        return path

    def test_missing_file(self):
        with self.assertRaises(FAQDataError):
            FAQChatbot(csv_path="data/does_not_exist.csv")

    def test_empty_file(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FAQDataError):
                FAQChatbot(csv_path=self._write(folder, "empty.csv", ""))

    def test_missing_columns(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self._write(folder, "bad.csv", "q,a\nhello,world\n")
            with self.assertRaises(FAQDataError):
                FAQChatbot(csv_path=path)

    def test_no_usable_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self._write(folder, "blank.csv", "question,answer\n,\n")
            with self.assertRaises(FAQDataError):
                FAQChatbot(csv_path=path)

    def test_excel_style_bom_file_is_read(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bom.csv"
            path.write_bytes("question,answer\nWhere is the shop?,Main street.\n".encode("utf-8-sig"))
            bot = FAQChatbot(csv_path=path, threshold=0.2)
            self.assertEqual(bot.get_answer("where is the shop").message, "Main street.")

    def test_non_utf8_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "latin.csv"
            path.write_bytes("question,answer\ncaf\xe9 hours?,9 to 5\n".encode("cp1252"))
            with self.assertRaises(FAQDataError):
                FAQChatbot(csv_path=path)

    def test_directory_instead_of_file(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FAQDataError):
                FAQChatbot(csv_path=folder)

    def test_ambiguous_question_asks_user(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self._write(
                folder, "amb.csv",
                "question,answer\nHostel fee payment,Pay at accounts.\n"
                "Hostel fee refund,Ask the warden.\nLibrary timings,9 to 5.\n",
            )
            bot = FAQChatbot(csv_path=path, threshold=0.2)
            response = bot.get_answer("hostel fee")
            self.assertEqual(response.status, "ambiguous")
            self.assertIn("Hostel fee payment", response.message)
            self.assertIn("Hostel fee refund", response.message)

    def test_small_custom_dataset_works(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self._write(
                folder, "tiny.csv",
                "question,answer\nWhat are your opening hours?,We open at 9.\n"
                "Where is the shop?,Main street.\n",
            )
            bot = FAQChatbot(csv_path=path, threshold=0.2)
            self.assertEqual(bot.get_answer("opening hours").message, "We open at 9.")


if __name__ == "__main__":
    unittest.main()
