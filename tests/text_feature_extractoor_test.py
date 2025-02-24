import unittest
from src.services.text_feature_extractor import TextQueryProcessor

class TestTextQueryProcessor(unittest.TestCase):

    def test_remove_punctuation(self):
        text = "Hello, world!"
        result = TextQueryProcessor.remove_punctuation(text)
        self.assertEqual(result, "Hello world")

    def test_remove_whitespace(self):
        text = "  Hello   world  "
        result = TextQueryProcessor.remove_whitespace(text)
        self.assertEqual(result, "Hello world")

    def test_remove_stopwords(self):
        text = "this is a test"
        result = TextQueryProcessor.remove_stopwords(text)
        self.assertEqual(result, "test")

    def test_get_tokenized_list(self):
        text = "Hello world"
        result = TextQueryProcessor.get_tokenized_list(text)
        self.assertEqual(result, ["Hello", "world"])

    def test_word_stemmer(self):
        tokens = ["running", "jumps", "easily"]
        result = TextQueryProcessor.word_stemmer(tokens)
        self.assertEqual(result, ["run", "jump", "easili"])

    def test_preprocessing_text(self):
        query = "  Hello, world! This is a test.  "
        result = TextQueryProcessor.preprocessing_text(query)
        self.assertEqual(result, "hello world test")

if __name__ == '__main__':
    unittest.main()