from nltk.tokenize import word_tokenize
import nlplib.utils.vocab as vocab


class TestVocab:
    def setup_method(self, method):
        _train = [
            "You look like my sister.",
            "I had some custard pudding for an afternoon snack.",
            "He turned to his friend for help.",
            "What do people usually do in the summer in Los Angeles?",
        ]
        _test = ["Do you like pizza?"]

        self.vocab = vocab.Vocab()
        self.train = [word_tokenize(sent) for sent in _train]
        self.test = [word_tokenize(sent) for sent in _test]

    def test_build_vocab(self):
        self.vocab.build_vocab(self.train, 1)

    def test_w2i(self):
        self.vocab.build_vocab(self.train)
        expected = [[6, 8, 10, 1, 34]]
        assert self.vocab.w2i(self.test) == expected

    def test_i2w(self):
        self.vocab.build_vocab(self.train)
        expected = [["do", "you", "like", "<UNK>", "?"]]
        assert self.vocab.i2w(self.vocab.w2i(self.test)) == expected
