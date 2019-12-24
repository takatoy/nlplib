DEFAULT_SP_TOKENS = {"<pad>": 0, "<unk>": 1, "<s>": 2, "</s>": 3}


class Vocab:
    def __init__(
        self,
        word2id=DEFAULT_SP_TOKENS,
        pad_token="<pad>",
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        cased=False,
    ):
        self.word2id = dict(word2id)
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.cased = cased

        self.word2id.setdefault(unk_token, len(self.word2id))
        self.id2word = {v: k for k, v in self.word2id.items()}

        self.pad_idx = self.word2id[self.pad_token]
        self.unk_idx = self.word2id[self.unk_token]
        self.bos_idx = self.word2id[self.bos_token]
        self.eos_idx = self.word2id[self.eos_token]

        self.n_unk = 0

    def __len__(self):
        return len(self.word2id)

    def build_vocab(self, sents, cutoff=-1, cased=False):
        word_cnt = {}

        def count(sent):
            for word in sent:
                if isinstance(word, list):
                    # if word is a list go deeper
                    count(word)
                    continue
                if not self.cased:
                    word = word.lower()
                word_cnt[word] = word_cnt.get(word, 0) + 1

        count(sents)

        for word, cnt in sorted(word_cnt.items(), key=lambda x: -x[1]):
            if cutoff != -1 and cnt <= cutoff:
                self.n_unk = len(word_cnt) - len(self.word2id)
                break
            if not self.cased:
                word = word.lower()
            _id = len(self.word2id)
            self.word2id.setdefault(word, _id)
            self.id2word[_id] = word

    def w2i(self, word):
        if isinstance(word, list) or isinstance(word, tuple):
            return [self.w2i(w) for w in word]
        else:
            if not self.cased:
                word = word.lower()
            return self.word2id.get(word, self.word2id[self.unk_token])

    def i2w(self, _id):
        if isinstance(_id, list) or isinstance(_id, tuple):
            return [self.i2w(i) for i in _id]
        else:
            return self.id2word[_id]
