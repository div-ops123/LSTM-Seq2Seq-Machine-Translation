"""Per-language vocabulary.

One `Vocab` instance per language (English, Igbo) -- never shared. See
docs/embedding_and_tokenization_reasoning.md for why separate vocabularies
were chosen over a single combined one.
"""


class Vocab:
    PAD_token = 0
    SOS_token = 1
    EOS_token = 2
    UNK_token = 3

    def __init__(self, name: str):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {
            self.PAD_token: "<PAD>",
            self.SOS_token: "<SOS>",
            self.EOS_token: "<EOS>",
            self.UNK_token: "<UNK>",
        }
        self.n_words = 4

    def add_sentence(self, sentence: str) -> None:
        for word in sentence.split(" "):
            if word:
                self.add_word(word)

    def add_word(self, word: str) -> None:
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def sentence_to_ids(self, sentence: str, max_length: int) -> list[int]:
        tokens = [w for w in sentence.split(" ") if w]
        ids = [self.word2index.get(w, self.UNK_token) for w in tokens]
        ids = ids[: max_length - 1]  # leave room for EOS
        ids.append(self.EOS_token)
        ids = ids + [self.PAD_token] * (max_length - len(ids))
        return ids

    def ids_to_sentence(self, ids) -> str:
        words = []
        for idx in ids:
            idx = int(idx)
            if idx in (self.PAD_token, self.SOS_token):
                continue
            if idx == self.EOS_token:
                break
            words.append(self.index2word.get(idx, "<UNK>"))
        return " ".join(words)

    @classmethod
    def from_file(cls, name: str, path: str, normalize_fn) -> "Vocab":
        vocab = cls(name)
        with open(path, encoding="utf-8") as f:
            for line in f:
                vocab.add_sentence(normalize_fn(line))
        return vocab
