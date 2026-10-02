import csv
import json
import os
from typing import Union

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

NgramKey = Union[tuple, str]

UNK = "<unk>"  # stands in for any bigram the vocab has never seen


def tokenize(text):
    """Lowercase and split on whitespace. Simple on purpose."""
    return text.lower().split()


def get_ngrams(tokens):
    """Unigrams (single words) + bigrams (adjacent word-pairs), combined.

    Unigrams matter because a message like "idiot" or "noob" has ZERO
    bigrams (bigrams need a pair of adjacent words) — without unigrams,
    every single-word message produces an all-zero input vector, and
    the model has no signal to work with at all.
    """

    # TODO: Create a list containing each individual token.
    # Example: ["hello", "world"] → [("hello",), ("world",)]
    unigrams = []
    for t in tokens:
        pass

    # TODO: Create a list of adjacent word pairs (bigrams). Each bigram should be a tuple of two strings.
    bigrams = []
    for i in range(len(tokens) - 1):
        pass  

    return unigrams + bigrams


def load_csv(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((row["text"], int(row["label"])))
    return rows


class BigramVectorizer:
    """
    Builds a bigram vocabulary from training text, then converts any
    message into a fixed-length bag-of-uni+bigrams count vector.
    """

    def __init__(self):
        self.bigram_to_index: dict[NgramKey, int] = {}



    def fit(self, texts, min_count=1):
        """
        Build the vocabulary from a list of raw text strings.
        Only call this on TRAINING text — that's what creates the
        train/test gap. Every unique unigram/bigram gets its own index.
        Index 0 is reserved for UNK (unseen n-grams at inference time).

        min_count: drop any n-gram appearing fewer than this many times
        across the whole training set. On small hand-built datasets,
        min_count=1 (keep everything) is fine. On real-world text,
        the vocabulary otherwise explodes — a huge long tail of typos,
        names, and one-off phrases that appear exactly once and just
        bloat memory without giving the model anything reusable to learn.
        """

        counts = {}

        for text in texts:
            tokens = tokenize(text)
            for ngram in get_ngrams(tokens):

                # TODO: Count how many times each n-gram appears across the whole training set.
                counts[ngram] = None

        self.bigram_to_index = {UNK: 0}

        for ngram, count in counts.items():
            if count >= min_count:

                # TODO: Give this n-gram the next available vocabulary index. The first n-gram after UNK should get index 1, the next one index 2, etc.
                self.bigram_to_index[ngram] = None

    @property
    def vocab_size(self):
        return len(self.bigram_to_index)



    def transform(self, text):
        """
        Convert one message into a bag-of-(uni+bi)grams count vector,
        length == vocab_size. N-grams not seen during fit() fall
        into the UNK slot (index 0) instead of being dropped silently.
        """
        vector = [0] * self.vocab_size
        tokens = tokenize(text)

        for ngram in get_ngrams(tokens):
            
            # TODO: Find this n-gram's vocabulary index (or 0 if it's UNK) and increment that slot in the vector.
            index = None
            vector[index] += None

        return vector



    def transform_batch(self, texts):
        return [self.transform(text) for text in texts]



    def save(self, path):
        """
        Save the fitted vocab to JSON. N-grams are tuples, which aren't
        valid JSON dict keys, so we join their words with a separator.
        """
        serializable = {}
        for ngram, index in self.bigram_to_index.items():
            if isinstance(ngram, tuple):
                key = "\u0001".join(ngram)
            else:
                key = ngram  # the UNK string
            serializable[key] = index

        with open(path, "w", encoding="utf-8") as f:
            json.dump(serializable, f)



    def load(self, path):
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        self.bigram_to_index = {}
        for key, index in raw.items():
            if key == UNK:
                self.bigram_to_index[key] = index
            else:
                self.bigram_to_index[tuple(key.split("\u0001"))] = index


if __name__ == "__main__":
    train_rows = load_csv(os.path.join(SCRIPT_DIR, "..", "data", "train.csv"))
    test_rows = load_csv(os.path.join(SCRIPT_DIR, "..", "data", "test.csv"))

    train_texts = [text for text, label in train_rows]
    test_texts = [text for text, label in test_rows]

    vectorizer = BigramVectorizer()
    vectorizer.fit(train_texts)  # fit on TRAIN ONLY

    print(f"vocab size (including UNK): {vectorizer.vocab_size}")

    sample = train_texts[0]
    vector = vectorizer.transform(sample)
    print(f"\nsample train message: {sample!r}")
    print(f"vector length: {len(vector)}")
    print(f"nonzero slots: {sum(1 for v in vector if v > 0)}")

    # deliberately check a test message that likely uses held-out words
    sample_test = test_texts[0]
    vector_test = vectorizer.transform(sample_test)
    unk_count = vector_test[0]
    print(f"\nsample test message: {sample_test!r}")
    print(f"UNK count in this message's vector: {unk_count}")
