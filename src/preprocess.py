import json


class PreprocessPipeline:

    """
    Handles preprocessing of raw name data.

    Includes loading data, building vocabulary, encoding,
    and saving vocabulary mappings.
    """

    def __init__(self, data_path, save_file, logger=None):
        """
        Initialize pipeline.

        Args:
            data_path: path to raw data file
            save_file: file to save vocabulary
            logger: optional logger
        """
        self.data_path = data_path
        self.save_file = save_file

        self.data = []
        self.vocab = []
        self.char2idx = {}
        self.idx2char = {}
        self.encoded = []

        self.encoded_inputs = []
        self.encoded_outputs = []

        self.logger = logger

    def load_data(self):
        """
        Load data from file and clean it.

        Converts all names to lowercase and removes duplicates.
        """

        with open(self.data_path, "r") as f:
            self.data = [line.strip().lower() for line in f]

        """
        Remove duplicate names.
        """
        self.data = list(set(self.data))

        if self.logger:
            self.logger.info("Loaded the Data")

    def build_vocab(self):
        """
        Build vocabulary from dataset.

        Includes special tokens and all unique characters.
        """

        chars = sorted(list(set("".join(self.data))))

        """
        Add special tokens.
        """
        self.vocab = ["<pad>", "<sos>", "<eos>"] + chars

        """
        Create mappings between characters and indices.
        """
        self.char2idx = {c: i for i, c in enumerate(self.vocab)}
        self.idx2char = {i: c for i, c in enumerate(self.vocab)}

        if self.logger:
            self.logger.info("Built the Vocabulary")

    def encode(self):
        """
        Convert names into sequences of indices.

        Adds <sos> to inputs and <eos> to targets.
        """

        for name in self.data:

            inp = ["<sos>"] + list(name)
            tgt = list(name) + ["<eos>"]

            inp_ids = [self.char2idx[x] for x in inp]
            tgt_ids = [self.char2idx[x] for x in tgt]

            self.encoded_inputs.append(inp_ids)
            self.encoded_outputs.append(tgt_ids)

        if self.logger:
            self.logger.info("Encoded the outputs and the inputs")

    def save_vocab(self):
        """
        Save vocabulary and mappings to file.
        """

        vocab_data = {
            "vocab": self.vocab,
            "char2idx": self.char2idx,
            "idx2char": self.idx2char
        }

        with open(self.save_file, "w") as f:
            json.dump(vocab_data, f, indent=4)

        if self.logger:
            self.logger.info(
                "Saved the vocabulary file, with char2idx and idx2char"
            )

    def get_data(self):
        """
        Return encoded inputs and outputs.
        """

        return self.encoded_inputs, self.encoded_outputs

    def run(self):
        """
        Run full preprocessing pipeline.
        """

        self.load_data()
        self.build_vocab()
        self.encode()
        self.save_vocab()

        return self.get_data()
