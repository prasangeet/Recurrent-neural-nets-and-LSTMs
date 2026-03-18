import torch
import json


class NameGenerator:

    """
    Generates names using a trained sequence model.

    Supports optional prefix-based generation and sampling controls.
    """

    def __init__(self, model, vocab_path, device="cpu"):
        """
        Initialize generator.

        Args:
            model: trained model
            vocab_path: path to vocabulary json
            device: cpu or cuda
        """

        self.model = model
        self.device = device

        """
        Move model to device and set evaluation mode.
        """
        self.model.to(device)
        self.model.eval()

        """
        Load vocabulary mappings.
        """
        with open(vocab_path, "r") as f:
            vocab_data = json.load(f)

        self.char2idx = vocab_data["char2idx"]
        self.idx2char = {int(k): v for k, v in vocab_data["idx2char"].items()}

        """
        Store special token indices.
        """
        self.sos = self.char2idx["<sos>"]
        self.eos = self.char2idx["<eos>"]
        self.pad = self.char2idx["<pad>"]

    def generate(self, prefix="", max_len=16, temperature=1.0, top_k=10):
        """
        Generate a single name.

        Args:
            prefix: starting string for generation
            max_len: maximum generated length
            temperature: controls randomness
            top_k: limits sampling to top k tokens

        Returns:
            generated name string
        """

        hidden = None

        """
        Start with <sos> token.
        """
        x = torch.tensor([[self.sos]], dtype=torch.long, device=self.device)

        name = []

        with torch.no_grad():

            """
            Feed prefix characters to the model to initialize hidden state.
            """
            for ch in prefix.lower():

                if ch not in self.char2idx:
                    continue

                idx = self.char2idx[ch]

                x = torch.tensor([[idx]], dtype=torch.long, device=self.device)

                logits, hidden = self.model(x, hidden)

                name.append(ch)

            """
            Generate characters step by step.
            """
            for _ in range(max_len):

                logits, hidden = self.model(x, hidden)

                logits = logits[:, -1, :]

                """
                Apply temperature scaling.
                """
                logits = logits / temperature

                probs = torch.softmax(logits, dim=-1)

                """
                Prevent sampling special tokens.
                """
                probs[0, self.pad] = 0
                probs[0, self.sos] = 0

                """
                Apply top-k sampling.
                """
                top_probs, top_idx = torch.topk(probs, top_k)

                """
                Normalize probabilities.
                """
                top_probs = top_probs / top_probs.sum()

                sample = torch.multinomial(top_probs[0], 1)

                next_char = top_idx[0, sample]

                idx = next_char.item()

                """
                Stop if end token is generated.
                """
                if idx == self.eos:
                    break

                char = self.idx2char[idx]

                name.append(char)

                """
                Feed predicted character back into model.
                """
                x = torch.tensor([[idx]], dtype=torch.long, device=self.device)

        return "".join(name)
