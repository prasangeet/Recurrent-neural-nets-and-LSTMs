from model_classes.vanilla_rnn import VanillaRNN
from model_classes.blstm import BLSTM
from model_classes.attention_rnn import AttentionRNN


class ModelFactory:

    """
    Factory class to create different sequence models.

    This helps in selecting and initializing models based on a name string.
    """

    @staticmethod
    def create(model_name, vocab_size, embed_size=64, hidden_size=128):
        """
        Create and return a model instance.

        Args:
            model_name: name of the model (rnn, blstm, attention)
            vocab_size: size of vocabulary
            embed_size: embedding dimension
            hidden_size: hidden state size

        Returns:
            initialized model
        """

        model_name = model_name.lower()

        """
        Select model based on name.
        """
        if model_name == "rnn":
            return VanillaRNN(vocab_size, embed_size, hidden_size)

        elif model_name == "blstm":
            return BLSTM(vocab_size, embed_size, hidden_size)

        elif model_name == "attention":
            return AttentionRNN(vocab_size, embed_size, hidden_size)

        else:
            """
            Raise error if model name is not recognized.
            """
            raise ValueError(f"Unknown model type: {model_name}")
