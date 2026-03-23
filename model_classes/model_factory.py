from model_classes.vanilla_rnn import VanillaRNN
from model_classes.blstm import BLSTM
from model_classes.attention_rnn import AttentionRNN


class ModelFactory:

    """
    Factory class to create different sequence models.
    This helps in selecting and initializing models based on a name string.
    """

    @staticmethod
    def create(model_name, vocab_size, embed_size=64, hidden_size=128, num_layers=1, dropout=0.0):
        """
        Create and return a model instance.

        Args:
            model_name: name of the model (rnn, blstm, attention)
            vocab_size: size of vocabulary
            embed_size: embedding dimension
            hidden_size: hidden state size
            num_layers: number of stacked layers
            dropout: dropout probability between layers

        Returns:
            initialized model
        """

        model_name = model_name.lower()

        """
        Select and initialize the model based on the name string.
        All models receive the same set of hyperparameters.
        """
        if model_name == "rnn":
            return VanillaRNN(vocab_size, embed_size, hidden_size, num_layers, dropout)

        elif model_name == "blstm":
            return BLSTM(vocab_size, embed_size, hidden_size, num_layers, dropout)

        elif model_name == "attention":
            return AttentionRNN(vocab_size, embed_size, hidden_size, num_layers, dropout)

        else:
            """
            Raise an error if the model name isn't one we recognize.
            """
            raise ValueError(
                f"Unknown model type: '{model_name}'. "
                f"Expected one of: 'rnn', 'blstm', 'attention'."
            )
