import torch
import torch.nn as nn
from model_classes.base_model import BaseSequenceModel


class VanillaRNN(BaseSequenceModel):
    """
    A simple RNN implemented from scratch using basic matrix operations.
    It processes input sequences character by character and maintains
    a hidden state to capture sequential information.
    Supports stacking multiple layers with optional dropout between them.
    """

    def __init__(self, vocab_size, embed_size=64, hidden_size=128, num_layers=1, dropout=0.0):
        """
        Initialize model parameters.

        Args:
            vocab_size: number of unique tokens
            embed_size: size of embedding vectors
            hidden_size: size of hidden state
            num_layers: how many RNN layers to stack
            dropout: dropout probability applied between layers (0 = off)
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        """
        Embedding layer to convert token indices to dense vectors.
        """
        self.embedding = nn.Embedding(vocab_size, embed_size, 0)

        """
        One set of weight matrices per layer.
        First layer takes embed_size as input, subsequent layers take hidden_size.
        """
        self.Wxh = nn.ParameterList()
        self.Whh = nn.ParameterList()
        self.bh = nn.ParameterList()

        for layer in range(num_layers):
            input_size = embed_size if layer == 0 else hidden_size
            self.Wxh.append(nn.Parameter(torch.randn(input_size, hidden_size) * 0.01))
            self.Whh.append(nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01))
            self.bh.append(nn.Parameter(torch.zeros(hidden_size)))

        """
        Dropout applied between layers during training.
        """
        self.dropout = nn.Dropout(p=dropout)

        """
        Fully connected layer to map hidden state to vocabulary logits.
        """
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        """
        Forward pass through the stacked RNN.

        Args:
            x: input tensor of shape (batch_size, seq_len)
            hidden: optional initial hidden state

        Returns:
            logits: predictions for each timestep (batch_size, seq_len, vocab_size)
            h: final hidden state of the last layer
        """
        batch_size, seq_len = x.shape

        """
        Convert input indices to embeddings.
        """
        x = self.embedding(x)

        """
        Initialize one hidden state per layer, all zeros.
        """
        h_layers = [
            torch.zeros(batch_size, self.hidden_size, device=x.device)
            for _ in range(self.num_layers)
        ]

        outputs = []

        """
        Iterate over each timestep.
        """
        for t in range(seq_len):

            """
            Input to the first layer is the embedding at this timestep.
            Input to subsequent layers is the hidden state of the layer below.
            """
            layer_input = x[:, t, :]

            for layer in range(self.num_layers):
                h = torch.tanh(
                    layer_input @ self.Wxh[layer] +
                    h_layers[layer] @ self.Whh[layer] +
                    self.bh[layer]
                )
                h_layers[layer] = h

                """
                Apply dropout between layers but not after the last one.
                """
                if layer < self.num_layers - 1:
                    layer_input = self.dropout(h)
                else:
                    layer_input = h

            """
            Compute output logits from the top layer's hidden state.
            """
            logits = self.fc(h_layers[-1])
            outputs.append(logits.unsqueeze(1))

        """
        Concatenate outputs across all timesteps.
        """
        logits = torch.cat(outputs, dim=1)

        return logits, h_layers[-1]
