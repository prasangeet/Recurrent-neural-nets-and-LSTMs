import torch
import torch.nn as nn

from model_classes.base_model import BaseSequenceModel 


class VanillaRNN(BaseSequenceModel):

    """
    A simple RNN implemented from scratch using basic matrix operations.

    It processes input sequences character by character and maintains
    a hidden state to capture sequential information.
    """

    def __init__(
        self,
        vocab_size,
        embed_size=64,
        hidden_size=128,
    ):
        """
        Initialize model parameters.

        Args:
            vocab_size: number of unique tokens
            embed_size: size of embedding vectors
            hidden_size: size of hidden state
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        """
        Embedding layer to convert token indices to dense vectors.
        """
        self.embedding = nn.Embedding(vocab_size, embed_size, 0)

        """
        Weight matrices and bias for RNN cell.
        Wxh: input to hidden
        Whh: hidden to hidden
        bh: bias term
        """
        self.Wxh = nn.Parameter(torch.randn(embed_size, hidden_size) * 0.01)
        self.Whh = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.bh = nn.Parameter(torch.zeros(hidden_size))

        """
        Fully connected layer to map hidden state to vocabulary logits.
        """
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        """
        Forward pass through the RNN.

        Args:
            x: input tensor of shape (batch_size, seq_len)
            hidden: optional initial hidden state

        Returns:
            logits: predictions for each timestep (batch_size, seq_len, vocab_size)
            h: final hidden state
        """

        batch_size, seq_len = x.shape

        """
        Convert input indices to embeddings.
        """
        x = self.embedding(x)

        """
        Initialize hidden state with zeros.
        """
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)

        outputs = []

        """
        Iterate over each timestep.
        """
        for t in range(seq_len):

            xt = x[:, t, :]

            """
            Update hidden state using tanh activation.
            """
            h = torch.tanh(
                xt @ self.Wxh +
                h @ self.Whh + 
                self.bh
            )

            """
            Compute output logits from hidden state.
            """
            logits = self.fc(h)

            outputs.append(logits.unsqueeze(1))

        """
        Concatenate outputs across all timesteps.
        """
        logits = torch.cat(outputs, dim=1)

        return logits, h
