import torch
import torch.nn as nn
from model_classes.base_model import BaseSequenceModel


class AttentionRNN(BaseSequenceModel):

    """
    RNN with attention mechanism.

    The model computes attention over all previous hidden states
    at each timestep to create a context vector.
    """

    def __init__(self, vocab_size, embed_size=64, hidden_size=128) -> None:
        """
        Initialize model parameters.

        Args:
            vocab_size: number of tokens
            embed_size: embedding size
            hidden_size: hidden state size
        """
        super().__init__()

        self.hidden_size = hidden_size

        """
        Embedding layer to convert indices to vectors.
        """
        self.embedding = nn.Embedding(vocab_size, embed_size)

        """
        RNN parameters.
        """
        self.Wxh = nn.Parameter(torch.randn(embed_size, hidden_size) * 0.01)
        self.Whh = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.bh = nn.Parameter(torch.zeros(hidden_size))

        """
        Final layer takes hidden state and context vector.
        """
        self.fc = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x, hidden=None):
        """
        Forward pass through Attention RNN.

        Args:
            x: input tensor (batch_size, seq_len)

        Returns:
            logits: (batch_size, seq_len, vocab_size)
            h: final hidden state
        """

        batch_size, seq_len = x.shape

        """
        Convert input indices to embeddings.
        """
        x = self.embedding(x)

        """
        Initialize hidden state.
        """
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)

        hidden_states = []
        outputs = []

        """
        Iterate over each timestep.
        """
        for t in range(seq_len):

            xt = x[:, t, :]

            """
            Update hidden state.
            """
            h = torch.tanh(
                xt @ self.Wxh + 
                h @ self.Whh + 
                self.bh
            )

            hidden_states.append(h)

            """
            Stack all previous hidden states.
            """
            H = torch.stack(hidden_states, dim=1)

            """
            Compute attention scores using dot product.
            """
            scores = torch.bmm(
                H,
                h.unsqueeze(2)
            ).squeeze(2)

            """
            Normalize scores to get attention weights.
            """
            weights = torch.softmax(scores, dim=1)

            """
            Compute context vector as weighted sum of hidden states.
            """
            context = torch.bmm(
                weights.unsqueeze(1),
                H
            ).squeeze(1)

            """
            Combine current hidden state and context.
            """
            combined = torch.cat([h, context], dim=1)

            """
            Compute output logits.
            """
            logits = self.fc(combined)

            outputs.append(logits.unsqueeze(1))
        
        """
        Concatenate outputs across timesteps.
        """
        logits = torch.cat(outputs, dim=1)

        return logits, h
