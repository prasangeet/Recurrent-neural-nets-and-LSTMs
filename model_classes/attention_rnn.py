import torch
import torch.nn as nn
from model_classes.base_model import BaseSequenceModel


class AttentionRNN(BaseSequenceModel):
    """
    RNN with attention mechanism, now supporting multiple stacked layers and dropout.
    The model computes attention over all previous hidden states of the top layer
    at each timestep to create a context vector.
    """

    def __init__(self, vocab_size, embed_size=64, hidden_size=128, num_layers=1, dropout=0.0):
        """
        Initialize model parameters.

        Args:
            vocab_size: number of tokens
            embed_size: embedding size
            hidden_size: hidden state size
            num_layers: number of stacked RNN layers
            dropout: dropout probability applied between layers
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        """
        Embedding layer to convert indices to vectors.
        """
        self.embedding = nn.Embedding(vocab_size, embed_size)

        """
        One set of RNN weights per layer.
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
        Dropout applied between layers.
        """
        self.dropout = nn.Dropout(p=dropout)

        """
        Final layer takes the top hidden state concatenated with the context vector.
        """
        self.fc = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x, hidden=None):
        """
        Forward pass through the stacked Attention RNN.

        Args:
            x: input tensor (batch_size, seq_len)

        Returns:
            logits: (batch_size, seq_len, vocab_size)
            h: final hidden state of the top layer
        """
        batch_size, seq_len = x.shape

        """
        Convert input indices to embeddings.
        """
        x = self.embedding(x)

        """
        Initialize hidden states for all layers.
        """
        h_layers = [
            torch.zeros(batch_size, self.hidden_size, device=x.device)
            for _ in range(self.num_layers)
        ]

        """
        We only track hidden states of the top layer for attention.
        """
        top_hidden_states = []
        outputs = []

        """
        Iterate over each timestep.
        """
        for t in range(seq_len):

            layer_input = x[:, t, :]

            for layer in range(self.num_layers):
                h = torch.tanh(
                    layer_input @ self.Wxh[layer] +
                    h_layers[layer] @ self.Whh[layer] +
                    self.bh[layer]
                )
                h_layers[layer] = h

                """
                Dropout between layers only, not after the top one.
                """
                if layer < self.num_layers - 1:
                    layer_input = self.dropout(h)
                else:
                    layer_input = h

            """
            Grab the top layer's hidden state for attention computation.
            """
            top_h = h_layers[-1]
            top_hidden_states.append(top_h)

            """
            Stack all top hidden states seen so far.
            """
            H = torch.stack(top_hidden_states, dim=1)

            """
            Compute attention scores using dot product.
            """
            scores = torch.bmm(H, top_h.unsqueeze(2)).squeeze(2)

            """
            Normalize scores to get attention weights.
            """
            weights = torch.softmax(scores, dim=1)

            """
            Compute context vector as weighted sum of past hidden states.
            """
            context = torch.bmm(weights.unsqueeze(1), H).squeeze(1)

            """
            Combine current hidden state and context vector, then project to vocab.
            """
            combined = torch.cat([top_h, context], dim=1)
            outputs.append(self.fc(combined).unsqueeze(1))

        """
        Concatenate outputs across all timesteps.
        """
        logits = torch.cat(outputs, dim=1)

        return logits, h_layers[-1]
