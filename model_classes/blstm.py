import torch
import torch.nn as nn 
from model_classes.base_model import BaseSequenceModel


class BLSTM(BaseSequenceModel):

    """
    Bidirectional LSTM implemented from scratch.

    It processes the sequence in both forward and backward directions
    and combines the hidden states from both directions.
    """

    def __init__(self, vocab_size, embed_size=64, hidden_size=128) -> None:
        """
        Initialize model parameters.

        Args:
            vocab_size: number of tokens
            embed_size: size of embeddings
            hidden_size: size of hidden state
        """
        super().__init__()

        self.hidden_size = hidden_size

        """
        Embedding layer to convert token indices to vectors.
        """
        self.embeddings = nn.Embedding(vocab_size, embed_size)

        """
        Forward LSTM parameters.
        """
        self.Wxi = nn.Parameter(torch.randn(embed_size, hidden_size) * 0.01)
        self.Whi = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.bi = nn.Parameter(torch.zeros(hidden_size))

        self.Wxf = nn.Parameter(torch.randn(embed_size, hidden_size) * 0.01)
        self.Whf = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.bf = nn.Parameter(torch.zeros(hidden_size))

        self.Wxo = nn.Parameter(torch.randn(embed_size, hidden_size) * 0.01)
        self.Who = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.bo = nn.Parameter(torch.zeros(hidden_size))

        self.Wxg = nn.Parameter(torch.randn(embed_size, hidden_size) * 0.01)
        self.Whg = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.bg = nn.Parameter(torch.zeros(hidden_size))

        """
        Backward LSTM parameters.
        """
        self.Wxi_b = nn.Parameter(torch.randn(embed_size, hidden_size) * 0.01)
        self.Whi_b = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.bi_b = nn.Parameter(torch.zeros(hidden_size))

        self.Wxf_b = nn.Parameter(torch.randn(embed_size, hidden_size) * 0.01)
        self.Whf_b = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.bf_b = nn.Parameter(torch.zeros(hidden_size))

        self.Wxo_b = nn.Parameter(torch.randn(embed_size, hidden_size) * 0.01)
        self.Who_b = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.bo_b = nn.Parameter(torch.zeros(hidden_size))

        self.Wxg_b = nn.Parameter(torch.randn(embed_size, hidden_size) * 0.01)
        self.Whg_b = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.bg_b = nn.Parameter(torch.zeros(hidden_size))

        """
        Final layer to map concatenated hidden states to logits.
        """
        self.fc = nn.Linear(hidden_size * 2, vocab_size)

    def lstm_step(self, xt, h, c,
                  Wx_i, Wh_i, b_i,
                  Wx_f, Wh_f, b_f,
                  Wx_o, Wh_o, b_o,
                  Wx_g, Wh_g, b_g):
        """
        Single LSTM step.

        Args:
            xt: input at timestep
            h: previous hidden state
            c: previous cell state

        Returns:
            updated hidden state and cell state
        """

        i = torch.sigmoid(xt @ Wx_i + h @ Wh_i + b_i)
        f = torch.sigmoid(xt @ Wx_f + h @ Wh_f + b_f)
        o = torch.sigmoid(xt @ Wx_o + h @ Wh_o + b_o)
        g = torch.tanh(xt @ Wx_g + h @ Wh_g + b_g)

        c = f * c + i * g
        h = o * torch.tanh(c)

        return h, c

    def forward(self, x, hidden=None):
        """
        Forward pass through BiLSTM.

        Args:
            x: input tensor (batch_size, seq_len)

        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """

        batch_size, seq_len = x.shape

        """
        Convert input indices to embeddings.
        """
        x = self.embeddings(x)

        """
        Initialize forward and backward hidden and cell states.
        """
        hf = torch.zeros(batch_size, self.hidden_size, device=x.device)
        cf = torch.zeros(batch_size, self.hidden_size, device=x.device)

        hb = torch.zeros(batch_size, self.hidden_size, device=x.device)
        cb = torch.zeros(batch_size, self.hidden_size, device=x.device)

        forward_states = []
        backward_states = []

        """
        Forward pass through sequence.
        """
        for t in range(seq_len):

            xt = x[:, t, :]

            hf, cf = self.lstm_step(
                xt, hf, cf,
                self.Wxi, self.Whi, self.bi,
                self.Wxf, self.Whf, self.bf,
                self.Wxo, self.Who, self.bo,
                self.Wxg, self.Whg, self.bg
            )

            forward_states.append(hf)

        """
        Backward pass through sequence.
        """
        for t in reversed(range(seq_len)):

            xt = x[:, t, :]

            hb, cb = self.lstm_step(
                xt, hb, cb,
                self.Wxi_b, self.Whi_b, self.bi_b,
                self.Wxf_b, self.Whf_b, self.bf_b,
                self.Wxo_b, self.Who_b, self.bo_b,
                self.Wxg_b, self.Whg_b, self.bg_b
            )

            backward_states.append(hb)

        """
        Reverse backward states to align with forward states.
        """
        backward_states.reverse()

        outputs = []

        """
        Combine forward and backward hidden states.
        """
        for hf, hb in zip(forward_states, backward_states):

            h = torch.cat([hf, hb], dim=1)

            outputs.append(self.fc(h).unsqueeze(1))

        """
        Concatenate outputs across all timesteps.
        """
        logits = torch.cat(outputs, dim=1)

        return logits, None
