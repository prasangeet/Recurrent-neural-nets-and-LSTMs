import torch
import torch.nn as nn
from model_classes.base_model import BaseSequenceModel


class BLSTM(BaseSequenceModel):
    """
    Bidirectional LSTM implemented from scratch, with support for
    stacked layers and dropout between them.

    Each layer processes the sequence in both directions. The forward
    and backward hidden states are concatenated before being passed
    up to the next layer.
    """

    def __init__(self, vocab_size, embed_size=64, hidden_size=128, num_layers=1, dropout=0.0):
        """
        Initialize model parameters.

        Args:
            vocab_size: number of tokens
            embed_size: size of embeddings
            hidden_size: size of hidden state per direction
            num_layers: number of stacked BiLSTM layers
            dropout: dropout probability applied between layers
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        """
        Embedding layer to convert token indices to vectors.
        """
        self.embeddings = nn.Embedding(vocab_size, embed_size)

        """
        Each layer has its own set of forward and backward LSTM weights.
        The first layer takes embed_size as input.
        Subsequent layers take hidden_size * 2 (concatenated forward + backward).
        """
        self.fwd_params = nn.ModuleList()
        self.bwd_params = nn.ModuleList()

        for layer in range(num_layers):
            input_size = embed_size if layer == 0 else hidden_size * 2
            self.fwd_params.append(self._make_lstm_params(input_size, hidden_size))
            self.bwd_params.append(self._make_lstm_params(input_size, hidden_size))

        """
        Dropout applied between layers.
        """
        self.dropout = nn.Dropout(p=dropout)

        """
        Final layer maps the top layer's concatenated hidden states to logits.
        """
        self.fc = nn.Linear(hidden_size * 2, vocab_size)

    def _make_lstm_params(self, input_size, hidden_size):
        """
        Creates a single LSTM cell's worth of parameters as a ModuleDict.
        Doing this per layer keeps the parameter lists clean and indexable.
        """
        return nn.ParameterDict({
            "Wxi": nn.Parameter(torch.randn(input_size, hidden_size) * 0.01),
            "Whi": nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01),
            "bi":  nn.Parameter(torch.zeros(hidden_size)),
            "Wxf": nn.Parameter(torch.randn(input_size, hidden_size) * 0.01),
            "Whf": nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01),
            "bf":  nn.Parameter(torch.zeros(hidden_size)),
            "Wxo": nn.Parameter(torch.randn(input_size, hidden_size) * 0.01),
            "Who": nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01),
            "bo":  nn.Parameter(torch.zeros(hidden_size)),
            "Wxg": nn.Parameter(torch.randn(input_size, hidden_size) * 0.01),
            "Whg": nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01),
            "bg":  nn.Parameter(torch.zeros(hidden_size)),
        })

    def lstm_step(self, xt, h, c, p):
        """
        Single LSTM step using a ParameterDict.

        Args:
            xt: input at this timestep
            h: previous hidden state
            c: previous cell state
            p: ParameterDict for this direction and layer

        Returns:
            updated hidden state and cell state
        """
        i = torch.sigmoid(xt @ p["Wxi"] + h @ p["Whi"] + p["bi"])
        f = torch.sigmoid(xt @ p["Wxf"] + h @ p["Whf"] + p["bf"])
        o = torch.sigmoid(xt @ p["Wxo"] + h @ p["Who"] + p["bo"])
        g = torch.tanh(xt @ p["Wxg"] + h @ p["Whg"] + p["bg"])

        c = f * c + i * g
        h = o * torch.tanh(c)

        return h, c

    def forward(self, x, hidden=None):
        """
        Forward pass through the stacked BiLSTM.

        Args:
            x: input tensor (batch_size, seq_len)

        Returns:
            logits: (batch_size, seq_len, vocab_size)
            None: no single hidden state to return for a bidirectional model
        """
        batch_size, seq_len = x.shape

        """
        Convert input indices to embeddings.
        """
        layer_input = self.embeddings(x)

        """
        Run through each stacked layer.
        """
        for layer in range(self.num_layers):

            hf = torch.zeros(batch_size, self.hidden_size, device=x.device)
            cf = torch.zeros(batch_size, self.hidden_size, device=x.device)
            hb = torch.zeros(batch_size, self.hidden_size, device=x.device)
            cb = torch.zeros(batch_size, self.hidden_size, device=x.device)

            forward_states = []
            backward_states = []

            """
            Forward pass through the sequence for this layer.
            """
            for t in range(seq_len):
                xt = layer_input[:, t, :]
                hf, cf = self.lstm_step(xt, hf, cf, self.fwd_params[layer])
                forward_states.append(hf)

            """
            Backward pass through the sequence for this layer.
            """
            for t in reversed(range(seq_len)):
                xt = layer_input[:, t, :]
                hb, cb = self.lstm_step(xt, hb, cb, self.bwd_params[layer])
                backward_states.append(hb)

            """
            Reverse backward states so they line up with forward states by timestep.
            """
            backward_states.reverse()

            """
            Concatenate forward and backward states at each timestep.
            This becomes the input to the next layer (or the final output).
            """
            combined = torch.stack(
                [torch.cat([f, b], dim=1) for f, b in zip(forward_states, backward_states)],
                dim=1
            )

            """
            Apply dropout between layers but leave the top layer's output clean.
            """
            if layer < self.num_layers - 1:
                layer_input = self.dropout(combined)
            else:
                layer_input = combined

        """
        Project each timestep's combined hidden state to vocabulary logits.
        """
        logits = self.fc(layer_input)

        return logits, None
