import torch
from torch.utils.data import Dataset


class NameDataset(Dataset):

    """
    Dataset class for name sequences.

    It stores input and target sequences and applies padding
    so that all sequences have the same length.
    """

    def __init__(self, inputs, targets, pad_idx=0):
        """
        Initialize dataset.

        Args:
            inputs: list of input sequences
            targets: list of target sequences
            pad_idx: index used for padding
        """

        self.inputs = inputs
        self.targets = targets
        self.pad_idx = pad_idx

        """
        Determine maximum sequence length for padding.
        """
        self.max_len = max(len(x) for x in inputs)

    def pad_sequence(self, seq):
        """
        Pad a sequence to max length.

        Args:
            seq: input sequence

        Returns:
            padded sequence
        """

        padded = seq + [self.pad_idx] * (self.max_len - len(seq))
        return padded

    def __len__(self):
        """
        Return total number of samples.
        """

        return len(self.inputs)

    def __getitem__(self, idx):
        """
        Get one sample from dataset.

        Args:
            idx: index

        Returns:
            x: input tensor
            y: target tensor
        """

        x = self.pad_sequence(self.inputs[idx])
        y = self.pad_sequence(self.targets[idx])

        x = torch.tensor(x, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)

        return x, y
