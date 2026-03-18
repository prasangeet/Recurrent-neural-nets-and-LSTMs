from logging import Logger
from torch.utils.data import DataLoader, random_split

from src.preprocess import PreprocessPipeline
from src.dataset import NameDataset

class NameDataModule:

    """
    Handles data loading, preprocessing, and dataset splitting.

    This class prepares training and validation dataloaders.
    """

    def __init__(
        self,
        data_path="TrainingNames.txt",
        vocab_file="vocab.json",
        batch_size=32,
        val_split=0.2,
        logger=None
    ):
        """
        Initialize data module.

        Args:
            data_path: path to raw data file
            vocab_file: file to save vocabulary
            batch_size: batch size for training
            val_split: fraction of data used for validation
            logger: optional logger
        """
        self.data_path = data_path
        self.vocab_file = vocab_file
        self.batch_size = batch_size
        self.val_split = val_split
        self.logger = logger
        self.vocab_size = None

    def setup(self):
        """
        Run preprocessing and prepare dataloaders.

        Returns:
            train_loader, val_loader, vocab_size
        """

        """
        Run preprocessing pipeline to get inputs and targets.
        """
        pipeline = PreprocessPipeline(
            data_path=self.data_path,
            save_file=self.vocab_file,
            logger=self.logger
        )

        inputs, targets = pipeline.run()

        """
        Store vocabulary size.
        """
        self.vocab_size = len(pipeline.vocab)

        """
        Create dataset.
        """
        dataset = NameDataset(inputs, targets)

        dataset_size = len(dataset)

        """
        Split dataset into train and validation sets.
        """
        val_size = int(dataset_size * self.val_split)
        train_size = dataset_size - val_size

        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size]
        )

        """
        Create dataloaders.
        """
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )

        """
        Log dataset split information.
        """
        if self.logger:
            self.logger.info(
                f"Dataset Split -> Train: {train_size} | Val: {val_size}"
            )

        return train_loader, val_loader, self.vocab_size
