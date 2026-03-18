import torch
from torch.optim import Adam
import torch.nn as nn
from tqdm import tqdm

from model_classes.model_factory import ModelFactory


class Trainer:

    """
    Handles model training, validation, and saving.

    Uses a data module to load data and trains the selected model.
    """

    def __init__(
        self,
        data_module,
        model_name,
        model_save_path,
        logger=None,
        lr=0.01,
        epochs=30
    ) -> None:
        """
        Initialize trainer.

        Args:
            data_module: data module instance
            model_name: type of model to train
            model_save_path: path to save model
            logger: optional logger
            lr: learning rate
            epochs: number of epochs
        """

        self.data_module = data_module
        self.model_name = model_name
        self.model_save_path = model_save_path

        self.lr = lr
        self.epochs = epochs
        self.logger = logger

        """
        Select device (GPU if available, else CPU).
        """
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def build_model(self, vocab_size):
        """
        Initialize model, loss function, and optimizer.
        """

        self.model = ModelFactory.create(
            self.model_name,
            vocab_size
        ).to(self.device)

        """
        Cross entropy loss ignoring padding token.
        """
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

        """
        Adam optimizer.
        """
        self.optimizer = Adam(
            self.model.parameters(),
            lr=self.lr
        )

        if self.logger:
            self.logger.info(
                f"Initialized model: {self.model_name}"
            )

    def train(self):
        """
        Main training loop.
        """

        train_loader, val_loader, vocab_size = self.data_module.setup()

        if self.logger:
            self.logger.info(f"Vocabulary size: {vocab_size}")

        self.build_model(vocab_size)

        for epoch in range(self.epochs):

            """
            Set model to training mode.
            """
            self.model.train()

            total_loss = 0

            for x, y in tqdm(
                train_loader,
                desc=f"Epoch {epoch+1}/{self.epochs}"
            ):
                x = x.to(self.device)
                y = y.to(self.device)

                """
                Forward pass.
                """
                logits, _ = self.model(x)

                """
                Compute loss.
                """
                loss = self.criterion(
                    logits.reshape(-1, vocab_size),
                    y.reshape(-1)
                )

                """
                Backpropagation and optimization.
                """
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)

            """
            Evaluate on validation set.
            """
            val_loss = self.evaluate(val_loader, vocab_size)

            if self.logger:
                self.logger.info(
                    f"Epoch {epoch+1}/{self.epochs} | "
                    f"Train Loss: {avg_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f}"
                )

        """
        Save trained model.
        """
        self.save()

    def evaluate(self, loader, vocab_size):
        """
        Evaluate model on validation data.

        Args:
            loader: validation dataloader
            vocab_size: size of vocabulary

        Returns:
            average validation loss
        """

        self.model.eval()

        total_loss = 0

        with torch.no_grad():

            for x, y in loader:

                x = x.to(self.device)
                y = y.to(self.device)

                logits, _ = self.model(x)

                loss = self.criterion(
                    logits.reshape(-1, vocab_size),
                    y.reshape(-1)
                )

                total_loss += loss.item()

        return total_loss / len(loader)

    def save(self):
        """
        Save the full model to disk.
        """

        torch.save(self.model, self.model_save_path)

        if self.logger:
            self.logger.info(
                f"Full model saved to: {self.model_save_path}"
            )
