import torch
from torch.optim import Adam
import torch.nn as nn
from tqdm import tqdm

from model_classes.model_factory import ModelFactory


class Trainer:

    """
    Handles model training, validation, and saving.

    Uses a data module to load data and trains the selected model
    with the given hyperparameter configuration.
    """

    def __init__(
        self,
        data_module,
        model_name,
        model_save_path,
        logger=None,
        learning_rate=0.001,
        epochs=30,
        hidden_size=128,
        num_layers=1,
        dropout=0.0,
        batch_size=64,
    ) -> None:
        """
        Initialize trainer.

        Args:
            data_module: data module instance
            model_name: type of model to train
            model_save_path: path to save model weights
            logger: optional logger
            learning_rate: optimizer learning rate
            epochs: number of training epochs
            hidden_size: hidden state size passed to the model
            num_layers: number of stacked layers in the model
            dropout: dropout probability between layers
            batch_size: number of samples per training batch
        """

        self.data_module = data_module
        self.model_name = model_name
        self.model_save_path = model_save_path
        self.logger = logger

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_size = batch_size

        """
        Select device — GPU if available, otherwise fall back to CPU.
        """
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        if self.logger:
            self.logger.info(f"Using device: {self.device}")

    def build_model(self, vocab_size):
        """
        Initialize model, loss function, and optimizer using
        the hyperparameters set during __init__.
        """

        self.model = ModelFactory.create(
            self.model_name,
            vocab_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)

        """
        Cross entropy loss — padding token index 0 is ignored.
        """
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

        """
        Adam optimizer with the configured learning rate.
        """
        self.optimizer = Adam(
            self.model.parameters(),
            lr=self.learning_rate
        )

        if self.logger:
            total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self.logger.info(f"  Model: {self.model_name}")
            self.logger.info(f"  Hidden size: {self.hidden_size} | Layers: {self.num_layers} | Dropout: {self.dropout}")
            self.logger.info(f"  Trainable parameters: {total_params:,}")

    def train(self):
        """
        Main training loop.

        Runs for the configured number of epochs, logs train and
        validation loss each epoch, saves the best checkpoint by
        validation loss, and returns the final validation loss for
        hyperparameter comparison in main.py.
        """

        train_loader, val_loader, vocab_size = self.data_module.setup(
            batch_size=self.batch_size
        )

        if self.logger:
            self.logger.info(f"  Vocab size: {vocab_size} | Batch size: {self.batch_size}")

        self.build_model(vocab_size)

        best_val_loss = float("inf")
        final_val_loss = float("inf")

        for epoch in range(self.epochs):

            """
            Training phase — model is in train mode so dropout is active.
            """
            self.model.train()

            total_loss = 0

            for x, y in tqdm(
                train_loader,
                desc=f"  Epoch {epoch+1}/{self.epochs}",
                leave=False
            ):
                x = x.to(self.device)
                y = y.to(self.device)

                """
                Forward pass.
                """
                logits, _ = self.model(x)

                """
                Compute loss over the full sequence.
                """
                loss = self.criterion(
                    logits.reshape(-1, vocab_size),
                    y.reshape(-1)
                )

                """
                Backpropagate and update weights.
                """
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)

            """
            Validation phase — no gradients needed here.
            """
            val_loss = self._evaluate(val_loader, vocab_size)
            final_val_loss = val_loss

            if self.logger:
                self.logger.info(
                    f"  Epoch {epoch+1}/{self.epochs} | "
                    f"Train Loss: {avg_train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f}"
                )

            """
            Save the model whenever validation loss improves.
            This way we keep the best checkpoint, not just the last one.
            """
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save()

                if self.logger:
                    self.logger.info(
                        f"  Checkpoint saved (val loss improved to {best_val_loss:.4f})"
                    )

        if self.logger:
            self.logger.info(
                f"  Training complete — best val loss: {best_val_loss:.4f}"
            )

        """
        Return the best validation loss so main.py can compare runs.
        """
        return best_val_loss

    def _evaluate(self, loader, vocab_size):
        """
        Evaluate the model on a validation dataloader.

        Args:
            loader: validation dataloader
            vocab_size: size of vocabulary

        Returns:
            average validation loss across all batches
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
        Save the full model to disk at the configured path.
        """

        torch.save(self.model, self.model_save_path)

        if self.logger:
            self.logger.info(
                f"  Model saved to: {self.model_save_path}"
            )
