from src.data_module import NameDataModule
from train.train import Trainer
from test.evaluator import Evaluator
from src.logger import setup_logger
import constants
from datetime import datetime


def main():
    """
    Main function to run training and evaluation for all models.
    """

    """
    Define model names and display titles.
    """
    model_titles = ["Vanilla RNN", "BiLSTM", "Attention RNN"]
    model_names = ["rnn", "blstm", "attention"]

    """
    Initialize logger.
    """
    logger = setup_logger()

    logger.info("Starting training pipeline")

    """
    Initialize data module.
    """
    data_module = NameDataModule(
        constants.DATA_DIR,
        constants.SAVE_DIR,
        logger=logger
    )

    """
    Load training names for novelty calculation.
    """
    with open(constants.DATA_DIR, "r") as f:
        train_names = [line.strip().lower() for line in f]

    train_names = set(train_names)

    """
    Create timestamp for model saving.
    """
    currtime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    """
    Train and evaluate each model.
    """
    for i in range(len(model_titles)):

        logger.info(f"TRAINING: {model_titles[i]}")

        """
        Define model save path.
        """
        model_path = constants.get_model_save_path(
            f"{model_names[i]}_{currtime}.pt"
        )

        """
        Initialize trainer.
        """
        trainer = Trainer(
            data_module=data_module,
            model_name=model_names[i],
            model_save_path=model_path,
            logger=logger,
            epochs=constants.EPOCHS
        )

        """
        Train the model.
        """
        trainer.train()

        logger.info(f"{model_titles[i]} training finished")

        logger.info("Starting evaluation")

        """
        Initialize evaluator.
        """
        evaluator = Evaluator(
            model_path=model_path,
            vocab_path=constants.SAVE_DIR,
            train_names=train_names,
            logger=logger
        )

        """
        Evaluate generated names.
        """
        evaluator.evaluate(num_samples=1000)

        logger.info("\n--------------------------------------\n")

    logger.info("All models trained and evaluated successfully")


if __name__ == "__main__":
    main()
