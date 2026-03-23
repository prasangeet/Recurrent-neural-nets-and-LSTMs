from src.data_module import NameDataModule
from train.train import Trainer
from test.evaluator import Evaluator
from src.logger import setup_logger
import constants
from datetime import datetime


"""
Hyperparameter configurations to try for each model.
Each dict is one run — we loop through all of them per model.
"""
HYPERPARAMETER_GRID = [
    {
        "learning_rate": 0.001,
        "hidden_size": 128,
        "num_layers": 1,
        "dropout": 0.0,
        "batch_size": 64,
    },
    {
        "learning_rate": 0.0005,
        "hidden_size": 256,
        "num_layers": 2,
        "dropout": 0.3,
        "batch_size": 32,
    },
    {
        "learning_rate": 0.002,
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.2,
        "batch_size": 128,
    },
]


def format_hparams(hparams):
    """
    Quick helper to print hyperparams in a readable one-liner for the logs.
    """
    return (
        f"lr={hparams['learning_rate']}, "
        f"hidden={hparams['hidden_size']}, "
        f"layers={hparams['num_layers']}, "
        f"dropout={hparams['dropout']}, "
        f"batch={hparams['batch_size']}"
    )


def main():
    """
    Main function to run hyperparameter tuning, training, and evaluation
    for all three model architectures.
    """

    """
    Human-readable names and internal model keys for the three architectures.
    """
    model_titles = ["Vanilla RNN", "BiLSTM", "Attention RNN"]
    model_names = ["rnn", "blstm", "attention"]

    """
    Set up the logger first so everything that follows gets recorded.
    """
    logger = setup_logger()
    logger.info("Starting hyperparameter tuning pipeline")
    logger.info(f"Total configurations to try per model: {len(HYPERPARAMETER_GRID)}")

    """
    Load and prepare the dataset once — no need to reload it for every run.
    """
    data_module = NameDataModule(
        constants.DATA_DIR,
        constants.SAVE_DIR,
        logger=logger
    )

    """
    Read training names into a set for fast novelty checking during evaluation.
    """
    with open(constants.DATA_DIR, "r") as f:
        train_names = {line.strip().lower() for line in f}

    """
    Timestamp used to keep saved model files from overwriting each other.
    """
    currtime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    """
    We'll track the best config per model so we can log a summary at the end.
    """
    best_results = {}

    """
    Outer loop: one pass per model architecture.
    """
    for title, name in zip(model_titles, model_names):
        logger.info(f"\n{'='*50}")
        logger.info(f"MODEL: {title}")
        logger.info(f"{'='*50}")

        best_loss = float("inf")
        best_hparams = None
        best_model_path = None

        """
        Inner loop: try every hyperparameter combination for this model.
        """
        for run_idx, hparams in enumerate(HYPERPARAMETER_GRID):
            logger.info(f"\n  Run {run_idx + 1}/{len(HYPERPARAMETER_GRID)}")
            logger.info(f"  Config: {format_hparams(hparams)}")

            """
            Build a unique filename so each run's weights are saved separately.
            """
            run_tag = f"{name}_run{run_idx + 1}_{currtime}"
            model_path = constants.get_model_save_path(f"{run_tag}.pt")

            """
            Set up the trainer with this run's hyperparameters.
            """
            trainer = Trainer(
                data_module=data_module,
                model_name=name,
                model_save_path=model_path,
                logger=logger,
                epochs=constants.EPOCHS,
                learning_rate=hparams["learning_rate"],
                hidden_size=hparams["hidden_size"],
                num_layers=hparams["num_layers"],
                dropout=hparams["dropout"],
                batch_size=hparams["batch_size"],
            )

            """
            Train the model and get back the final loss for comparison.
            """
            final_loss = trainer.train()

            logger.info(
                f"  Run {run_idx + 1} done — "
                f"final loss: {final_loss:.4f} | "
                f"{format_hparams(hparams)}"
            )

            """
            Keep track of which config gave the lowest loss for this model.
            """
            if final_loss < best_loss:
                best_loss = final_loss
                best_hparams = hparams
                best_model_path = model_path
                logger.info(f"  *** New best for {title}: loss {best_loss:.4f} ***")

        """
        Once all runs are done, evaluate only the best-performing config.
        """
        logger.info(f"\n  Best config for {title}: {format_hparams(best_hparams)}")
        logger.info(f"  Best loss: {best_loss:.4f}")
        logger.info(f"  Running evaluation on best checkpoint...")

        evaluator = Evaluator(
            model_path=best_model_path,
            vocab_path=constants.SAVE_DIR,
            train_names=train_names,
            logger=logger
        )

        """
        Generate 1000 names and log quality metrics like novelty and diversity.
        """
        eval_results = evaluator.evaluate(num_samples=1000)

        """
        Log evaluation metrics so they show up clearly in the run log.
        """
        logger.info(f"  Evaluation results for {title} (best config):")
        if isinstance(eval_results, dict):
            for metric, value in eval_results.items():
                logger.info(f"    {metric}: {value}")
        else:
            logger.info(f"    {eval_results}")

        """
        Store the best result for this model for the final summary.
        """
        best_results[title] = {
            "best_loss": best_loss,
            "best_hparams": best_hparams,
            "eval_results": eval_results,
        }

        logger.info("\n--------------------------------------\n")

    """
    Print a clean summary at the very end so it's easy to compare all models.
    """
    logger.info("HYPERPARAMETER TUNING COMPLETE — SUMMARY")
    logger.info("=" * 50)
    for title, result in best_results.items():
        logger.info(f"\n{title}")
        logger.info(f"  Best loss    : {result['best_loss']:.4f}")
        logger.info(f"  Best config  : {format_hparams(result['best_hparams'])}")
        if isinstance(result["eval_results"], dict):
            for metric, value in result["eval_results"].items():
                logger.info(f"  {metric:15s}: {value}")

    logger.info("\nAll models trained and evaluated successfully")


if __name__ == "__main__":
    main()
