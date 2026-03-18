import torch
import random
from src.generator import NameGenerator


class Evaluator:

    def __init__(self, model_path, vocab_path, train_names, logger, device="cpu"):

        self.device = device
        self.logger = logger

        self.model = torch.load(
            model_path,
            map_location=device,
            weights_only=False
        )

        self.generator = NameGenerator(self.model, vocab_path, device)

        self.train_names = set(train_names)

        # prefixes to encourage better generation
        self.prefixes = [
            "a", "an", "ar",
            "sh", "pr", "ra",
            "vi", "ka", "de",
            "ni", "sa"
        ]

    def evaluate(self, num_samples=1000):

        generated = []

        attempts = 0
        max_attempts = num_samples * 3

        while len(generated) < num_samples and attempts < max_attempts:

            prefix = random.choice(self.prefixes)

            name = self.generator.generate(prefix=prefix)

            attempts += 1

            if name and len(name) > 2:
                generated.append(name)

        if len(generated) == 0:
            self.logger.error("No names generated.")
            return 0.0, 0.0

        unique_generated = set(generated)

        novelty = sum(name not in self.train_names for name in generated) / len(generated)

        diversity = len(unique_generated) / len(generated)

        self.logger.info(f"Generated names: {len(generated)}")
        self.logger.info(f"Unique names: {len(unique_generated)}")

        self.logger.info(f"Novelty Rate: {novelty:.4f}")
        self.logger.info(f"Diversity: {diversity:.4f}")

        self.logger.info("Sample generated names:")

        samples = random.sample(list(unique_generated), min(10, len(unique_generated)))

        for name in samples:
            self.logger.info(name)

        self.logger.info("\n--------------------------------------\n")

        return novelty, diversity
