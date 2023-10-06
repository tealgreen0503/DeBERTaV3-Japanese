import argparse
from pathlib import Path

import datasets
import yaml
from dotenv import load_dotenv

from scripts import save_pre_tokenized_text, train_model, train_tokenizer

if __name__ == "__main__":
    # Use save_to_disk() and load_from_disk() instead of using the cache
    datasets.disable_caching()
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--pre_tokenize", action="store_true")
    parser.add_argument("--train_tokenizer", action="store_true")
    parser.add_argument("--train_model", action="store_true")
    args = parser.parse_args()

    with (Path("config") / f"{args.model_name}.yaml").open(mode="r") as f:
        config = yaml.safe_load(f)

    save_pre_tokenized_text(config)
    train_tokenizer(config)
    train_model(config)
