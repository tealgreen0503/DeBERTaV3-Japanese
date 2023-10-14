import argparse
import gc
import os
import re
from collections.abc import Generator
from pathlib import Path

import datasets
import sudachipy
import tokenizers
import yaml
from tokenizers import Regex
from tqdm import tqdm

from src.data import download_dataset


def save_pre_tokenized_text(config: dict) -> None:
    tokenizer = sudachipy.Dictionary().create()
    dataset_dict = download_dataset(config["dataset_names"], seed=config["seed"])
    os.makedirs("data/pre_tokenized", exist_ok=True)
    with open("data/pre_tokenized/train.txt", "w") as f:
        for example in tqdm(dataset_dict["train"]):
            for sentence in split_long_sentence(preprocess_text(example["text"])):
                f.write(pre_tokenize(sentence, tokenizer) + "\n")

    del dataset_dict, tokenizer
    gc.collect()


def preprocess_text(text: str) -> str:
    text = text.replace("_NEWLINE_", "")
    text = tokenizers.normalizers.Sequence(
        [
            tokenizers.normalizers.Strip(),
            tokenizers.normalizers.Nmt(),
            tokenizers.normalizers.NFKC(),
            tokenizers.normalizers.Replace(Regex(" {2,}"), " "),
        ]
    ).normalize_str(text)
    return text


def split_long_sentence(text: str) -> Generator[str, None, None]:
    max_length = 2048

    start_index = 0
    end_index = max_length
    while start_index < len(text):
        if len(text) < end_index:
            yield text[start_index:]
            break

        split_point = text.rfind("。", start_index, end_index)
        if split_point == -1:
            yield text[start_index:end_index]
            start_index = end_index
            end_index = start_index + max_length
        else:
            yield text[start_index : split_point + 1]
            start_index = split_point + 1
            end_index = start_index + max_length


def pre_tokenize(text: str, tokenizer: sudachipy.Tokenizer) -> str:
    """split by sudachi and space"""
    return "<|dlm|>".join(
        [
            "<|dlm|>".join(re.split("( )", m.surface())).removeprefix("<|dlm|>").removesuffix("<|dlm|>")
            for m in tokenizer.tokenize(text, mode=sudachipy.SplitMode.A)
        ]
    ).lower()


def main() -> None:
    # Use save_to_disk() and load_from_disk() instead of using the cache
    datasets.disable_caching()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)
    args = parser.parse_args()
    with Path(args.config_file).open(mode="r") as f:
        config = yaml.safe_load(f)

    save_pre_tokenized_text(config)


if __name__ == "__main__":
    main()
