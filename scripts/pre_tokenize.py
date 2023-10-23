import argparse
import gc
import os
import re
from pathlib import Path
from typing import Any

import pysbd
import sudachipy
import tokenizers
import yaml
from tokenizers import Regex
from tqdm import tqdm

from src.data import download_dataset

DELIMITER = "｜"  # noqa: RUF001  # FULLWIDTH VERTICAL LINE


def save_pre_tokenized_text(config: dict[str, Any]) -> None:
    tokenizer = sudachipy.Dictionary().create()
    segmenter = pysbd.Segmenter(language="ja", clean=False)
    max_bytes = config["sentencepiece"]["max_sentence_length"] + len(DELIMITER.encode())

    dataset_dict = download_dataset(config["dataset_names"], seed=config["seed"], is_training_tokenizer=True)

    os.makedirs("data/pre_tokenized", exist_ok=True)
    with open("data/pre_tokenized/train.txt", "w") as f:
        for example in tqdm(dataset_dict["train"]):
            text = ""
            text_bytes = 0
            for _sentence in segmenter.segment(preprocess_text(example["text"])):
                try:
                    sentence = pre_tokenize(_sentence, tokenizer) + DELIMITER
                except Exception:
                    if text != "":
                        f.write(text.removesuffix(DELIMITER) + "\n")
                        text = ""
                        text_bytes = 0
                        continue
                    else:
                        # text = ""
                        # text_bytes = 0
                        continue
                finally:
                    sentence_bytes = len(sentence.encode())
                    if text_bytes + sentence_bytes < max_bytes:
                        text += sentence
                        text_bytes += sentence_bytes
                        continue
                    else:
                        if text != "":
                            f.write(text.removesuffix(DELIMITER) + "\n")
                            if sentence_bytes < max_bytes:
                                text = sentence
                                text_bytes = sentence_bytes
                                continue
                            else:
                                # ignore too long sentence
                                text = ""
                                text_bytes = 0
                                continue
                        else:
                            # text = ""
                            # text_bytes = 0
                            continue
        if text != "":
            f.write(text.removesuffix(DELIMITER) + "\n")

    del dataset_dict, tokenizer
    gc.collect()


def preprocess_text(text: str) -> str:
    text = tokenizers.normalizers.Sequence(
        [
            tokenizers.normalizers.Strip(),
            tokenizers.normalizers.Nmt(),
            tokenizers.normalizers.NFKC(),
            tokenizers.normalizers.Replace(Regex(" {2,}"), " "),
        ]
    ).normalize_str(text)
    return text


def pre_tokenize(text: str, tokenizer: sudachipy.Tokenizer) -> str:
    """split by sudachi and space"""
    return DELIMITER.join(
        [
            DELIMITER.join(re.split("( )", m.surface())).removeprefix(DELIMITER).removesuffix(DELIMITER)
            for m in tokenizer.tokenize(text, mode=sudachipy.SplitMode.A)
        ]
    ).lower()  # lowercase after pre tokenization


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)
    args = parser.parse_args()
    with Path(args.config_file).open(mode="r") as f:
        config = yaml.safe_load(f)

    save_pre_tokenized_text(config)


if __name__ == "__main__":
    main()
