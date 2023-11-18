import argparse
import os
import re
import unicodedata
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import pysbd
import sudachipy
import yaml
from tokenizers.normalizers import Nmt
from tqdm import tqdm

from src.data import download_dataset
from src.utils import cpu_count

DELIMITER = "<dlm>"


def save_pre_tokenized_text(config: dict[str, Any]) -> None:
    segmenter = pysbd.Segmenter(language="ja", clean=False)
    max_bytes = config["sentencepiece"]["max_sentence_length"] + len(DELIMITER.encode())

    dataset_dict = download_dataset(config["dataset_names"], seed=config["seed"], is_training_tokenizer=True)
    raw_texts = dataset_dict["train"]["text"]

    os.makedirs("data/pre_tokenized", exist_ok=True)
    with open("data/pre_tokenized/train.txt", "w") as f:
        with Pool(cpu_count()) as pool:
            for pre_tokenized_texts in tqdm(
                pool.imap(partial(pre_tokenize_process, segmenter=segmenter, max_bytes=max_bytes), raw_texts),
                total=len(raw_texts),
            ):
                for pre_tokenized_text in pre_tokenized_texts:
                    f.write(pre_tokenized_text + "\n")


def pre_tokenize_process(raw_text: str, segmenter: pysbd.Segmenter, max_bytes: int) -> list[str]:
    # Create Sudachi Tokenizer here because it cannot be serialized.
    # Disable IgnoreYomiganaPlugin by specifying config_path.
    tokenizer = sudachipy.Dictionary(config_path="config/sudachi.json").create(mode=sudachipy.SplitMode.A)

    pre_tokenized_texts = []
    text = ""
    text_bytes = 0
    for sentence in segmenter.segment(preprocess(raw_text)):
        try:
            sentence = pre_tokenize(sentence, tokenizer) + DELIMITER
        except Exception:
            if text != "":
                pre_tokenized_texts.append(text.removesuffix(DELIMITER))
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
                    pre_tokenized_texts.append(text.removesuffix(DELIMITER))
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
        pre_tokenized_texts.append(text.removesuffix(DELIMITER))

    return pre_tokenized_texts


nmt_normalize = Nmt().normalize_str
nfkc_normalize = partial(unicodedata.normalize, "NFKC")
space_normalize = partial(re.compile(r" {2,}").sub, " ")


def preprocess(text: str) -> str:
    return space_normalize(nfkc_normalize(nmt_normalize(text.strip())))


def pre_tokenize(text: str, tokenizer: sudachipy.Tokenizer) -> str:
    """Split by Sudachi tokenizer and whitespace"""
    return DELIMITER.join([split_by_space(m.surface()) for m in tokenizer.tokenize(text.strip())])


def split_by_space(text: str) -> str:
    return text[0] + text[1:-1].replace(" ", f"{DELIMITER} {DELIMITER}") + text[-1] if " " in text[1:-1] else text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)
    args = parser.parse_args()
    with Path(args.config_file).open(mode="r") as f:
        config = yaml.safe_load(f)

    save_pre_tokenized_text(config)


if __name__ == "__main__":
    main()
