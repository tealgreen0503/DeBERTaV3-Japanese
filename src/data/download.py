import logging
import os
from typing import Literal

import datasets
from datasets import Dataset, DatasetDict, load_dataset

from src.data.filters import (
    extract_japanese_text,
    is_good_compression_ratio,
    is_japanese,
    is_not_ad_content,
    is_not_empty,
    is_not_footer_header_noisy_oscar,
    is_valid_domain,
    remove_empty_parenthesis,
    remove_wikipedia_footnote,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def download_dataset(
    dataset_names: list[Literal["wikipedia", "cc100", "oscar"]], seed: int = 42, is_training_tokenizer: bool = False
) -> DatasetDict:
    if is_training_tokenizer:
        dataset_names = list(set(dataset_names))

    dataset_dicts: list[DatasetDict] = []
    for dataset_name in dataset_names:
        match dataset_name:
            case "wikipedia":
                dataset_dicts.append(download_wikipedia(seed))
            case "cc100":
                dataset_dicts.append(download_cc100(seed))
            case "oscar":
                dataset_dicts.append(download_oscar(seed))
            case _:
                raise ValueError(f"Unsupported dataset: {dataset_name}")

    dataset_dict = DatasetDict()
    if is_training_tokenizer:
        sampled_datasets: list[Dataset] = []
        for _dataset_dict in dataset_dicts:
            # Sample 1GB of data from each train dataset
            dataset = _dataset_dict["train"]
            sample_size = 1e9 / dataset.size_in_bytes
            assert sample_size <= 1
            sampled_dataset, _ = dataset.train_test_split(train_size=sample_size, shuffle=True, seed=seed).values()
            sampled_datasets.append(sampled_dataset)
        dataset_dict["train"] = datasets.concatenate_datasets(sampled_datasets, split=datasets.Split.TRAIN)
    else:
        for split in ["train", "validation", "test"]:
            dataset_dict[split] = datasets.concatenate_datasets(
                [dataset_dict[split] for dataset_dict in dataset_dicts], split=datasets.Split(split)
            )
    return dataset_dict


def download_wikipedia(seed: int) -> DatasetDict:
    if os.path.isdir("data/filtered/wikipedia"):
        return datasets.load_from_disk("data/filtered/wikipedia")
    else:
        dataset = load_dataset(
            "wikipedia", language="ja", date="20231020", beam_runner="DirectRunner", split=datasets.Split.TRAIN
        ).remove_columns(["id", "url", "title"])
        logger.info(f"Completed downloading {dataset.info.dataset_name}: size={dataset.size_in_bytes / 1e9:.2f}GB")

        dataset = dataset.filter(is_not_empty())
        dataset = dataset.map(remove_wikipedia_footnote(), batched=True)
        dataset = dataset.map(remove_empty_parenthesis(), batched=True)

        dataset_dict = DatasetDict()
        dataset_dict["train"], valid_test_dataset = dataset.train_test_split(test_size=0.1, seed=seed).values()
        dataset_dict["validation"], dataset_dict["test"] = valid_test_dataset.train_test_split(
            test_size=0.5, seed=seed
        ).values()

        dataset_dict.save_to_disk("data/filtered/wikipedia")
    return dataset_dict


def download_cc100(seed: int) -> DatasetDict:
    if os.path.isdir("data/filtered/cc100"):
        return datasets.load_from_disk("data/filtered/cc100")
    else:
        dataset = load_dataset("cc100", lang="ja", split=datasets.Split.TRAIN, streaming=True).remove_columns("id")
        logger.info(f"Completed downloading {dataset.info.dataset_name}: size={dataset.size_in_bytes / 1e9:.2f}GB")

        dataset = dataset.filter(is_not_empty())
        dataset = dataset.filter(is_japanese())
        dataset = dataset.filter(is_not_ad_content())
        dataset = dataset.filter(is_good_compression_ratio())
        dataset = dataset.map(extract_japanese_text(), batched=True)

        dataset_dict = DatasetDict()
        dataset_dict["train"], valid_test_dataset = dataset.train_test_split(test_size=0.1, seed=seed).values()
        dataset_dict["validation"], dataset_dict["test"] = valid_test_dataset.train_test_split(
            test_size=0.5, seed=seed
        ).values()

        dataset_dict.save_to_disk("data/filtered/cc100")
    return dataset_dict


def download_oscar(seed: int) -> DatasetDict:
    if os.path.isdir("data/filtered/oscar"):
        return datasets.load_from_disk("data/filtered/oscar")
    else:
        dataset = load_dataset(
            "oscar-corpus/OSCAR-2301", language="ja", split=datasets.Split.TRAIN, streaming=True
        ).remove_columns("id")
        logger.info(f"Completed downloading {dataset.info.dataset_name}: size={dataset.size_in_bytes / 1e9:.2f}GB")

        dataset = dataset.filter(is_not_empty())
        dataset = dataset.filter(is_japanese())
        dataset = dataset.filter(is_not_footer_header_noisy_oscar())
        dataset = dataset.filter(is_valid_domain("oscar"))
        dataset = dataset.filter(is_not_ad_content())
        dataset = dataset.filter(is_good_compression_ratio())
        dataset = dataset.remove_columns(["meta"])
        dataset = dataset.map(extract_japanese_text(), batched=True)

        dataset_dict = DatasetDict()
        dataset_dict["train"], valid_test_dataset = dataset.train_test_split(test_size=0.1, seed=seed).values()
        dataset_dict["validation"], dataset_dict["test"] = valid_test_dataset.train_test_split(
            test_size=0.5, seed=seed
        ).values()

        dataset_dict.save_to_disk("data/filtered/oscar")
    return dataset_dict
