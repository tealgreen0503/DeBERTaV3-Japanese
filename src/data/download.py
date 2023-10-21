import os
from typing import Literal

import datasets
from datasets import DatasetDict, load_dataset

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


def download_dataset(
    dataset_names: list[Literal["wikipedia", "cc100", "oscar", "mc4"]], unique: bool = False, seed: int = 42
) -> DatasetDict:
    if unique:
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
            case "mc4":
                dataset_dicts.append(download_mc4(seed))
            case _:
                raise Exception(f"Unsupported dataset: {dataset_name}")
    dataset_dict = DatasetDict()
    for split in ["train", "validation", "test"]:
        dataset_dict[split] = datasets.concatenate_datasets(
            [dataset_dict[split] for dataset_dict in dataset_dicts], split=datasets.Split(split)
        )
    return dataset_dict


def download_wikipedia(seed: int) -> DatasetDict:
    if os.path.isdir("data/filtered/wikipedia"):
        return datasets.load_from_disk("data/filtered/wikipedia")
    else:
        dataset = load_dataset("wikipedia", language="ja", date="20231001", beam_runner="DirectRunner").remove_columns(
            ["id", "url", "title"]
        )

        dataset = dataset.filter(is_not_empty())
        dataset = dataset.map(remove_wikipedia_footnote())
        dataset = dataset.map(remove_empty_parenthesis())

        dataset_dict = DatasetDict()
        dataset_dict["train"], valid_test_dataset = dataset.train_test_split(test_size=0.1, seed=seed)
        dataset_dict["validation"], dataset_dict["test"] = valid_test_dataset.train_test_split(
            test_size=0.5, seed=seed
        )

        dataset_dict.save_to_disk("data/filtered/wikipedia")
    return dataset_dict


def download_cc100(seed: int) -> DatasetDict:
    if os.path.isdir("data/filtered/cc100"):
        return datasets.load_from_disk("data/filtered/cc100")
    else:
        dataset = load_dataset("cc100", lang="ja").remove_columns("id")

        dataset = dataset.filter(is_not_empty())
        dataset = dataset.filter(is_japanese())
        dataset = dataset.filter(is_not_ad_content())
        dataset = dataset.filter(is_good_compression_ratio())
        dataset = dataset.map(extract_japanese_text())

        dataset_dict = DatasetDict()
        dataset_dict["train"], valid_test_dataset = dataset.train_test_split(test_size=0.1, seed=seed)
        dataset_dict["validation"], dataset_dict["test"] = valid_test_dataset.train_test_split(
            test_size=0.5, seed=seed
        )

        dataset_dict.save_to_disk("data/filtered/cc100")
    return dataset_dict


def download_oscar(seed: int) -> DatasetDict:
    if os.path.isdir("data/filtered/oscar"):
        return datasets.load_from_disk("data/filtered/oscar")
    else:
        dataset = load_dataset("oscar-corpus/OSCAR-2301", language="ja").remove_columns("id")

        dataset = dataset.filter(is_not_empty())
        dataset = dataset.filter(is_japanese())
        dataset = dataset.filter(is_not_footer_header_noisy_oscar())
        dataset = dataset.filter(is_valid_domain("oscar"))
        dataset = dataset.filter(is_not_ad_content())
        dataset = dataset.filter(is_good_compression_ratio())
        dataset = dataset.map(extract_japanese_text())
        dataset = dataset.remove_columns(["meta"])

        dataset_dict = DatasetDict()
        dataset_dict["train"], valid_test_dataset = dataset.train_test_split(test_size=0.1, seed=seed)
        dataset_dict["validation"], dataset_dict["test"] = valid_test_dataset.train_test_split(
            test_size=0.5, seed=seed
        )

        dataset_dict.save_to_disk("data/filtered/oscar")
    return dataset_dict


def download_mc4(seed: int) -> DatasetDict:
    if os.path.isdir("data/filtered/mc4"):
        return datasets.load_from_disk("data/filtered/mc4")
    else:
        dataset = load_dataset("mc4", "ja").remove_columns("timestamp")

        dataset = dataset.filter(is_not_empty())
        dataset = dataset.filter(is_japanese())
        dataset = dataset.filter(is_valid_domain("mc4"))
        dataset = dataset.filter(is_not_ad_content())
        dataset = dataset.filter(is_good_compression_ratio())
        dataset = dataset.map(extract_japanese_text())
        dataset = dataset.remove_columns(["url"])

        dataset_dict = DatasetDict()
        dataset_dict["train"], valid_test_dataset = dataset.train_test_split(test_size=0.1, seed=seed)
        dataset_dict["validation"], dataset_dict["test"] = valid_test_dataset.train_test_split(
            test_size=0.5, seed=seed
        )

        dataset_dict.save_to_disk("data/filtered/mc4")
    return dataset_dict
