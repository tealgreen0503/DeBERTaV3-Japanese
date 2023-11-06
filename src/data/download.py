import os
import warnings
from typing import Literal

import datasets
from datasets import Dataset, DatasetDict, load_dataset

from src.data.filters import (
    is_not_empty,
    is_not_footer_header_noisy_for_oscar,
    is_valid_domain_for_oscar,
    is_valid_japanese,
    remove_empty_parenthesis,
    remove_wikipedia_footnote,
)
from src.utils import cpu_count

warnings.simplefilter("ignore", FutureWarning)


def download_dataset(
    dataset_names: list[Literal["wikipedia", "cc100", "oscar"]], seed: int = 42, is_training_tokenizer: bool = False
) -> DatasetDict:
    if is_training_tokenizer:
        dataset_names = sorted(set(dataset_names), key=dataset_names.index)

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
        for dataset_dict_ in dataset_dicts:
            # Sample 3GB of data from each train dataset
            dataset = dataset_dict_["train"]
            sample_size = 3e9 / dataset.size_in_bytes
            if sample_size < 1:
                sampled_dataset, _ = dataset.train_test_split(train_size=sample_size, shuffle=True, seed=seed).values()
                sampled_datasets.append(sampled_dataset)
            else:
                sampled_datasets.append(dataset)
        dataset_dict["train"] = datasets.concatenate_datasets(sampled_datasets, split=datasets.Split.TRAIN)
    else:
        dataset_dict["train"] = datasets.concatenate_datasets(
            [dataset_dict_["train"] for dataset_dict_ in dataset_dicts], split=datasets.Split.TRAIN
        )
        validation_datasets: list[Dataset] = []
        validation_dataset_names: set[str] = set()
        for dataset_dict_ in dataset_dicts:
            dataset = dataset_dict_["validation"]
            if dataset.info.dataset_name not in validation_dataset_names:
                validation_datasets.append(dataset)
                validation_dataset_names.add(dataset.info.dataset_name)
            else:
                del dataset
        dataset_dict["validation"] = datasets.concatenate_datasets(
            validation_datasets, split=datasets.Split.VALIDATION
        )

    return dataset_dict


def download_wikipedia(seed: int) -> DatasetDict:
    if os.path.isdir("data/filtered/wikipedia"):
        return datasets.load_from_disk("data/filtered/wikipedia")
    else:
        dataset = load_dataset(
            "wikipedia", language="ja", date="20231101", beam_runner="DirectRunner", split=datasets.Split.TRAIN
        ).select_columns("text")

        dataset = dataset.filter(is_not_empty(), batched=True, load_from_cache_file=False, num_proc=cpu_count())
        dataset = dataset.filter(is_valid_japanese(), batched=True, load_from_cache_file=False, num_proc=cpu_count())
        dataset = dataset.map(
            remove_wikipedia_footnote(), batched=True, load_from_cache_file=False, num_proc=cpu_count()
        )
        dataset = dataset.map(
            remove_empty_parenthesis(), batched=True, load_from_cache_file=False, num_proc=cpu_count()
        )

        dataset_dict = DatasetDict()
        dataset_dict["train"], dataset_dict["validation"] = dataset.train_test_split(
            test_size=5000, seed=seed
        ).values()

        dataset_dict.save_to_disk("data/filtered/wikipedia", num_proc=cpu_count())
    return dataset_dict


def download_cc100(seed: int) -> DatasetDict:
    if os.path.isdir("data/filtered/cc100"):
        return datasets.load_from_disk("data/filtered/cc100")
    else:
        dataset = load_dataset("range3/cc100-ja", split=datasets.Split.TRAIN, num_proc=cpu_count()).select_columns(
            "text"
        )

        dataset = dataset.filter(is_not_empty(), batched=True, load_from_cache_file=False, num_proc=cpu_count())
        dataset = dataset.filter(is_valid_japanese(), batched=True, load_from_cache_file=False, num_proc=cpu_count())

        dataset_dict = DatasetDict()
        dataset_dict["train"], dataset_dict["validation"] = dataset.train_test_split(
            test_size=5000, seed=seed
        ).values()

        dataset_dict.save_to_disk("data/filtered/cc100", num_proc=cpu_count())
    return dataset_dict


def download_oscar(seed: int) -> DatasetDict:
    if os.path.isdir("data/filtered/oscar"):
        return datasets.load_from_disk("data/filtered/oscar")
    else:
        from dotenv import load_dotenv

        load_dotenv()

        dataset = load_dataset(
            "oscar-corpus/OSCAR-2301",
            language="ja",
            split=datasets.Split.TRAIN,
            token=os.getenv("HUGGINGFACE_ACCESS_TOKEN", None),
            num_proc=cpu_count(),
        ).select_columns(["text", "meta"])

        dataset = dataset.filter(
            is_not_footer_header_noisy_for_oscar(), batched=True, load_from_cache_file=False, num_proc=cpu_count()
        )
        dataset = dataset.filter(
            is_valid_domain_for_oscar(), batched=True, load_from_cache_file=False, num_proc=cpu_count()
        )
        dataset = dataset.remove_columns("meta")
        dataset = dataset.filter(is_not_empty(), batched=True, load_from_cache_file=False, num_proc=cpu_count())
        dataset = dataset.filter(is_valid_japanese(), batched=True, load_from_cache_file=False, num_proc=cpu_count())

        dataset_dict = DatasetDict()
        dataset_dict["train"], dataset_dict["validation"] = dataset.train_test_split(
            test_size=5000, seed=seed
        ).values()

        dataset_dict.save_to_disk("data/filtered/oscar", num_proc=cpu_count())
    return dataset_dict
