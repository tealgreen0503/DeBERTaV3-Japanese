import os
from collections.abc import Generator
from typing import Literal

import datasets
import tensorflow_datasets as tfds
from datasets import Dataset, DatasetDict


def download_dataset(dataset_names: list[Literal["wiki40b", "cc100", "oscar"]], seed: int = 42) -> DatasetDict:
    dataset_dicts: list[DatasetDict] = []
    for dataset_name in dataset_names:
        if "wiki40b" in dataset_name:
            dataset_dicts.append(download_wiki40b())
        elif "cc100" in dataset_name:
            dataset_dicts.append(download_cc100(seed=seed))
        elif "oscar" in dataset_name:
            dataset_dicts.append(download_oscar(seed=seed))
        else:
            raise Exception(f"Unsupported dataset: {dataset_name}")
    dataset_dict = DatasetDict()
    for split in ["train", "validation", "test"]:
        dataset_dict[split] = datasets.concatenate_datasets(
            [dataset_dict[split] for dataset_dict in dataset_dicts], split=datasets.Split(split)
        )
    return dataset_dict


def download_wiki40b() -> DatasetDict:
    def tf_wiki40b_generator(split: str) -> Generator[str, None, None]:
        tf_dataset = tfds.load("wiki40b/ja", split=split)
        for aritcle in tf_dataset.as_numpy_iterator():
            is_paragraph = False
            for line in aritcle["text"].decode().split("\n"):
                match line:
                    case "_START_ARTICLE_":
                        is_paragraph = False
                    case "_START_SECTION_":
                        is_paragraph = False
                    case "_START_PARAGRAPH_":
                        is_paragraph = True
                    case _:
                        if is_paragraph and len(line) > 0:
                            yield {"text": line}

    if os.path.isdir("data/raw/wiki40b"):
        return datasets.load_from_disk("data/raw/wiki40b")
    else:
        dataset_dict: DatasetDict = DatasetDict()
        for split in ["train", "validation", "test"]:
            dataset_dict[split] = Dataset.from_generator(
                tf_wiki40b_generator, gen_kwargs={"split": datasets.Split(split)}
            )
        dataset_dict.save_to_disk("data/raw/wiki40b")
    return dataset_dict


def download_cc100(seed: int) -> DatasetDict:
    def tf_cc100_generator() -> Generator[str, None, None]:
        tf_dataset = tfds.load("huggingface:cc100/lang=ja")
        for sample in tf_dataset.as_numpy_iterator():
            for line in sample["text"].decode().split("\n"):
                if len(line) > 0:
                    yield {"text": line}

    if os.path.isdir("data/raw/cc100"):
        return datasets.load_from_disk("data/raw/cc100")
    else:
        dataset = Dataset.from_generator(tf_cc100_generator)
        dataset_dict: DatasetDict = DatasetDict()
        dataset_dict["train"], valid_test_dataset = dataset.train_test_split(test_size=0.1, seed=seed)
        dataset_dict["validation"], dataset_dict["test"] = valid_test_dataset.train_test_split(
            test_size=0.5, seed=seed
        )
        dataset_dict.save_to_disk("data/raw/cc100")
    return dataset_dict


def download_oscar(seed: int) -> DatasetDict:
    def tf_oscar_generator() -> Generator[str, None, None]:
        tf_dataset = tfds.load("oscar/ja")
        for sample in tf_dataset.as_numpy_iterator():
            for line in sample["text"].decode().split("\n"):
                if len(line) > 0:
                    yield {"text": line}

    if os.path.isdir("data/raw/oscar"):
        return datasets.load_from_disk("data/raw/oscar")
    else:
        dataset = Dataset.from_generator(tf_oscar_generator)
        dataset_dict: DatasetDict = DatasetDict()
        dataset_dict["train"], valid_test_dataset = dataset.train_test_split(test_size=0.1, seed=seed)
        dataset_dict["validation"], dataset_dict["test"] = valid_test_dataset.train_test_split(
            test_size=0.5, seed=seed
        )
        dataset_dict.save_to_disk("data/raw/oscar")
    return dataset_dict
