import os
from collections.abc import Generator
from typing import Literal

import datasets
from datasets import Dataset, DatasetDict, load_dataset


def download_dataset(
    dataset_names: list[Literal["wikipedia", "wiki40b", "cc100", "oscar"]], unique: bool = False, seed: int = 42
) -> DatasetDict:
    if unique:
        dataset_names = list(set(dataset_names))
    dataset_dicts: list[DatasetDict] = []
    for dataset_name in dataset_names:
        match dataset_name:
            case "wikipedia":
                dataset_dicts.append(download_wikipedia(seed))
            case "wiki40b":
                dataset_dicts.append(download_wiki40b())
            case "cc100":
                dataset_dicts.append(download_cc100(seed))
            case "oscar":
                dataset_dicts.append(download_oscar(seed))
            case _:
                raise Exception(f"Unsupported dataset: {dataset_name}")
    dataset_dict = DatasetDict()
    for split in ["train", "validation", "test"]:
        dataset_dict[split] = datasets.concatenate_datasets(
            [dataset_dict[split] for dataset_dict in dataset_dicts], split=datasets.Split(split)
        )
    return dataset_dict


def download_wikipedia(seed: int) -> DatasetDict:
    if os.path.isdir("data/raw/wikipedia"):
        return datasets.load_from_disk("data/raw/wikipedia")
    else:
        dataset = load_dataset("wikipedia", language="ja", date="20231001", beam_runner="DirectRunner").remove_columns(
            ["id", "url", "title"]
        )
        dataset_dict = DatasetDict()
        dataset_dict["train"], valid_test_dataset = dataset.train_test_split(test_size=0.1, seed=seed)
        dataset_dict["validation"], dataset_dict["test"] = valid_test_dataset.train_test_split(
            test_size=0.5, seed=seed
        )
        dataset_dict.save_to_disk("data/raw/wikipedia")
    return dataset_dict


def download_wiki40b() -> DatasetDict:
    def tf_wiki40b_generator(split: str) -> Generator[dict[str, str], None, None]:
        import tensorflow_datasets as tfds

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
                            yield {"text": line.replace("_NEWLINE_", "")}

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
    if os.path.isdir("data/raw/cc100"):
        return datasets.load_from_disk("data/raw/cc100")
    else:
        dataset = load_dataset("cc100", lang="ja").remove_columns("id")
        dataset_dict = DatasetDict()
        dataset_dict["train"], valid_test_dataset = dataset.train_test_split(test_size=0.1, seed=seed)
        dataset_dict["validation"], dataset_dict["test"] = valid_test_dataset.train_test_split(
            test_size=0.5, seed=seed
        )
        dataset_dict.save_to_disk("data/raw/cc100")
    return dataset_dict


def download_oscar(seed: int) -> DatasetDict:
    if os.path.isdir("data/raw/oscar"):
        return datasets.load_from_disk("data/raw/oscar")
    else:
        dataset = load_dataset("oscar-corpus/OSCAR-2301", language="ja")
        dataset = dataset.filter(
            lambda x: len(set(x["meta"]["quality_warnings"]) & {"header", "footer", "noisy"}) == 0
        )
        dataset = dataset.remove_columns(["id", "meta"])
        dataset_dict = DatasetDict()
        dataset_dict["train"], valid_test_dataset = dataset.train_test_split(test_size=0.1, seed=seed)
        dataset_dict["validation"], dataset_dict["test"] = valid_test_dataset.train_test_split(
            test_size=0.5, seed=seed
        )
        dataset_dict.save_to_disk("data/raw/oscar")
    return dataset_dict
