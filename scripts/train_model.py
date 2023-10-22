import argparse
import os
import shutil
from pathlib import Path
from typing import Any

import datasets
import torch
import yaml
from dotenv import load_dotenv
from transformers import (
    DataCollatorForLanguageModeling,
    DebertaV2Config,
    DebertaV2TokenizerFast,
    Trainer,
    TrainingArguments,
)

from src.data import download_dataset
from src.models import DebertaV3ForPreTraining

# Use save_to_disk() and load_from_disk() instead of using the cache
datasets.disable_caching()


def train_model(config: dict[str, Any], local_rank: int = -1) -> None:
    load_dotenv()
    config_discriminator = DebertaV2Config(**config["model"]["discriminator"])
    config_generator = DebertaV2Config(**config["model"]["generator"])

    tokenizer = DebertaV2TokenizerFast.from_pretrained(Path("models") / config["model_name"])

    if os.path.isdir("data/encoded"):
        dataset_dict = datasets.load_from_disk("data/encoded")
    else:
        dataset_dict = download_dataset(config["dataset_names"], seed=config["seed"])
        dataset_dict = dataset_dict.map(
            lambda x: tokenizer(x["text"], truncation=True, return_overflowing_tokens=True),
            batched=True,
            remove_columns="text",
        )
        dataset_dict.save_to_disk("data/encoded")

    model = DebertaV3ForPreTraining._from_config(
        config=config_discriminator,
        config_generator=config_generator,
        torch_dtype=getattr(torch, config["model"]["torch_dtype"]),
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer)

    tmp_dir = Path("tmp")
    training_args = TrainingArguments(output_dir=tmp_dir, local_rank=local_rank, **config["trainer"])
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["validation"],
        tokenizer=tokenizer,
        preprocess_logits_for_metrics=lambda logits, labels: torch.softmax(logits, dim=-1),
    )
    trainer.train()

    shutil.rmtree(tmp_dir)
    save_path = Path("models") / config["model_name"]
    os.makedirs(save_path, exist_ok=True)
    trainer.save_model(save_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()
    with Path(args.config_file).open(mode="r") as f:
        config = yaml.safe_load(f)

    train_model(config, local_rank=args.local_rank)


if __name__ == "__main__":
    main()
