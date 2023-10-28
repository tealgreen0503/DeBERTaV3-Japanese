import argparse
import os
import shutil
from pathlib import Path
from typing import Any

import datasets
import pysbd
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

from src.data import batch_preprocess, download_dataset
from src.models import DebertaV3ForPreTraining
from src.utils import cpu_count


def train_model(config: dict[str, Any]) -> None:
    load_dotenv()

    config_discriminator = DebertaV2Config(**config["model"]["discriminator"])
    config_generator = DebertaV2Config(**config["model"]["generator"])

    tokenizer = DebertaV2TokenizerFast.from_pretrained(Path("models") / config["model_name"])
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
    text_segmenter = pysbd.Segmenter(language="ja", clean=False, char_span=True)

    if os.path.isdir("data/encoded"):
        dataset_dict = datasets.load_from_disk("data/encoded")
    else:
        dataset_dict = download_dataset(config["dataset_names"], seed=config["seed"])
        dataset_dict = dataset_dict.map(
            batch_preprocess,
            batched=True,
            remove_columns="text",
            fn_kwargs={"tokenizer": tokenizer, "text_segmenter": text_segmenter},
            num_proc=cpu_count(),
        )
        dataset_dict.save_to_disk("data/encoded", num_proc=cpu_count())

    model = DebertaV3ForPreTraining._from_config(
        config=config_discriminator,
        config_generator=config_generator,
        torch_dtype=getattr(torch, config["model"]["torch_dtype"]),
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer)

    tmp_dir = Path("tmp")
    training_args = TrainingArguments(output_dir=tmp_dir, dataloader_num_workers=cpu_count(), **config["trainer"])
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
    args = parser.parse_args()
    with Path(args.config_file).open(mode="r") as f:
        config = yaml.safe_load(f)

    train_model(config)


if __name__ == "__main__":
    main()
