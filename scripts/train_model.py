import argparse
import os
import tempfile
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

import wandb
from src.data import batch_preprocess, download_dataset
from src.models import DebertaV3ForPreTraining
from src.utils import cpu_count


def train_model(config: dict[str, Any], resume_from_run_id: str | None = None, debug: bool = False) -> None:
    load_dotenv()

    discriminator_config = DebertaV2Config(**config["model"]["discriminator"])
    generator_config = DebertaV2Config(**config["model"]["generator"])

    tokenizer = DebertaV2TokenizerFast.from_pretrained("models/tokenizer")
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
            fn_kwargs={"tokenizer": tokenizer, "text_segmenter": text_segmenter, "max_length": config["max_length"]},
            load_from_cache_file=False,
            num_proc=cpu_count(),
        )
        dataset_dict.save_to_disk("data/encoded", num_proc=cpu_count())
    if debug:
        dataset_dict["train"] = dataset_dict["train"].select(range(32000))
        dataset_dict["validation"] = dataset_dict["validation"].select(range(1600))

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, **config["data_collator"])

    torch_dtype = config["model"].get("torch_dtype", None)
    model = DebertaV3ForPreTraining._from_config(
        config=discriminator_config,
        generator_config=generator_config,
        torch_dtype=getattr(torch, torch_dtype) if torch_dtype is not None else None,
    )

    if resume_from_run_id is not None:
        wandb.init(id=resume_from_run_id, name=config["model_name"] + f"-{resume_from_run_id}", resume="must")
        checkpoint_dir = Path("checkpoints") / config["model_name"] / f"run-{resume_from_run_id}"
    else:
        run_id = wandb.util.generate_id()
        wandb.init(id=run_id, name=config["model_name"] + f"-{run_id}")
        checkpoint_dir = Path("checkpoints") / config["model_name"] / f"run-{run_id}"

    training_args = TrainingArguments(
        output_dir=checkpoint_dir, dataloader_num_workers=cpu_count(), **config["trainer"]
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["validation"],
        tokenizer=tokenizer,
        preprocess_logits_for_metrics=lambda logits, labels: torch.softmax(logits, dim=-1),
    )

    if resume_from_run_id is not None:
        trainer.train(resume_from_checkpoint=checkpoint_dir)
    else:
        trainer.train()

    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer.model.save_pretrained(tmp_dir)
        model = DebertaV3ForPreTraining.from_pretrained(tmp_dir)
        model.save_pretrained(Path("models") / config["model_name"])
        tokenizer.save_pretrained(Path("models") / config["model_name"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--resume_run_id", type=str, default=None)
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()

    with Path(args.config_file).open(mode="r") as f:
        config = yaml.safe_load(f)

    train_model(config, resume_from_run_id=args.resume_run_id, debug=args.debug)


if __name__ == "__main__":
    main()
