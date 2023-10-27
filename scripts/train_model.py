import argparse
import bisect
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
    BatchEncoding,
    DataCollatorForLanguageModeling,
    DebertaV2Config,
    DebertaV2TokenizerFast,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
)

from src.data import download_dataset
from src.models import DebertaV3ForPreTraining


def batch_preprocess(
    batch_example: dict[str, list[Any]],
    tokenizer: PreTrainedTokenizerFast,
    text_segmenter: pysbd.Segmenter,
    max_length: int = 512,
) -> dict[str, list[Any]]:
    batch_encoding = tokenizer.batch_encode_plus(
        batch_example["text"], add_special_tokens=False, padding=False, truncation=False
    )
    batch_encoding = batch_segment_text_into_sentences(
        batch_encoding, batch_example["text"], tokenizer, text_segmenter, max_length=max_length
    )
    return batch_encoding


def batch_segment_text_into_sentences(
    batch_encoding: BatchEncoding,
    batch_text: list[str],
    tokenizer: PreTrainedTokenizerFast,
    text_segmenter: pysbd.Segmenter,
    max_length: int = 512,
) -> dict[str, list[Any]]:
    batch_outputs: dict[str, list[Any]] = {}
    for encoding, text in zip(batch_encoding.encodings, batch_text, strict=True):
        end_char_index_candidates = [text_span.end for text_span in text_segmenter.segment(text)]

        encoding_length = len(encoding.ids)
        start_token_index = 0
        while start_token_index < encoding_length:
            end_token_index_limit = start_token_index + max_length - 2

            if end_token_index_limit < encoding_length:
                end_char_index_limit = encoding.token_to_chars(end_token_index_limit)[1]
                end_char_index_candidate_index = bisect.bisect(end_char_index_candidates, end_char_index_limit) - 1
                if end_char_index_candidate_index >= 0:
                    end_char_index = end_char_index_candidates[end_char_index_candidate_index]
                    end_token_index = encoding.char_to_token(end_char_index)
                    end_char_index_candidates = end_char_index_candidates[end_char_index_candidate_index + 1 :]
                else:
                    end_token_index = end_token_index_limit
            else:
                end_token_index = encoding_length

            ids = encoding.ids[start_token_index:end_token_index]
            outputs = tokenizer.prepare_for_model(ids, add_special_tokens=True, padding=False, truncation=False)

            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)

            start_token_index = end_token_index

    return BatchEncoding(batch_outputs)


def train_model(config: dict[str, Any], local_rank: int = -1) -> None:
    load_dotenv()
    config_discriminator = DebertaV2Config(**config["model"]["discriminator"])
    config_generator = DebertaV2Config(**config["model"]["generator"])

    tokenizer = DebertaV2TokenizerFast.from_pretrained(Path("models") / config["model_name"])
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
