import argparse
import os
import shutil
import warnings
from pathlib import Path
from typing import Any

import sentencepiece as spm
import yaml
from tokenizers import Regex, Tokenizer, decoders, normalizers, processors
from tokenizers.models import BPE, Unigram
from transformers import DebertaV2TokenizerFast, PreTrainedTokenizerFast
from transformers.convert_slow_tokenizer import SentencePieceExtractor, import_protobuf
from transformers.utils.sentencepiece_model_pb2 import ModelProto

from scripts.pre_tokenize import DELIMITER


def train_tokenizer(config: dict[str, Any]) -> None:
    spm_model_prefix = "models/sentencepiece/spm"
    os.makedirs(Path(spm_model_prefix).parent, exist_ok=True)
    spm_kwargs = convert_spm_kwargs(
        input="data/pre_tokenized/train.txt",
        model_prefix=spm_model_prefix,
        pretokenization_delimiter=DELIMITER,
        **config["sentencepiece"],
    )
    spm.SentencePieceTrainer.train(spm_kwargs)

    tokenizer = create_tokenizer(f"{spm_model_prefix}.model", **config["tokenizer"])
    tokenizer.save_pretrained(Path("models") / config["model_name"])
    shutil.copy(spm_model_prefix + ".model", Path("models") / config["model_name"])
    shutil.copy(spm_model_prefix + ".vocab", Path("models") / config["model_name"])


def convert_spm_kwargs(**kwargs: Any) -> str:
    kwarg_texts: list[str] = []
    for key, value in kwargs.items():
        if isinstance(value, bool):
            kwarg_texts.append(f"{key}={str(value).lower()}")
        else:
            kwarg_texts.append(f"{key}={value}")
    return "--" + " --".join(kwarg_texts)


def create_tokenizer(vocab_file: str, **kwargs: Any) -> PreTrainedTokenizerFast:
    # from transformers.utils import sentencepiece_model_pb2 as model_pb2
    model_pb2 = import_protobuf()

    m = model_pb2.ModelProto()
    with open(vocab_file, "rb") as f:
        m.ParseFromString(f.read())
    proto = m

    if proto.trainer_spec.byte_fallback:
        warnings.warn(
            "The sentencepiece tokenizer that you are converting to a fast tokenizer uses the byte fallback option"
            " which is not implemented in the fast tokenizers. In practice this means that the fast version of the"
            " tokenizer can produce unknown tokens whereas the sentencepiece version would have converted these "
            "unknown tokens into a sequence of byte tokens matching the original piece of text."
        )

    tokenizer = get_backend_tokenizer(proto, vocab_file)
    tokenizer.normalizer = get_normalizer(proto)
    tokenizer.decoder = get_docoder()
    tokenizer.post_processor = get_post_processor()
    return DebertaV2TokenizerFast(tokenizer_object=tokenizer, **kwargs)


def get_backend_tokenizer(proto: ModelProto, vocab_file: str) -> Tokenizer:
    model_type = proto.trainer_spec.model_type
    vocab_scores = [(piece.piece, piece.score) for piece in proto.pieces]
    unk_id = proto.trainer_spec.unk_id

    if model_type == 1:
        tokenizer = Tokenizer(Unigram(vocab_scores, unk_id))
    elif model_type == 2:
        _, merges = SentencePieceExtractor(vocab_file).extract()
        bpe_vocab = {word: i for i, (word, score) in enumerate(vocab_scores)}
        tokenizer = Tokenizer(BPE(bpe_vocab, merges, unk_token=proto.trainer_spec.unk_piece, fuse_unk=True))
    else:
        raise Exception(
            "You're trying to run a `Unigram` model but you're file was trained with a different algorithm"
        )

    return tokenizer


def get_normalizer(proto: ModelProto) -> normalizers.Normalizer:
    precompiled_charsmap = proto.normalizer_spec.precompiled_charsmap
    normalizer_list = [
        normalizers.Strip(),
        normalizers.Nmt(),
        normalizers.NFKC(),
        normalizers.Replace(Regex(" {2,}"), " "),
        normalizers.Lowercase(),
    ]
    if precompiled_charsmap:
        normalizer_list.append(normalizers.Precompiled(precompiled_charsmap))
    normalizer_list.append(normalizers.Replace(Regex(" "), "▁"))
    return normalizers.Sequence(normalizer_list)


def get_docoder() -> decoders.Decoder:
    return decoders.Metaspace(replacement="▁", add_prefix_space=False)


def get_post_processor() -> processors.PostProcessor:
    return processors.TemplateProcessing(
        single="[CLS]:0 $A:0 [SEP]:0",
        pair="[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
        special_tokens=[("[CLS]", 1), ("[SEP]", 2)],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)
    args = parser.parse_args()
    with Path(args.config_file).open(mode="r") as f:
        config = yaml.safe_load(f)

    train_tokenizer(config)


if __name__ == "__main__":
    main()
