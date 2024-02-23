import bisect
from collections.abc import Iterable
from itertools import chain
from typing import Any

import pysbd
from transformers import BatchEncoding, PreTrainedTokenizerFast
from transformers.tokenization_utils_base import EncodingFast


def batch_preprocess(
    batch_example: dict[str, list[Any]],
    tokenizer: PreTrainedTokenizerFast,
    text_segmenter: pysbd.Segmenter,
    max_length: int = 512,
) -> dict[str, list[Any]]:
    batch_encoding_fast = tokenizer.batch_encode_plus(
        batch_example["text"], add_special_tokens=False, padding=False, truncation=False
    ).encodings
    batch_input_ids = batch_segment_text_into_sentences(
        batch_encoding_fast, batch_example["text"], text_segmenter, max_length=max_length
    )
    batch_encoding = batch_prepare_for_model(batch_input_ids, tokenizer, max_length=max_length)
    return batch_encoding


def batch_segment_text_into_sentences(
    batch_encoding_fast: list[EncodingFast],
    batch_text: list[str],
    text_segmenter: pysbd.Segmenter,
    max_length: int = 512,
) -> Iterable[list[int]]:
    max_length_without_special_tokens = max_length - 2
    batch_input_ids = [
        segment_text_into_sentences(
            encoding, text, text_segmenter, max_length_without_special_tokens=max_length_without_special_tokens
        )
        for encoding, text in zip(batch_encoding_fast, batch_text, strict=True)
    ]
    return chain.from_iterable(batch_input_ids)


def segment_text_into_sentences(
    encoding: EncodingFast, text: str, text_segmenter: pysbd.Segmenter, max_length_without_special_tokens: int = 510
) -> list[list[int]]:
    batch_input_ids: list[list[int]] = []

    end_char_index_candidates = [text_span.end - 1 for text_span in text_segmenter.segment(text)]
    encoding_length = len(encoding.ids)

    start_token_index = 0
    while start_token_index < encoding_length:
        end_token_index_limit = start_token_index + max_length_without_special_tokens

        if end_token_index_limit < encoding_length:
            end_char_index_limit = encoding.token_to_chars(end_token_index_limit)[1] - 1
            # Search for the last char index that is less than or equal to end_char_index_limit
            end_char_index_candidates_index = bisect.bisect(end_char_index_candidates, end_char_index_limit) - 1
            if end_char_index_candidates_index >= 0:
                end_char_index = end_char_index_candidates[end_char_index_candidates_index]
                end_token_index = encoding.char_to_token(end_char_index_candidates[end_char_index_candidates_index])
                # When the end_token_index is None, search for the next end_char_index
                while end_token_index is None:
                    end_char_index_candidates_index -= 1
                    if end_char_index_candidates_index < 0:
                        end_token_index = end_token_index_limit
                        break
                    end_char_index = end_char_index_candidates[end_char_index_candidates_index]
                    end_token_index = encoding.char_to_token(end_char_index)
                end_char_index_candidates = end_char_index_candidates[end_char_index_candidates_index + 1 :]
            else:
                end_token_index = end_token_index_limit
        else:
            end_token_index = encoding_length

        batch_input_ids.append(encoding.ids[start_token_index:end_token_index])
        start_token_index = end_token_index

    return batch_input_ids


def batch_prepare_for_model(
    batch_input_ids: Iterable[list[int]], tokenizer: PreTrainedTokenizerFast, max_length: int = 512
) -> BatchEncoding:
    encoding_dict_list = [
        tokenizer.prepare_for_model(
            input_ids, add_special_tokens=True, padding=False, truncation=True, max_length=max_length
        )
        for input_ids in batch_input_ids
    ]
    return BatchEncoding(
        {key: [encoding_dict[key] for encoding_dict in encoding_dict_list] for key in encoding_dict_list[0].keys()}
    )
