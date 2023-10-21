import math
import zlib
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import regex
from hojichar import Document
from hojichar.filters.document_filters import AcceptJapanese, DiscardAds, DiscardRareKuten


def is_not_empty() -> Callable[..., bool]:
    def is_valid(example: dict[str, Any]) -> bool:
        return example["text"].strip() != ""

    return is_valid


def is_japanese() -> Callable[..., bool]:
    filters = [AcceptJapanese(), DiscardRareKuten()]

    def is_valid(example: dict[str, Any]) -> bool:
        doc = Document(example["text"])
        for f in filters:
            doc = f.apply(doc)
            if doc.is_rejected:
                return False
        return True

    return is_valid


def is_valid_domain(dataset_name: Literal["oscar", "mc4"]) -> Callable[..., bool]:
    dict_path = Path(__file__).parent.joinpath("valid_domains.txt")
    valid_domains = set(dict_path.read_text().splitlines())

    if dataset_name == "oscar":

        def is_valid(example: dict[str, Any]) -> bool:
            if example["meta"]["warc_headers"]["warc-target-uri"].startswith("https://ja.wikipedia.org/"):
                return False
            domain = urlparse(example["meta"]["warc_headers"]["warc-target-uri"]).hostname
            return domain.split(".")[-1] in valid_domains

    elif dataset_name == "mc4":

        def is_valid(example: dict[str, Any]) -> bool:
            if example["url"].startswith("https://ja.wikipedia.org/"):
                return False
            domain = urlparse(example["url"]).hostname
            return domain.split(".")[-1] in valid_domains

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return is_valid


def is_not_ad_content() -> Callable[..., bool]:
    content_filter = DiscardAds()

    def is_valid(example: dict[str, Any]) -> bool:
        doc = content_filter.apply(Document(example["text"]))
        return not doc.is_rejected

    return is_valid


def is_not_footer_header_noisy_oscar() -> Callable[..., bool]:
    filterd_tag = {"header", "footer", "noisy"}

    def is_valid(example: dict[str, Any]) -> bool:
        return not bool(set(example["meta"]["quality_warnings"]) & filterd_tag)

    return is_valid


def is_good_compression_ratio() -> Callable[..., bool]:
    """Checks if data compression (deflate) yields a desired size of data stream.

    NOTE:
    Ths judgment is based on an assumption that a "natual" sentence has an entropy
    within a certain range, and both "too simple" (low entropy) and "too complex" (high
    entropy) sentences don't reflect human's usual writing.
    This function calculates the data compression ratio (calculated by the Deflate
    algorithm) of the original stream, and compares if the resulting ratio is in-between
    the specified range.
    This criterion is somewhat sensitive against the length of the original stream (e.g.
    if the input is long, the resulting compression ratio tends to be small).
    This function also has a mechanism to consider the original length (adjusted by the
    `length_factor` parameter).

    Args:
        min_score: The lower bound of the compression ratio.
        max_score: The upper bound of the compression ratio.
        length_factor: Penalty factor of log(original_byte_length), usually set to
            something larger than 0. Using 0 falls back to a simple compression ratio.

    Returns:
        Judgment function, bound with `min` and `max`.

    Example:
        >>> is_valid = has_good_compression_ratio(0.1, 1.0, 0.0)
        >>> is_valid({"text": "LbdJA66Ufy4Pr6ffQEIo0DL60OL7kQl6y6ohAhqYKf3laCruuR"})
        False  # 1.16
        >>> is_valid({"text": "a" * 200})
        False  # 0.06
        >>> is_valid({"text": "This is a usual sentence. This sentence should pass this judgment."})
        True  # 0.92
    """

    min_score = 0.3
    max_score = 0.7
    length_factor = 0.0

    def is_valid(example: dict[str, Any]) -> bool:
        encoded = example["text"].encode("utf-8")
        compressed = zlib.compress(encoded, level=9)
        encoded_length = len(encoded)
        compressed_length = len(compressed)
        ratio = compressed_length / encoded_length
        length_penalty = length_factor * math.log(encoded_length)
        score = ratio + length_penalty
        return min_score <= score <= max_score

    return is_valid


def extract_japanese_text() -> Callable[..., dict[str, Any]]:
    ja_regex = regex.compile(r"[\p{Script=Hiragana}\p{Script=Katakana}ー]+")
    script_regex = regex.compile(r"[\u0000-\u007F\u0020-\u002F\u003A-\u0040\u005B-\u0060\u007B-\u007E]{100,}")
    url_regex = regex.compile(r"https?://[\w/:%#\$&\?\(\)~\.=\+\-]+")

    def regex_filter(sentence: str, regex_pattern: regex.Pattern) -> str:
        valid_text = ""
        index = 0
        for m in regex_pattern.finditer(sentence):
            valid_text += sentence[index : m.start()]
            index = m.end()
        valid_text += sentence[index:]
        return valid_text

    def extract(example: dict[str, Any]) -> dict[str, Any]:
        valid_text = ""
        for sentence in example["text"].split("\n"):
            if ja_regex.search(sentence):
                sentence = regex_filter(sentence, url_regex)
                sentence = regex_filter(sentence, script_regex)
                valid_text += sentence
        example["text"] = valid_text
        return example

    return extract


def remove_wikipedia_footnote() -> Callable[..., dict[str, Any]]:
    footnote_sections: list[str] = ["脚注", "関連項目", "日本国内の関連項目", "出典", "出典・脚注", "参照", "外部リンク", "参考文献", "その他関連事項"]
    footnote_regex = regex.compile(rf"\n({'|'.join(footnote_sections)})\s*\n")

    def remove(example: dict[str, Any]) -> dict[str, Any]:
        if m := footnote_regex.search(example["text"]):
            example["text"] = example["text"][: m.start()]
        return example

    return remove


def remove_empty_parenthesis() -> Callable[..., dict[str, Any]]:
    def remove(example: dict[str, Any]) -> dict[str, Any]:
        example["text"] = regex.sub(r"（[\s,，、;；]*", "（", example["text"])  # noqa: RUF001
        example["text"] = regex.sub(r"[\s,，、;；]*）", "）", example["text"])  # noqa: RUF001
        example["text"] = regex.sub(r"（\s*）", "", example["text"])  # noqa: RUF001
        example["text"] = regex.sub(r"\([\s,;]*", "(", example["text"])
        example["text"] = regex.sub(r"[\s,;]*\)", ")", example["text"])
        example["text"] = regex.sub(r"\s?\(\s*\)", "", example["text"])
        return example

    return remove
