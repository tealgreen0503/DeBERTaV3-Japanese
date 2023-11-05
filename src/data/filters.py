from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import regex


def is_not_empty() -> Callable[[dict[str, Any]], bool]:
    def is_valid(example: dict[str, Any]) -> bool:
        return example["text"].strip() != ""

    return is_valid


def is_valid_japanese() -> Callable[[dict[str, Any]], bool]:
    hiragana_katakana_regex = regex.compile(r"[ぁ-んァ-ン]")

    def is_valid(example: dict[str, Any]) -> bool:
        text = example["text"]
        return "。" in text and bool(hiragana_katakana_regex.search(text[:100]))

    return is_valid


def is_valid_domain_for_oscar() -> Callable[[dict[str, Any]], bool]:
    dict_path = Path(__file__).parent.joinpath("valid_domains.txt")
    valid_domains = set(dict_path.read_text().splitlines())

    def is_valid(example: dict[str, Any]) -> bool:
        url = example["meta"]["warc_headers"]["warc-target-uri"]
        if isinstance(url, str) and url.startswith("https://ja.wikipedia.org/"):
            return False
        domain = urlparse(url).hostname
        return domain.rsplit(".", 1)[-1] in valid_domains if domain is not None else False

    return is_valid


def is_not_footer_header_noisy_for_oscar() -> Callable[[dict[str, Any]], bool]:
    filterd_tag = {"header", "footer", "noisy"}

    def is_valid(example: dict[str, Any]) -> bool:
        return (quality_warnings := example["meta"]["quality_warnings"]) is None or not bool(
            set(quality_warnings) & filterd_tag
        )

    return is_valid


def remove_wikipedia_footnote() -> Callable[[dict[str, list[Any]]], dict[str, list[Any]]]:
    footnote_sections: list[str] = ["脚注", "関連項目", "日本国内の関連項目", "出典", "出典・脚注", "参照", "外部リンク", "参考文献", "その他関連事項"]
    footnote_regex = regex.compile(rf"\n({'|'.join(footnote_sections)})\s*\n")

    def remove(text: str) -> str:
        if m := footnote_regex.search(text):
            text = text[: m.start()]
        return text

    def batch_remove(examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        examples["text"] = [remove(text) for text in examples["text"]]
        return examples

    return batch_remove


def remove_empty_parenthesis() -> Callable[[dict[str, Any]], dict[str, Any]]:
    def remove(text: str) -> str:
        text = regex.sub(r"（[\s,，、;；]*", "（", text)
        text = regex.sub(r"[\s,，、;；]*）", "）", text)
        text = regex.sub(r"（\s*）", "", text)
        text = regex.sub(r"\([\s,;]*", "(", text)
        text = regex.sub(r"[\s,;]*\)", ")", text)
        text = regex.sub(r"\s?\(\s*\)", "", text)
        return text

    def batch_remove(examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        examples["text"] = [remove(text) for text in examples["text"]]
        return examples

    return batch_remove
