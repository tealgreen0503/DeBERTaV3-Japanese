# DeBERTaV3-Japanese

## Usage
```sh
poetry install
poetry shell
pip install multiprocess==0.70.15  # cf. https://github.com/huggingface/datasets/issues/5613
python -m run --model_name deberta-v3-xsmall --pre_tokenize --train_tokenizer --train_model
```

## Reference
- [microsoft/DeBERTa: The implementation of DeBERTa](https://github.com/microsoft/DeBERTa)
- [Sentencepiece の分割を MeCab っぽくする - Qiita](https://qiita.com/taku910/items/fbaeab4684665952d5a9)
