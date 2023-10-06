# DeBERTaV3-Japanese

## Usage
### Setup
```sh
cp .env.example .env  # add your WandB API Key
poetry install
poetry shell
pip install multiprocess==0.70.15  # cf. https://github.com/huggingface/datasets/issues/5613
```
### Training
```sh
python -m run --model_name deberta-v3-xsmall
```
or
```sh
python -m pre_tokenize --model_name deberta-v3-xsmall
python -m train_tokenizer --model_name deberta-v3-xsmall
python -m train_model --model_name deberta-v3-xsmall
```

## Reference
- [microsoft/DeBERTa: The implementation of DeBERTa](https://github.com/microsoft/DeBERTa)
- [Sentencepiece の分割を MeCab っぽくする - Qiita](https://qiita.com/taku910/items/fbaeab4684665952d5a9)
