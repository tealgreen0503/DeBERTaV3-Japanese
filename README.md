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
./run.sh config/deberta-v3-xsmall.yaml
```
or
```sh
python -m pre_tokenize --config_file config/deberta-v3-xsmall.yaml
python -m train_tokenizer --config_file config/deberta-v3-xsmall.yaml
python -m train_model --config_file config/deberta-v3-xsmall.yaml
```

## Reference
- [microsoft/DeBERTa: The implementation of DeBERTa](https://github.com/microsoft/DeBERTa)
- [Sentencepiece の分割を MeCab っぽくする - Qiita](https://qiita.com/taku910/items/fbaeab4684665952d5a9)
