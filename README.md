# DeBERTaV3-Japanese

## Usage
### Setup
```sh
cp .env.example .env  # add your WandB API Key
poetry install
poetry run pip install multiprocess==0.70.15  # cf. https://github.com/huggingface/datasets/issues/5613
```
### Training
```sh
poetry run source run.sh --config_file config/deberta-v3-xsmall.yaml --num_gpus 1
```
or
```sh
poetry run python -m pre_tokenize --config_file config/deberta-v3-xsmall.yaml
poetry run python -m train_tokenizer --config_file config/deberta-v3-xsmall.yaml
poetry run deepspeed --module --num_gpus=1 scripts.train_model --config_file config/deberta-v3-xsmall.yaml
```

## References
- [microsoft/DeBERTa: The implementation of DeBERTa](https://github.com/microsoft/DeBERTa)
- [llm-jp/llm-jp-corpus](https://github.com/llm-jp/llm-jp-corpus)
- [Sentencepiece の分割を MeCab っぽくする - Qiita](https://qiita.com/taku910/items/fbaeab4684665952d5a9)
