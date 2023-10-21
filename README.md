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
source run.sh --config_file config/deberta-v3-xsmall.yaml --num_gpus 1
```
or
```sh
python -m pre_tokenize --config_file config/deberta-v3-xsmall.yaml
python -m train_tokenizer --config_file config/deberta-v3-xsmall.yaml
deepspeed --module --num_gpus=1 scripts.train_model --config_file config/deberta-v3-xsmall.yaml
```

## References
- [microsoft/DeBERTa: The implementation of DeBERTa](https://github.com/microsoft/DeBERTa)
- [llm-jp/llm-jp-corpus](https://github.com/llm-jp/llm-jp-corpus)
- [Sentencepiece の分割を MeCab っぽくする - Qiita](https://qiita.com/taku910/items/fbaeab4684665952d5a9)
