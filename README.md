# DeBERTaV3-Japanese

## Usage
```sh
poetry install
poetry shell
pip install multiprocess==0.70.15  # cf. https://github.com/huggingface/datasets/issues/5613
python -m run --model_name deberta-v3-xsmall --pre_tokenize --train_tokenizer --train_model
```
