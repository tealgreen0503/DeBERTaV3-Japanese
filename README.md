# DeBERTaV3-Japanese

## Usage
### Setup
```sh
cp .env.example .env  # set your WandB API Key
poetry install
poetry run pip install multiprocess==0.70.15  # cf. https://github.com/huggingface/datasets/issues/5613
```
### Training
```sh
poetry run python -m scripts.pre_tokenize --config_file config/deberta-v3-xsmall.yaml
poetry run python -m scripts.train_tokenizer --config_file config/deberta-v3-xsmall.yaml
poetry run accelerate launch --config_file config/accelerate_config_zero3.yaml -m scripts.train_model --config_file config/deberta-v3-xsmall.yaml
```

### Loading Pre-trained Model
- The pre-trained DeBERTaV3 model can be loaded as a DeBERTaV2 model using the `AutoModel` interface.
```python
discriminator_config = DebertaV2Config(**config_kwargs)
generator_config = DebertaV2Config(**config_kwargs)
pretrained_model = DebertaV3ForPreTraining._from_config(config=discriminator_config, generator_config=generator_config)

# Pretraining

pretrained_model.save_pretrained("path/to/model")
model = AutoModel.from_pretrained("path/to/model")
print(type(model))
# <class 'transformers.models.deberta_v2.modeling_deberta_v2.DebertaV2Model'>
```

## Feature
- The `DeBERTaV3ForPretraining` is designed for compatibility with both DeBERTaV2 and DeBERTaV3 models, allowing for seamless fine-tuning with DeBERTaV2 or further pre-training with DeBERTaV3 (Replace Token Detection).
- Pre-tokenization is not necessary:
  - Although Sentencepiece and Sudachi were utilized in the training of the Tokenizer, loading a pre-trained Tokenizer does not require Sudachi. For further details, refer to [this blog post](https://qiita.com/taku910/items/fbaeab4684665952d5a9).

## Evaluation
- WIP

## References
- [microsoft/DeBERTa: The implementation of DeBERTa](https://github.com/microsoft/DeBERTa)
- [llm-jp/llm-jp-corpus](https://github.com/llm-jp/llm-jp-corpus)
- [Sentencepiece の分割を MeCab っぽくする - Qiita](https://qiita.com/taku910/items/fbaeab4684665952d5a9)
