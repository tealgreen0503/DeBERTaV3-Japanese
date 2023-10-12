#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <config_file>"
    exit 1
fi

CONFIG_FILE=$1

python -m scripts.save_pre_tokenized_text --config_file "$CONFIG_FILE"
python -m scripts.train_tokenizer --config_file "$CONFIG_FILE"
python -m scripts.train_model --config_file "$CONFIG_FILE"
