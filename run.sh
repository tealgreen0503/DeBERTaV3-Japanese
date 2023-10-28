#!/bin/bash

accelerate_config_file="config/accelerate_config_zero3.yaml"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --config_file) config_file="$2"; shift ;;
        --accelerate_config_file) accelerate_config_file="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [[ -z "$config_file" ]]; then
    echo "Configuration file is not provided. Use --config_file to specify it."
    exit 1
fi

poetry run python -m scripts.pre_tokenize --config_file "$config_file"
poetry run python -m scripts.train_tokenizer --config_file "$config_file"
poetry run accelerate launch --config_file "$accelerate_config_file" -m scripts.train_model --config_file "$config_file"
