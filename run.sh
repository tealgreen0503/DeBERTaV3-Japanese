#!/bin/bash

num_gpus=1

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --config_file) config_file="$2"; shift ;;
        --num_gpus) num_gpus="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [[ -z "$config_file" ]]; then
    echo "Configuration file is not provided. Use --config_file to specify it."
    exit 1
fi

poetry run python -m scripts.save_pre_tokenized_text --config_file "$config_file"
poetry run python -m scripts.train_tokenizer --config_file "$config_file"
poetry run deepspeed --module --num_gpus="$num_gpus" scripts.train_model --config_file "$config_file"
