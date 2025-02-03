#!/bin/bash -ex
python llama_finetune.py --run-name 1b-test-run --model_name 1B --batch-size 4 --llama3 --use-grpo --resume-dir exp/1b-test-run/checkpoint-10000
