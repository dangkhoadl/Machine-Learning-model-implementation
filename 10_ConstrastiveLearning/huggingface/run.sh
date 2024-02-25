#!/usr/bin/env bash
#
set -u           # Detect unset variable
set -e           # Exit on non-zero return code from any command
set -o pipefail  # Exit if any of the commands in the pipeline will
                 # return non-zero return code

export CUDA_VISIBLE_DEVICES="0,1";

python 1_pretrain.py "conf/SupCon.yaml" \
    --exp_name "SupConLoss"
python 2_finetune.py "conf/SupCon.yaml" \
    --exp_name "SupConLoss"
