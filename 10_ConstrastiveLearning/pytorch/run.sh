#!/usr/bin/env bash
#
set -u           # Detect unset variable
set -e           # Exit on non-zero return code from any command
set -o pipefail  # Exit if any of the commands in the pipeline will
                 # return non-zero return code

export CUDA_VISIBLE_DEVICES="0";

python train_baseline.py "conf/1_baseline.yaml"
python train_contrastive.py "conf/2_SimCLR.yaml"
python train_contrastive.py "conf/3_SupCon.yaml"
python train_contrastive.py "conf/4_SupCon_simclr.yaml"
