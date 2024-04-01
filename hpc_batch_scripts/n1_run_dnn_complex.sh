#!/bin/bash
#BSUB -n 1
#BSUB -W 900
#BSUB -J IoT_regression
#BSUB -q gpu
#BSUB =gpu "num=1:mode=shared:mps=no"
#BSUB -o dnn_outputs/stdout.%J
#BSUB -e dnn_outputs/stderr.%J

source ~/.bashrc
conda activate /usr/local/usrapps/csc522s24/lrwilli7/env_iot
python ../neuralnetwork/dnn_complex.py
conda deactivate