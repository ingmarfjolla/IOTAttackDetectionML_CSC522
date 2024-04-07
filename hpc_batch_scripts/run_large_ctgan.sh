#!/bin/bash
#BSUB -n 1
#BSUB -W 36:00
#BSUB -J CTGAN
#BSUB -q gpu
#BSUB -gpu "num=1:mode=shared:mps=no"
#BSUB -o dnn_outputs/stdout.%J
#BSUB -e dnn_outputs/stderr.%J

source ~/.bashrc
conda activate /usr/local/usrapps/csc522s24/lrwilli7/env_synth
python ../generator_custom/large_run.py
conda deactivate