#!/bin/bash
#BSUB -n 1
#BSUB -W 15:00
#BSUB -J IoT_regression
#BSUB -R "rusage[mem=128]"
#BSUB -o regression_outputs/stdout.%J
#BSUB -e regression_outputs/stderr.%J

source ~/.bashrc
conda activate /usr/local/usrapps/csc522s24/lrwilli7/env_synth
python ../synthetic_data/CGsynth_regression.py
conda deactivate