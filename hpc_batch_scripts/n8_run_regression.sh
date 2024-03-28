#!/bin/bash
#BSUB -n 8
#BSUB -W 60
#BSUB -J IoT_regression
#BSUB -o regression_outputs/stdout.%J
#BSUB -e regression_outputs/stderr.%J

source ~/.bashrc
conda activate /usr/local/usrapps/csc522s24/lrwilli7/env_iot
python ../regression/regression_script.py
conda deactivate