#!/bin/bash
#BSUB -n 1
#BSUB -W 30
#BSUB -J IoT_regression
#BSUB -o stdout.%J
#BSUB -e stderr.%J

conda activate /usr/local/usrapps/csc522s24/lrwilli7/env_iot
python ../regression/regression_script.py
conda deactivate