#!/bin/bash
#BSUB -n 1
#BSUB -W 30
#BSUB -J IoT_regression
#BSUB -o stdout.%J
#BSUB -e stderr.%J

module load conda
python regression_script.py