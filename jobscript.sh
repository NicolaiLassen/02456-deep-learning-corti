#!/bin/sh
 #BSUB -q gpuv100
 #BSUB -gpu "num=1"
 #BSUB -J myJob
 #BSUB -n 1
 #BSUB -W 10:00
 #BSUB -R "rusage[mem=32GB]"
 #BSUB -o logs/%J.out
 #BSUB -e logs/%J.err
 pip3 install --user -r requirements.txt
 echo "Traning..."
 python3 main.py