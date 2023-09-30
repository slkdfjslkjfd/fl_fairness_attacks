#!/bin/bash
#SBATCH --job-name=fairness_attacks
#SBATCH -c 4
#SBATCH --gres=gpu:1
cd /nfs-share/jr897/FL-fairness-attacks
source ../miniconda3/bin/activate workspace
[ -d "outputs" ] || (echo "added directory 'outputs'" && mkdir outputs)
python src/main.py > outputs/output