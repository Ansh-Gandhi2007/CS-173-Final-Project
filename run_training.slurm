#!/bin/bash
#SBATCH --job-name=grpo_camelcase
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --partition=gpu

module load anaconda
conda activate rl_env

# Run unit tests and training
pytest test_grpo.py
python train_grpo.py
