#!/bin/bash
#SBATCH --job-name=training_job
#SBATCH --output=logs/output_%j.log
#SBATCH --error=logs/error_%j.log
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

source venv/bin/activate
cd ~/qml-trackmania

python training/quantum_sac.py