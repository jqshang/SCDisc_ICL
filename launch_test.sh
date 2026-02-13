#!/bin/bash

#SBATCH --account=aip-rgrosse
#SBATCH --output=slurm/output/%j_%x.out

#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=5G

project_dir="/home/$USER/projects/aip-rgrosse/$USER/SCDisc_ICL"

export HF_HOME="/scratch/$USER/hf_cache"

module load python/3.11.5 cuda/12.2 gcc arrow/21.0.0

cd $project_dir
source .venv/bin/activate

python3 -m gpt2_test 
