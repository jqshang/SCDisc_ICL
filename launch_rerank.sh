#!/bin/bash

#SBATCH --account=aip-rgrosse
#SBATCH --output=slurm/output/%j_%x.out

#SBATCH --time=08:00:00
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

# gpt-4.1-2025-04-14 gpt-4o-2024-05-13   Qwen/Qwen3-4B-Thinking-2507  Qwen/Qwen2.5-72B-Instruct
# python -m icl.run_reranking --dataset semeval_en --tokenizer-model bert-base-uncased --llm-model gpt4 --llm-checkpoint gpt-4o-2024-05-13  --n-icl-examples 16
python -m icl.run_reranking --dataset semeval_en --tokenizer-model bert-base-uncased --llm-model qwen --llm-checkpoint Qwen/Qwen3-4B-Thinking-2507 --n-icl-examples 4
# python -m icl.run_reranking --dataset semeval_en --tokenizer-model bert-base-uncased --llm-model gemma3 --llm-checkpoint google/gemma-3-4b-it --scaling-curve --bucket-sizes 0,1,2,4,8,16 --n-bucket-seeds 1

