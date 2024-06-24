#!/bin/bash -f
#

#SBATCH -p gpu-a
#SBATCH --exclude=bcl-gpu15
#SBATCH --gres=gpu:4
#SBATCH --mem 128G
#SBATCH -t 7-0:0
#SBATCH --mail-type="ALL"
#SBATCH --mail-user="alexandria.leto@intel.com"

#SBATCH --job-name=mistral_generation
#SBATCH --output=logs/mistral-%j.out
#SBATCH --error=logs/mistral-%j.err

source activate alce
cache_path="/export/data/aleto/hf_cache_overflow/"
export PYTHONPATH=/home/aleto/projects/ALCE/


python run.py --config configs/asqa_llama-13b_shot2_ndoc10_gtr_extraction.yaml --model mistralai/Mistral-7B-v0.1
