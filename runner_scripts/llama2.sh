#!/bin/bash -f
#

#SBATCH -p gpu-a
#SBATCH --exclude=bcl-gpu15
#SBATCH --gres=gpu:2
#SBATCH --mem 128G
#SBATCH -t 7-0:0
#SBATCH --mail-type="ALL"
#SBATCH --mail-user="alexandria.leto@intel.com"

#SBATCH --job-name=llama2_generation
#SBATCH --output=logs/llama2-%j.out
#SBATCH --error=logs/llama2-%j.err

source activate alce
cache_path="/export/data/aleto/hf_cache_overflow/"
export PYTHONPATH=/home/aleto/projects/ALCE/


python run.py --config configs/asqa_llama2_shot2_ndoc5_gtr_default.yaml
