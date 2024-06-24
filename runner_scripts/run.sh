#!/bin/bash -f
#

#SBATCH -p gpu-a
#SBATCH --exclude=bcl-gpu15
#SBATCH --gres=gpu:4
#SBATCH --mem 256G
#SBATCH -t 7-0:0
#SBATCH --mail-type="ALL"
#SBATCH --mail-user="alexandria.leto@intel.com"

#SBATCH --job-name=llama_generation
#SBATCH --output=logs/llama-%j.out
#SBATCH --error=logs/llama-%j.err

source activate alce
cache_path="/export/data/aleto/hf_cache_overflow/"
export PYTHONPATH=/home/aleto/projects/ALCE/


python run.py --config configs/eli5_llama2_shot2_ndoc5_bm25_default.yaml
