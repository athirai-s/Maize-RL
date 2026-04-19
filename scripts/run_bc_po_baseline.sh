#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --job-name=maze_bc_po
#SBATCH --output=logs/maze_bc_po_%j.out
#SBATCH --error=logs/maze_bc_po_%j.err

module load conda
source activate LLM_RL || eval "$(conda shell.bash hook)" && conda activate LLM_RL

echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"

export OPENBLAS_NUM_THREADS=1
export PATH=/home1/ashanmug/.conda/envs/LLM_RL/bin:$PATH

CONDA_ENV=/home1/ashanmug/.conda/envs/LLM_RL
NVDIR=$CONDA_ENV/lib/python3.9/site-packages/nvidia
export LD_LIBRARY_PATH=$NVDIR/cudnn/lib:$NVDIR/cublas/lib:$NVDIR/cuda_runtime/lib:$NVDIR/cuda_cupti/lib:$NVDIR/cuda_nvrtc/lib:$NVDIR/cufft/lib:$NVDIR/cusolver/lib:$NVDIR/cusparse/lib:$LD_LIBRARY_PATH
export XLA_PYTHON_CLIENT_PREALLOCATE=false

cd /project2/jieyuz_1727/Maize-RL/LMRL-Gym

mkdir -p logs outputs/bc_po_baseline data

DATA_FILE=/project2/jieyuz_1727/Maize-RL/LMRL-Gym/data/partially_observed_filtered_maze_data.jsonl

if [ ! -f "$DATA_FILE" ]; then
    echo "Downloading PO maze data..."
    wget -O "$DATA_FILE" https://rail.eecs.berkeley.edu/datasets/rl-llm-bench-dataset/maze/partially_observed_filtered_maze_data.jsonl
fi

/home1/ashanmug/.conda/envs/LLM_RL/bin/python -m llm_rl_scripts.maze.bc.partially_observed_bc \
    HF gpt2 \
    "$DATA_FILE" \
    0.9 \
    --outputs-path=/project2/jieyuz_1727/Maize-RL/LMRL-Gym/outputs/bc_po_baseline/ \
    --exp-name=po_bc_gpt2_small \
    --epochs=500 \
    --lr=1e-4 \
    --train-bsize=128 \
    --eval-every-steps=256 \
    --save-at-end
