#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --job-name=maze_bc
#SBATCH --output=logs/maze_bc_%j.out
#SBATCH --error=logs/maze_bc_%j.err

module load conda
source activate LLM_RL || eval "$(conda shell.bash hook)" && conda activate LLM_RL

echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"

export OPENBLAS_NUM_THREADS=1
export PATH=/home1/ashanmug/.conda/envs/LLM_RL/bin:$PATH

cd /project2/jieyuz_1727/Maize-RL/LMRL-Gym

mkdir -p logs outputs

/home1/ashanmug/.conda/envs/LLM_RL/bin/python -m llm_rl_scripts.maze.bc.fully_observed_bc \
    HF gpt2 \
    data/fully_observed_filtered_maze_data.jsonl \
    --outputs-path=outputs/baseline/ \
    --exp-name=fo_bc_gpt2_small \
    --epochs 500 \
    --lr 1e-4 \
    --train-bsize 128 \
    --eval-every-steps 256 \
    --save-at-end
