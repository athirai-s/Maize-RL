#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --job-name=maze_ppo
#SBATCH --output=logs/maze_ppo_%j.out
#SBATCH --error=logs/maze_ppo_%j.err

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

mkdir -p logs outputs/ppo_baseline

/home1/ashanmug/.conda/envs/LLM_RL/bin/python -m llm_rl_scripts.maze.ppo.train_ppo_online \
    HF gpt2 \
    --outputs-path=/project2/jieyuz_1727/Maize-RL/LMRL-Gym/outputs/ppo_baseline/ \
    --exp-name=ppo_gpt2_small \
    --maze-name=double_t_maze \
    --describe-function=describe_observation_give_position \
    --reward-function=standard_reward \
    --n-rounds=50 \
    --epochs=2 \
    --lr=1e-6 \
    --train-bsize=8 \
    --grad-accum-steps=4 \
    --rollout-bsize=16 \
    --n-rollouts=64 \
    --ppo-data-bsize=16 \
    --max-input-length=128 \
    --max-output-length=8 \
    --init-kl-coef=0.05 \
    --gamma=0.99 \
    --lam=0.95 \
    --cliprange=0.2 \
    --cliprange-value=0.2 \
    --value-loss-coef=0.5 \
    --save-every-rounds=10 \
    --log-every=32
