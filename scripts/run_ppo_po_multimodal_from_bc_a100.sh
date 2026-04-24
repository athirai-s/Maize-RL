#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --job-name=maze_ppo_po_mm_bc_a100
#SBATCH --output=logs/maze_ppo_po_mm_bc_a100_%j.out
#SBATCH --error=logs/maze_ppo_po_mm_bc_a100_%j.err

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
export XLA_FLAGS="--xla_gpu_graph_level=0"

cd /project2/jieyuz_1727/Maize-RL/LMRL-Gym

mkdir -p logs
mkdir -p /scratch1/ashanmug/maize-rl/outputs/ppo_po_mm_from_bc_a100

BC_CKPT=/scratch1/ashanmug/maize-rl/outputs/bc_po_baseline/po_bc_gpt2_small.2026-04-22-19-16-19.436.bdfc01ac3e7f11f18ff770b5e8f03870/best

/home1/ashanmug/.conda/envs/LLM_RL/bin/python -m llm_rl_scripts.maze.ppo.partially_observed_ppo_online_trainvision_revised \
    --model-load-mode=PARAMS \
    --model-load-path="$BC_CKPT" \
    --outputs-path=/scratch1/ashanmug/maize-rl/outputs/ppo_po_mm_from_bc_a100/ \
    --exp-name=ppo_po_mm_from_bc_a100 \
    --maze-name=double_t_maze \
    --describe-function=describe_observation_only_walls \
    --reward-function=standard_reward \
    --n-rounds=30 \
    --epochs=1 \
    --lr=1e-7 \
    --train-bsize=4 \
    --grad-accum-steps=32 \
    --rollout-bsize=16 \
    --n-rollouts=256 \
    --ppo-data-bsize=16 \
    --gradient-checkpointing \
    --use-fp16-activations \
    --max-input-length=1012 \
    --max-output-length=8 \
    --init-kl-coef=0.05 \
    --kl-target=6.0 \
    --kl-horizon=10000 \
    --gamma=0.99 \
    --lam=0.95 \
    --cliprange=0.2 \
    --cliprange-value=0.2 \
    --value-loss-coef=0.5 \
    --policy-temperature=0.7 \
    --save-every-rounds=10 \
    --log-every=32 \
    --no-save-train-state \
    --patch-size=3 \
    --num-visual-tokens=4 \
    --no-print-local-patch
