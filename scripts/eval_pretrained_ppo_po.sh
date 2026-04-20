#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --job-name=eval_ppo_po
#SBATCH --output=logs/eval_ppo_po_%j.out
#SBATCH --error=logs/eval_ppo_po_%j.err

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
export XLA_FLAGS="--xla_gpu_enable_command_buffer="

cd /project2/jieyuz_1727/Maize-RL/LMRL-Gym

mkdir -p logs outputs/eval_ppo_po_pretrained

CKPT_DIR=/project2/jieyuz_1727/Maize-RL/LMRL-Gym/checkpoints/ppo_po_pretrained
BASE_URL=https://rail.eecs.berkeley.edu/datasets/rl-llm-bench-dataset/maze/checkpoints/partially_observed/ppo

mkdir -p "$CKPT_DIR"
cd "$CKPT_DIR"

for f in config.json params.msgpack train_state.msgpack; do
    if [ ! -f "$f" ]; then
        echo "Downloading $f..."
        wget -q --show-progress "$BASE_URL/$f"
    else
        echo "$f already downloaded."
    fi
done

ls -la "$CKPT_DIR"

cd /project2/jieyuz_1727/Maize-RL/LMRL-Gym

/home1/ashanmug/.conda/envs/LLM_RL/bin/python -m llm_rl_scripts.maze.bc.eval_bc \
    PARAMS "$CKPT_DIR" \
    --outputs-path=/project2/jieyuz_1727/Maize-RL/LMRL-Gym/outputs/eval_ppo_po_pretrained/ \
    --no-fully-observed \
    --policy-n-rollouts=32 \
    --policy-bsize=1 \
    --policy-max-input-length=256 \
    --policy-max-output-length=8
