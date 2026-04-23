#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --job-name=maze_ilql_po
#SBATCH --output=logs/maze_ilql_po_%j.out
#SBATCH --error=logs/maze_ilql_po_%j.err

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
mkdir -p /scratch1/ashanmug/maize-rl/outputs/ilql_po_baseline

DATA_FILE=/project2/jieyuz_1727/Maize-RL/LMRL-Gym/data/partially_observed_filtered_maze_data.jsonl

/home1/ashanmug/.conda/envs/LLM_RL/bin/python -m llm_rl_scripts.maze.ilql.partially_observed_ilql \
    HF gpt2 \
    "$DATA_FILE" \
    --outputs-path=/scratch1/ashanmug/maize-rl/outputs/ilql_po_baseline/ \
    --exp-name=ilql_po_gpt2_small \
    --epochs=20 \
    --lr=1e-4 \
    --train-bsize=16 \
    --grad-accum-steps=8 \
    --tau=0.95 \
    --cql-weight=0.0 \
    --gamma=0.99 \
    --max-length=1024 \
    --gradient-checkpointing \
    --log-every=256 \
    --eval-every-epochs=2 \
    --save-every-epochs=5 \
    --save-at-end \
    --no-save-train-state
