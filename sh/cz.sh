#!/bin/bash -l
# FILE: cz.sh

#### Choose Partition
#SBATCH --partition=debug

#### Cluster-specific settings
#SBATCH --qos=normal
#SBATCH --mem=256G
#SBATCH --time=48:00:00

#### Number of nodes and tasks
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=10

#### Job-specific info
#SBATCH --job-name="multimodal_train"
#SBATCH --output="./out/cz-%j.out"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=asandhu9@jh.edu

########################
### Set Environment  ###
########################
setup_time=$(date +%s)
source ~/.bashrc
source .venv/bin/activate

export PATH="/usr/local/cuda-12.2/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH"
export TORCH_CPP_LOG_LEVEL=INFO
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# NCCL settings
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eno1
export NCCL_P2P_LEVEL=TRACE

# Automatic variables from SLURM
NUM_MACHINES=${SLURM_NNODES:-1}
NUM_PROCESSES=$(( ${SLURM_GPUS_PER_NODE:-2} * ${SLURM_NNODES:-1} ))
MACHINE_RANK=${SLURM_NODEID:-0}
CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-10}
GPU_IDS=$(seq 0 $(( SLURM_GPUS_PER_NODE - 1 )) | paste -sd,)

export OMP_NUM_THREADS=${CPUS_PER_TASK}

MAIN_IP=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

echo "==== Job Debug Info ===="
echo "NUM_MACHINES=${NUM_MACHINES}"
echo "NUM_PROCESSES=${NUM_PROCESSES}"
echo "MACHINE_RANK=${MACHINE_RANK}"
echo "GPU_IDS=${GPU_IDS}"
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"
echo "MAIN_IP=${MAIN_IP}"
echo "========================"

# Set port for Accelerate communication
BASE_PORT=29550
MAX_PORT=$((BASE_PORT + 50))
PORT=$BASE_PORT
while [ $PORT -le $MAX_PORT ]; do
    if ! lsof -i :$PORT > /dev/null 2>&1; then
        echo "Port $PORT is available."
        break
    fi
    PORT=$((PORT + 1))
done
if [ $PORT -gt $MAX_PORT ]; then
    echo "Error: No available port found."
    exit 1
fi
echo "Selected port: $PORT"

###########################
### Run Training Script ###
###########################
echo "Job started on $(date)"
start_time=$(date +%s)

INFER_SCRIPT="main.py"
SCRIPT_ARGS="--census-version 2023-12-15 --organism homo_sapiens --measurement-name RNA \
--output-dir output --batch-size 64 --num-epochs 50 --local-model-path /home/asandhu9/cz-biohub-test/models/dmis-lab/biobert/1.1 --use-wandb"
"--search-hparams --train-model"

srun python -m accelerate.commands.launch \
    --num_processes=${NUM_PROCESSES} \
    --num_machines=${NUM_MACHINES} \
    --machine_rank=${MACHINE_RANK} \
    --gpu_ids="${GPU_IDS}" \
    --main_process_ip=${MAIN_IP} \
    --main_process_port=${PORT} \
    --deepspeed_config_file configs/ds_config.json \
    $INFER_SCRIPT $SCRIPT_ARGS

echo "Job completed on $(date)"
end_time=$(date +%s)
elapsed=$((end_time - start_time))
days=$((elapsed / 86400))
hours=$(( (elapsed % 86400) / 3600 ))
minutes=$(( (elapsed % 3600) / 60 ))
seconds=$((elapsed % 60))
echo "Process Time Elapsed: ${days}d ${hours}h ${minutes}m ${seconds}s"
