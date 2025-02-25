#!/bin/bash -l
# FILE: sh_i/bridge/llama_pi.sh

#### Choose Partition
#SBATCH --partition=debug

#### cluster specific settings
#SBATCH --qos=normal
#SBATCH --mem=256G
#SBATCH --time=48:00:00

#### number of nodes and tasks
# nodes
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# GPUs
#SBATCH --gpus-per-node=1
# CPUs
#SBATCH --cpus-per-task=10

#### job specific info
#SBATCH --job-name="cz"
#SBATCH --output="./out/cz-%j.out" # Path to store logs
#SBATCH --mail-type=ALL
#SBATCH --mail-user=asandhu9@jh.edu

######################
### Set environment ###
######################

# Record setup start time
setup_time=$(date +%s)

source ~/.bashrc
source .venv/bin/activate  # Your virtual environment

# Set CUDA 12.2 bin and library paths
export PATH="/usr/local/cuda-12.2/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH"

######################
### Environment Vars #
######################
export TORCH_CPP_LOG_LEVEL=INFO
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# NCCL / communication settings
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eno1
export NCCL_P2P_LEVEL=TRACE

# Configure OpenMPI to use TCP, Shared Memory, and Self
export OMPI_MCA_btl=tcp,sm,self

# Check for srun
which srun || { echo "ERROR: srun not found in PATH."; exit 1; }

###########################
######## Run Job ##########
###########################
echo "Job started on $(date)"
start_time=$(date +%s)

elapsed=$((start_time - setup_time))
days=$((elapsed / 86400))
hours=$(( (elapsed % 86400) / 3600 ))
minutes=$(( (elapsed % 3600) / 60 ))
seconds=$((elapsed % 60))
echo "Setup Time elapsed: ${days}d ${hours}h ${minutes}m ${seconds}s"

INFER_SCRIPT="test"

srun python -m $INFER_SCRIPT

###########################
### Post-Job Actions    ###
###########################
echo "Job completed on $(date)"
end_time=$(date +%s)

elapsed=$((end_time - start_time))
days=$((elapsed / 86400))
hours=$(( (elapsed % 86400) / 3600 ))
minutes=$(( (elapsed % 3600) / 60 ))
seconds=$((elapsed % 60))
echo "Process Time Elapsed: ${days}d ${hours}h ${minutes}m ${seconds}s"