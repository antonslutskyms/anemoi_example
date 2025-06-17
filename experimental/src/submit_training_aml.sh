DATA_PATH=$1
OUTPUT_DIR=$2

echo "Data path: $DATA_PATH"
echo "Output directory: $OUTPUT_DIR"

echo "--------------- ENVIRONMENT --------------------"
env
echo "------------------------------------------------"

mkdir -p /pscratch/sd/t/timothys/anemoi-house/replay/atmosphere-subsampled

ln -s $DATA_PATH /pscratch/sd/t/timothys/anemoi-house/replay/atmosphere-subsampled/p0
ln -s $OUTPUT_DIR /pscratch/sd/t/timothys/anemoi-house/replay/atmosphere-subsampled/training-output

export RANK=$OMPI_COMM_WORLD_NODE_RANK || $AZUREML_CR_NODE_RANK
export WORLD_RANK=$OMPI_COMM_WORLD_NODE_RANK || $AZUREML_CR_NODE_RANK
export GLOBAL_RANK=$OMPI_COMM_WORLD_NODE_RANK || $AZUREML_CR_NODE_RANK
export LOCAL_RANK=$OMPI_COMM_WORLD_NODE_RANK || $AZUREML_CR_NODE_RANK
export WORLD_SIZE=2
export ANEMOI_BASE_SEED=1000
export SLURM_GPUS_PER_NODE=1
export SLURM_NNODES=1

echo "--------------- ENVIRONMENT #2 --------------------"
env
echo "------------------------------------------------"

# move to environment
pip install flash-attn

#mpiexec --allow-run-as-root -n $WORLD_SIZE anemoi-training train --config-name=config
anemoi-training train --config-name=config
