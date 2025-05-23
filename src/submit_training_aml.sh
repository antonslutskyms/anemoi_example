DATA_PATH=$1
OUTPUT_DIR=$2

echo "Data path: $DATA_PATH"
echo "Output directory: $OUTPUT_DIR"

mkdir -p /pscratch/sd/t/timothys/anemoi-house/replay/atmosphere-subsampled

ln -s $DATA_PATH /pscratch/sd/t/timothys/anemoi-house/replay/atmosphere-subsampled/p0
ln -s $OUTPUT_DIR /pscratch/sd/t/timothys/anemoi-house/replay/atmosphere-subsampled/training-output

export RANK=0
export WORLD_RANK=0
export GLOBAL_RANK=0
export LOCAL_RANK=0
export WORLD_SIZE=1
export ANEMOI_BASE_SEED=1000
export SLURM_GPUS_PER_NODE=1
export SLURM_NNODES=1

# move to environment
pip install flash-attn

#mpiexec --allow-run-as-root -n $WORLD_SIZE anemoi-training train --config-name=config
anemoi-training train --config-name=config
