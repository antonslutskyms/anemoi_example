DATA_PATH=$1
OUTPUT_DIR=$2

echo "Data path: $DATA_PATH"
echo "Output directory: $OUTPUT_DIR"

mkdir -p /pscratch/sd/t/timothys/anemoi-house/replay/atmosphere-subsampled

ln -s $DATA_PATH /pscratch/sd/t/timothys/anemoi-house/replay/atmosphere-subsampled/p0

mkdir -p /pscratch/sd/t/timothys/anemoi-house/replay/atmosphere-subsampled/training-output
#ln -s $OUTPUT_DIR /pscratch/sd/t/timothys/anemoi-house/replay/atmosphere-subsampled/training-output

# export RANK=$OMPI_COMM_WORLD_NODE_RANK 
# export WORLD_RANK=$OMPI_COMM_WORLD_NODE_RANK 
# export GLOBAL_RANK=$OMPI_COMM_WORLD_NODE_RANK
# export LOCAL_RANK=$OMPI_COMM_WORLD_LOCAL_RANK
export WORLD_SIZE=2
export ANEMOI_BASE_SEED=1000
#export SLURM_GPUS_PER_NODE=2
#export SLURM_NNODES=1

echo "--------------- ENVIRONMENT --------------------"
env
echo "------------------------------------------------"

echo "---------------- CONDA -------------------------"
conda env list
echo "------------------------------------------------"

# move to environment
pip install flash-attn==2.7.3

echo "***********************************************"

#mpiexec --allow-run-as-root -n $WORLD_SIZE anemoi-training train --config-name=config
anemoi-training train --config-name=config

echo "Copying data to output dir: $OUTPUT_DIR"
cp -R /pscratch/sd/t/timothys/anemoi-house/replay/atmosphere-subsampled/training-output/* $OUTPUT_DIR
echo "Done copying data to output dir: $OUTPUT_DIR"
exit 0