#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1

DATA_DIR=$1

# Start TensorBoard in background
tensorboard --logdir "../logs" &
TENSORBOARD_PID=$!
echo "Started Tensorboard with PID: $TENSORBOARD_PID"

# Start training script
python ../run-gqn.py \
    --data_dir $DATA_DIR \
    --log_dir "../logs" \
    --data_parallel "True" \
    --batch_size 1 \
    --workers 6
