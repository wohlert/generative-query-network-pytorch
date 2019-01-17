#!/usr/bin/env bash

LOCATION=$1
BATCH_SIZE=$2

echo "Downloading data"
gsutil -m cp -R gs://gqn-dataset/shepard_metzler_5_parts $LOCATION

echo "Deleting small records"
TRAIN_PATH="$LOCATION/shepard_metzler_5_parts/train"
find "$TRAIN_PATH/*.tfrecord" -type f -size -10M | xargs rm # remove smaller than 10mb

echo "Converting data"
python tfrecord-converter.py $LOCATION shepard_metzler_5_parts -b $BATCH_SIZE -m "train"
echo "Training data: done"
python tfrecord-converter.py $LOCATION shepard_metzler_5_parts -b $BATCH_SIZE -m "test"
echo "Testing data: done"