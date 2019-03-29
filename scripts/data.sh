#!/usr/bin/env bash

LOCATION=$1   # example: /tmp/data
BATCH_SIZE=$2 # example: 64

echo "Downloading data"
gsutil -m cp -R gs://gqn-dataset/shepard_metzler_5_parts $LOCATION

echo "Deleting small records" # less than 10MB
DATA_PATH="$LOCATION/shepard_metzler_5_parts/**/*.tfrecord"
find $DATA_PATH -type f -size -10M | xargs rm

echo "Converting data"
python tfrecord-converter.py $LOCATION shepard_metzler_5_parts -b $BATCH_SIZE -m "train"
echo "Training data: done"
python tfrecord-converter.py $LOCATION shepard_metzler_5_parts -b $BATCH_SIZE -m "test"
echo "Testing data: done"

echo "Removing original records"
rm -rf "$LOCATION/shepard_metzler_5_parts/**/*.tfrecord"