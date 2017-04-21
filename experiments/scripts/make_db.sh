#!/usr/bin/env bash
# usage: scripts/make_db.sh /path/to/the/downloaded/dataset.zip

DATA=data/express
CAFFE=caffe-fast-rcnn

cd $(dirname ${BASH_SOURCE[0]})/../

# Parse arguments.
if [[ $# -ne 1 ]]; then
  echo "Usage: $(basename $0) dataset"
  echo "    dataset    Path to the downloaded dataset.zip"
  exit
fi


# Create the pretrain database
echo "Create pretrain database ..."
mkdir -p $DATA/pretrain_db
python2 tools/make_pretrain_dataset.py
for subset in train val; do
  $CAFFE/build/tools/convert_imageset \
    -encoded -resize_height 100 -resize_width 500 \
    $DATA/pretrain_db/ $DATA/pretrain_db/${subset}.txt \
    $DATA/pretrain_db/${subset}_lmdb
done
