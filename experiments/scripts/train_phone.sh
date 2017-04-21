#!/usr/bin/env bash

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=0
NET=$1
DATASET=phone

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:1:$len}
# EXTRA_ARGS=${array[@]:2:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

case $DATASET in
  phone)
    TRAIN_IMDB="phone_train"
    TEST_IMDB="phone_test"
    PT_DIR="phone"
    ITERS=60000
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

LOG="experiments/logs/${DATASET}_train_${NET}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"


time python tools/train_phone.py --gpu ${GPU_ID} \
  --solver models/${PT_DIR}/${NET}/solver.prototxt \
  --imdb ${TRAIN_IMDB} \
  --iters ${ITERS} \
  --cfg experiments/cfgs/train_phone.yml \
  ${EXTRA_ARGS}

# set +x
# NET_FINAL=`grep -B 1 "done solving" ${LOG} | grep "Wrote snapshot" | awk '{print $4}'`
# set -x

# NET_FINAL="output/express_train/vgg16_faster_rcnn_iter_25000.caffemodel"
# time python tools/test_phone.py --gpu ${GPU_ID} \
#   --net ${NET_FINAL} \
#   --test_def models/${PT_DIR}/${NET}/test.prototxt \
#   --imdb ${TEST_IMDB} \
#   --cfg experiments/cfgs/train.yml \
#   ${EXTRA_ARGS}