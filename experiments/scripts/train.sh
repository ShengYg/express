#!/usr/bin/env bash
# Usage:
# ./experiments/scripts/train.sh GPU [options args to {train,test}_net.py]
#
# Example:
# ./experiments/scripts/train.sh 0 \
#   --set EXP_DIR foobar RNG_SEED 42 TRAIN.SCALES "[400, 500, 600, 700]"

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=0
NET="VGG16"
DATASET=express


case $DATASET in
  express)
    TRAIN_IMDB="express_train"
    TEST_IMDB="express_test"
    PT_DIR="express"
    ITERS=40000
    YML="train.yml"
    ;;
  phone)
    TRAIN_IMDB="phone_train"
    TEST_IMDB="phone_test"
    PT_DIR="phone"
    ITERS=60000
    YML="train.yml"
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

LOG="experiments/logs/${DATASET}_train_${NET}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

# LOG="experiments/logs/phone_train.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
# python tools/pretrain_mnist.py all 2>&1 | tee "$LOG"


# python tools/train_net.py --gpu ${GPU_ID} \
#   --solver models/${PT_DIR}/${NET}/solver.prototxt \
#   --imdb ${TRAIN_IMDB} \
#   --iters ${ITERS} \
#   --cfg experiments/cfgs/${YML} \
#   ${EXTRA_ARGS}

# set +x
# NET_FINAL=`grep -B 1 "done solving" ${LOG} | grep "Wrote snapshot" | awk '{print $4}'`
# set -x

NET_FINAL="output/express_train/express_iter_25000.caffemodel"
python tools/test_net.py --gpu ${GPU_ID} \
  --net ${NET_FINAL} \
  --test_def models/${PT_DIR}/${NET}/test.prototxt \
  --imdb ${TEST_IMDB} \
  --cfg experiments/cfgs/${YML} \
  ${EXTRA_ARGS}

####################################################################################################

# python tools/train_net.py --gpu ${GPU_ID} \
#   --solver models/${PT_DIR}/${NET}/solver.prototxt \
#   --imdb ${TRAIN_IMDB} \
#   --iters ${ITERS} \
#   --cfg experiments/cfgs/${YML} \
#   --weights output/mnist_out/mnist_iter_10000.caffemodel
#   ${EXTRA_ARGS}

# set +x
# NET_FINAL=`grep -B 1 "done solving" ${LOG} | grep "Wrote snapshot" | awk '{print $4}'`
# set -x


# NET_FINAL="output/phone_out/phone_iter_60000.caffemodel"
# python tools/test_net.py --gpu ${GPU_ID} \
#   --net ${NET_FINAL} \
#   --test_def models/${PT_DIR}/${NET}/test.prototxt \
#   --imdb ${TEST_IMDB} \
#   --cfg experiments/cfgs/${YML} \
#   ${EXTRA_ARGS}


