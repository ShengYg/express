#!/usr/bin/env bash

CAFFE=caffe-fast-rcnn
DATASET=psdb_softmax
NET=$1
SNAPSHOTS_DIR=output/${DATASET}_pretrain

LOG="experiments/logs/${DATASET}_pretrain_${NET}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"

cd $(dirname ${BASH_SOURCE[0]})/../../

mkdir -p ${SNAPSHOTS_DIR}

GLOG_logtostderr=1 ${CAFFE}/build/tools/caffe train \
  -solver models/${DATASET}/${NET}/pretrain_solver.prototxt \
  -weights data/imagenet_models/${NET}.caffemodel 2>&1 | tee ${LOG}

ITER=$2
GLOG_logtostderr=1 ${CAFFE}/build/tools/caffe test \
  -model models/${DATASET}/${NET}/pretrain.prototxt \
  -weights output/psdb_train/ResNet-50_iter_${ITER}.caffemodel \
  -gpu 0 -iterations 1000 2>&1 | tee ${LOG}