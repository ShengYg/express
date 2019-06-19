#!/usr/bin/env bash

CUDA_PATH=/usr/local/cuda/

python setup.py build_ext --inplace
rm -rf build

# cd roi_pooling/src/cuda
# echo "Compiling roi pooling kernels by nvcc..."
# nvcc -c -o roi_pooling.cu.o roi_pooling_kernel.cu \
# 	 -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -arch=sm_52

# g++ -std=c++11 -shared -o roi_pooling.so roi_pooling_op.cc \
# 	roi_pooling_op.cu.o -I $TF_INC -fPIC -lcudart -L $CUDA_PATH/lib64
# cd ../../
# python build.py

# rm nms/cpu_nms.c
# rm nms/cpu_nms.so
# rm nms/gpu_nms.cpp
# rm nms/gpu_nms.so
# rm roi_pooling/_ext/roi_pooling/_roi_pooling.so
# rm roi_pooling/src/cuda/roi_pooling.cu.o
# rm utils/bbox.c
# rm utils/cython_bbox.so
# rm utils/cython_nms.so
# rm utils/nms.c

