#!/usr/bin/env bash

CUDA_PATH=/usr/local/cuda/

cd src
echo "Compiling my_lib kernels by nvcc..."
nvcc -c -o roi_pooling.cu.o roi_pooling_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_50

cd ../
python build.py

