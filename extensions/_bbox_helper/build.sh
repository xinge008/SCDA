cd src/cuda
echo "Compiling nms kernels by nvcc..."
nvcc -c -o iou_overlap_kernel.cu.o iou_overlap_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_50
cd ../../
python build.py
