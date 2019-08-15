cd src/cuda
echo "Compiling focal_loss kernels by nvcc..."
nvcc -c -o focal_loss_sigmoid_kernel.cu.o focal_loss_sigmoid_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_50
nvcc -c -o focal_loss_softmax_kernel.cu.o focal_loss_softmax_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_50
cd ../../
python build.py
