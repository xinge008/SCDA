#include <stdio.h>
#include <math.h>
#include <float.h>
#include "focal_loss_softmax_kernel.h"

#define DIVUP(m, n) ((m) / (m) + ((m) % (n) > 0))

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

__global__ void SpatialSoftmaxKernel(const int N, const float* Xdata, float* Pdata,
    const int num_classes) {
  CUDA_1D_KERNEL_LOOP(index, N / num_classes) {
    int base = index * num_classes; //base index

    // Subtract max on each cell for numerical reasons
    float max_val = -FLT_MAX;
    for(int c = 0; c < num_classes; ++c) {
      max_val = max(max_val, Xdata[base + c]);
    }
    // Exponentiate
    float expsum = 0.0f;
    for(int c = 0; c < num_classes; ++c) {
      float expx = expf(Xdata[base + c] - max_val);
      Pdata[base + c] = expx;
      expsum += expx;
    }
    // Normalize
    for(int c = 0; c < num_classes; ++c) {
      Pdata[base + c] /= expsum;
    }
  }
}

__global__ void SoftmaxFocalLossKernel(
    const int N, 
    const float* Pdata, const int* targets, float* losses,
    const float weight_pos, const float gamma, const float alpha,
    const int num_classes) {
  CUDA_1D_KERNEL_LOOP(i, N / num_classes) {

    int base = i * num_classes;
    const int label = static_cast<int>(targets[i]);

    float Np = max(weight_pos, 1.0);
    float z = (label == 0) * (1 - alpha) / Np +
              (label >= 1) * alpha / Np;

    losses[i] = 0.0;
    if (label >= 0) {
      losses[i] =
          -(powf(1.0 - Pdata[base + label], gamma) *
          log(max(Pdata[base + label], FLT_MIN))) * z;
    }
  }
}

__global__ void SoftmaxFocalLossGradientWeightKernel(
    const int N,
    const float* Pdata, const int* targets, float* buff,
    const float weight_pos, const float gamma, const float alpha,
    const int num_classes) {
  CUDA_1D_KERNEL_LOOP(i, N / num_classes) {

    int base = i * num_classes;
    const int label = static_cast<int>(targets[i]);
    float Np = max(weight_pos, 1.0);
    float z =  (label == 0) * (1 - alpha) / Np +
               (label >= 1) * alpha / Np;

    buff[i] = 0.0;
    if (label >= 0) {
      float onemp = 1. - Pdata[base + label];
      float p = Pdata[base + label];
      buff[i] =
          (-powf(onemp, gamma) +
          gamma * powf(onemp, gamma - 1) * p * log(max(p, FLT_MIN))) * z;
    }
  }
}


__global__ void SoftmaxFocalLossGradientKernel(
    const int N,
    const float* Pdata, const int* targets, const float* buff,
    float* dX, const int num_classes) {
  CUDA_1D_KERNEL_LOOP(i, N) {

    int ind = i / num_classes;
    int cls = i % num_classes;

    const int label = static_cast<int>(targets[ind]);

    float c1 = (label >= 0) * 1.0;
    float c2 = (label == cls) * 1.0;
    dX[i] = 0.0;
    dX[i] = c1 * buff[ind] * (c2 - Pdata[i]);
  }
}

int SoftmaxFocalLossForwardLaucher(
    const int N, const float* logits,
    const int* targets, const float weight_pos,
    const float gamma, const float alpha,
    const int num_classes, float* losses,
    float* priors, cudaStream_t stream){

    const int kThreadsPerBlock = 1024;
    int output_size = N;
    cudaError_t err;

    err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf( stderr, "%s#%d: cudaCheckError() failed : %s\n", __FILE__,
                __LINE__, cudaGetErrorString( err ) );
        exit( -1 );
    }
    SpatialSoftmaxKernel<<<(output_size / num_classes + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(
      N, logits, priors, num_classes);

    SoftmaxFocalLossKernel<<<(output_size / num_classes + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(
      N, priors, targets, losses, weight_pos, gamma, alpha, num_classes);


    err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf( stderr, "%s#%d: cudaCheckError() failed : %s\n", __FILE__,
                __LINE__, cudaGetErrorString( err ) );
        exit( -1 );
    }

    return 1;
}


int SoftmaxFocalLossBackwardLaucher(
    const int N, const float* logits, const int* targets,
    float* dX_data, const float weight_pos,
    const float gamma, const float alpha, const int num_classes, 
    const float* priors, float* buff, cudaStream_t stream){

    const int kThreadsPerBlock = 1024;
    int output_size = N;
    cudaError_t err;

    SoftmaxFocalLossGradientWeightKernel<<<(output_size / num_classes + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(
        N, priors, targets, buff, weight_pos, gamma, alpha, num_classes);

    SoftmaxFocalLossGradientKernel<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(
        N, priors, targets, buff, dX_data, num_classes);

    err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf( stderr, "%s#%d: cudaCheckError() failed : %s\n", __FILE__,
                __LINE__, cudaGetErrorString( err ) );
        exit( -1 );
    }

    return 1;
}


