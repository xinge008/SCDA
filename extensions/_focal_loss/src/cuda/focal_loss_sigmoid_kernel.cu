#include <stdio.h>
#include <math.h>
#include <float.h>
#include "focal_loss_sigmoid_kernel.h"

#define DIVUP(m, n) ((m) / (m) + ((m) % (n) > 0))

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

__global__ void SigmoidFocalLossKernel(
    const int N, const float* logits,
    const int* targets, const float weight_pos,
    const float gamma, const float alpha,
    const int num_classes, float* losses) {
  CUDA_1D_KERNEL_LOOP(i, N) {
      int d = i % num_classes;   //current class
      int tmp = i / num_classes; //targets index
      int t = targets[tmp];

    // check whether the class is true class or not.
    // The target classes are in range 1 - 81 and the d is in range 0-80
    // because we predict A*80 dim, so for comparison purpose, compare t and (d+1)
    float c1 = (t == (d + 1));
    float c2 = (t != -1 & t != (d + 1));

    float Np = max(weight_pos, 1.0);
    float zn = (1.0 - alpha) / Np;
    float zp = alpha / Np;

    // p = 1. / 1. + expf(-x)
    float p = 1. / (1. + expf(-logits[i]));

    // (1 - p)**gamma * log(p) where
    float term1 = powf((1. - p), gamma) * logf(max(p, FLT_MIN));
    // p**gamma * log(1 - p)
    float term2 =
        powf(p, gamma) *
        (-1. * logits[i] * (logits[i] >= 0) -
         logf(1. + expf(logits[i] - 2. * logits[i] * (logits[i] >= 0))));

    losses[i] = 0.0;
    losses[i] += -c1 * term1 * zp;
    losses[i] += -c2 * term2 * zn;
  }
}

__global__ void SigmoidFocalLossGradientKernel(
    const int N, const float* logits,
    const int* targets, float* dX_data, const float weight_pos,
    const float gamma, const float alpha, const int num_classes) {
  CUDA_1D_KERNEL_LOOP(i, N) {
      int d = i % num_classes;   //current class
      int tmp = i / num_classes; //targets index
      int t = targets[tmp];

      float Np = max(weight_pos, 1.0);
      float zn = (1.0 - alpha) / Np;
      float zp = alpha / Np;
      //int t = targets[n * (H * W * A) + a * (H * W) + y * W + x];

      float c1 = (t == (d + 1));
      float c2 = (t != -1 & t != (d + 1));
      float p = 1. / (1. + expf(-logits[i]));

      // (1-p)**g * (1 - p - g*p*log(p))
      float term1 =
          powf((1. - p), gamma) *
          (1. - p - (p * gamma * logf(max(p, FLT_MIN))));
      // (p**g) * (g*(1-p)*log(1-p) - p)
      float term2 =
          powf(p, gamma) *
          ((-1. * logits[i] * (logits[i] >= 0) -
           logf(1. + expf(logits[i] - 2. * logits[i] * (logits[i] >= 0)))) *
           (1. - p) * gamma - p);
      dX_data[i] = 0.0;
      dX_data[i] += -c1 * zp * term1;
      dX_data[i] += -c2 * zn * term2;
  }
}

int SigmoidFocalLossForwardLaucher(
    const int N, const float* logits,
    const int* targets, const float weight_pos,
    const float gamma, const float alpha,
    const int num_classes, float* losses, cudaStream_t stream){

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
    SigmoidFocalLossKernel<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(
      N, logits, targets, weight_pos, gamma, alpha, num_classes, losses);
    err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf( stderr, "%s#%d: cudaCheckError() failed : %s\n", __FILE__,
                __LINE__, cudaGetErrorString( err ) );
        exit( -1 );
    }

    return 1;
}


int SigmoidFocalLossBackwardLaucher(
    const int N, const float* logits, const int* targets,
    float* dX_data, const float weight_pos,
    const float gamma, const float alpha, const int num_classes,
    cudaStream_t stream){

    const int kThreadsPerBlock = 1024;
    int output_size = N;
    cudaError_t err;

    SigmoidFocalLossGradientKernel<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(
        N, logits, targets, dX_data, weight_pos, gamma, alpha, num_classes);
    err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf( stderr, "%s#%d: cudaCheckError() failed : %s\n", __FILE__,
                __LINE__, cudaGetErrorString( err ) );
        exit( -1 );
    }

    return 1;
}


