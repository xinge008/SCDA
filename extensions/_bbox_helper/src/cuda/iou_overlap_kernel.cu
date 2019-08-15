// #ifdef __cplusplus
// extern "C" {
// #endif

#include <math.h>
#include <stdio.h>
#include <float.h>
#include "iou_overlap_kernel.h"


#define DIVUP(m, n) ((m) / (m) + ((m) % (n) > 0))

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

//__device__ inline float devIoU(float const * const a, float const * const b) {
//  float left = fmaxf(a[0], b[0]), right = fminf(a[2], b[2]);
//  float top = fmaxf(a[1], b[1]), bottom = fminf(a[3], b[3]);
//  float width = fmaxf(right - left + 1, 0.f), height = fmaxf(bottom - top + 1, 0.f);
//  float interS = width * height;
//  float Sa = (a[2] - a[0] + 1) * (a[3] - a[1] + 1);
//  float Sb = (b[2] - b[0] + 1) * (b[3] - b[1] + 1);
//  return interS / (Sa + Sb - interS);
//}

__global__ void IOUOverlapKernel(
    const float* bbox1,
    const float* bbox2,
    const int size_bbox,
    const int num_bbox1,
    const int num_bbox2,
    float* top_data){
    CUDA_KERNEL_LOOP(index, num_bbox1 * num_bbox2){
        int b1 = index / num_bbox2;
        int b2 = index % num_bbox2;

        int base1 = b1 * size_bbox;
        float b1_x1 = bbox1[base1];
        float b1_y1 = bbox1[base1 + 1];
        float b1_x2 = bbox1[base1 + 2];
        float b1_y2 = bbox1[base1 + 3];
        float b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1); 

        int base2 = b2 * size_bbox;
        float b2_x1 = bbox2[base2];
        float b2_y1 = bbox2[base2 + 1];
        float b2_x2 = bbox2[base2 + 2];
        float b2_y2 = bbox2[base2 + 3];
        float b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1); 

        float left = fmaxf(b1_x1, b2_x1), right  = fminf(b1_x2, b2_x2);
        float top  = fmaxf(b1_y1, b2_y1), bottom = fminf(b1_y2, b2_y2);
        float width = fmaxf(right - left, 0.f), height = fmaxf(bottom - top, 0.f);
        float interS = width * height;
        float unionS = fmaxf(b1_area + b2_area - interS, 1.0);
        top_data[b1 * num_bbox2 + b2] = interS / unionS;
    }
}

int IOUOverlap(
    const float* bboxes1_data, 
    const float* bboxes2_data, 
    const int size_bbox,
    const int num_bbox1,
    const int num_bbox2,
    float* top_data,
    cudaStream_t stream){
        const int kThreadsPerBlock = 1024;
        int output_size = num_bbox1 * num_bbox2;
        //int output_size = num_bbox1;
        cudaError_t err;

        err = cudaGetLastError();
        if(cudaSuccess != err)
        {
            fprintf( stderr, "%s#%d: cudaCheckError() failed : %s\n", __FILE__,
                __LINE__, cudaGetErrorString( err ) );
            exit( -1 );
        }

        IOUOverlapKernel<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(
                     bboxes1_data, bboxes2_data, size_bbox, num_bbox1, num_bbox2, top_data);

    err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf( stderr, "%s#%d: cudaCheckError() failed : %s\n", __FILE__,
                __LINE__, cudaGetErrorString( err ) );
        exit( -1 );
    }

    return 1;
}

// #ifdef __cplusplus
// }
// #endif
