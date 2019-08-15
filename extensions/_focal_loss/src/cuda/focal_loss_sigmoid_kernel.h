#ifndef _FOCAL_LOSS_SIGMOID_KERNEL
#define _FOCAL_LOSS_SIGMOID_KERNEL

#ifdef __cplusplus
extern "C" {
#endif

int SigmoidFocalLossForwardLaucher(
    const int N, const float* logits,
    const int* targets, const float weight_pos,
    const float gamma, const float alpha,
    const int num_classes, float* losses, cudaStream_t stream);

int SigmoidFocalLossBackwardLaucher(
    const int N, const float* logits, 
    const int* targets, float* dX_data, const float weight_pos,
    const float gamma, const float alpha, const int num_classes,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif
