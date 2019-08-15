#ifndef _FOCAL_LOSS_SOFTMAX_KERNEL
#define _FOCAL_LOSS_SOFTMAX_KERNEL

#ifdef __cplusplus
extern "C" {
#endif

int SoftmaxFocalLossForwardLaucher(
    const int N, const float* logits,
    const int* targets, const float weight_pos,
    const float gamma, const float alpha,
    const int num_classes, float* losses,
    float* priors, cudaStream_t stream);

int SoftmaxFocalLossBackwardLaucher(
    const int N, const float* logits, 
    const int* targets, float* dX_data, const float weight_pos,
    const float gamma, const float alpha, const int num_classes,
    const float* priors, float* buff, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif
