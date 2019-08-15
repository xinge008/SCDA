#include <math.h>
#include <THC/THC.h>
#include <assert.h>
#include <stdio.h>
#include "cuda/focal_loss_sigmoid_kernel.h"
#include "cuda/focal_loss_softmax_kernel.h"

extern THCState *state;

int focal_loss_sigmoid_forward_cuda(
                           int N,
                           THCudaTensor * logits,
                           THCudaIntTensor * targets,
                           float weight_pos,
                           float gamma, 
                           float alpha,
                           int num_classes,
                           THCudaTensor * losses){
    // Grab the input tensor
    float * logits_flat = THCudaTensor_data(state, logits);
    int * targets_flat = THCudaIntTensor_data(state, targets);

    float * losses_flat = THCudaTensor_data(state, losses);

    cudaStream_t stream = THCState_getCurrentStream(state);

    SigmoidFocalLossForwardLaucher(
        N, logits_flat, targets_flat, weight_pos, 
        gamma, alpha, num_classes, losses_flat, stream);

    return 1;
}

int focal_loss_sigmoid_backward_cuda(
                           int N,
                           THCudaTensor * logits,
                           THCudaIntTensor * targets,
                           THCudaTensor * dX_data,
                           float weight_pos,
                           float gamma,
                           float alpha,
                           int num_classes){
    // Grab the input tensor
    float * logits_flat = THCudaTensor_data(state, logits);
    int * targets_flat = THCudaIntTensor_data(state, targets);

    float * dX_data_flat = THCudaTensor_data(state, dX_data);

    cudaStream_t stream = THCState_getCurrentStream(state);
    SigmoidFocalLossBackwardLaucher(
        N, logits_flat, targets_flat, dX_data_flat,
        weight_pos, gamma, alpha, num_classes, stream);

    return 1;
}

int focal_loss_softmax_forward_cuda(
                           int N,
                           THCudaTensor * logits,
                           THCudaIntTensor * targets,
                           float weight_pos,
                           float gamma, 
                           float alpha,
                           int num_classes,
                           THCudaTensor * losses,
                           THCudaTensor * priors){
    // Grab the input tensor
    float * logits_flat = THCudaTensor_data(state, logits);
    int * targets_flat = THCudaIntTensor_data(state, targets);

    float * losses_flat = THCudaTensor_data(state, losses);
    float * priors_flat = THCudaTensor_data(state, priors);

    cudaStream_t stream = THCState_getCurrentStream(state);

    SoftmaxFocalLossForwardLaucher(
        N, logits_flat, targets_flat, weight_pos, 
        gamma, alpha, num_classes, losses_flat, priors_flat, stream);

    return 1;
}

int focal_loss_softmax_backward_cuda(
                           int N,
                           THCudaTensor * logits,
                           THCudaIntTensor * targets,
                           THCudaTensor * dX_data,
                           float weight_pos,
                           float gamma,
                           float alpha,
                           int num_classes,
                           THCudaTensor * priors,
                           THCudaTensor * buff){
    // Grab the input tensor
    float * logits_flat = THCudaTensor_data(state, logits);
    int * targets_flat = THCudaIntTensor_data(state, targets);

    float * dX_data_flat = THCudaTensor_data(state, dX_data);
    float * priors_flat = THCudaTensor_data(state, priors);
    float * buff_flat = THCudaTensor_data(state, buff);

    cudaStream_t stream = THCState_getCurrentStream(state);
    SoftmaxFocalLossBackwardLaucher(
        N, logits_flat, targets_flat, dX_data_flat,
        weight_pos, gamma, alpha, num_classes, priors_flat, buff_flat, stream);

    return 1;
}
