
int focal_loss_sigmoid_forward_cuda(
                           int N,
                           THCudaTensor * logits,
                           THCudaIntTensor * targets,
                           float weight_pos,
                           float gamma, 
                           float alpha,
                           int num_classes,
                           THCudaTensor * losses);

int focal_loss_sigmoid_backward_cuda(
                           int N,
                           THCudaTensor * logits,
                           THCudaIntTensor * targets,
                           THCudaTensor * dX_data,
                           float weight_pos,
                           float gamma,
                           float alpha,
                           int num_classes);

int focal_loss_softmax_forward_cuda(
                           int N,
                           THCudaTensor * logits,
                           THCudaIntTensor * targets,
                           float weight_pos,
                           float gamma, 
                           float alpha,
                           int num_classes,
                           THCudaTensor * losses,
                           THCudaTensor * priors);

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
                           THCudaTensor * buff);
