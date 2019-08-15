#ifndef _IOU_OVERLAP_KERNEL
#define _IOU_OVERLAP_KERNEL

#ifdef __cplusplus
extern "C" {
#endif

int IOUOverlap(
    const float* bboxes1_data, const float* bboxes2_data, 
    const int size_bbox,
    const int num_bbox1,
    const int num_bbox2,
    float* top_data, 
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif

