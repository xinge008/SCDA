// ------------------------------------------------------------------
// Faster R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Shaoqing Ren
// ------------------------------------------------------------------
#include <THC/THC.h>
#include <TH/TH.h>
#include <math.h>
#include <assert.h>
#include <stdio.h>

#include "cuda/iou_overlap_kernel.h"


extern THCState *state;

int gpu_iou_overlaps(THCudaTensor * bboxes1, THCudaTensor * bboxes2, THCudaTensor * output){
    // Grad the input tensor
    float * bboxes1_data = THCudaTensor_data(state, bboxes1);
    float * bboxes2_data = THCudaTensor_data(state, bboxes2);
    float * output_data = THCudaTensor_data(state, output);

    // Number of boxes
    int num_bbox1 = THCudaTensor_size(state, bboxes1, 0);
    int num_bbox2 = THCudaTensor_size(state, bboxes2, 0);
    int size_bbox1 = THCudaTensor_size(state, bboxes1, 1);
    int size_bbox2 = THCudaTensor_size(state, bboxes2, 1);
    
    assert(size_bbox1 == 4);
    assert(size_bbox2 == 4);
    if(size_bbox1 != 4 || size_bbox2 != 4){
        exit(1);
        return 0;
    }
 
    cudaStream_t stream = THCState_getCurrentStream(state);
    IOUOverlap(
               bboxes1_data,
               bboxes2_data,
               size_bbox1,
               num_bbox1,
               num_bbox2,
               output_data,
               stream);
  return 1;
}
