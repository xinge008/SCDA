#include <TH/TH.h>
#include <math.h>

int cpu_iou_overlaps(THFloatTensor * bboxes1, THFloatTensor * bboxes2, THFloatTensor * output){

    float * bboxes1_flat = THFloatTensor_data(bboxes1);
    float * bboxes2_flat = THFloatTensor_data(bboxes2);

    // TO BE IMPLEMENTED
}
