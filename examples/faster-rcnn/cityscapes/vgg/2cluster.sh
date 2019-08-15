#!/bin/bash
ROOT=../../../..
export PYTHONPATH=$ROOT:$PYTHONPATH
#--------------------------
job_name=training_2cluster
ckdir=2cluster
mkdir -p ./${ckdir}/${job_name}
#--------------------------
PARTITION=$1
GPUS=${5:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}

srun -p ${PARTITION} --ntasks=${GPUS} --gres=gpu:${GPUS_PER_NODE} \
		--ntasks-per-node=${GPUS_PER_NODE} \
        --job-name=${job_name} \
python -u -W ignore $ROOT/tools/faster_rcnn_train_val.py \
  --config=config_512.json \
  --dist=1 \
  --fix_num=0 \
  --L1=1 \
  --cluster_num=2 \
  --threshold=256 \
  --recon_size=512 \
  --port=21603 \
  --arch=vgg16_FasterRCNN \
  --warmup_epochs=1 \
  --lr=0.0000125 \
  --step_epochs=16,22 \
  --batch-size=1 \
  --epochs=25 \
  --dataset=cityscapes \
  --train_meta_file=/path/to/train.txt \
  --target_meta_file=/path/to/foggy_train.txt \
  --val_meta_file=/path/to/foggy_val.txt \
  --datadir=/path/to/leftImg8bit/ \
  --pretrained=/path/to/torchvision_models/vgg16-397923af.pth \
  --results_dir=${ckdir}/${job_name}/results_dir \
  --save_dir=${ckdir}/${job_name} \
  2>&1 | tee ${ckdir}/${job_name}/train.log
