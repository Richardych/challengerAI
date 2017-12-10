#!/bin/sh

# Compute the mean image from the imagenettraining lmdb
# N.B. this is available in data/ilsvrc12

DATA=/home/yuchaohui/ych/caffe_ych/data/chaai/
TOOLS=/home/yuchaohui/ych/caffe_ych/build/tools
 

$TOOLS/compute_image_mean $DATA/img_train_lmdb \
  $DATA/mean.binaryproto


echo "Done."
