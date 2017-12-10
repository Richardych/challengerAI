#!/bin/sh

#sudo ./build/tools/caffe train -solver /home/superhui/caffe/models/chaai/solver.prototxt
./build/tools/caffe train -solver /home/yuchaohui/ych/caffe_ych/models/ResNet/tmp/newsplitnotzs/solver.prototxt -weights /home/yuchaohui/ych/caffe_ych/models/ResNet/ResNet-50-model.caffemodel -gpu 1
#./build/tools/caffe train -solver /home/yuchaohui/ych/caffe_ych/models/vgg/vgg19/solver.prototxt -snapshot /home/yuchaohui/ych/caffe_ych/models/vgg/vgg19/caffe_vgg_train_iter_45000.solverstate
