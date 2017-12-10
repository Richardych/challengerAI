#!/bin/sh

#sudo ./build/tools/caffe train -solver /home/superhui/caffe/models/chaai/solver.prototxt
./build/tools/caffe train -solver /home/yuchaohui/ych/caffe_ych/models/ResNet/tmp/newsplit/solver.prototxt -weights /home/yuchaohui/ych/caffe_ych/models/ResNet/ResNet-50-model.caffemodel -gpu 0
#./build/tools/caffe train -solver /home/yuchaohui/ych/caffe_ych/models/ResNet/tmp/newdatalay/solver.prototxt -snapshot /home/yuchaohui/ych/caffe_ych/models/ResNet/tmp/newdatalay/resnet_iter_20000.solverstate
