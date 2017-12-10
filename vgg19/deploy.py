# coding=utf-8
import os
import sys
root='/home/yuchaohui/ych/caffe_ych/'   #根目录 
sys.path.insert(0,root+'build/python')
import caffe 
import numpy as np 

deploy=root + 'models/vgg/vgg19/VGG_ILSVRC_19_layers_deploy.prototxt'    	#deploy文件 
caffe_model=root + 'models/vgg/vgg19/caffe_vgg_train_iter_60000.caffemodel'  	#训练好的 caffemodel 
labels_filename = root +'models/vgg/vgg19/scene_classes.csv'    		#类别名称文件，将数字标签转换回类别名称  
#mean_file = 'mean.npy'  #root + 'models/vgg/VGG_mean.binaryproto'

import os
dir = root+'scene_test_a_images_20170922'

filelist=[]
filenames = os.listdir(dir)
for fn in filenames:
   fullfilename = os.path.join(dir,fn)
   filelist.append(fullfilename)
 
#img=root+'testaip/20e7c0534d98b626b7faa9c41df00e2bc65e6cc5.jpg'   #随机找的一张待测图片 

def Test(img):
    caffe.set_mode_gpu()
    caffe.set_device(0)
    net = caffe.Net(deploy,caffe_model,caffe.TEST)   #加载model和network 
    #图片预处理设置 
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})  #设定图片的shape格式(1,3,28,28) 
    transformer.set_transpose('data', (2,0,1))    #改变维度的顺序，由原始图片(28,28,3)变为(3,28,28) 
    #transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))    #减去均值，前面训练模型时没有减均值，这儿就不用 
    transformer.set_raw_scale('data', 255)    # 缩放到【0，255】之间 
    transformer.set_channel_swap('data', (2,1,0))   #交换通道，将图片由RGB变为BGR 
       
    im=caffe.io.load_image(img)                   #加载图片 
    net.blobs['data'].data[...] = transformer.preprocess('data',im)      #执行上面设置的图片预处理操作，并将图片载入到blob中 
       
    #执行测试 
    out = net.forward() 
       
    labels = np.loadtxt(labels_filename, str, delimiter='\t')   #读取类别名称文件 
    prob= net.blobs['prob'].data[0].flatten() #取出最后一层（prob）属于某个类别的概率值，并打印,'prob'为最后一层的名称
    #print 'prob:>>>>',prob,'\n'
    print len(prob)
    order=prob.argsort()[-3:]    #[4]  #将概率值排序，取出最大值所在的序号 ,9指的是分为0-9十类 
    print 'order:>>>>>>>>',order,'\n'
    #print len(order)
    #argsort()函数是从小到大排列 
    #print 'the class is:',labels[order]   #将该序号转换成对应的类别名称，并打印 
    f=file(root+"models/vgg/vgg19/resultlabel.txt","a+")
    f.writelines(img+' '+str(order)+'\n')


for i in range(2196, len(filelist)):
    img= filelist[i]
    print img
    Test(img)

"""
def meanprotoTonpy(meanprotofile):
    import numpy as np
    MEAN_PROTO_PATH = meanprotofile                    # 待转换的pb格式图像均值文件路径
    MEAN_NPY_PATH = 'mean.npy'                         # 转换后的numpy格式图像均值文件路径

    blob = caffe.proto.caffe_pb2.BlobProto()           # 创建protobuf blob   
    data = open(MEAN_PROTO_PATH, 'rb' ).read()         # 读入mean.binaryproto文件内容
    blob.ParseFromString(data)                         # 解析文件内容到blob
    array = np.array(caffe.io.blobproto_to_array(blob))# 将blob中的均值转换成numpy格式，array的shape （mean_number，channel, hight, width）
    mean_npy = array[0]                                # 一个array中可以有多组均值存在，故需要通过下标选择其中一组均值
    np.save(MEAN_NPY_PATH ,mean_npy)
"""
