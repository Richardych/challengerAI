# coding=utf-8
import os
import sys
root='/home/yuchaohui/ych/caffe_ych/'   #根目录 
sys.path.insert(0,root+'python')
import caffe
import numpy as np 
import cv2
from PIL import Image


deploy=root + 'models/ResNet/tmp/newsplit/ResNet-50-deploy.prototxt'    	#deploy文件 
caffe_model=root + 'models/ResNet/tmp/newsplit/resnet_iter_20000.caffemodel'  	#训练好的 caffemodel 
labels_filename = root +'models/ResNet/tmp/newsplit/scene_classes.csv'    		#类别名称文件，将数字标签转换回类别名称  
#mean_file = 'mean.npy'  #root + 'models/vgg/VGG_mean.binaryproto'

import os
dir = root+'scene_test_b_images_20170922'

filelist=[]
filenames = os.listdir(dir)
for fn in filenames:
   fullfilename = os.path.join(dir,fn)
   filelist.append(fullfilename)
 
#img=root+'testaip/20e7c0534d98b626b7faa9c41df00e2bc65e6cc5.jpg'   #随机找的一张待测图片 


def cal_prob(net, data):
    output = net.forward(**{"data":data})
    prob = output["prob"][0]
    return prob

def nimg_to_img(nimg):
    img = Image.fromarray(np.array(nimg.swapaxes(0,1).swapaxes(1,2), dtype=np.uint8))
    return img
def nimg_to_cvimg(nimg):
    cvimg = np.array(nimg.swapaxes(0,1).swapaxes(1,2), dtype=np.uint8)
    return cvimg
def cvimg_to_nimg(cvimg):
    nimg = np.array(cvimg.swapaxes(1,2).swapaxes(0,1), dtype=np.float32)
    return nimg
def img_to_nimg(img):
    nimg = np.array(img, dtype=np.float32)
    nimg = nimg.swapaxes(1,2).swapaxes(0,1)
    return nimg


def histeq(nimg):
    cvimg = nimg_to_cvimg(nimg)
    equ = cvimg[:,:,:]
    for i in range(3):
        equ[:,:,i] = cv2.equalizeHist(cvimg[:,:,i])
    return cvimg_to_nimg(equ)


def Test(img,net):
    """  
    im=caffe.io.load_image(img)                   #加载图片
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})  #设定图片的shape格式(1,3,28,28) 
    transformer.set_transpose('data', (2,0,1))    #改变维度的顺序，由原始图片(28,28,3)变为(3,28,28) 
    transformer.set_raw_scale('data', 255)    # 缩放到【0，255】之间 
    transformer.set_channel_swap('data', (2,1,0))   #交换通道，将图片由RGB变为BGR 
    #net.blobs['data'].reshape(1,3,240,240)
    im=caffe.io.load_image(img)                   #加载图片 
    net.blobs['data'].data[...] = transformer.preprocess('data',im)      #执行上面设置的图片预处理操作，并将图片载入到blob中 
    
    out = net.forward()
    prob= net.blobs['prob'].data[0].flatten() #取出最后一层（prob）属于某个类别的概率值，并打印,'prob'为最后一层的名称
    print prob
    """
    cvimg = cv2.imread(img)
    crop = cv2.resize(cvimg,(input_size, input_size))
    data = np.array(crop, dtype=np.float32)
    data = data.swapaxes(1,2).swapaxes(0,1)
    data = histeq(data)
    data -= 128.0
    data /= 100.0
    data = data.reshape((1, 3, input_size, input_size))
    list_data = []
    i = (input_size - target_size) / 2
    j = (input_size - target_size) / 2
    list_data.append(data[:,:,i:(i+target_size),j:(j+target_size)])
    prob = cal_prob(net, list_data[0])
    cls = prob.argsort()[-3:]
    print img
    print cls
    f=file(root+"models/ResNet/tmp/newsplit/resultlabelres50b.txt","a+")
    f.writelines(img+' '+str(cls)+'\n')

"""
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
    f=file(root+"models/ResNet/tmp/resultlabel-res50.txt","a+")
    f.writelines(img+' '+str(order)+'\n')
"""
input_size = 256
target_size = 240

caffe.set_mode_gpu()
caffe.set_device(1)
net = caffe.Net(deploy,caffe_model,caffe.TEST)   #加载model和network 
"""
for name in net.blobs.keys():
    print name
    print net.blobs[name].data
for para in net.params.keys():
    print para
    print net.params[para][0].data
    print net.params[para][1].data
"""
for i in range(0, len(filelist)):
    img= filelist[i]
    Test(img, net)






