import sys, os
sys.path.insert(0, "../caffe_ms/python")
import caffe
import lmdb
import cv2
import numpy as np
from caffe.proto import caffe_pb2
from PIL import Image

'''
if len(sys.argv) <= 2:
    print 'usage'
    print './main model_folder test_gt'
    exit(-1)
'''

cls_2_label_path = 'cls_2_label'
model_path = '../trained_models/caffe/car_python_res/car_python_res50_ms_iter_60000.caffemodel'
deploy_path = 'ResNet-50-deploy.prototxt'
test_gt = '../deploy/test_data/test_gt_3_steps'

cls_list = [''] * len(open(cls_2_label_path).readlines())
cls_exist = {}

map_sub_cls_to_idx = {}
map_main_cls_to_idx = {}
cnt_sub = 0
cnt_main = 0
    
for line in open(cls_2_label_path):
    cls, idx = line.strip().rsplit(' ', 1)
    cls_list[int(idx)] = cls
    cls_exist[cls] = 1
    if not map_sub_cls_to_idx.has_key(cls):
        map_sub_cls_to_idx[cls] = cnt_sub;
        cnt_sub += 1
    if not map_main_cls_to_idx.has_key(cls.split('-')[0]):
        map_main_cls_to_idx[cls.split('-')[0]] = cnt_main;
        cnt_main += 1

print cnt_sub,  cnt_main
caffe.set_mode_gpu()

CONF=deploy_path
MODEL=model_path
net = caffe.Net(CONF, MODEL, caffe.TEST)

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
    #equ.save('histeq.jpg')
    return cvimg_to_nimg(equ)
 
    
def sub_local(nimg):
    a = nimg_to_cvimg(nimg)
    a = cv2.addWeighted(a, 1, cv2.GaussianBlur(a, (0,0), a.shape[1]/20), -1, 128)
    return cvimg_to_nimg(a)


cnt = 0
main_acc = 0
mid_acc = 0
sub_acc = 0

img_idx = 0

conf_mat_sub = np.zeros((cnt_sub, cnt_sub))
conf_mat_main = np.zeros((cnt_main, cnt_main))

input_size = 256
target_size = 240
test_lists = []


for line in open(test_gt):
    img, label = line.strip().split()
    if cls_exist.has_key(label) or cls_exist.has_key('back'+label):
        test_lists.append(line)

for line in test_lists:
    #print line.strip()
    img_filename, gt = line.strip().split()
    cvimg = cv2.imread(img_filename)
    if cvimg == None:
        print line
        continue
    crop = cv2.resize(cvimg,(input_size, input_size))

    data = np.array(crop, dtype=np.float32)
    #print data.shape
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

    cls = np.argsort(prob)
    pred = cls[::-1][0]

    #gt = line.strip().split('/')[-2]

    

    main_cls = cls_list[pred].split('-')[0]
    mid_cls = '-'.join(cls_list[pred].split('-')[:2])
    sub_cls = cls_list[pred]

    main_cls = main_cls.replace('back', '')
    mid_cls = mid_cls.replace('back', '')
    sub_cls = sub_cls.replace('back', '')
    
    #print 'cls', pred, 'conf', prob[pred]
    #print 'pred_cls', sub_cls, 'gt', gt

    if sub_cls == gt:
        sub_acc += 1
    if main_cls == gt.split('-')[0]:
        main_acc += 1
    else:
        print line.strip().split()[0]
        print 'cls', pred, 'conf', prob[pred]
        print 'pred_cls', sub_cls, 'gt', gt
    if mid_cls == '-'.join(gt.split('-')[:2]):
        mid_acc += 1

    #try:
    #    conf_mat_sub[map_sub_cls_to_idx[sub_cls]][map_sub_cls_to_idx[gt]] += 1
    #    conf_mat_main[map_main_cls_to_idx[main_cls]][map_main_cls_to_idx[gt.split('-')[0]]] += 1
    #except:
    #    print map_sub_cls_to_idx[sub_cls]
    #    print map_sub_cls_to_idx[gt]
    #    print map_main_cls_to_idx[main_cls]
    #    print map_main_cls_to_idx[gt.split('-')[0]]
    #    conf_mat_sub[map_sub_cls_to_idx[sub_cls]][map_sub_cls_to_idx[gt]] += 1
    #    conf_mat_main[map_main_cls_to_idx[main_cls]][map_main_cls_to_idx[gt.split('-')[0]]] += 1

    #if sub_cls_exist.has_key(gt):
    #    sub_img_idx += 1
    #    if sub_img_idx % 100 == 0:
            #print 'sub acc', sub_acc*1.0/sub_img_idx, sub_acc, sub_img_idx

    img_idx += 1
    #if img_idx % 100 == 0:
    #    print 'main acc', main_acc*1.0/img_idx, main_acc, img_idx
    #    print 'mid acc', mid_acc*1.0/img_idx, mid_acc, img_idx
    #    print 'sub acc', sub_acc*1.0/img_idx, sub_acc, img_idx
 

print 'main acc', main_acc*1.0/img_idx, main_acc, img_idx
print 'mid acc', mid_acc*1.0/img_idx, mid_acc, img_idx
print 'sub acc', sub_acc*1.0/img_idx, sub_acc, img_idx


 #0.70
 #0.76
