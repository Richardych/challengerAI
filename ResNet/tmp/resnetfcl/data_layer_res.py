import numpy as np
import os
import sys
import cv2
import h5py 
import json

#caffe_root='/mnt/data3/zdb/DeepV_cls/v1.6_and_v1.5/caffe'
caffe_root=os.getcwd()
sys.path.insert(0,caffe_root+'/python')
import caffe
import lmdb

class PythonDataLayer(caffe.Layer):
    #typ represents the tpye of flip images
    def flip(self,nimg):
        typ=np.random.randint(2)
        if typ==1:# flip the image left and right
            nimg=nimg[::,::,::-1]
        return nimg
    def random_crop(self, nimg):
        start_x = np.random.randint(self.BORDER)
        start_y = np.random.randint(self.BORDER)
        end_x = start_x + self.IMAGE_SIZE - self.BORDER
        end_y = start_y + self.IMAGE_SIZE - self.BORDER
        return nimg[:, start_x:end_x, start_y:end_y]

    def sub_crop(self, nimg):
        typ=np.random.randint(4)
        crop_ratio_width = 0.4
        crop_ratio_height = 0.5
        if typ==0:
            start_x = np.random.randint(crop_ratio_width*self.IMAGE_SIZE)
            end_x = self.IMAGE_SIZE
            start_y = 0
            end_y = self.IMAGE_SIZE
        elif typ==1:
            start_x = 0
            end_x = self.IMAGE_SIZE - np.random.randint(crop_ratio_width*self.IMAGE_SIZE)
            start_y = 0
            end_y = self.IMAGE_SIZE
        elif typ==2:
            start_x = 0
            end_x = self.IMAGE_SIZE
            start_y = np.random.randint(crop_ratio_height*self.IMAGE_SIZE)
            end_y = self.IMAGE_SIZE
        else:
            start_x = 0
            end_x = self.IMAGE_SIZE
            start_y = 0
            end_y = self.IMAGE_SIZE
        nimg = nimg[:, start_y:end_y, start_x:end_x]
        cvimg = self.nimg_to_cvimg(nimg)
        cvimg = cv2.resize(cvimg, (self.IMAGE_SIZE, self.IMAGE_SIZE))
        return self.cvimg_to_nimg(cvimg)
        
            


    def center_crop(self, nimg):
        start_x = self.BORDER / 2
        start_y = self.BORDER / 2
        end_x = start_x + self.IMAGE_SIZE - self.BORDER
        end_y = start_y + self.IMAGE_SIZE - self.BORDER
        return nimg[:, start_x:end_x, start_y:end_y]


    def nimg_to_img(self, nimg):
        from PIL import Image
        img = Image.fromarray(np.array(nimg.swapaxes(0,1).swapaxes(1,2), dtype=np.uint8))
        return img
    def nimg_to_cvimg(self, nimg):
        cvimg = np.array(nimg.swapaxes(0,1).swapaxes(1,2), dtype=np.uint8)
        return cvimg
    def cvimg_to_nimg(self, cvimg):
        nimg = np.array(cvimg.swapaxes(1,2).swapaxes(0,1), dtype=np.float32)
        return nimg

    def img_to_nimg(self, img):
        nimg = np.array(img, dtype=np.float32)
        nimg = nimg.swapaxes(1,2).swapaxes(0,1)
        return nimg

    def random_rotate(self, nimg):
        img = self.nimg_to_img(nimg)
        max_degree = 5
        angle = (np.random.random() - 0.5)*2 * max_degree
        img = img.rotate(angle)
        return self.img_to_nimg(img)


    def histeq(self, nimg):
        #definity do this if called
        typ=np.random.randint(2)
        if typ == 0:
            cvimg = self.nimg_to_cvimg(nimg)
            equ = cvimg[:,:,:]
            for i in range(3):
                equ[:,:,i] = cv2.equalizeHist(cvimg[:,:,i])
            #equ.save('histeq.jpg')
            try:
                assert(equ.shape==(self.IMAGE_SIZE, self.IMAGE_SIZE, 3))
            except:
                print equ.shape
                assert(equ.shape==(self.IMAGE_SIZE, self.IMAGE_SIZE, 3))
            return self.cvimg_to_nimg(equ)
        else:
            return nimg
    
    
    def sub_local(self, nimg):
        a = self.nimg_to_cvimg(nimg)
        a = cv2.addWeighted(a, 1, cv2.GaussianBlur(a, (0,0), self.IMAGE_SIZE/20), -1, 128)
        return self.cvimg_to_nimg(a)


    def setup(self,bottom,top):
        print 'set up entering'
        self._name_to_top_map={'data':0,'label':1}

        param = json.loads(self.param_str)
        self.path = param['path']
        self.batch_size = int(param['batch_size'])
        self.input_size = int(param['input_size'])
        self.target_size = int(param['target_size'])
        self.mode = param['mode']

        self.env=lmdb.Environment(self.path,readonly=False,map_size=4 * 1024 *1024,metasync=False,sync=True,map_async=True)
        self.index = 0
        self.key_list = []
        self.txn = self.env.begin()
        self.cursor = self.txn.cursor()
        self.IMAGE_SIZE = self.input_size
        self.BORDER = self.input_size -  self.target_size
        for idx, key in enumerate(self.cursor.iternext_nodup()):
            self.key_list.append(key)
        print 'train num', len(self.key_list)
        print 'setup is over'

    def reshape(self,bottom,top):
        top[0].reshape(self.batch_size,3,self.IMAGE_SIZE - self.BORDER, self.IMAGE_SIZE - self.BORDER)
        top[1].reshape(self.batch_size,1)        
        #top[2].reshape(self.batch_size,1)        


    def forward(self,bottom,top):
        for i in range(self.batch_size):
            choose_key = self.key_list[(self.index + i)%len(self.key_list)]
            raw_datum=self.cursor.get(choose_key)

            datum = caffe.proto.caffe_pb2.Datum()
            datum.ParseFromString(raw_datum)
            data=np.fromstring(datum.data, dtype=np.uint8)
            data=np.reshape(data,(datum.channels, datum.height, datum.width))
            data=np.array(data, dtype=np.float32)

            if self.mode == "train":
                #data=self.sub_local(data)
                data=self.sub_crop(data)
                data=self.flip(data)
                data=self.histeq(data)
                data=self.random_rotate(data)
                data=self.random_crop(data)
                #data=self.center_crop(data)
            else:
                data=self.histeq(data)
                #data=self.sub_local(data)
                data=self.center_crop(data)
            data -= 128.0
            data /= 100.0
            top[0].data[i,0,:,:]=data[0,:,:]
            top[0].data[i,1,:,:]=data[1,:,:]
            top[0].data[i,2,:,:]=data[2,:,:]
            
            #example of flipping images

            #top[1].data[i,0]=int(datum.label) % 1000
            top[1].data[i,0]=datum.label
            #print datum.label / 1000
            #top[2].data[i,0]=int(datum.label) / 1000
        self.index += self.batch_size
        self.index %= len(self.key_list)
           
    def backward(self,top,propagate_Down,bottom):
        pass
