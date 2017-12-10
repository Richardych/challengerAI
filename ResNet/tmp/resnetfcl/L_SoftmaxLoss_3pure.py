# -*- coding: utf-8 -*-
import sys
sys.path.append('/usr/lib/python2.7/dist-packages/')
import caffe
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.linalg.misc import norm
import operator
import math
import os
import cv2
import json

caffe_root=os.getcwd()
class L_SoftmaxLossLayer(caffe.Layer):
	def c(self,n,k):
		# calculate combinatorial number
		if k==0:
			return 1
		return reduce(operator.mul, range(n - k + 1, n + 1)) /reduce(operator.mul, range(1, k +1))
	
	def cal_k(self,cos, m):
		# calculate k for Large-Margin Softmax
		table = []
		pi = 3.14159
		for i in range(m+1):
			table.append(np.cos(i*pi/m))
		for i in range(m):
			if cos<=table[i] and cos>=table[i+1]:
				return i
		return m-1
	
	def calSoftmaxProb(self,score):
		# calculate softmax probability
		M = np.shape(score)[0]
		scoreMax = np.max(score,1)
		scoreMax = np.resize(scoreMax,[M,1])
		score -= scoreMax
		score = np.exp(score)
		prob = score*1.0/np.resize(np.sum(score,1),[M,1])
		return prob 

	def combineProb(self,main_list,mid_list,sub_prob):
		# calculate main brand probability and mid brand probability
		main_prob = np.zeros([np.shape(sub_prob)[0],len(main_list)],dtype=np.float32)
		mid_prob = np.zeros([np.shape(sub_prob)[0],len(mid_list)],dtype=np.float32)
		for i in range(len(main_list)):
			main_prob[:,i] = np.sum(sub_prob[:,main_list[i]],1)
		for i in range(len(mid_list)):
			mid_prob[:,i] = np.sum(sub_prob[:,mid_list[i]],1)
		return [main_prob,mid_prob]

	def calAccuracy(self,prob,groundtruth):
		M = np.shape(groundtruth)[0]
		result = np.ndarray.argmax(prob,1)
		result = np.resize(result,[M,1])
		re = result == groundtruth
		re = np.ndarray.tolist(re)
		return re.count([True])*1.0/M

	def calSoftmaxLoss(self,prob,groundtruth):
		M = np.shape(prob)[0]
		loss = 0
		for i in range(M):
			loss += -np.log(prob[i,int(groundtruth[i,0])])
		loss  = loss/M
		return loss
	
	def cal_L_SoftmaxLoss(self, weight, x ,groundtruth, m, ratio):
		M = np.shape(groundtruth)[0]
		score = np.dot(x, weight)
		scoreMax = np.max(score,1)
		scoreMax = np.resize(scoreMax,[M,1])
		score -= scoreMax
		score = np.exp(score)
		sum_score = np.sum(score,1)
		loss = 0
		for i in range(M):
			norm_mul = norm(weight[:,int(groundtruth[i,0])])*norm(x[i,:])
			cos = np.dot(np.transpose(weight[:,int(groundtruth[i,0])]), np.transpose(x[i,:])) / norm_mul
			k = self.cal_k(cos,m)
			cos_change = 0
			for n in range(int(m) / 2 + 1):
				cos_change += np.power(-1,n) * self.c(m,2*n) * np.power(cos,m-2*n) * np.power(1-cos*cos,n)
			angle = np.power(-1,k)*cos_change-2*k
			numerator = (ratio*score[i,int(groundtruth[i,0])]+np.exp(norm_mul*angle-scoreMax[i,0]))/(1+ratio)
			denominator = numerator + sum_score[i] - score[i,int(groundtruth[i,0])]
			loss += -np.log(numerator/denominator)
		loss /= M
		return loss
	
	def calSoftmaxDerivative_weight(self, weight, x, groundtruth):
		M = np.shape(groundtruth)[0]
		diff_L_f = self.calSoftmaxProb(np.dot(x,weight)) # M*labels
		for i in range(M):
			# diff_L_f of groundtruth
			diff_L_f[i,int(groundtruth[i,0])]-=1
		diff_L_weight = np.dot(np.transpose(x),diff_L_f) # num_filters_l-1 * labels
		diff_L_weight /= M
		return diff_L_weight
	
	def calSoftmaxDerivative_x(self, weight, x, groundtruth):
		M = np.shape(groundtruth)[0]
		diff_L_f = self.calSoftmaxProb(np.dot(x,weight)) # M*labels
		for i in range(M):
			# diff_L_f of groundtruth
			diff_L_f[i,int(groundtruth[i,0])]-=1
		diff_L_x = np.dot(diff_L_f,np.transpose(weight)) # M*num_filters_l-1
		diff_L_x /= M
		return diff_L_x

	def combineWeight(self, weight, label_list):
		weight_label = np.zeros([np.shape(weight)[0],len(label_list)],dtype=np.float32)
		for i in range(len(label_list)):
			weight_label[:,i] = np.sum(weight[:,label_list[i]],1)/len(label_list[i])
		return weight_label
	
	def divideWeight(self, weight, label_list, number):
		diff_L_weight_result = np.zeros([np.shape(weight)[0],number],dtype=np.float32)
		for i in range(len(label_list)):
			diff_L_weight_result[:,label_list[i]] = np.resize(weight[:,i],[np.shape(weight)[0],1])/len(label_list[i])
		return diff_L_weight_result
	
	def cal_L_SoftmaxLoss_MainMid(self, weight, x, groundtruth, label_list, m, ratio):
		weight_label = self.combineWeight(weight, label_list)
		return self.cal_L_SoftmaxLoss(weight_label, x ,groundtruth, m, ratio)
	
	def cal_L_SoftmaxDerivative_weight_MainMid(self, weight, x, groundtruth, label_list, m, ratio):
		weight_label = self.combineWeight(weight, label_list)
		diff_L_weight = self.cal_L_SoftmaxDerivative_weight(weight_label, x, groundtruth, m, ratio)
		diff_L_weight = self.divideWeight(diff_L_weight, label_list, np.shape(weight)[1])
		return diff_L_weight
	
	def cal_L_SoftmaxDerivative_x_MainMid(self, weight, x, groundtruth, label_list, m, ratio):
		weight_label = self.combineWeight(weight, label_list)
		diff_L_x = self.cal_L_SoftmaxDerivative_x(weight_label, x, groundtruth, m, ratio)
		return diff_L_x
	
	def cal_L_SoftmaxDerivative_weight(self, weight, x, groundtruth, m, ratio):
		M = np.shape(groundtruth)[0]
		diff_L_f_nogt = self.calSoftmaxProb(np.dot(x,weight)) # M*labels
		diff_L_f_gt = np.zeros_like(diff_L_f_nogt, dtype=np.float32) # M*labels
		diff_L_weight = np.zeros_like(weight, dtype=np.float32) # num_filters_l-1 * labels
		diff_f_weight = np.transpose(x) # num_filters_l-1 * M
		diff_f_weight_gt = np.zeros_like(diff_f_weight, dtype=np.float32) # num_filters_l-1 * M
	
		for i in range(M):
			norm_weight = norm(weight[:,int(groundtruth[i,0])])
			norm_x = norm(x[i,:])
			norm_mul = norm_weight*norm_x
			vector_weight = weight[:,int(groundtruth[i,0])]
			vector_x = np.transpose(x[i,:])
			vector_mul = np.dot(np.transpose(vector_weight), vector_x)
			cos = vector_mul*1.0/norm_mul
			k = self.cal_k(cos,m)
			temp = 0
			n = int(m)/2
	
			# diff_L_f of groundtruth
			diff_L_f_gt[i,int(groundtruth[i,0])] = diff_L_f_nogt[i,int(groundtruth[i,0])] - 1
			diff_L_f_nogt[i,int(groundtruth[i,0])] = 0
	
			# diff_f_weight of groundtruth
			temp = m*1.0*np.dot(np.power(vector_mul,m-1),vector_x)/np.power(norm_mul,m-1)
			temp -= (m-1)*1.0*np.dot(np.power(vector_mul,m),vector_weight)/np.power(norm_weight,m+1)/np.power(norm_x,m-1)
			temp *= self.c(m,0)
			if n==1:
				temp1 = -(m-2)*1.0*np.dot(np.power(vector_mul,m-3),vector_x)/np.power(norm_mul,m-3)
				temp1 += (m-3)*1.0*np.dot(np.power(vector_mul,m-2),vector_weight)/np.power(norm_weight,m-1)/np.power(norm_x,m-3)
				temp1 += m*1.0*np.dot(np.power(vector_mul,m-1),vector_x)/np.power(norm_mul,m-1)
				temp1 -= (m-1)*1.0*np.dot(np.power(vector_mul,m),vector_weight)/np.power(norm_weight,m+1)/np.power(norm_x,m-1)
				temp1 *= self.c(m,2)
				temp += temp1
			diff_f_weight_gt[:,i] = np.power(-1,k)*temp-2.0*k*norm_x*vector_weight/norm_weight # num_filters_l-1 * 1
		diff_f_weight_gt = (ratio*np.transpose(x)+diff_f_weight_gt)/(1+ratio)
		diff_L_weight = np.dot(diff_f_weight,diff_L_f_nogt)+np.dot(diff_f_weight_gt,diff_L_f_gt) # number_filters_l-1 * labels
		diff_L_weight /= M
		return diff_L_weight
	
	def cal_L_SoftmaxDerivative_x(self, weight, x, groundtruth, m, ratio):
		M = np.shape(groundtruth)[0]
		labels = np.shape(weight)[1]
		diff_L_f_nogt = self.calSoftmaxProb(np.dot(x,weight)) # M*labels
		diff_L_f_gt = np.zeros_like(diff_L_f_nogt, dtype=np.float32) # M*labels
		diff_L_x = np.zeros_like(x, dtype=np.float32) # M * number_filters_l-1
		diff_f_x = np.transpose(weight) # labels * number_filters_l-1
		diff_f_x_gt = np.zeros_like(diff_f_x, dtype=np.float32) # labels * number_filters_l-1
		for i in range(M):
			norm_weight = norm(weight[:,int(groundtruth[i,0])])
			norm_x = norm(x[i,:])
			norm_mul = norm_weight*norm_x
			vector_weight = weight[:,int(groundtruth[i,0])] #  number_filters_l-1 * 1
			vector_x = np.transpose(x[i,:])
			vector_mul = np.dot(np.transpose(vector_weight), vector_x)
			cos = vector_mul*1.0/norm_mul
			k = self.cal_k(cos,m)
			temp = 0
			n = int(m)/2
	
			# diff_L_f of groundtruth
			diff_L_f_gt[i,int(groundtruth[i,0])] = diff_L_f_nogt[i,int(groundtruth[i,0])] - 1
			diff_L_f_nogt[i,int(groundtruth[i,0])] = 0
	
			# diff_f_x of groundtruth
			temp = m*1.0*np.dot(np.power(vector_mul,m-1),vector_weight)/np.power(norm_mul,m-1)
			temp -= (m-1)*1.0*np.dot(np.power(vector_mul,m),vector_x)/np.power(norm_weight,m-1)/np.power(norm_x,m+1)
			temp *= self.c(m,0)
			if n==1:
				temp1 = -(m-2)*1.0*np.dot(np.power(vector_mul,m-3),vector_weight)/np.power(norm_mul,m-3)
				temp1 += (m-3)*1.0*np.dot(np.power(vector_mul,m-2),vector_x)/np.power(norm_weight,m-3)/np.power(norm_x,m-1)
				temp1 += m*1.0*np.dot(np.power(vector_mul,m-1),vector_weight)/np.power(norm_mul,m-1)
				temp1 -= (m-1)*1.0*np.dot(np.power(vector_mul,m),vector_x)/np.power(norm_weight,m-1)/np.power(norm_x,m+1)
				temp1 *= self.c(m,2)
				temp += temp1
			temp = np.power(-1,k)*temp-2.0*k*norm_weight*vector_x/norm_x # 1 * num_filters_l-1
			temp = (ratio*np.transpose(vector_weight)+temp)/(1+ratio)
			diff_f_x_temp = np.zeros_like(diff_f_x, dtype=np.float32)
			diff_f_x_temp[int(groundtruth[i,0]),:] += np.transpose(temp)  
			diff_L_x[i,:] = np.dot(diff_L_f_gt[i,:],diff_f_x_temp)  # 1 * num_filters_l-1 
		diff_L_x += np.dot(diff_L_f_nogt,diff_f_x) # M * number_filters_l-1
		diff_L_x /= M
		return diff_L_x
	 
	def genMainMidDict(self, main_labels, mid_labels):
		main_dict={}
		mid_dict={}
		main_2_label_path=caffe_root+'/python/main_2_label'
		mid_2_label_path=caffe_root+'/python/mid_2_label'
		cls_2_label_path=caffe_root+'/python/cls_2_label'
		cls_list={}

		main_list=[[] for i in range(main_labels)]
		mid_list=[[] for i in range(mid_labels)]

		for line in open(main_2_label_path,'r'):
			main_cls, idx = line.strip().rsplit(' ', 1)
			main_dict[main_cls]=int(idx)
		for line in open(mid_2_label_path,'r'):
			mid_cls, idx = line.strip().rsplit(' ', 1)
			mid_dict[mid_cls]=int(idx)
		for line in open(cls_2_label_path,'r'):
			cls, idx = line.strip().rsplit(' ', 1)
			cls_list[int(idx)] = cls
			main_label = cls.split('-')[0]
			mid_label = '-'.join(cls.split('-')[:2])
			main_idx = main_dict[main_label]
			mid_idx = mid_dict[mid_label]
			main_list[main_idx].append(idx)
			mid_list[mid_idx].append(idx)
		return [main_list,mid_list,cls_list,main_dict,mid_dict]

	def setup(self, bottom, top):
		print 'L_SoftmaxLoss entering'
		self._name_to_bottom_map={'fc7':0,'label':1}

		param = json.loads(self.param_str)
		self.init_type = param['init_type']
		self.std = float(param['gaussian_std'])
		self.mainLossWeight = float(param['mainLossWeight'])
		self.midLossWeight = float(param['midLossWeight'])
		self.subLossWeight = float(param['subLossWeight'])
		self.main_labels = int(param['main_labels'])
		self.mid_labels = int(param['mid_labels'])
		self.sub_labels = int(param['sub_labels'])
		self.m = int(param['m'])
		self.weight_file = param['weight_file']

		self.ratio = 10000
		self.iteration = 0

		self.blobs.add_blob(np.shape(bottom[0].data)[1],self.sub_labels)
		self.blobs[0].data[...] = np.zeros([np.shape(bottom[0].data)[1],self.sub_labels], dtype = np.float32)
		self.diff = np.zeros_like(bottom[0].data,dtype = np.float32)
		
		if self.init_type == 'gaussian':
			np.random.seed(0)
			s = np.random.normal(0, self.std, np.size(self.blobs[0].data))
			self.blobs[0].data[...] = np.resize(s,np.shape(self.blobs[0].data))
		[self.main_list,self.mid_list,self.cls_list,self.main_dict,self.mid_dict] = self.genMainMidDict(self.main_labels,self.mid_labels)
		print 'L_SoftmaxLoss end'

	def reshape(self, bottom, top):
		self.bottom_shape = np.shape(bottom[0].data)
		#print self.bottom_shape
		top[0].reshape(1)
		
	def forward(self, bottom, top):
		self.iteration += 1
		ratio_min = 5
		if self.iteration%10000==0 and self.ratio>ratio_min:
			self.ratio *= 0.2
			if self.ratio<ratio_min:
				self.ratio=ratio_min

		
		# divide groundtruth label
		M = np.shape(bottom[1].data)[0]
		self.main_gt = np.zeros([M,self.main_labels],dtype=np.int)
		self.mid_gt = np.zeros([M,self.mid_labels],dtype=np.int)
		for i in range(M):
			cls_label = bottom[1].data[i,0]
			cls_name = self.cls_list[int(cls_label)]
			main_label = self.main_dict[cls_name.split('-')[0]]
			mid_label = self.mid_dict['-'.join(cls_name.split('-')[:2])]
			self.main_gt[i,0] = int(main_label)
			self.mid_gt[i,0] = int(mid_label)
			#print cls_label, self.main_gt[i,0], self.mid_gt[i,0]

		x = bottom[0].data
		main_loss = self.cal_L_SoftmaxLoss_MainMid(self.blobs[0].data, x, self.main_gt, self.main_list, self.m, self.ratio)
		mid_loss = self.cal_L_SoftmaxLoss_MainMid(self.blobs[0].data, x, self.mid_gt, self.mid_list, self.m, self.ratio)
		sub_loss = self.cal_L_SoftmaxLoss(self.blobs[0].data, x, bottom[1].data, self.m, self.ratio)
		top[0].data[...] = self.mainLossWeight*main_loss+self.midLossWeight*mid_loss+self.subLossWeight*sub_loss

		if self.iteration%20==0:
			fc_result = np.dot(x, self.blobs[0].data)
			sub_prob = self.calSoftmaxProb(fc_result)
			[self.main_prob, self.mid_prob] = self.combineProb(self.main_list,self.mid_list,sub_prob)
			main_accuracy = self.calAccuracy(self.main_prob,self.main_gt)
			mid_accuracy = self.calAccuracy(self.mid_prob,self.mid_gt)
			sub_accuracy = self.calAccuracy(sub_prob,bottom[1].data)
			print self.main_gt[0,0], self.mid_gt[0,0], bottom[1].data[0,0]
			print self.main_prob[0,:], self.mid_prob[0,:], sub_prob[0,:]
			print "loss: ", top[0].data[...], "   MainAccuracy: ", main_accuracy, "   MidAccuracy: ", mid_accuracy, "   SubAccuracy: ", sub_accuracy
		
		main_x_diff = self.cal_L_SoftmaxDerivative_x_MainMid(self.blobs[0].data, x, self.main_gt, self.main_list, self.m, self.ratio)
		mid_x_diff = self.cal_L_SoftmaxDerivative_x_MainMid(self.blobs[0].data, x, self.mid_gt, self.mid_list, self.m, self.ratio)
		sub_x_diff = self.cal_L_SoftmaxDerivative_x(self.blobs[0].data, x, bottom[1].data, self.m, self.ratio)
		self.diff[...] = main_x_diff + mid_x_diff + sub_x_diff
		#self.gradientChecking(self.blobs[0].data, x, self.main_gt, self.m, self.ratio, 0.0001)

	def backward(self, top, propagate_down, bottom):
		x = bottom[0].data
		main_weight_diff = self.cal_L_SoftmaxDerivative_weight_MainMid(self.blobs[0].data, x, self.main_gt, self.main_list, self.m, self.ratio)
		mid_weight_diff = self.cal_L_SoftmaxDerivative_weight_MainMid(self.blobs[0].data, x, self.mid_gt, self.mid_list, self.m, self.ratio)
		sub_weight_diff = self.cal_L_SoftmaxDerivative_weight(self.blobs[0].data, x, bottom[1].data, self.m, self.ratio)
		self.blobs[0].diff[...] = main_weight_diff + mid_weight_diff + sub_weight_diff
		bottom[0].diff[...] = self.diff[...]
		
