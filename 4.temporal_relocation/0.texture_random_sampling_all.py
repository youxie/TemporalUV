#******************************************************************************
#
# TemporalUV: Capturing Loose Clothing with Temporally Coherent UV Coordinates
# Copyright 2022 You Xie, Huiqi Mao, Angela Yao, Nils Thuerey
#
# This program is free software, distributed under the terms of the
# Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#
#******************************************************************************

# before temporal relocation, we achieve texture random sampling for all frames.



import tensorflow as tf
import os
import numpy as np
import cv2 as cv
import sys
sys.path.append(os.path.abspath('../tools'))
from ops import *
os.environ["CUDA_VISIBLE_DEVICES"]="4"


start_ite = 1
total_run = 101
imgheight = 224
imgwidth = 176
first_run = True
img = tf.placeholder(tf.float32,[None,imgheight,imgwidth,3])
IUV = tf.placeholder(tf.float32,[None,imgheight,imgwidth,3])
update = []
if first_run:
	texture_map1,tup = TransferTextureback_tf(1,img,IUV)
	update.append(tup)
else:
	texture_map1 = tf.placeholder(tf.float32,[None,1200,800,3])

texture_map = texture_map1
def add_zero_bound(img):
	img1 = tf.zeros([tf.shape(img)[0],1,tf.shape(img)[2]+2,3])
	img2 = tf.concat([tf.zeros_like(img[:,:,0:1,:]),img,tf.zeros_like(img[:,:,0:1,:])],2)
	img = tf.concat([img1,img2,img1],1)
	return img

def bilinear_gen(img,IUV):
	img_withb = add_zero_bound(img)
	img_l = img_withb[:,1:-1,1:-1,:]
	img_r = tf.roll(img_withb,shift=-1, axis=2)[:,1:-1,1:-1,:]
	IUV_withb = add_zero_bound(IUV)
	IUV_l = IUV_withb[:,1:-1,1:-1,:]
	IUV_r = tf.roll(IUV_withb,shift=-1, axis=2)[:,1:-1,1:-1,:]

	ratio = tf.random.uniform([tf.shape(img)[0],tf.shape(img)[1],tf.shape(img)[2],1],minval=0, maxval=1)
	ratio = tf.concat([ratio,ratio,ratio],-1)
	img_x = tf.multiply(1-ratio,tf.cast(img_l,tf.float32))+tf.multiply(ratio,tf.cast(img_r,tf.float32))
	IUV_x = tf.multiply(1-ratio,tf.cast(IUV_l,tf.float32))+tf.multiply(ratio,tf.cast(IUV_r,tf.float32))


	img_x = add_zero_bound(img_x)
	img_u = img_x[:,1:-1,1:-1,:]
	img_b = tf.roll(img_x,shift=-1, axis=1)[:,1:-1,1:-1,:]

	IUV_x = add_zero_bound(IUV_x)
	IUV_u = IUV_x[:,1:-1,1:-1,:]
	IUV_b = tf.roll(IUV_x,shift=-1, axis=1)[:,1:-1,1:-1,:]

	ratio = tf.random.uniform([tf.shape(img)[0],tf.shape(img)[1],tf.shape(img)[2],1],minval=0, maxval=1)
	ratio = tf.concat([ratio,ratio,ratio],-1)
	img_y = tf.multiply(1-ratio,tf.cast(img_b,tf.float32))+tf.multiply(ratio,tf.cast(img_u,tf.float32))
	IUV_y = tf.multiply(1-ratio,tf.cast(IUV_b,tf.float32))+tf.multiply(ratio,tf.cast(IUV_u,tf.float32))
	return img_y,IUV_y

img2,IUV2 = bilinear_gen(img,IUV)
texture_map2,tup = TransferTextureback_tf(1,img2,IUV2)
update.append(tup)
texture_map = tf.where(tf.equal(texture_map,tf.constant([0.,0.,0.])),texture_map2,texture_map)

data_name = "A1jFMj0n1JS"
input_IUV_path = "/mnt/netdisk1/youxie/Fashion_extrapolation/%s/%s_optimization/"%(data_name,data_name)
input_img_path = "/mnt/netdisk1/youxie/Fashion_extrapolation/%s/rmbg/"%(data_name)
save_coor_path = "rs_%s_noseg/"%(data_name)

if not os.path.exists(save_coor_path):
    os.makedirs(save_coor_path)

with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())
	for i in range(1,5,1):
		first_run = True
		if os.path.exists(save_coor_path + "%04d.png"%i):
			continue
		IUV_data = np.load("%s%04d.npy"%(input_IUV_path,i))[22:-22,8:-8,:][np.newaxis,::4,::4,:]
		img_data = np.array(cv.imread("%s%04d.png"%(input_img_path,i)),np.float32)[22:-22,8:-8,:][np.newaxis,::4,::4,:]#
		for ite in range(start_ite,total_run):
		    print("frame:%d, %d iteration starts"%(i,ite))
		    if not first_run:
		        t_map = np.load(save_coor_path + "%04d.npy"%(i))[np.newaxis,:,:,:]
		        texture_out,img_2,IUV_2,_ = sess.run([texture_map,img2,IUV2,update],feed_dict={img:img_data, IUV:IUV_data, texture_map1: t_map})
		    else:
		        print(np.shape(IUV_data),np.shape(img_data))
		        texture_out,img_2,IUV_2,_ = sess.run([texture_map,img2,IUV2,update],feed_dict={img:img_data, IUV:IUV_data})
		        first_run = False
		    cv.imwrite(save_coor_path + "%04d.png"%(i), texture_out[0])
		    np.save(save_coor_path + "%04d.npy"%(i), texture_out[0])
