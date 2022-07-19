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

#optimization
#Here we build differentiable transormations between images and textures via UV maps. UV maps are initialized with UVs after extrapolation and optimized to minimize differences between input image and recovered image.


import tensorflow as tf
import cv2 as cv
import os
import numpy as np
import sys
sys.path.append(os.path.abspath('../tools'))
from ops import *
from tensorflow.python.ops.parallel_for.gradients import jacobian
import argparse

import subprocess as sp
def get_gpu_memory():
    _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
    ACCEPTABLE_AVAILABLE_MEMORY = 1024 *5
    COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    available_index = -1
    for i in range(len(memory_free_values)):
        if memory_free_values[i] > ACCEPTABLE_AVAILABLE_MEMORY:
            available_index = i
    if available_index < 0:
        print("no available GPU (all GPUs are occupied)!!!")
        exit()
    return str(available_index)

gpu_index = get_gpu_memory()

os.environ["CUDA_VISIBLE_DEVICES"]=gpu_index


parser = argparse.ArgumentParser()
parser.add_argument("--frame_index", type=int, default=0, help="frame index")
parser.add_argument("--iter", type=int, default=1001, help="optimization iteration")
parser.add_argument("--factor", type=float, default=0., help="factor")
parser.add_argument("--IUV_initial_path",type=str, required=True)
args = parser.parse_args()

texture_height = 1200
texture_width = 800
img_height = 940
img_width = 720
max_index = 24
update = []
batch_size = 1
smooth = 1

def get_pixel_value(img, x, y):
    shape = tf.shape(x)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]

    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
    b = tf.tile(batch_idx, (1, height, width))
    indices = tf.concat([b[:,:,:,tf.newaxis],x[:,:,:,tf.newaxis],y[:,:,:,tf.newaxis]],-1)
    return tf.gather_nd(img, indices)

def bilinear_sampler(img, x, y):
    """
    Performs bilinear sampling of the input images according to the
    normalized coordinates provided by the sampling grid. Note that
    the sampling is done identically for each channel of the input.
    To test if the function works properly, output image should be
    identical to input image when theta is initialized to identity
    transform.
    Input
    -----
    - img: batch of images in (B, H, W, C) layout.
    - grid: x, y which is the output of affine_grid_generator.
    Returns
    -------
    - out: interpolated images according to grids. Same size as grid.
    """
    H = texture_width
    W = texture_height
    max_y = tf.cast(H - 1, 'int32')
    max_x = tf.cast(W - 1, 'int32')
    zero = tf.zeros([], dtype='int32')

    # rescale x and y to [0, W-1/H-1]
    x = tf.cast(x, 'float32')
    y = tf.cast(y, 'float32')

    # grab 4 nearest corner points for each (x_i, y_i)
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1

    # clip to range [0, H-1/W-1] to not violate img boundaries
    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)

    # get pixel value at corner coords
    Ia = get_pixel_value(img, x0, y0)
    Ib = get_pixel_value(img, x0, y1)
    Ic = get_pixel_value(img, x1, y0)
    Id = get_pixel_value(img, x1, y1)

    # recast as float for delta calculation
    x0 = tf.cast(x0, 'float32')
    x1 = tf.cast(x1, 'float32')
    y0 = tf.cast(y0, 'float32')
    y1 = tf.cast(y1, 'float32')

    # calculate deltas
    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    # add dimension for addition
    wa = tf.expand_dims(wa, axis=3)
    wb = tf.expand_dims(wb, axis=3)
    wc = tf.expand_dims(wc, axis=3)
    wd = tf.expand_dims(wd, axis=3)

    # compute output
    out = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])

    return out

def get_mask(image):
    with tf.variable_scope("Mask") as scope:
        tmpVar = tf.Variable(tf.zeros([image.get_shape()[0],image.get_shape()[1], image.get_shape()[2],1],tf.float32), trainable=False)
        position = tf.where(tf.logical_or(tf.logical_or(tf.greater(image[:,:,:,0],0),tf.greater(image[:,:,:,1],0)),tf.greater(image[:,:,:,2],0)))
        xyzvalue = tf.ones_like(position[:,:1],dtype=tf.float32)
        output = tf.scatter_nd_update(tmpVar, tf.cast(position,tf.int32), xyzvalue)
        update = tmpVar.assign(tf.zeros([image.get_shape()[0], image.get_shape()[1], image.get_shape()[2],1],tf.float32))
        return output,update


original_grid_np = np.zeros([batch_size,texture_height,texture_width,2])
for i in range(texture_height):
    for j in range(texture_width):
        original_grid_np[0,i,j] = [i,j]
original_grid = tf.convert_to_tensor(original_grid_np,dtype = tf.float32)

IUV_data_orig = np.zeros([batch_size,texture_height,texture_width,3])
img_data_orig = np.zeros([batch_size,texture_height,texture_width,3])
IUV_initial_orig = np.zeros([batch_size,texture_height,texture_width,3])
new_part_orig = np.zeros([batch_size,texture_height,texture_width,1])
old_part_orig = np.ones([batch_size,texture_height,texture_width,1])

frame = args.frame_index
if smooth:
    output_path = "A1wwPTTzVGS_optimization/"
else:
    output_path = "A1wwPTTzVGS_optimization/"
print("%dth frame optimization starts!"%frame)
if not os.path.exists(output_path):
    os.mkdir(output_path)
    os.system("cp %s %s"%(__file__,output_path))

IUV_data_np = np.array(cv.imread("/mnt/netdisk1/youxie/Fashion_extrapolation/A1wwPTTzVGS/densepose_IUV/%04d.png"%frame),np.float32)[np.newaxis,:,:,:]
image_data_np = np.array(cv.imread("/mnt/netdisk1/youxie/Fashion_extrapolation/A1wwPTTzVGS/rmbg/%04d.png"%frame),np.float32)[np.newaxis,:,:,:]
IUV_initial_np = np.array(cv.imread(args.IUV_initial_path + "%04d.png"%frame),np.float32)[np.newaxis,:,:,:]

for i in range(img_height):
    for j in range(img_width):
        if (IUV_data_np[0,i,j]>0).any():
            IUV_data_orig[0,i,j] = IUV_data_np[0,i,j]
        if (image_data_np[0,i,j]>0).any():
            img_data_orig[0,i,j] = image_data_np[0,i,j]
        if (IUV_initial_np[0,i,j]>0).any():
            IUV_initial_orig[0,i,j] = IUV_initial_np[0,i,j]
        if (IUV_initial_np[0,i,j]>0).any():
            new_part_orig[0,i,j] = 1

old_part_orig = old_part_orig - new_part_orig

old_part_IUV = np.multiply(IUV_data_orig,old_part_orig)
new_part_IUV = np.multiply(IUV_initial_orig,new_part_orig)

config = tf.ConfigProto()
config.allow_soft_placement=True
sess = tf.Session(config=config)

IUV_var = tf.Variable(initial_value=IUV_initial_orig, trainable=True, dtype = tf.float32)

mask_new_part = tf.convert_to_tensor(new_part_orig)
mask_old_part = tf.convert_to_tensor(old_part_orig)

IUV_data = tf.multiply(IUV_var,tf.cast(tf.concat([tf.zeros_like(IUV_initial_orig[:,:,:,:1]),mask_new_part,mask_new_part],-1),tf.float32))

IUV_data = IUV_data + tf.cast(tf.multiply(tf.convert_to_tensor(IUV_initial_orig),mask_old_part),tf.float32)

IUV_data = tf.concat([IUV_data[:,:,:,:1] + tf.cast(tf.convert_to_tensor(new_part_IUV[:,:,:,:1]),tf.float32),tf.cast(IUV_data[:,:,:,1:2],tf.float32),tf.cast(IUV_data[:,:,:,2:],tf.float32)],-1)

gradient_loss = tf.constant(0.)
if smooth:
    img_gradient_x, img_gradient_y = tf.image.image_gradients(IUV_data)
    img_gradient_xx, img_gradient_xy = tf.image.image_gradients(img_gradient_x)
    img_gradient_yx, img_gradient_yy = tf.image.image_gradients(img_gradient_y)
    gradient_loss = tf.reduce_mean(tf.square(img_gradient_x-1)*100 + tf.square(img_gradient_y-1)*100)
    gradient_gradient_loss_tmp = tf.gradients(gradient_loss, IUV_data)[0]
    gradient_gradient_loss_large = tf.gradients(gradient_loss, IUV_data)[0]
    for ax in range(2):
        for leng in [-2,-1,1,2]:
            gradient_gradient_loss_neighbor = tf.roll(gradient_gradient_loss_tmp,leng,ax)
            gradient_gradient_loss_large = tf.where(tf.less(tf.abs(gradient_gradient_loss_large),tf.abs(gradient_gradient_loss_neighbor)),gradient_gradient_loss_large,gradient_gradient_loss_neighbor)

    for x_leng in [-2,-1,1,2]:
        for y_leng in [-2,-1,1,2]:
            gradient_gradient_loss_neighbor = tf.roll(gradient_gradient_loss_tmp,[x_leng,y_leng],[1,2])
            gradient_gradient_loss_large = tf.where(tf.less(tf.abs(gradient_gradient_loss_large),tf.abs(gradient_gradient_loss_neighbor)),gradient_gradient_loss_large,gradient_gradient_loss_neighbor)

    gradient_gradient_loss = tf.where(tf.greater(tf.abs(gradient_gradient_loss_tmp),5e-4),gradient_gradient_loss_large/100.,gradient_gradient_loss_tmp)
    IUV_loss = (tf.reduce_mean(tf.square(img_gradient_xx) + tf.square(img_gradient_xy) + tf.square(img_gradient_yx) + tf.square(img_gradient_yy))) * 10
    gradient_gradient_IUV_loss = tf.gradients(IUV_loss, IUV_data)[0]
    gradient_gradient_IUV_loss_large = tf.gradients(IUV_loss, IUV_data)[0]
    for ax in range(2):
        for leng in [-2,-1,1,2]:
            gradient_gradient_IUV_loss_neighbor = tf.roll(gradient_gradient_IUV_loss,leng,ax)
            gradient_gradient_IUV_loss_large = tf.where(tf.less(tf.abs(gradient_gradient_IUV_loss),tf.abs(gradient_gradient_IUV_loss_neighbor)),gradient_gradient_IUV_loss,gradient_gradient_IUV_loss_neighbor)

    for x_leng in [-2,-1,1,2]:
        for y_leng in [-2,-1,1,2]:
            gradient_gradient_IUV_loss_neighbor = tf.roll(gradient_gradient_IUV_loss,[x_leng,y_leng],[1,2])
            gradient_gradient_IUV_loss_large = tf.where(tf.less(tf.abs(gradient_gradient_IUV_loss),tf.abs(gradient_gradient_IUV_loss_neighbor)),gradient_gradient_IUV_loss,gradient_gradient_IUV_loss_neighbor)

    gradient_gradient_IUV_loss = tf.where(tf.greater(tf.abs(gradient_gradient_IUV_loss),5e-4),gradient_gradient_IUV_loss_large/100.,gradient_gradient_IUV_loss)

image_data = tf.convert_to_tensor(img_data_orig,dtype=tf.float32)

gradient_back_IUV_data2IUV_var = tf.cast(tf.gradients(IUV_data, IUV_var)[0],tf.float32)

IUV_data = tf.concat([tf.clip_by_value(IUV_data[:,:,:,0:1],0,max_index),tf.clip_by_value(IUV_data[:,:,:,1:3],0,255)],-1)

real_grid_x_hard = tf.cast((tf.clip_by_value(IUV_data[:,:,:,0:1]-1,0,max_index))/4,tf.int32)*200+tf.cast(tf.cast(255 - IUV_data[:,:,:,2:],tf.float32)*199./255.,tf.int32)
real_grid_y_hard = tf.cast((tf.clip_by_value(IUV_data[:,:,:,0:1]-1,0,max_index))%4,tf.int32)*200+tf.cast(tf.cast(IUV_data[:,:,:,1:2],tf.float32)*199./255.,tf.int32)

real_grid_x = ((tf.clip_by_value(IUV_data[:,:,:,0:1]-1,0,max_index))/4)*200+(255 - IUV_data[:,:,:,2:])*199./255.
real_grid_y = ((tf.clip_by_value(IUV_data[:,:,:,0:1]-1,0,max_index))%4)*200+(IUV_data[:,:,:,1:2])*199./255.

real_grid_x = tf.stop_gradient(tf.cast(real_grid_x_hard,tf.float32) - tf.cast(real_grid_x,tf.float32)) + tf.cast(real_grid_x,tf.float32)
real_grid_y = tf.stop_gradient(tf.cast(real_grid_y_hard,tf.float32) - tf.cast(real_grid_y,tf.float32)) + tf.cast(real_grid_y,tf.float32)

real_grid = tf.concat([real_grid_x,real_grid_y, tf.zeros_like(real_grid_y)],-1)

vel_grid = tf.cast(real_grid[:,:,:,:2],tf.float32) - original_grid
vel_grid = tf.concat([vel_grid,tf.zeros_like(vel_grid[:,:,:,:1])],-1)

gradient_back_vel_grid2IUV_data = tf.gradients(vel_grid, IUV_data)[0]



############################
position_valid_IUV = tf.where(tf.greater(IUV_data[:,:,:,0],0))
iuv_current_points = tf.gather_nd(IUV_data,position_valid_IUV)
vel_values = tf.gather_nd(vel_grid[:,:,:,:2],position_valid_IUV)

uv_current_points = iuv_current_points[:,1:]
x_current_points = tf.cast(255 - uv_current_points[:,1:2],tf.float32)*199./255.
y_current_points = tf.cast(uv_current_points[:,0:1],tf.float32)*199./255.
i_current_points = iuv_current_points[:,0:1]
x_current_points = tf.cast((iuv_current_points[:,0:1]-1)/4,tf.int32)*200+tf.cast(x_current_points,tf.int32)
y_current_points = tf.cast((iuv_current_points[:,0:1]-1)%4,tf.int32)*200+tf.cast(y_current_points,tf.int32)
xy_current = tf.concat([x_current_points,y_current_points],-1)

tmpVar = tf.Variable(tf.zeros([texture_height,texture_width,2],tf.float32), trainable=False, name="tmp")
position = tf.where(tf.logical_or(tf.logical_or(tf.greater(image_data[:,:,:,0],0),tf.greater(image_data[:,:,:,1],0)),tf.greater(image_data[:,:,:,2],0)))
with tf.control_dependencies([tf.assign(tmpVar,tf.zeros([texture_height,texture_width,2],tf.float32))]):
    new_values = tf.concat([tf.cast(position[:,1:],tf.float32),vel_values],0)
    new_position = tf.concat([tf.cast(position[:,1:],tf.int32),xy_current],0)
    vel_texture = tf.scatter_nd_update(tmpVar, new_position, new_values)[tf.newaxis,:,:,:]

vel_texture = tf.concat([vel_texture,tf.zeros_like(vel_texture[:,:,:,:1])],-1)
original_grid = tf.concat([original_grid,tf.zeros_like(original_grid[:,:,:,:1])],-1)

end_grid = original_grid - vel_texture

gradient_back_end_grid2vel_texture = tf.cast(tf.gradients(end_grid,vel_texture)[0],tf.float32)
############################

texture = bilinear_sampler(image_data, end_grid[:,:,:,0], end_grid[:,:,:,1])

gradient_back_vel_texture2IUV_data = tf.cast(tf.contrib.image.dense_image_warp(gradient_back_vel_grid2IUV_data, vel_texture[:,:,:,:2]),tf.float32)

image_data_back = bilinear_sampler(texture, (original_grid+vel_grid)[:,:,:,0], (original_grid+vel_grid)[:,:,:,1])

gradient_back_texture2IUV_data = tf.gradients(texture,end_grid)[0]*gradient_back_end_grid2vel_texture*gradient_back_vel_texture2IUV_data

loss = tf.reduce_mean(tf.square(image_data_back - image_data))
gradient_back_loss_2_image_data_back = tf.gradients(loss,image_data_back)[0]


gradient_back_image_data_back_2_IUV_data = tf.gradients(image_data_back, vel_grid)[0]*tf.cast(tf.gradients(vel_grid, IUV_data)[0],tf.float32) + tf.cast(tf.gradients(image_data_back,texture)[0],tf.float32)*gradient_back_texture2IUV_data


fullgradient = gradient_back_loss_2_image_data_back * gradient_back_image_data_back_2_IUV_data * gradient_back_IUV_data2IUV_var
smooth_part = tf.constant(0.)
IUV_part = tf.constant(0.)
if smooth:
    smooth_part = gradient_gradient_loss * gradient_back_IUV_data2IUV_var[0]
    IUV_part = gradient_gradient_IUV_loss * gradient_back_IUV_data2IUV_var[0]
    fullgradient = smooth_part + IUV_part + args.factor * fullgradient
learning_rate = tf.placeholder(tf.float32)

opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

gen_optimizer = opt.apply_gradients([(fullgradient,IUV_var)])
sess.run(tf.initialize_all_variables())
previous = IUV_initial_orig

interval = 100
total_loss = 0
for i in range(1,args.iter):
    if i > 30000:
        lr = 10.0
    elif i > 20000:
        lr = 10.0
    elif i > 10000:
        lr = 10.0
    else:
        lr = 10.0
    _, ls, ls_gradient, optimized_IUV,texture_gen,img_back,full_g,smooth_g,ls_IUV,g_gradient_IUV_loss = sess.run([gen_optimizer,loss,gradient_loss,IUV_data,texture,image_data_back,fullgradient,smooth_part,IUV_loss,gradient_gradient_IUV_loss],feed_dict={learning_rate:lr})
    total_loss += ls
    if i%interval == 0:
        print("steps:",i,"l2 loss:",ls,"mean ls loss:",total_loss/interval,"gradient:",np.mean(full_g),"gradient loss",ls_gradient,"max smooth_g:",np.amax(smooth_g),"mean smooth_g:",np.mean(smooth_g),"IUV_loss:",ls_IUV,"IUV loss max gradient:",np.amax(g_gradient_IUV_loss),"IUV loss mean gradient:",np.mean(g_gradient_IUV_loss))
        print("biggest change:",np.amax(np.abs(optimized_IUV-previous)))
        previous = optimized_IUV
        total_loss = 0
cv.imwrite(output_path + "%04d.png"%frame,optimized_IUV[0,:img_height,:img_width,:])
np.save(output_path + "%04d.npy"%frame,optimized_IUV[0,:img_height,:img_width,:])
texture_map = TransferTextureback_np(image_data_np[0],optimized_IUV[0,:img_height,:img_width,:])
result = TransferTexture_np(texture_map/255.,np.zeros_like(IUV_data_np[0]),optimized_IUV[0,:img_height,:img_width,:])
cv.imwrite(output_path + "texture_%04d.png"%frame,texture_map)
cv.imwrite(output_path + "img_back_%04d.png"%frame,result)
