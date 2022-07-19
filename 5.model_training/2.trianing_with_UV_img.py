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

# train with IUV and images

import tensorflow as tf
import cv2 as cv
import os
import numpy as np
import sys
sys.path.append(os.path.abspath('../tools'))
from net import *
from ops import *
import time
import logging
import subprocess as sp
def get_gpu_memory():
    _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
    ACCEPTABLE_AVAILABLE_MEMORY = 1024 *6
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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

index_start = 0
index_end = 121


texture_height = 1200
texture_width = 800
max_index = 24
update = []
headpath = "/mnt/netdisk1/youxie/Fashion_completion/"
# mode = 'inference'
mode = 'train'
downsample = 1
add_poschannel = 0
num_resblock = 30
batch_norm = False
input_channel = 27
use_UV_GAN = 1
use_IMG_GAN = 1
use_TEMP_GAN = 1
if use_TEMP_GAN:
    recurrent_frames = 3
else:
    recurrent_frames = 1
if add_poschannel:
    input_channel += 2
if mode == 'train':
    skip = 1
    if use_TEMP_GAN:
        batch_size = 32
    else:
        batch_size = 32
    is_training = True
    piecesize_x = 32
    piecesize_y = 32
    max_neighbor_x = 1
    max_neighbor_y = 1
    data_mode = 1
    input_path = "/mnt/netdisk1/youxie/Fashion_extrapolation/IUV/"
    IUV_path = "/mnt/netdisk1/youxie/Fashion_coordinate_2/data/OF_texture_IUV_optimization_all_smooth_local_22_IUV/"
    texture_reference = np.array(cv.imread("/mnt/netdisk1/youxie/Fashion_coordinate_2/data/ex_IUV_optimization_all_smooth/texture_0000_ex.png"),np.float32)
    image_path = "/mnt/netdisk1/youxie/Fashion_extrapolation/example_rmbg_refined/"
    model_path = "/mnt/netdisk1/youxie/Fashion_completion/test_0526/model_0043.ckpt"
else:
    skip = 1
    recurrent_frames = 1
    is_training = False
    batch_size = 1
    piecesize = 0
    testnum = 529
    modelnum = 219
    data_mode = 1
    max_neighbor_x = 1
    max_neighbor_y = 1
    input_path = "/mnt/netdisk1/youxie/Fashion_extrapolation/IUV/"
    image_path = "/mnt/netdisk1/youxie/Fashion_extrapolation/example_rmbg_refined/"
    texture_reference = np.array(cv.imread("/mnt/netdisk1/youxie/Fashion_coordinate_2/data/ex_IUV_optimization_all_smooth/texture_0000_ex.png"),np.float32)
    texture_reference = np.array(cv.imread("/mnt/netdisk1/youxie/cvpr2022/texture_exchange/A1jFMj0n1JS/optimization/texture_0000_ex_combine_3.png"),np.float32)
    texture_grid = np.array(cv.imread("/mnt/netdisk1/youxie/cvpr2022/texture_grid.png"),np.float32)
    input_path = "/mnt/netdisk1/huiqi/fashion_dataset/train_IUV/A1jFMj0n1JS/"
    image_path = "/mnt/netdisk1/huiqi/fashion_dataset/train_images/A1jFMj0n1JS/"
    index_end = 120
    model_path = headpath + "test_%04d/backup/model_%04d.ckpt"%(testnum,modelnum)

learningrate = 1e-6
decay_step = 2000
decay_rate = 1.0
stair = False
beta = 0.5

max_iter = 200000

input_IUV = []
ref_IUV = []
ref_IUV_3 = []
image_list = []



def maketestdir(headpath=""):
    count = 0
    if mode == 'train':
        while os.path.exists(headpath+"test_%04d/"%count):
            count += 1
        savepath = headpath+"test_%04d/"%count
    elif mode == 'inference':
        while os.path.exists(headpath+"inference_%04d/"%count):
            count += 1
        savepath = headpath+"inference_%04d/"%count
    else:
        print("wrong mode!")
        exit()
    os.makedirs(savepath)
    outputpath = savepath+"imgs/"
    if mode == 'train':
        os.mkdir(outputpath)
    return savepath, outputpath

def cropping(img,piecesize_x,piecesize_y,x_start,y_start):
    return img[x_start:x_start+piecesize_x,y_start:y_start+piecesize_y,:]

def shifting(img,x_shift,y_shift):
    return np.roll(img,[x_shift,y_shift],axis=[0,1])

def scaling(img,x_scale,y_scale):
    img_height = np.shape(img)[0]
    img_width = np.shape(img)[1]
    img_scale = cv.resize(img,(int(img_width*y_scale),int(img_height*x_scale)),interpolation = cv.INTER_AREA)
    img_height_scale = np.shape(img_scale)[0]
    img_width_scale = np.shape(img_scale)[1]

    output = np.zeros_like(img)
    if img_height_scale > img_height:
        img_scale = img_scale[int((img_height_scale-img_height)/2):int((img_height_scale-img_height)/2)+img_height,:,:]
    if img_width_scale > img_width:
        img_scale = img_scale[:,int((img_width_scale-img_width)/2):int((img_width_scale-img_width)/2)+img_width,:]
    img_height_scale = np.shape(img_scale)[0]
    img_width_scale = np.shape(img_scale)[1]
    output[int((img_height-img_height_scale)/2):int((img_height-img_height_scale)/2)+img_height_scale,int((img_width-img_width_scale)/2):int((img_width-img_width_scale)/2)+img_width_scale] = img_scale
    return output

def rotate_image(image,image_center,angle):
  rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv.INTER_LINEAR)
  return result

if mode == 'train':
    for start_ds_x in range(data_mode):
        for start_ds_y in range(data_mode):
            for i in range(index_start,index_end,skip):
                IUV_in = np.array(cv.imread("%s%04d.png"%(input_path,i)),np.float32)[22:-22,8:-8,:]
                if downsample:
                    IUV_in = IUV_in[start_ds_x::4,start_ds_y::4,:]
                input_IUV.append(IUV_in)


            for i in range(index_start,index_end,skip):
                IUV_ref_3 = np.load("%sIUV_%04d.npy"%(IUV_path,(i)))
                ref_IUV_3.append(IUV_ref_3)
    ref_IUV_3 = np.array(ref_IUV_3).astype(np.float32)
    ref_I = np.reshape(ref_IUV_3[:,:,:,0],[-1])
    ref_shape = np.shape(ref_IUV_3)
    one_hot_I = np.reshape(np.eye(25)[ref_I.astype(np.int32)],[ref_shape[0],ref_shape[1],ref_shape[2],25])
    ref_IUV = np.concatenate([one_hot_I,ref_IUV_3[:,:,:,1:]],-1)
    print("reference shape:%s"%(str(np.shape(ref_IUV))))
else:
    for start_ds_x in range(data_mode):
        for start_ds_y in range(data_mode):
            for i in range(index_start,index_end,skip):
                IUV_in = np.array(cv.imread("%s%04d.png"%(input_path,i)),np.float32)
                if downsample:
                    IUV_in = IUV_in[22:-22,8:-8,:]
                    IUV_in = IUV_in[start_ds_x::4,start_ds_y::4,:]
                else:
                    IUV_in = IUV_in[6:-6,:,:]
                input_IUV.append(IUV_in)
    print("input_IUV shape:%s"%(str(np.shape(input_IUV))))

for start_ds_x in range(data_mode):
    for start_ds_y in range(data_mode):
        for i in range(index_start,index_end,skip):
            image_in = np.array(cv.imread("%s%04d.png"%(image_path,i)),np.float32)
            if downsample:
                image_in = image_in[22:-22,8:-8,:]
                image_in = image_in[start_ds_x::4,start_ds_y::4,:]
            else:
                image_in = image_in[6:-6,:,:]
            image_list.append(image_in)
image_list = np.array(image_list).astype(np.float32)
print("image shape:%s"%(str(np.shape(image_list))))

input_IUV = np.array(input_IUV).astype(np.float32)
input_I = np.reshape(input_IUV[:,:,:,0],[-1])
input_shape = np.shape(input_IUV)
one_hot_I = np.reshape(np.eye(25)[input_I.astype(np.int32)],[input_shape[0],input_shape[1],input_shape[2],25])
input_IUV = np.concatenate([one_hot_I,input_IUV[:,:,:,1:]],-1)
print("input shape:%s"%(str(np.shape(input_IUV))))

img_height = np.shape(input_IUV)[1]
img_width = np.shape(input_IUV)[2]

if add_poschannel:
    coordinate_image = []
    coordinate_image_1 = np.ones([img_height,img_width,2])
    for i in range(img_height):
        for j in range(img_width):
            coordinate_image_1[i,j] = [i*255./img_height,j*255./img_width]
    for i in range(recurrent_frames):
        coordinate_image.append(coordinate_image_1)
    coordinate_image = np.array(coordinate_image)

def mask_np(img):
    output = np.zeros([piecesize,piecesize,1])
    for mask_height in range(piecesize):
        for mask_width in range(piecesize):
            if (img[:,mask_height,mask_width,:]>0).any():
                output[mask_height,mask_width] = 1
    return output

def next_train_batch(shuffel = True):
    input_part_shuffle = []
    ref_part_shuffle = []
    ref_temp_part_shuffle = []
    ref_img_shuffle = []

    for i in range(batch_size):
        occupied_rate = -1
        start_idx = np.random.randint(np.shape(input_IUV)[0]-recurrent_frames+1)
        end_idx = start_idx + recurrent_frames
        x_start = np.random.randint(0,img_height - piecesize_x)
        y_start = np.random.randint(0,img_width - piecesize_y)
        input_part_image = input_IUV[start_idx:end_idx,x_start:x_start+piecesize_x,y_start:y_start+piecesize_y,:]
        if add_poschannel:
            input_part_image = np.concatenate([input_part_image,coordinate_image[:,x_start:x_start+piecesize_x,y_start:y_start+piecesize_y,:]],-1)
        input_part_shuffle.append(input_part_image)
        ref_part_image = ref_IUV[start_idx:end_idx,x_start:x_start+piecesize_x,y_start:y_start+piecesize_y,:]
        ref_part_shuffle.append(ref_part_image)
        ref_image = image_list[start_idx:end_idx,x_start:x_start+piecesize_x,y_start:y_start+piecesize_y,:]
        ref_img_shuffle.append(ref_image)

        IUV_0 = ref_IUV_3[start_idx]
        x_shift = np.random.randint(-3,4)
        y_shift = np.random.randint(-3,4)

        x_scale = np.random.randint(900,1100)/1000
        y_scale = np.random.randint(900,1100)/1000

        angle = np.random.randint(-2,2)
        image_center = (np.random.randint(0,img_height/3),np.random.randint(0,img_width))

        IUV_1 = np.copy(IUV_0)
        IUV_2 = np.copy(IUV_0)

        shift_switch = np.random.random()
        scale_switch = np.random.random()
        rotate_switch = np.random.random()

        if scale_switch>0.5:
            IUV_1 = scaling(IUV_1,x_scale,y_scale)
            IUV_2 = scaling(IUV_1,x_scale,y_scale)
        elif rotate_switch>0.5:
            IUV_1 = rotate_image(IUV_1,image_center,angle)
            IUV_2 = rotate_image(IUV_1,image_center,angle)
        elif shift_switch>0.2:
            IUV_1 = shifting(IUV_1,x_shift,y_shift)
            IUV_2 = shifting(IUV_1,x_shift,y_shift)

        IUV_0 = cropping(IUV_0,piecesize_x,piecesize_y,x_start,y_start)
        IUV_1 = cropping(IUV_1,piecesize_x,piecesize_y,x_start,y_start)
        IUV_2 = cropping(IUV_2,piecesize_x,piecesize_y,x_start,y_start)
        if recurrent_frames == 3:
            IUV_temp = np.concatenate([IUV_0[np.newaxis,:,:,:],IUV_1[np.newaxis,:,:,:],IUV_2[np.newaxis,:,:,:]],0)
        else:
            IUV_temp = IUV_0[np.newaxis,:,:,:]
        ref_temp_part_shuffle.append(IUV_temp)

    return np.array(input_part_shuffle),np.array(ref_part_shuffle),np.array(ref_img_shuffle), np.array(ref_temp_part_shuffle)

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
    final_outputs = tf.zeros([batch_size*recurrent_frames,piecesize_x,piecesize_y,3])
    for neighbor_x in range(max_neighbor_x):
        for neighbor_y in range(max_neighbor_y):
            x0 = tf.cast(tf.floor(x), 'int32') - neighbor_x
            x1 = x0 + 1 + neighbor_x
            y0 = tf.cast(tf.floor(y), 'int32') - neighbor_y
            y1 = y0 + 1 + neighbor_y

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
            final_outputs += out/(max_neighbor_x*max_neighbor_y)

    return final_outputs

def get_mask(image):
    with tf.variable_scope("Mask") as scope:
        tmpVar = tf.Variable(tf.zeros([image.get_shape()[0],image.get_shape()[1], image.get_shape()[2],1],tf.float32), trainable=False)
        position = tf.where(tf.logical_or(tf.logical_or(tf.greater(image[:,:,:,0],0),tf.greater(image[:,:,:,1],0)),tf.greater(image[:,:,:,2],0)))
        xyzvalue = tf.ones_like(position[:,:1],dtype=tf.float32)
        output = tf.scatter_nd_update(tmpVar, tf.cast(position,tf.int32), xyzvalue)
        update = tmpVar.assign(tf.zeros([image.get_shape()[0], image.get_shape()[1], image.get_shape()[2],1],tf.float32))
        return output,update

def softargmax(x, beta=1e10):
    x_range = tf.range(x.shape.as_list()[-1], dtype=x.dtype)
    output = tf.reduce_sum(tf.nn.softmax(x*beta) * x_range, axis=-1)
    return output

if mode == "inference":

    input_IUV_tf = tf.placeholder(tf.float32, shape=[batch_size, img_height, img_width, input_channel])
    IUV_fulfilled_raw_pre = tf.placeholder(tf.float32, shape=[batch_size, img_height, img_width, 27])

    IUV_fulfilled_raw = generator(tf.concat([input_IUV_tf[:,:,:,:25],input_IUV_tf[:,:,:,25:]/255],-1), num_resblock = num_resblock, batch_norm = batch_norm, is_training = is_training, printinfo = False,gen_output_channels=27)
    IUV_fulfilled_raw = tf.concat([tf.nn.softmax(IUV_fulfilled_raw[:,:,:,:25]),IUV_fulfilled_raw[:,:,:,25:]],-1)
    IUV_fulfilled = tf.concat([IUV_fulfilled_raw[:,:,:,:25],IUV_fulfilled_raw[:,:,:,25:]*255],-1)


    config = tf.ConfigProto()
    config.allow_soft_placement=True
    sess = tf.Session(config=config)

    var = tf.trainable_variables()
    gen_var = [v for v in var if "generator" in v.name]
    disc_var = [v for v in var if "discriminator" in v.name]
    saver = tf.train.Saver(max_to_keep=1,var_list=gen_var+disc_var)
    sess.run(tf.initialize_all_variables())

    saver.restore(sess,model_path)
    print("model restored from %s"%model_path)
    savepath,_ =maketestdir(headpath+"/test_%04d/"%(testnum))
    os.system("cp %s %s/%s"%(__file__,savepath,__file__))
    os.system("cp %s %s/%s"%("net.py",savepath,"net.py"))
    os.system("cp %s %s/%s"%("ops.py",savepath,"ops.py"))
    IUV_raw_pre = np.zeros_like(input_IUV[0:1,:,:,:])
    output_genIUV = []
    for i in range(0,np.shape(input_IUV)[0]):
        start_time = time.time()
        input = input_IUV[i:i+1,:,:,:]
        if add_poschannel:
            input = np.concatenate([input_IUV[i:i+1,:,:,:],coordinate_image],-1)
        output, IUV_raw_pre  = sess.run([IUV_fulfilled,IUV_fulfilled_raw],feed_dict={input_IUV_tf:input, IUV_fulfilled_raw_pre:IUV_raw_pre})
        gen_I_value = np.argmax(output[:,:,:,:25],axis = 3)[:,:,:,np.newaxis]
        output = np.concatenate([gen_I_value,output[:,:,:,25:]],-1)
        output_genIUV.append(output[0])
        if data_mode == 1:
            IUV_output = np.clip(output[0],0,255)
            for x in range(img_height):
                for y in range(img_width):
                    if IUV_output[x,y,0] == 0:
                        IUV_output[x,y] = [0,0,0]
            cv.imwrite(savepath + "gen_%04d.png"%(i), IUV_output)
            np.save(savepath + "gen_%04d.npy"%(i), IUV_output)
            print(np.amax(IUV_output))
            image = image_list[i,:,:,:]
            texture_map = TransferTextureback_np(image,IUV_output)
            image_back = TransferTexture_np(texture_map/255.,np.zeros_like(image),IUV_output)
            image_with_t0 = TransferTexture_np(texture_reference/255.,np.ones_like(image)*255,IUV_output)
            image_grid_with_t0 = TransferTexture_np(texture_grid/255.,np.zeros_like(image),IUV_output)
            cv.imwrite(savepath + "/texture_%04d.png"%(i), texture_map)
            cv.imwrite(savepath + "/image_back_%04d.png"%(i), image_back)
            cv.imwrite(savepath + "/image_with_t0_%04d.png"%(i), image_with_t0)
            cv.imwrite(savepath + "/image_grid_with_t0_%04d.png"%(i), image_grid_with_t0)
        print("finish %dth frame! time cost:%f"%(i,time.time()-start_time))

if mode == 'train':
    l2_factor = tf.placeholder(tf.float32)
    img_l2_factor = tf.placeholder(tf.float32)
    uv_factor = tf.placeholder(tf.float32)
    img_factor = tf.placeholder(tf.float32)
    temp_factor = tf.placeholder(tf.float32)
    gradient_factor = tf.placeholder(tf.float32)
    uv_gradient_factor = tf.placeholder(tf.float32)
    img_gradient_factor = tf.placeholder(tf.float32)
    input_IUV_tf = tf.placeholder(tf.float32, shape=[batch_size, recurrent_frames, piecesize_x, piecesize_y, input_channel])
    ref_IUV_tf = tf.placeholder(tf.float32, shape=[batch_size, recurrent_frames, piecesize_x, piecesize_y, input_channel])
    temporal_IUV = tf.placeholder(tf.float32, shape=[batch_size, recurrent_frames, piecesize_x, piecesize_y, 3])

    ref_img_tf = tf.placeholder(tf.float32, shape=[batch_size, recurrent_frames, piecesize_x, piecesize_y, 3])
    ref_img_tf_rh = tf.reshape(tf.transpose(ref_img_tf,perm=[0,2,3,1,4]),[batch_size, piecesize_x, piecesize_y, recurrent_frames*3])
    gen_input = input_IUV_tf[:,0,:,:,:]
    out_ref = ref_IUV_tf[:,0,:,:,:]
    input_col = gen_input
    out_col = out_ref
    IUV_fulfilled_raw = generator(tf.concat([gen_input[:,:,:,:25],gen_input[:,:,:,25:]/255],-1), num_resblock = num_resblock, batch_norm = batch_norm, is_training = is_training, printinfo = True,gen_output_channels=27)
    IUV_fulfilled = tf.concat([IUV_fulfilled_raw[:,:,:,:25],IUV_fulfilled_raw[:,:,:,25:]*255],-1)

    ref_IUV_tf_raw = tf.concat([out_ref[:,:,:,:25],out_ref[:,:,:,25:]/255],-1)


    for i in range(1,recurrent_frames):
        gen_input = input_IUV_tf[:,i,:,:,:]
        out_ref = ref_IUV_tf[:,i,:,:,:]
        input_col = tf.concat([input_col,gen_input],0)
        out_col = tf.concat([out_col,out_ref],0)
        IUV_fulfilled_raw_rec = generator(tf.concat([gen_input[:,:,:,:25],gen_input[:,:,:,25:]/255],-1), num_resblock = num_resblock, batch_norm = batch_norm, is_training = is_training, printinfo = False, reuse = True,gen_output_channels=27)
        IUV_fulfilled_raw = tf.concat([IUV_fulfilled_raw,IUV_fulfilled_raw_rec],0)
        IUV_fulfilled_rec = tf.concat([IUV_fulfilled_raw_rec[:,:,:,:25],IUV_fulfilled_raw_rec[:,:,:,25:]*255],-1)
        IUV_fulfilled = tf.concat([IUV_fulfilled,IUV_fulfilled_rec],0)
        ref_IUV_tf_raw_rec = tf.concat([out_ref[:,:,:,:25],out_ref[:,:,:,25:]/255],-1)
        ref_IUV_tf_raw = tf.concat([ref_IUV_tf_raw,ref_IUV_tf_raw_rec],0)

    update = []

    dx_gen_image, dy_gen_image = tf.image.image_gradients(IUV_fulfilled[:batch_size,:,:,25:])
    dx_image, dy_image = tf.image.image_gradients(out_col[:batch_size,:,:,25:])
    loss_gradient =  (tf.reduce_mean(tf.square(dx_image-dx_gen_image)) + tf.reduce_mean(tf.square(dy_image-dy_gen_image))) * 2 / 10
    loss_l2 = tf.reduce_mean(tf.square(IUV_fulfilled_raw[:batch_size,:,:,25:] - ref_IUV_tf_raw[:batch_size,:,:,25:]))*255*255
    loss_I = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = IUV_fulfilled[:batch_size,:,:,:25],labels = out_col[:batch_size,:,:,:25]))#*300

    if use_TEMP_GAN:
        IUV_value_gen = IUV_fulfilled[:,:,:,25:]
        IUV_value_gen = tf.reshape(tf.transpose(tf.reshape(IUV_value_gen,[recurrent_frames,batch_size,piecesize_x,piecesize_y,2]),perm=[1,2,3,0,4]),[batch_size,piecesize_x,piecesize_y,recurrent_frames*2])
        temporal_IUV_ref = tf.reshape(tf.transpose(temporal_IUV[:,:,:,:,1:],perm=[0,2,3,1,4]),[batch_size,piecesize_x,piecesize_y,recurrent_frames*2])

        I_softmax = tf.nn.softmax(IUV_fulfilled[:,:,:,:25])
        I_value = tf.stop_gradient(tf.cast(tf.argmax(I_softmax,axis=3),tf.float32) - softargmax(I_softmax)) + softargmax(I_softmax)
        I_value = tf.reshape(tf.transpose(tf.reshape(I_value,[recurrent_frames,batch_size,piecesize_x,piecesize_y,1]),perm=[1,2,3,0,4]),[batch_size,piecesize_x,piecesize_y,recurrent_frames*1])
        dx_I_1, dy_I_1 = tf.image.image_gradients(I_value[:,:,:,0:1])
        dx_I_2, dy_I_2 = tf.image.image_gradients(I_value[:,:,:,1:2])
        dx_I_3, dy_I_3 = tf.image.image_gradients(I_value[:,:,:,2:3])
        for x in range(-3,4):
            for y in range(-3,4):
                dx_I_1_1, dy_I_1_1 = tf.image.image_gradients(tf.roll(I_value[:,:,:,0:1],[x,y],[1,2]))
                dx_I_1 = tf.where(tf.greater(dx_I_1,dx_I_1_1),dx_I_1,dx_I_1_1)
                dy_I_1 = tf.where(tf.greater(dy_I_1,dy_I_1_1),dy_I_1,dy_I_1_1)

                dx_I_2_1, dy_I_2_1 = tf.image.image_gradients(tf.roll(I_value[:,:,:,1:2],[x,y],[1,2]))
                dx_I_2 = tf.where(tf.greater(dx_I_2,dx_I_2_1),dx_I_2,dx_I_2_1)
                dy_I_2 = tf.where(tf.greater(dy_I_2,dy_I_2_1),dy_I_2,dy_I_2_1)

                dx_I_3_1, dy_I_3_1 = tf.image.image_gradients(tf.roll(I_value[:,:,:,2:3],[x,y],[1,2]))
                dx_I_3 = tf.where(tf.greater(dx_I_3,dx_I_3_1),dx_I_3,dx_I_3_1)
                dy_I_3 = tf.where(tf.greater(dy_I_3,dy_I_3_1),dy_I_3,dy_I_3_1)


        dx_IUV, dy_IUV = tf.image.image_gradients(IUV_value_gen)

        condition_1_1 = tf.logical_or(tf.logical_or(tf.equal(I_value[:,:,:,0:1],2),tf.equal(I_value[:,:,:,0:1],9)),tf.equal(I_value[:,:,:,0:1],10))
        condition_1 = tf.logical_and(tf.logical_and(tf.equal(dx_I_1,0),tf.equal(dy_I_1,0)),condition_1_1)
        condition_2_1 = tf.logical_or(tf.logical_or(tf.equal(I_value[:,:,:,1:2],2),tf.equal(I_value[:,:,:,1:2],9)),tf.equal(I_value[:,:,:,1:2],10))
        condition_2 = tf.logical_and(tf.logical_and(tf.equal(dx_I_2,0),tf.equal(dy_I_2,0)),condition_2_1)
        condition_3_1 = tf.logical_or(tf.logical_or(tf.equal(I_value[:,:,:,2:3],2),tf.equal(I_value[:,:,:,2:3],9)),tf.equal(I_value[:,:,:,2:3],10))
        condition_3 = tf.logical_and(tf.logical_and(tf.equal(dx_I_3,0),tf.equal(dy_I_3,0)),condition_3_1)

        mask_1 = tf.where(condition_1,tf.ones_like(IUV_value_gen[:,:,:,:1]),tf.zeros_like(IUV_value_gen[:,:,:,:1]))
        mask_2 = tf.where(condition_2,tf.ones_like(IUV_value_gen[:,:,:,:1]),tf.zeros_like(IUV_value_gen[:,:,:,:1]))
        mask_3 = tf.where(condition_3,tf.ones_like(IUV_value_gen[:,:,:,:1]),tf.zeros_like(IUV_value_gen[:,:,:,:1]))

        loss_uv_gradient_spatial = tf.reduce_mean(tf.square(tf.multiply(dx_IUV[:,:,:,:2],mask_1))) + tf.reduce_mean(tf.square(tf.multiply(dy_IUV[:,:,:,:2],mask_1)))\
        + tf.reduce_mean(tf.square(tf.multiply(dx_IUV[:,:,:,2:4],mask_2))) + tf.reduce_mean(tf.square(tf.multiply(dy_IUV[:,:,:,2:4],mask_2)))\
        + tf.reduce_mean(tf.square(tf.multiply(dx_IUV[:,:,:,4:],mask_3))) + tf.reduce_mean(tf.square(tf.multiply(dy_IUV[:,:,:,4:],mask_3)))

        mask_12 = tf.where(tf.logical_and(condition_1,condition_2),tf.ones_like(IUV_value_gen[:,:,:,:1]),tf.zeros_like(IUV_value_gen[:,:,:,:1]))
        mask_23 = tf.where(tf.logical_and(condition_2,condition_3),tf.ones_like(IUV_value_gen[:,:,:,:1]),tf.zeros_like(IUV_value_gen[:,:,:,:1]))
        loss_uv_gradient_temporal = tf.reduce_mean(tf.square(tf.multiply(IUV_value_gen[:,:,:,:2] - IUV_value_gen[:,:,:,2:4],mask_12))) + tf.reduce_mean(tf.square(tf.multiply(IUV_value_gen[:,:,:,2:4] - IUV_value_gen[:,:,:,4:],mask_23)))
        mask_123 = tf.where(tf.logical_and(tf.logical_and(condition_1,condition_2),condition_3),tf.ones_like(IUV_value_gen[:,:,:,:1]),tf.zeros_like(IUV_value_gen[:,:,:,:1]))
        loss_uv_second_gradient_temporal = tf.reduce_mean(tf.square(tf.multiply(IUV_value_gen[:,:,:,:2] - 2*IUV_value_gen[:,:,:,2:4] + IUV_value_gen[:,:,:,4:],mask_123)))
        loss_uv_gradient = loss_uv_gradient_spatial * 1e-1 + loss_uv_gradient_temporal * 1 + loss_uv_second_gradient_temporal * 1

    if use_IMG_GAN:
        I_softmax = tf.nn.softmax(IUV_fulfilled[:,:,:,:25])
        I_value = tf.stop_gradient(tf.cast(tf.argmax(I_softmax,axis=3),tf.float32) - softargmax(I_softmax)) + softargmax(I_softmax)
        UV_value = IUV_fulfilled[:,:,:,25:]

        real_grid_x_hard = tf.cast(tf.round(tf.clip_by_value(I_value-1,0,24))/4,tf.int32)*200+tf.cast(tf.cast(255 - IUV_fulfilled[:,:,:,26],tf.float32)*199./255.,tf.int32)
        real_grid_y_hard = tf.cast(tf.round(tf.clip_by_value(I_value-1,0,24))%4,tf.int32)*200+tf.cast(tf.cast(IUV_fulfilled[:,:,:,25],tf.float32)*199./255.,tf.int32)

        real_grid_x = ((tf.clip_by_value(I_value-1,0,24))/4)*200+(255 - IUV_fulfilled[:,:,:,26])*199./255.
        real_grid_y = ((tf.clip_by_value(I_value-1,0,24))%4)*200+(IUV_fulfilled[:,:,:,25])*199./255.

        real_grid_x = tf.stop_gradient(tf.cast(real_grid_x_hard,tf.float32) - tf.cast(real_grid_x,tf.float32)) + tf.cast(real_grid_x,tf.float32)
        real_grid_y = tf.stop_gradient(tf.cast(real_grid_y_hard,tf.float32) - tf.cast(real_grid_y,tf.float32)) + tf.cast(real_grid_y,tf.float32)

        texture_0_np = np.zeros([batch_size*recurrent_frames,texture_height,texture_width,3])
        for b_index in range(batch_size*recurrent_frames):
            texture_0_np[b_index] = texture_reference
        texture_0_tf = tf.convert_to_tensor(texture_0_np,tf.float32)

        img_mask,udp = get_mask(ref_img_tf_rh)
        update.append(udp)
        img_output_raw = bilinear_sampler(texture_0_tf, real_grid_x, real_grid_y)
        img_output = tf.multiply(tf.reshape(tf.transpose(tf.reshape(img_output_raw,[recurrent_frames,batch_size,piecesize_x,piecesize_y,3]),perm=[1,2,3,0,4]),[batch_size,piecesize_x,piecesize_y,recurrent_frames*3]),img_mask)

        img_loss = tf.reduce_mean(tf.square(img_output-ref_img_tf_rh)) + tf.reduce_mean(tf.square(img_output_raw[:batch_size,:,:,:]-img_output_raw[batch_size:2*batch_size,:,:,:])) + tf.reduce_mean(tf.square(img_output_raw[batch_size:2*batch_size,:,:,:]-img_output_raw[2*batch_size:3*batch_size,:,:,:]))

    loss = loss_l2 * l2_factor + loss_gradient * gradient_factor + loss_I# + img_loss * 1e-2
    if use_IMG_GAN:
        loss_img_opt = img_loss * 1e-2 * img_l2_factor
    if use_TEMP_GAN:
        loss += loss_uv_gradient * uv_gradient_factor

    config = tf.ConfigProto()
    config.allow_soft_placement=True
    sess = tf.Session(config=config)


    if use_UV_GAN:
        disc_UV_true,_ = discriminator(ref_IUV_tf_raw[:batch_size,:,:,25:], name_scope='discriminator',printinfo = True)
        disc_UV_fake,_ = discriminator(IUV_fulfilled_raw[:batch_size,:,:,25:], name_scope='discriminator',reuse=True, printinfo = False)
    if use_IMG_GAN:
        disc_img_true,_ = discriminator(ref_img_tf_rh[:,:,:,:], name_scope='discriminator_img',printinfo = True)
        disc_img_fake,_ = discriminator(img_output[:,:,:,:], name_scope='discriminator_img',reuse=True, printinfo = False)
    if use_TEMP_GAN:
        disc_temp_true,_ = discriminator(temporal_IUV_ref, name_scope='discriminator_temp',printinfo = True)
        disc_temp_fake,_ = discriminator(IUV_value_gen, name_scope='discriminator_temp',reuse=True, printinfo = False)

    var = tf.trainable_variables()
    gen_var = [v for v in var if "generator" in v.name]
    disc_UV_var = [v for v in var if "discriminator_unit" in v.name]
    disc_img_var = [v for v in var if "discriminator_img" in v.name]
    disc_temp_var = [v for v in var if "discriminator_temp" in v.name]

    global_st = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(learningrate,global_st, decay_step, decay_rate, staircase=stair)
    with tf.control_dependencies(update):
        if use_UV_GAN:
            disc_loss_disc_UV = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_UV_true, labels=tf.ones_like(disc_UV_true)))
            disc_loss_gen_UV = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_UV_fake, labels=tf.zeros_like(disc_UV_fake)))
            gen_GAN_loss_UV = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_UV_fake, labels=tf.ones_like(disc_UV_fake)))
            disc_loss_UV = disc_loss_disc_UV + disc_loss_gen_UV
            disc_optimizer_UV = tf.train.AdamOptimizer(learning_rate, beta1=beta).minimize(disc_loss_UV, var_list=disc_UV_var,global_step=global_st)
            loss = loss + gen_GAN_loss_UV * 30 * uv_factor
        if use_IMG_GAN:
            disc_loss_disc_img = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_img_true, labels=tf.ones_like(disc_img_true)))
            disc_loss_gen_img = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_img_fake, labels=tf.zeros_like(disc_img_fake)))
            gen_GAN_loss_img = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_img_fake, labels=tf.ones_like(disc_img_fake)))
            disc_loss_img = disc_loss_disc_img + disc_loss_gen_img
            disc_optimizer_img = tf.train.AdamOptimizer(learning_rate, beta1=beta).minimize(disc_loss_img, var_list=disc_img_var,global_step=global_st)
            loss_img_opt = loss_img_opt + gen_GAN_loss_img * 10 * img_factor

            gen_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta)
            grads_vals = gen_opt.compute_gradients(loss_img_opt,gen_var)
            for i, (g, v) in enumerate(grads_vals):
                if g is not None:
                    threshold = 1e2
                    grads_vals[i] = (tf.where(tf.less(tf.abs(g),threshold),tf.zeros_like(g),g), v)  # clip gradients
            gen_optimizer_img = gen_opt.apply_gradients(grads_vals)


        if use_TEMP_GAN:
            disc_loss_disc_temp = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_temp_true, labels=tf.ones_like(disc_temp_true)))
            disc_loss_gen_temp = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_temp_fake, labels=tf.zeros_like(disc_temp_fake)))
            gen_GAN_loss_temp = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_temp_fake, labels=tf.ones_like(disc_temp_fake)))
            disc_loss_temp = disc_loss_disc_temp + disc_loss_gen_temp
            disc_optimizer_temp = tf.train.AdamOptimizer(learning_rate, beta1=beta).minimize(disc_loss_temp, var_list=disc_temp_var,global_step=global_st)
            loss = loss + gen_GAN_loss_temp * 30 * temp_factor
        gen_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=beta).minimize(loss, var_list=gen_var,global_step=global_st)



    saver = tf.train.Saver(max_to_keep=1,var_list=gen_var+disc_UV_var+disc_img_var+disc_temp_var)

    sess.run(tf.initialize_all_variables())

    if model_path != "":
        saver.restore(sess,model_path)
        print("model restored from %s"%model_path)

    start_time = time.time()
    save_no = 0

    savepath,outputpath =maketestdir(headpath+"/")
    train_lr = tf.summary.scalar("training learning rate",     learning_rate)
    gradient_loss = tf.summary.scalar("gradient loss",     loss_gradient)
    train_loss = tf.summary.scalar("training loss",     loss)
    l2_loss = tf.summary.scalar("l2 loss",     loss_l2)
    I_loss = tf.summary.scalar("I value loss",     loss_I)
    if use_UV_GAN:
        gen_GAN_ls_UV = tf.summary.scalar("generator_GAN_loss_UV",     gen_GAN_loss_UV)
        disc_GAN_ls_UV = tf.summary.scalar("discriminator_GAN_loss_UV",     disc_loss_UV)
    if use_IMG_GAN:
        gen_GAN_ls_img = tf.summary.scalar("generator_GAN_loss_img",     gen_GAN_loss_img)
        disc_GAN_ls_img = tf.summary.scalar("discriminator_GAN_loss_img",     disc_loss_img)
    if use_TEMP_GAN:
        gen_GAN_ls_temp = tf.summary.scalar("generator_GAN_loss_temp",     gen_GAN_loss_temp)
        disc_GAN_ls_temp = tf.summary.scalar("discriminator_GAN_loss_temp",     disc_loss_temp)
    merge_summary = tf.summary.merge_all()
    print("test results will be stored in %s"%savepath)
    os.system("cp %s %s/%s"%(__file__,savepath,__file__))
    os.system("cp %s %s/%s"%("net.py",savepath,"net.py"))
    os.system("cp %s %s/%s"%("ops.py",savepath,"ops.py"))
    summary_writer = tf.summary.FileWriter(savepath, sess.graph)
    factor_l2 = 0.8
    factor_gradient = 0.8
    factor_uv = 0.8
    factor_img_gradient = 0.8
    factor_img = 0.8
    factor_temp = 0.8
    factor_img_l2 = 0.8
    factor_uv_gradient = 0.0
    for step in range(1,max_iter):
        if step % decay_step == 0:
            factor_l2 *= 0.99
            factor_gradient *= 0.99
            factor_uv *= 0.99
            factor_img *= 1.1
            factor_temp *= 1.1
            factor_img_l2 *= 1.1
            factor_uv_gradient *= 1.1
            factor_img_gradient *= 0.97
        if factor_l2 < 0.3:
            factor_l2 = 0.3
            factor_gradient = 0.3
            factor_uv = 0.3
        if factor_img_gradient<1e-3:
            factor_img_gradient = 1e-3
        if factor_img > 1:
            factor_img = 1
            factor_temp = 1
            factor_img_l2 = 1
            factor_uv_gradient = 1
        input, reference, img_ref, IUV_temporal = next_train_batch()
        if use_UV_GAN:
            _,_, ls_gen_UV,ls_disc_UV,ls, train_summary,lr,ls_gradient,ls_I,ls_l2 = sess.run([gen_optimizer,disc_optimizer_UV,gen_GAN_loss_UV,disc_loss_UV,loss,merge_summary,learning_rate,loss_gradient,loss_I,loss_l2],feed_dict={input_IUV_tf:input,ref_IUV_tf:reference, ref_img_tf:img_ref,temporal_IUV:IUV_temporal,l2_factor:factor_l2,gradient_factor:factor_gradient,uv_factor:factor_uv,img_factor:factor_img,temp_factor:factor_temp,img_l2_factor:factor_img_l2,img_gradient_factor:factor_img_gradient,uv_gradient_factor:factor_uv_gradient})
        if use_IMG_GAN:
            _,_,ls_img, ls_gen_img,ls_disc_img,img_gen,img_reference = sess.run([gen_optimizer_img,disc_optimizer_img,img_loss,disc_loss_UV,gen_GAN_loss_img,img_output,ref_img_tf_rh],feed_dict={input_IUV_tf:input,ref_IUV_tf:reference, ref_img_tf:img_ref,temporal_IUV:IUV_temporal,l2_factor:factor_l2,gradient_factor:factor_gradient,uv_factor:factor_uv,img_factor:factor_img,temp_factor:factor_temp,img_l2_factor:factor_img_l2,img_gradient_factor:factor_img_gradient,uv_gradient_factor:factor_uv_gradient})
        if use_TEMP_GAN:
            _,ls_gen_temp,ls_disc_temp,ls_uv_gradient_spatial,ls_uv_gradient_temporal,ls_uv_second_gradient_temporal,ls_uv_gradient = sess.run([disc_optimizer_temp,gen_GAN_loss_temp,disc_loss_temp,loss_uv_gradient_spatial,loss_uv_gradient_temporal,loss_uv_second_gradient_temporal,loss_uv_gradient],feed_dict={input_IUV_tf:input,ref_IUV_tf:reference, ref_img_tf:img_ref,temporal_IUV:IUV_temporal,l2_factor:factor_l2,gradient_factor:factor_gradient,uv_factor:factor_uv,img_factor:factor_img,temp_factor:factor_temp,img_l2_factor:factor_img_l2,img_gradient_factor:factor_img_gradient,uv_gradient_factor:factor_uv_gradient})
        if (not use_UV_GAN) and (not use_IMG_GAN) and (not use_TEMP_GAN):
            _, ls,IUV_gen,train_summary,lr,ls_gradient,ls_I,ls_l2 = sess.run([gen_optimizer,loss,IUV_fulfilled,merge_summary,learning_rate,loss_gradient,loss_I,loss_l2],feed_dict={input_IUV_tf:input,ref_IUV_tf:reference,ref_img_tf:img_ref,temporal_IUV:IUV_temporal,l2_factor:factor_l2,gradient_factor:factor_gradient,uv_factor:factor_uv,img_factor:factor_img,temp_factor:factor_temp,img_l2_factor:factor_img_l2,img_gradient_factor:factor_img_gradient,uv_gradient_factor:factor_uv_gradient})

        if step % 100 == 0:
            summary_writer.add_summary(train_summary,step)
        if step % 1000 == 0:
            output = "########################################\n"
            output = output + "l2 factor:"+'\t'+str(factor_l2)+'\t'+"gradient factor:"+'\t'+str(factor_gradient)+'\t'+"uv factor:"+str(factor_uv)+'\t'+"img factor:"+'\t'+str(factor_img)+'\t'+"temp factor:"+'\t'+str(factor_temp)+'\t'+"img l2 factor:"+'\t'+str(factor_img_l2)+'\t'+"img gradient factor:"+'\t'+str(factor_img_gradient)+'\t'+"uv_gradient_factor:"+'\t'+str(factor_uv_gradient) + '\n'
            if use_UV_GAN:
                output = output + "step:\t%d\t,loss:\t%f\t,gradient loss:\t%f\t, I loss:\t%f\t, l2 loss:\t%f\t, gen_GAN_loss_UV:\t%f\t, disc_GAN_loss_UV:\t%f\t, time cost:\t%f\t,learning rate:\t%f\t \n"%(step,ls,ls_gradient, ls_I,ls_l2,ls_gen_UV,ls_disc_UV,time.time()-start_time,lr)
            if use_IMG_GAN:
                output = output + "image loss(rgb):"+'\t'+str(ls_img)+'\t'+ "gen_GAN_loss_img:"+'\t'+str(ls_gen_img)+'\t'+"disc_GAN_loss_img:"+'\t'+str(ls_disc_img)+'\n'
            if use_TEMP_GAN:
                output = output + "gen_GAN_loss_temp:"+'\t'+str(ls_gen_temp)+'\t'+"disc_GAN_loss_temp:"+'\t'+str(ls_disc_temp)+'\t'+"ls_uv_gradient_spatial:"+'\t'+str(ls_uv_gradient_spatial)+'\t'+"ls_uv_gradient_temporal:"+'\t'+str(ls_uv_gradient_temporal)+'\t'+"ls_uv_second_gradient_temporal:"+'\t'+str(ls_uv_second_gradient_temporal)+'\t'+"ls_uv_gradient:"+'\t'+str(ls_uv_gradient) + '\n'
            if (not use_UV_GAN) and (not use_IMG_GAN) and (not use_TEMP_GAN):
                output = output+"oss:\t%f\t,gradient loss:\t%f\t, I loss:\t%f\t, l2 loss:\t%f\t, time cost:\t%f\t,learning rate:\t%f\t \n"%(step,ls,ls_gradient, ls_I,ls_l2,time.time()-start_time,lr)
            print(output)
            trainf = open(savepath + "trainloss.txt",'a+')
            trainf.write(output+'\r\n')
            trainf.close()
            start_time = time.time()
            saver.save(sess, savepath + 'model_%04d.ckpt' % save_no)
            save_no += 1
