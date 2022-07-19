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


### Extrapolation
##With raw UV as inputs, here we applied linear extrapolation and virtual mass-spring system to achieve meaningful UV extrapolation.

import tensorflow as tf
import numpy as np
import cv2 as cv
import os
import sys
sys.path.append(os.path.abspath('../tools'))
from ops import *
import time
from PIL import Image
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.interpolate import RegularGridInterpolator
from sklearn.linear_model import LinearRegression

os.environ["CUDA_VISIBLE_DEVICES"]="0"
imgheight = 940
imgwidth = 720
textureheight = 1200
texturewidth = 800
batch_size = 1
update = []
output_MS = 0 #debug flag
output_IMG = 1 #debug flag
epsilon = 0.05
MS_output_path = "test/"
if output_MS and (not os.path.exists(MS_output_path)):
    os.mkdir(MS_output_path)

def get_mask(image):
    with tf.variable_scope("Mask") as scope:
        tmpVar = tf.Variable(tf.zeros([batch_size,image.get_shape()[1], image.get_shape()[2],1],tf.float32), trainable=False)
        position = tf.where(tf.logical_or(tf.logical_or(tf.greater(image[:,:,:,0],0),tf.greater(image[:,:,:,1],0)),tf.greater(image[:,:,:,2],0)))
        xyzvalue = tf.ones_like(position[:,:1],dtype=tf.float32)
        output = tf.scatter_nd_update(tmpVar, tf.cast(position,tf.int32), xyzvalue)
        update = tmpVar.assign(tf.zeros([batch_size, image.get_shape()[1], image.get_shape()[2],1],tf.float32))
        return output,update

def mean_neighbor(img_input,IUV_input,k,s,tmp,neighbor_num,I_given = 0):
    neighbor = []
    tmp_IUV = np.copy((IUV_input)[0,k-neighbor_num:k+neighbor_num+1,s-neighbor_num:s+neighbor_num+1,:])
    if I_given == 0:
        for h_index in range(2*neighbor_num + 1):
            for w_index in range(2*neighbor_num + 1):
                if not (h_index == neighbor_num and w_index == neighbor_num):
                    neighbor.append(tmp_IUV[h_index,w_index,0])

        neighbor = np.array(neighbor)
        neighbor = np.reshape(neighbor,[-1])
        if np.amax(neighbor) == 0.:
            return [0,0,0]
        tmp_order = np.bincount(np.int64(neighbor))
        count = np.amax(tmp_order[1:])
        if count < 2:
            return [0,0,0]
        I_value = np.argmax(tmp_order[1:])+1
    elif I_given > 0:
        I_value = I_given
    else:
        print("Invalid case!")
        exit()

    output_I = I_value
    positive_position = []
    positive_value = []
    for h_index in range(2*neighbor_num + 1):
        for w_index in range(2*neighbor_num + 1):
            if tmp_IUV[h_index,w_index,0] == I_value:
                positive_position.append([h_index,w_index])
                positive_value.append(tmp_IUV[h_index,w_index,1:])

    output = [0,0,0]

    valid_num = len(positive_position)

    if valid_num < 2:
        return [0,0,0]

    positive_position = np.array(positive_position)
    positive_value = np.array(positive_value)
    output_U, output_V = LinearRegression().fit(positive_position, positive_value).predict(np.reshape([neighbor_num,neighbor_num],[1,2]))[0]

    output_extra = np.array([output_I,output_U,output_V]).astype(np.float32)
    output = np.copy(output_extra)

    if neighbor_num == 1:
        #push
        correction_length = 20
        correction_IUV = np.copy((IUV_input+tmp)[0,np.clip(k-correction_length,0,imgheight):np.clip(k+correction_length+1,0,imgheight),np.clip(s-correction_length,0,imgwidth):np.clip(s+correction_length+1,0,imgwidth),:])
        correction_height, correction_width = np.shape(correction_IUV[:,:,0])
        correction_position = []
        correction_value = []
        for h_index in range(correction_height):
            for w_index in range(correction_width):
                if correction_IUV[h_index,w_index,0] == I_value:
                    correction_position.append([h_index,w_index])
                    correction_value.append(correction_IUV[h_index,w_index,1:])
        correction_num = len(correction_position)
        correction_value = np.array(correction_value)

        IUV_test  = np.reshape((IUV_input+tmp)[:,np.clip(k-30,0,imgheight):np.clip(k+30,0,imgheight),np.clip(s-30,0,imgwidth):np.clip(s+30,0,imgwidth),:].astype(np.int32),[-1,3])


        if output[1]>255 or output[2]>255 or output[1]<0 or output[2]<0 or (output[1]==0 and output[2]==0) or correction_num == 0:
            return [0,0,0]

        sim_loop = 0
        if output_MS:
            MS_file = open(MS_output_path + "%04d_%04d.txt"%(k,s),"a+")
            MS_file.write("white push points:"+str(np.shape(IUV_test)[0])+"\r\n")
            MS_file.write("push points:"+str(correction_num)+"\r\n")
            MS_file.write("extrapolation points:"+str(valid_num)+"\r\n")
            MS_output = np.zeros([256,256,3])
            for white_point in range(np.shape(IUV_test)[0]):
                if IUV_test[white_point,0]  == I_value:
                    MS_output[int(IUV_test[white_point,1]),int(IUV_test[white_point,2])] = [255,255,255]
            for green_point in range(correction_num):
                MS_output[int(correction_value[green_point,0]),int(correction_value[green_point,1])] = [255,0,0]
            for red_point in range(valid_num):
                MS_output[int(positive_value[red_point,0]),int(positive_value[red_point,1])] = [0,0,255]
            cv.imwrite(MS_output_path + "/%04d_%04d_push_%04d.png"%(k,s,sim_loop), MS_output)
            MS_output_tmp = np.copy(MS_output)
            MS_output_tmp[int(output[1]),int(output[2])] = [0,255,0]
            cv.imwrite(MS_output_path + "/%04d_%04d_push_%04d.png"%(k,s,sim_loop+1), MS_output_tmp)
            MS_file.write(str(output)+"\r\n")

        positive_force = np.zeros([correction_num,2])
        all_distance = np.zeros([correction_num,2])
        coefficient = 1.0
        rest_length = 1

        for force_round in range(correction_num):
            force =[0,0]
            distance = np.array([output[1],output[2]]) - correction_value[force_round]
            if np.abs(distance[0]) < epsilon and np.abs(distance[1]) > epsilon:
                force = [0, np.sign(distance[1])]
            elif np.abs(distance[0]) > epsilon and np.abs(distance[1]) < epsilon:
                force = [np.sign(distance[0]),0]
            elif np.abs(distance[0]) < epsilon and np.abs(distance[1]) < epsilon:
                force = [((-1)**np.random.randint(0,2))/3,((-1)**np.random.randint(0,2))/3]
            else:
                force = np.sign(distance)
            positive_force[force_round] = force
            if output_MS:
                MS_file.write("push force:"+str(correction_value[force_round])+'\t'+str(force)+"\r\n")

        approved_distance = 2
        while (np.reshape(np.array(output).astype(np.int32),[3]).tolist() in IUV_test.tolist() and correction_num>0):
            if sim_loop > 10000:
                output = [0,0,0]
                break
            total_force = np.sum(positive_force,axis=0)
            old_output = np.copy(output)
            total_force /= correction_num

            if np.abs(total_force[0]) - np.abs(total_force[1]) > 0.2:
                total_force[1] = 0
            elif np.abs(total_force[1]) - np.abs(total_force[0]) > 0.2:
                total_force[0] = 0


            if output[0] > 250 and total_force[0] > 0:
                total_force[0] = 0
            if output[1] > 250 and total_force[1] > 0:
                total_force[1] = 0
            if output[0] < 5 and total_force[0] < 0:
                total_force[0] = 0
            if output[1] < 5 and total_force[1] < 0:
                total_force[1] = 0

            output[1:] += total_force
            sim_loop += 1

            if output[1]>255 or output[2]>255 or output[1]<0 or output[2]<0:
                return [0,0,0]

            if output_MS:
                MS_file.write("total_push_force:"+str(total_force)+"\r\n")
                MS_file.write("push update:"+str(output)+"\r\n")
                MS_output_tmp = np.copy(MS_output)
                MS_output_tmp[int(output[1]),int(output[2])] = [0,255,0]
                cv.imwrite(MS_output_path + "/%04d_%04d_push_%04d.png"%(k,s,sim_loop+1), MS_output_tmp)

            new_output = np.copy(output)
            if np.all(new_output == old_output):
                break
            for force_round in range(correction_num):
                force =[0,0]
                distance = np.array([output[1],output[2]]) - correction_value[force_round]
                if np.abs(distance[0]) < epsilon and np.abs(distance[1]) > epsilon:
                    force = [0, np.sign(distance[1])]
                elif np.abs(distance[0]) > epsilon and np.abs(distance[1]) < epsilon:
                    force = [np.sign(distance[0]),0]
                elif np.abs(distance[0]) < epsilon and np.abs(distance[1]) < epsilon:
                    force = [((-1)**np.random.randint(0,2))/3,((-1)**np.random.randint(0,2))/3]
                else:
                    force = np.sign(distance)
                positive_force[force_round] = force
                if output_MS:
                    MS_file.write("push force:"+str(correction_value[force_round])+'\t'+str(force)+"\r\n")



    if np.max(np.abs(output[1:] - positive_value)) > 8 or neighbor_num > 1:
        output = np.copy(output_extra)
        # pull

        correction_IUV = tmp_IUV
        correction_num = valid_num
        correction_value = positive_value

        IUV_test  = np.reshape((IUV_input+tmp)[:,np.clip(k-30,0,imgheight):np.clip(k+30,0,imgheight),np.clip(s-30,0,imgwidth):np.clip(s+30,0,imgwidth),:].astype(np.int32),[-1,3])

        sim_loop = 0
        if output_MS:
            MS_file = open(MS_output_path + "%04d_%04d.txt"%(k,s),"a+")
            MS_file.write("white pull points:"+str(np.shape(IUV_test)[0])+"\r\n")
            MS_file.write("pull points:"+str(correction_num)+"\r\n")
            MS_file.write("extrapolation points:"+str(valid_num)+"\r\n")
            MS_output = np.zeros([256,256,3])
            for white_point in range(np.shape(IUV_test)[0]):
                if IUV_test[white_point,0]  == I_value:
                    MS_output[int(IUV_test[white_point,1]),int(IUV_test[white_point,2])] = [255,255,255]
            for green_point in range(correction_num):
                MS_output[int(correction_value[green_point,0]),int(correction_value[green_point,1])] = [255,0,0]
            for red_point in range(valid_num):
                MS_output[int(positive_value[red_point,0]),int(positive_value[red_point,1])] = [0,0,255]
            cv.imwrite(MS_output_path + "/%04d_%04d_pull_%04d.png"%(k,s,sim_loop), MS_output)
            MS_output_tmp = np.copy(MS_output)
            MS_output_tmp[int(output[1]),int(output[2])] = [0,255,0]
            cv.imwrite(MS_output_path + "/%04d_%04d_pull_%04d.png"%(k,s,sim_loop+1), MS_output_tmp)
            MS_file.write(str(output)+"\r\n")

        positive_force = np.zeros([correction_num,2])
        all_distance = np.zeros([correction_num,2])
        coefficient = 1.0
        rest_length = 1

        for force_round in range(correction_num):
            distance = correction_value[force_round]-np.array([output[1],output[2]])
            all_distance[force_round] = np.abs(distance)
            if np.abs(distance[0]) < epsilon and np.abs(distance[1]) > epsilon:
                force = [0,(np.abs(distance[1])-rest_length)*distance[1]*coefficient/np.abs(distance[1])]
            elif np.abs(distance[0]) > epsilon and np.abs(distance[1]) < epsilon:
                force = [(np.abs(distance[0])-rest_length)*distance[0]*coefficient/np.abs(distance[0]),0]
            elif np.abs(distance[0]) > epsilon and np.abs(distance[1]) > epsilon:
                force = (np.abs(distance)-rest_length)*distance*coefficient/np.abs(distance)
            else:
                force = (np.abs(distance)-rest_length)*coefficient*[(-1)**np.random.randint(0,2),(-1)**np.random.randint(0,2)]
            positive_force[force_round] = force
            if output_MS:
                MS_file.write("pull force:"+str(correction_value[force_round])+'\t'+str(force)+"\r\n")

        approved_distance = 2
        while (np.reshape(np.array(output).astype(np.int32),[3]).tolist() in IUV_test.tolist() and correction_num>0):
            if sim_loop > 10000:
                output = [0,0,0]
                break
            total_force = np.sum(positive_force,axis=0)
            old_output = np.copy(output)
            output[1:] += total_force/(correction_num)
            sim_loop += 1

            if output[1]>255 or output[2]>255 or output[1]<0 or output[2]<0:
                return [0,0,0]

            if output_MS:
                MS_file.write("total_pull_force:"+str(total_force)+"\r\n")
                MS_file.write("pull update:"+str(output)+"\r\n")
                MS_output_tmp = np.copy(MS_output)
                MS_output_tmp[int(output[1]),int(output[2])] = [0,255,0]
                cv.imwrite(MS_output_path + "/%04d_%04d_pull_%04d.png"%(k,s,sim_loop+1), MS_output_tmp)

            new_output = np.copy(output)
            if np.all(new_output == old_output):
                break
            for force_round in range(correction_num):
                distance = correction_value[force_round]-np.array([output[1],output[2]])
                all_distance[force_round] = np.abs(distance)
                if np.abs(distance[0]) < epsilon and np.abs(distance[1]) > epsilon:
                    force = [0,(np.abs(distance[1])-rest_length)*distance[1]*coefficient/np.abs(distance[1])]
                elif np.abs(distance[0]) > epsilon and np.abs(distance[1]) < epsilon:
                    force = [(np.abs(distance[0])-rest_length)*distance[0]*coefficient/np.abs(distance[0]),0]
                elif np.abs(distance[0]) > epsilon and np.abs(distance[1]) > epsilon:
                    force = (np.abs(distance)-rest_length)*distance*coefficient/np.abs(distance)
                else:
                    force = (np.abs(distance)-rest_length)*coefficient*[(-1)**np.random.randint(0,2),(-1)**np.random.randint(0,2)]
                positive_force[force_round] = force
                if output_MS:
                    MS_file.write("pull force:"+str(correction_value[force_round])+'\t'+str(force)+"\r\n")

    if output[1]>255 or output[2]>255 or output[1]<0 or output[2]<0 or (output[1]==0 and output[2]==0):
        output = [0,0,0]
    output = np.array(output).astype(np.int32).astype(np.float64)
    if output_MS:
        MS_file.write("final result:"+str(output)+"\r\n")
        MS_file.close()
    return output

if output_IMG:
    output_IUV_path = "extrapolation"
    if not os.path.exists(output_IUV_path):
        os.mkdir(output_IUV_path)
    output_texture_path = "texture"
    if not os.path.exists(output_texture_path):
        os.mkdir(output_texture_path)
    output_img_back_path = "image_back"
    if not os.path.exists(output_img_back_path):
        os.mkdir(output_img_back_path)

for i in range(1):
    IUV = np.array(cv.imread("/data/youxie/IUVoptimization/IUV/%04d.png"%i),np.float32)[np.newaxis,:,:,:]
    I_manual = np.array(Image.open("/data/youxie/gradientswap/example_i_value_label_post_processed/%04d.png"%i),np.float32)[np.newaxis,:,:] #manually labelled different parts for the whole body (index from 1 to 24)
    img = np.array(cv.imread("/data/youxie/gradientswap/example_rmbg_refined/%04d.png"%i),np.float32)[np.newaxis,:,:,:]

    IUV_extrapolation = np.copy(IUV)
    previous = np.copy(IUV_extrapolation)
    start_time = time.time()
    for neighbor_num in range(1,3):
        for t in range(100):
            tmp = np.zeros_like(IUV)
            failed_points = 0
            succuss_points = 0
            for m in range(neighbor_num,imgheight-neighbor_num+1):
                for n in range(neighbor_num,imgwidth-neighbor_num+1):
                    if np.any(IUV_extrapolation[0,m,n,0] > 0) and np.all(img[0,m,n] == [0.,0.,0.]):
                        IUV_extrapolation[0,m,n] = [0.,0.,0.]
                        I_manual[0,m,n] = 0
                    elif np.any(img[0,m,n] > 0):
                        if I_manual[0,m,n] > 0:
                            if I_manual[0,m,n] == IUV_extrapolation[0,m,n,0]:
                                continue
                            else:
                                IUV_extrapolation[0,m,n,] = [0.,0.,0.]
                                tmp[0,m,n] = mean_neighbor(img,IUV_extrapolation,m,n,tmp,neighbor_num,I_given=I_manual[0,m,n])
                        elif I_manual[0,m,n] == 0.:
                            if I_manual[0,m,n] != IUV_extrapolation[0,m,n,0]:
                                IUV_extrapolation[0,m,n,] = [0.,0.,0.]
                            tmp[0,m,n,:] = mean_neighbor(img,IUV_extrapolation,m,n,tmp,neighbor_num)
                            if tmp[0,m,n,0] != 0.:
                                I_manual[0,m,n] = tmp[0,m,n,0]

                        if tmp[0,m,n,0] == 0.:
                            failed_points += 1
                        else:
                            succuss_points += 1
                    if np.any(tmp[0,m,n]<0) or np.any(tmp[0,m,n]>255):
                        print("!!!!!!!!!!!!!!!!!ERROR!!!!!!!!!!!!!!!!!!!!")
            print("round: %d, %d points are not well addressed!%d points are well addressed! time cost:%f"%(t,failed_points,succuss_points,time.time()-start_time))
            IUV_extrapolation += tmp
            if np.linalg.norm(previous-IUV_extrapolation)==0:
                previous = np.copy(IUV_extrapolation)
                print("extrapolation doesn't work anymore, increase neighbor range.")
                break
            previous = np.copy(IUV_extrapolation)
    print("img: %d, time cost:%f"%(i,time.time()-start_time))
    if output_IMG:
        cv.imwrite(output_IUV_path + "/%04d.png"%(i),IUV_extrapolation[0])
        texture_map = TransferTextureback_np(img[0],IUV_extrapolation[0])
        result = TransferTexture_np(texture_map/255.,np.zeros_like(IUV[0]),IUV_extrapolation[0])
        cv.imwrite(output_texture_path + "/%04d.png"%(i), texture_map)
        cv.imwrite(output_img_back_path + "/%04d.png"%(i), result)
