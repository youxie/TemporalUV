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

## temporal relocation, with optical flow and rgb matching, we can generate the full sequence with the fixed texture.

import cv2
import numpy as np
import os
import sys
sys.path.append(os.path.abspath('../tools'))
from ops import *
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.interpolate import RegularGridInterpolator
from sklearn.linear_model import LinearRegression
from sklearn import neighbors

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res

img_height = 224
img_width = 176

input_IUV_path = "/mnt/netdisk1/youxie/Fashion_extrapolation/IUV_optimization_all_smooth/"
input_img_path = "/mnt/netdisk1/youxie/Fashion_extrapolation/example_rmbg_refined/"
input_texture_path = "/mnt/netdisk1/youxie/Fashion_coordinate_2/data/rs_IUV_optimization_all_smooth/"
texture_reference = np.array(cv.imread("/mnt/netdisk1/youxie/Fashion_coordinate_2/data/ex_IUV_optimization_all_smooth/texture_0000_ex.png"),np.float32)

texture_grid = np.array(cv.imread("texture_grid.png"),np.float32)
output_path = "OF_texture_IUV_optimization_all_smooth_local_30/"
output_path_IUV = "OF_texture_IUV_optimization_all_smooth_local_30_IUV/"

if not os.path.exists(output_path):
    os.makedirs(output_path)
if not os.path.exists(output_path_IUV):
    os.makedirs(output_path_IUV)

coor_1_texture = np.zeros([1200,800,3])
for x in range(1200):
    for y in range(800):
        coor_1_texture[x,y] = [x,y,0]

base_frame = 0
IUV_1 = np.array(cv2.imread("%s/IUV_%04d.png"%(input_IUV_path,base_frame)),np.float32)[22:-22,8:-8,:][::4,::4,:3]
coor1_back = TransferTexture_np(coor_1_texture/255.,np.zeros([img_height,img_width,3]),IUV_1)
np.save("%s/map_%04d.npy"%(output_path,base_frame), coor_1_texture)
cv2.imwrite("%s/map_%04d.png"%(output_path,base_frame), coor_1_texture)
np.save("%s/%04d.npy"%(output_path,base_frame), coor1_back)
cv2.imwrite("%s/%04d.png"%(output_path,base_frame), coor1_back)
os.system("cp %s %s"%(__file__, output_path))

for i in range(0,121,1):
    if i == base_frame:
        continue
    texture_1 = np.array(cv2.imread("%s/%04d.png"%(input_texture_path,base_frame)),np.float32)
    frame1 = np.array(cv.imread(input_img_path + "%04d.png"%(base_frame)),np.float32)[22:-22,8:-8,:][::4,::4,:3]
    IUV_1 = np.array(cv2.imread("%s/IUV_%04d.png"%(input_IUV_path,base_frame)),np.float32)[22:-22,8:-8,:][::4,::4,:3]
    prvs = cv2.cvtColor(texture_1,cv2.COLOR_BGR2GRAY)

    texture_2 = np.array(cv2.imread("%s/%04d.png"%(input_texture_path,i)),np.float32)
    frame2 = np.array(cv.imread(input_img_path + "%04d.png"%(i)),np.float32)[22:-22,8:-8,:][::4,::4,:3]
    IUV_2 = np.array(cv2.imread("%s/IUV_%04d.png"%(input_IUV_path,i)),np.float32)[22:-22,8:-8,:][::4,::4,:3]
    texture_real_2 = TransferTextureback_np(frame2,IUV_2)
    next = cv2.cvtColor(texture_2,cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 5, 3, 5, 1.1, 0)
    coor_2_texture = warp_flow(coor_1_texture, flow)
    coor_2_texture[400:800,200:400] = coor_1_texture[400:800,200:400]

    for x in range(200,400):
        for y in range(0,200):
            if np.linalg.norm(texture_reference[int(coor_2_texture[x,y,0]),int(coor_2_texture[x,y,1])] - texture_real_2[x,y]) > 30:
                coor_2_texture[x,y] = [0,0,0]

    for x in range(800,1200):
        for y in range(600,800):
            if np.linalg.norm(texture_reference[int(coor_2_texture[x,y,0]),int(coor_2_texture[x,y,1])] - texture_real_2[x,y]) > 30:
                coor_2_texture[x,y] = [0,0,0]

    for x in range(400,800):
        for y in range(200,400):
            if np.linalg.norm(texture_reference[int(coor_2_texture[x,y,0]),int(coor_2_texture[x,y,1])] - texture_real_2[x,y]) > 40:
                coor_2_texture[x,y] = [0,0,0]

    for x in range(1200):
        for y in range(800):
            if np.all(texture_real_2[x,y]==0):
                coor_2_texture[x,y] = [x,y,0]


    flag = np.zeros([1200,800,1])
    sample_size = 20
    for ite in range(1):
        for x in range(200,400-sample_size):
            for y in range(0,200-sample_size):
                if np.all(coor_2_texture[x,y]==0) and np.any(texture_real_2[x,y]>0):
                    search_size = 25
                    dist_col = []
                    while ((dist_col == [] or dist_col[0][0]>500) and search_size < 40):
                        x_start = x
                        x_end = x+sample_size
                        y_start = y
                        y_end = y+sample_size
                        empty_count = 0
                        for xi in range(x_start,x_end):
                            for yi in range(y_start,y_end):
                                if np.all(coor_2_texture[xi,yi]==0) and np.any(texture_real_2[xi,yi]>0):
                                    empty_count += 1
                        if empty_count > 0:
                            if x > 400 - search_size - 1:
                                x_start = 400 - search_size - 1
                                x_end = x_start + search_size - sample_size
                            else:
                                x_start = x
                                x_end = x_start + search_size - sample_size
                            if y > 200 - search_size - 1:
                                y_start = 200 - search_size - 1
                                y_end = y_start + search_size - sample_size
                            else:
                                y_start = y
                                y_end = y_start + search_size - sample_size

                            dist_col = []
                            for xi in range(x_start,x_end):
                                for yi in range(y_start,y_end):
                                    distance = np.linalg.norm(texture_1[xi:xi+sample_size,yi:yi+sample_size]-texture_2[x:x+sample_size,y:y+sample_size])
                                    dist_col.append([distance,xi,yi])
                            dist_col.sort()
                            search_size += 1
                    coor_2_texture[x:x+sample_size,y:y+sample_size] = coor_1_texture[dist_col[0][1]:dist_col[0][1]+sample_size,dist_col[0][2]:dist_col[0][2]+sample_size]


    for ite in range(1):
        sample_size = 10
        for x in range(800,1200-sample_size):
            for y in range(600,800-sample_size):
                if np.all(coor_2_texture[x,y]==0) and np.any(texture_real_2[x,y]>0):
                    search_size = 25
                    dist_col = []
                    while ((dist_col == [] or dist_col[0][0]>500) and search_size < 40):
                        x_start = x
                        x_end = x+sample_size
                        y_start = y
                        y_end = y+sample_size
                        empty_count = 0
                        for xi in range(x_start,x_end):
                            for yi in range(y_start,y_end):
                                if np.all(coor_2_texture[xi,yi]==0) and np.any(texture_real_2[xi,yi]>0):
                                    empty_count += 1
                        if empty_count > 0:
                            if x > 1200 - search_size - 1:
                                x_start = 1200 - search_size - 1
                                x_end = x_start + search_size - sample_size
                            else:
                                x_start = x
                                x_end = x_start + search_size - sample_size
                            if y > 800 - search_size - 1:
                                y_start = 800 - search_size - 1
                                y_end = y_start + search_size - sample_size
                            else:
                                y_start = y
                                y_end = y_start + search_size - sample_size

                            dist_col = []
                            for xi in range(x_start,x_end):
                                for yi in range(y_start,y_end):
                                    distance = np.linalg.norm(texture_1[xi:xi+sample_size,yi:yi+sample_size]-texture_2[x:x+sample_size,y:y+sample_size])
                                    dist_col.append([distance,xi,yi])
                            dist_col.sort()
                            search_size += 1
                    coor_2_texture[x:x+sample_size,y:y+sample_size] = coor_1_texture[dist_col[0][1]:dist_col[0][1]+sample_size,dist_col[0][2]:dist_col[0][2]+sample_size]

    for ite in range(1):
        sample_size = 4
        for x in range(400,800-sample_size):
            for y in range(200,400-sample_size):
                if (np.all(coor_2_texture[x,y]==0) and np.any(texture_real_2[x,y]>0)):
                    dist_col = []
                    search_size = 25
                    while ((dist_col == [] or dist_col[0][0]>500) and search_size < 40):
                        x_start = x
                        x_end = x+sample_size
                        y_start = y
                        y_end = y+sample_size
                        empty_count = 0
                        for xi in range(x_start,x_end):
                            for yi in range(y_start,y_end):
                                if np.all(coor_2_texture[xi,yi]==0) and np.any(texture_real_2[xi,yi]>0):
                                    empty_count += 1
                        if empty_count > 0:
                            if x > 800 - search_size - 1:
                                x_start = 800 - search_size - 1
                                x_end = x_start + search_size - sample_size
                            else:
                                x_start = x
                                x_end = x_start + search_size - sample_size
                            if y > 400 - search_size - 1:
                                y_start = 400 - search_size - 1
                                y_end = y_start + search_size - sample_size
                            else:
                                y_start = y
                                y_end = y_start + search_size - sample_size

                            dist_col = []
                            for xi in range(x_start,x_end):
                                for yi in range(y_start,y_end):
                                    distance = np.linalg.norm(texture_1[xi:xi+sample_size,yi:yi+sample_size]-texture_2[x:x+sample_size,y:y+sample_size])
                                    dist_col.append([distance,xi,yi])
                            dist_col.sort()
                            search_size += 1
                    coor_2_texture[x:x+sample_size,y:y+sample_size] = coor_1_texture[dist_col[0][1]:dist_col[0][1]+sample_size,dist_col[0][2]:dist_col[0][2]+sample_size]
    coor_1_texture = coor_2_texture
    texture_2_gen = np.zeros_like(texture_2)
    texture_2_gen_grid = np.zeros_like(texture_2)
    for x in range(1200):
        for y in range(800):
            if np.all(texture_real_2[x,y]==0):
                coor_2_texture[x,y] = [x,y,0]
            else:
                texture_2_gen[x,y] = texture_reference[int(coor_2_texture[x,y,0]),int(coor_2_texture[x,y,1])]
                texture_2_gen_grid[x,y] = texture_grid[int(coor_2_texture[x,y,0]),int(coor_2_texture[x,y,1])]
    cv2.imwrite("%s/map_%04d.png"%(output_path,i), coor_2_texture)
    np.save("%s/map_%04d.npy"%(output_path,i), coor_2_texture)
    coor2_back = TransferTexture_np(coor_2_texture/255.,np.zeros([img_height,img_width,3]),IUV_2)
    cv2.imwrite("%s/%04d.png"%(output_path,i), coor2_back)
    np.save("%s/%04d.npy"%(output_path,i), coor2_back)
    frame2_back = TransferTexture_np(texture_2_gen/255.,np.zeros([img_height,img_width,3]),IUV_2)
    cv2.imwrite("%s/image_%04d.png"%(output_path,i), frame2_back)
    cv2.imwrite("%s/texture_%04d.png"%(output_path,i), texture_2_gen)
    frame2_back = TransferTexture_np(texture_2_gen_grid/255.,np.zeros([img_height,img_width,3]),IUV_2)
    cv2.imwrite("%s/image_grid_%04d.png"%(output_path,i), frame2_back)


    img = np.zeros([img_height,img_width,3])
    coor = coor2_back
    for x in range(img_height):
        for y in range(img_width):
            img[x,y] = texture_reference[int(coor[x,y,0]),int(coor[x,y,1])]
    cv.imwrite("%simage_%04d.png"%(output_path_IUV,i),img)
    IUV_output = np.zeros_like(coor)
    for x in range(img_height):
        for y in range(img_width):
            if np.any(coor[x,y]>0):
                row = coor[x,y,0]//200
                column = coor[x,y,1]//200
                i_value = column * 6 + row + 1
                IUV_output[x,y,0] = i_value
                x_coor = coor[x,y,0] - row * 200
                y_coor = coor[x,y,1] - column * 200
                v_value = 255 - x_coor * 255. / 199.
                u_value = y_coor * 255. / 199.
                IUV_output[x,y,1:] = [u_value,v_value]
    cv.imwrite("%sIUV_%04d.png"%(output_path_IUV,i),IUV_output)
    np.save("%sIUV_%04d.npy"%(output_path_IUV,i),IUV_output)
    texture = TransferTextureback_np(img,IUV_output)
    cv.imwrite("%stexture_%04d.png"%(output_path_IUV,i),texture)
    image_back = TransferTexture_np(texture/255.,np.zeros([img_height,img_width,3]),IUV_output)
    cv.imwrite("%simage_back_%04d.png"%(output_path_IUV,i),image_back)
    image_with_t0 = TransferTexture_np(texture_reference/255.,np.zeros([img_height,img_width,3]),IUV_output)
    cv.imwrite("%simage_with_t0_%04d.png"%(output_path_IUV,i),image_with_t0)

    print("frame %d finished!"%i)
