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


# After texture random sampling, texture extrapolation is applied to fill the blank area in the texture.


import numpy as np
import cv2 as cv
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.interpolate import RegularGridInterpolator
from sklearn.linear_model import LinearRegression
from sklearn import neighbors
import os

img_height = 1200
img_width = 800

def holefill_nn(image):
    image_copy = np.copy(image)
    for i in range(img_height):
        for j in range(img_width):
            if np.all(image[i,j] == 0):
                position = []
                value = []
                x_start_index = np.clip(i-2,int(i/200) * 200,(int(i/200)+1) * 200)
                x_end_index = np.clip(i+2,int(i/200) * 200,(int(i/200)+1) * 200)
                y_start_index = np.clip(j-2,int(j/200) * 200,(int(j/200)+1) * 200)
                y_end_index = np.clip(j+2,int(j/200) * 200,(int(j/200)+1) * 200)
                for x in range(x_start_index,x_end_index):
                    for y in range(y_start_index,y_end_index):
                        if np.any(image[x,y] > 0):
                            position.append([x,y])
                            value.append(image[x,y])
                if len(position) > 0:
                    position = np.array(position)
                    value = np.array(value)
                    knn = neighbors.KNeighborsRegressor(1, weights='uniform')
                    output = knn.fit(position, value).predict(np.reshape([i,j],[1,2]))[0]
                    image_copy[i,j] = output
    return image_copy

texture = np.array(cv.imread("A1wwPTTzVGS_optimization/texture_0000_rs.png"),np.float32)
output_path = "A1wwPTTzVGS_optimization/"
if not os.path.exists(output_path):
    os.mkdir(output_path)
os.system("cp %s %s"%(__file__,output_path))
for ite in range(200):
    print("round %d starts!"%(ite))
    texture = holefill_nn(texture)
    cv.imwrite(output_path+"texture_0000_ex.png",texture)
