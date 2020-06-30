# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 22:09:21 2020

@author: sarde
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt 

import pickle
import cv2


size=(60,60)
root='D:/00_NCSU/Spring2020/ECE763_ComputerVision/Project/project03/data/face/'

def readDumpImages():
    '''
    Read Images using PIL and Dump in .p object using pickle

    Returns
    -------
    None.

    '''
    trainFace=np.array([np.array(cv2.resize(cv2.imread(root+'pos_train/face'+str(i)+'.jpg'),size, interpolation=cv2.INTER_AREA)) for i in range(1000)])
    trainNonFace=np.array([np.array(cv2.resize(cv2.imread(root+'neg_train/non_face'+str(i)+'.jpg'),size,interpolation=cv2.INTER_AREA)) for i in range(1000)])
    testFace=np.array([np.array(cv2.resize(cv2.imread(root+'pos_test/face'+str(i)+'.jpg'),size,interpolation=cv2.INTER_AREA)) for i in range(1000,1100)])
    testNonFace=np.array([np.array(cv2.resize(cv2.imread(root+'neg_test/non_face'+str(i)+'.jpg'),size,interpolation=cv2.INTER_AREA)) for i in range(1000,1100)])
    return trainFace, trainNonFace, testFace, testNonFace