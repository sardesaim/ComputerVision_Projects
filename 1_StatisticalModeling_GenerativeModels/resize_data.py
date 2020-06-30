# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 12:50:56 2020

@author: sarde
"""

# import and convert images to defined size
import numpy as np
import glob 
from PIL import Image
import cv2
import pickle

root='D:/00_NCSU/Spring2020/ECE763_ComputerVision/Project/project01/Data/CBCL/'
proj = 'D:/00_NCSU/Spring2020/ECE763_ComputerVision/Project/project01/'
trainFace=np.array([np.array(Image.open(filename)) for filename in glob.glob(root+'pos_train/*.pgm')])
trainNonFace=np.array([np.array(Image.open(filename)) for filename in glob.glob(root+'neg_train/*.pgm')])
testFace=np.array([np.array(Image.open(filename)) for filename in glob.glob(root+'pos_test/*.pgm')])
testNonFace=np.array([np.array(Image.open(filename)) for filename in glob.glob(root+'neg_test/*.pgm')])

size = (20,20)
# trainFace=trainFace.reshape(trainFace.shape[0],-1)
# trainNonFace=trainNonFace.reshape(trainNonFace.shape[0], -1)
# testFace=testFace.reshape(testFace.shape[0],-1)
# testNonFace=testNonFace.reshape(testNonFace.shape[0],-1)

trainFace = np.array([np.array(cv2.resize(trainFace[i], \
                                          size, interpolation=cv2.INTER_AREA)) for i in range(trainFace.shape[0])])
    
trainNonFace = np.array([np.array(cv2.resize(trainNonFace[i], \
                                          size, interpolation=cv2.INTER_AREA)) for i in range(trainNonFace.shape[0])])
    
testFace = np.array([np.array(cv2.resize(testFace[i], \
                                          size, interpolation=cv2.INTER_AREA)) for i in range(testFace.shape[0])])
    
testNonFace = np.array([np.array(cv2.resize(testNonFace[i], \
                                          size, interpolation=cv2.INTER_AREA)) for i in range(testNonFace.shape[0])])
    
pickle.dump(trainFace, open(proj+'Data_dumps/CBCLtrainFace20.p', 'wb+'))
pickle.dump(trainNonFace, open(proj+'Data_dumps/CBCLtrainNonFace20.p', 'wb+'))
pickle.dump(testFace, open(proj+'Data_dumps/CBCLtestFace20.p', 'wb+'))
pickle.dump(testNonFace, open(proj+'Data_dumps/CBCLtestNonFace20.p', 'wb+'))

size = (10,10)
trainFace=trainFace.reshape(trainFace.shape[0],-1)
trainNonFace=trainNonFace.reshape(trainNonFace.shape[0], -1)
testFace=testFace.reshape(testFace.shape[0],-1)
testNonFace=testNonFace.reshape(testNonFace.shape[0],-1)

trainFace = np.array([np.array(cv2.resize(trainFace[i], \
                                          size, interpolation=cv2.INTER_AREA)) for i in range(trainFace.shape[0])])
    
trainNonFace = np.array([np.array(cv2.resize(trainNonFace[i], \
                                          size, interpolation=cv2.INTER_AREA)) for i in range(trainNonFace.shape[0])])
    
testFace = np.array([np.array(cv2.resize(testFace[i], \
                                          size, interpolation=cv2.INTER_AREA)) for i in range(testFace.shape[0])])
    
testNonFace = np.array([np.array(cv2.resize(testNonFace[i], \
                                          size, interpolation=cv2.INTER_AREA)) for i in range(testNonFace.shape[0])])
    
pickle.dump(trainFace, open(proj+'Data_dumps/CBCLtrainFace10.p', 'wb+'))
pickle.dump(trainNonFace, open(proj+'Data_dumps/CBCLtrainNonFace10.p', 'wb+'))
pickle.dump(testFace, open(proj+'Data_dumps/CBCLtestFace10.p', 'wb+'))
pickle.dump(testNonFace, open(proj+'Data_dumps/CBCLtestNonFace10.p', 'wb+'))

# root='D:/00_NCSU/Spring2020/ECE763_ComputerVision/Project/project01/'
# trainFace=np.array([np.array(Image.open(root+'pos_train/face'+str(i)+\
#                                         '.jpg')) for i in range(1000)])
# trainNonFace=np.array([np.array(Image.open(root+'neg_train/non_face'+str(i)+\
#                                         '.jpg')) for i in range(1000)])
# testFace=np.array([np.array(Image.open(root+'pos_test/face'+str(i)+\
#                                         '.jpg')) for i in range(1000,1100)])
# testNonFace=np.array([np.array(Image.open(root+'neg_test/non_face'+str(i)+\
#                                             '.jpg')) for i in range(1000,1100)])

# pickle.dump(trainFace, open(proj+'Data_dumps/WiderRGBtrainFace20.p', 'wb+'))
# pickle.dump(trainNonFace, open(proj+'Data_dumps/WiderRGBtrainNonFace20.p', 'wb+'))
# pickle.dump(testFace, open(proj+'Data_dumps/WiderRGBtestFace20.p', 'wb+'))
# pickle.dump(testNonFace, open(proj+'Data_dumps/WiderRGBtestNonFace20.p', 'wb+'))
