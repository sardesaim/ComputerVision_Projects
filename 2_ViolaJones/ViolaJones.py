# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 15:26:29 2020

@author: sarde
"""
# from time import time
from tqdm import tqdm 
import numpy as np
import matplotlib.pyplot as plt
import cv2 
import glob 
from PIL import Image
import pickle
import os
from dask import delayed
from skimage.feature import haar_like_feature
from skimage.feature import haar_like_feature_coord
from skimage.feature import draw_haar_like_feature
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from skimage.transform import integral_image


root = 'D:/00_NCSU/Spring2020/ECE763_ComputerVision/Project/project02/'

def dump_data():
    '''
    Read and dump data into .p files 
    Returns
    -------
    None.
    '''
    size=(10,10)
    face_data=np.array([cv2.resize(np.array(Image.open(filename)),size) for filename in glob.glob(root+'Face16/*.bmp')])
    nonface_data=np.array([cv2.resize(np.array(Image.open(filename)), size) for filename in glob.glob(root+'Nonface16/*.bmp')])
    split = 0.8
    train_face_shape = int(face_data.shape[0]*split)
    train_nonface_shape = int(nonface_data.shape[0]*split)
    trainFace = face_data[0:train_face_shape]
    trainNonFace = nonface_data[0:train_nonface_shape]
    testFace = face_data[train_face_shape-1:-1]
    testNonFace = nonface_data[train_nonface_shape-1:-1] 
    
    try:
        os.mkdir('Data_dumps')
    except:
        pass
    pickle.dump(trainFace, open(root+'Data_dumps/trainFace.p', 'wb+'))
    pickle.dump(trainNonFace, open(root+'Data_dumps/trainNonFace.p', 'wb+'))
    pickle.dump(testFace, open(root+'Data_dumps/testFace.p', 'wb+'))
    pickle.dump(testNonFace, open(root+'Data_dumps/testNonFace.p', 'wb+'))
    
def loadImageArraysFromDumps():
    '''
    load images (np arrays) from dumps using pickle

    Returns
    -------
    trainFace : np array
        train set for face.
    trainNonFace : np array
        train set for non face.
    testFace : np array
        test set for face.
    testNonFace : np array
        test set for non face.

    '''
    trainFace=pickle.load(open('Data_dumps/trainFace.p', 'rb'))
    trainNonFace=pickle.load(open('Data_dumps/trainNonFace.p', 'rb'))
    testFace=pickle.load(open('Data_dumps/testFace.p', 'rb'))
    testNonFace=pickle.load(open('Data_dumps/testNonFace.p', 'rb'))
    return trainFace, trainNonFace, testFace, testNonFace

def normalize_dataset(trainFace, trainNonFace, testFace, testNonFace):
    '''
    Normalize Data from 0 to 1 

    Parameters
    ----------
    trainFace : np array
        trainData for face.
    trainNonFace : np array
        trainData for non_face.
    testFace : np array
        trainData for face
    testNonFace : np array
        testData for non_face

    Returns
    -------
    np array
        trainData for face.
    np array
        trainData for non_face.
    np array
        trainData for face.
    np array
        testData for non_face.

    '''
    scale = 255
    return trainFace/scale , trainNonFace/scale , testFace/scale , testNonFace/scale 

def getIntegralImage(img):
    '''
    Returns intergral image 

    Parameters
    ----------
    img : np array 
        Input image.

    Returns
    -------
    np array
        Integral image.

    '''
    x = np.cumsum(np.cumsum(img,1),0)
    x_p = np.pad(x, pad_width=1, mode='constant', constant_values=0)
    return x_p[0:-1,0:-1]

def getFeatureValue(feature_kernel_sizes,int_image, feat_type, x, y, scale_x, scale_y):
    # feature_kernel_sizes = [ (2,1), (1,2), (1,3), (3,1), (2,2)]
    # bottom_x = x + feature_kernel_sizes[feat_type][0]*scale_x
    # bottom_y = y + feature_kernel_sizes[feat_type][1]*scale_y
    
    if feat_type==0:
        curr_feat = feature_kernel_sizes[feat_type]
        ht, wd = curr_feat[0]*scale_x, curr_feat[1]*scale_y
        a_x, a_y = x-1,y-1
        d_x,d_y= ( x+ht-1)//2, (y+wd-1)
        b_x, b_y = x-1, y+wd-1
        c_x, c_y = (x+ht-1)//2, y-1
        sum_top = int_image[d_x,d_y]+int_image[a_x,a_y]-int_image[b_x,b_y]-int_image[c_x,c_y]
        a_x, a_y = c_x, c_y
        b_x, b_y = d_x, d_y
        d_x,d_y= ( x+ht-1), (y+wd-1)
        c_x, c_y = x+ht-1, y-1
        sum_bottom = int_image[d_x,d_y]+int_image[a_x,a_y]-int_image[b_x,b_y]-int_image[c_x,c_y]
        value = sum_top - sum_bottom
    elif feat_type==1:
        curr_feat = feature_kernel_sizes[feat_type]
        ht, wd = curr_feat[0]*scale_x, curr_feat[1]*scale_y
        a_x, a_y = x-1,y-1
        d_x,d_y= ( x+ht-1), (y+wd-1)//2
        b_x, b_y = (x+ht-1), y-1
        c_x, c_y = x-1, (y+wd-1)//2
        sum_left = int_image[d_x,d_y]+int_image[a_x,a_y]-int_image[b_x,b_y]-int_image[c_x,c_y]
        a_x, a_y = b_x, b_y
        b_x, b_y = (x+ht-1), y-1
        c_x, c_y = d_x, d_y
        d_x,d_y= ( x+ht-1), (y+wd-1)
        sum_right = int_image[d_x,d_y]+int_image[a_x,a_y]-int_image[b_x,b_y]-int_image[c_x,c_y]
        value = sum_right - sum_left
    elif feat_type==2:
        curr_feat = feature_kernel_sizes[feat_type]
        ht, wd = curr_feat[0]*scale_x, curr_feat[1]*scale_y
        a_x, a_y = x-1,y-1
        d_x,d_y= ( x+ht-1)//3, (y+wd-1)
        b_x, b_y = x-1, y+wd-1
        c_x, c_y = (x+ht-1)//3, y-1
        sum_top = int_image[d_x,d_y]+int_image[a_x,a_y]-int_image[b_x,b_y]-int_image[c_x,c_y]
        a_x, a_y = c_x, c_y
        b_x, b_y = d_x, d_y
        d_x,d_y= ( x+ht-1)*2//3, (y+wd-1)
        c_x, c_y = (x+ht-1)*2//3, y-1
        sum_middle = int_image[d_x,d_y]+int_image[a_x,a_y]-int_image[b_x,b_y]-int_image[c_x,c_y]
        a_x, a_y = c_x, c_y
        b_x, b_y = d_x, d_y
        d_x,d_y= ( x+ht-1), (y+wd-1)
        c_x, c_y = x-1, (y+wd-1)
        sum_bottom = int_image[d_x,d_y]+int_image[a_x,a_y]-int_image[b_x,b_y]-int_image[c_x,c_y]
        value = sum_middle - sum_top - sum_bottom
    elif feat_type==3:        
        curr_feat = feature_kernel_sizes[feat_type]
        ht, wd = curr_feat[0]*scale_x, curr_feat[1]*scale_y
        a_x, a_y = x-1,y-1
        d_x,d_y= ( x+ht-1), (y+wd-1)//3
        b_x, b_y = x-1, (y+wd-1)//3
        c_x, c_y = x+ht-1, y-1
        sum_left = int_image[d_x,d_y]+int_image[a_x,a_y]-int_image[b_x,b_y]-int_image[c_x,c_y]
        a_x, a_y = b_x, b_y
        c_x, c_y = d_x, d_y
        b_x, b_y = (x-1), (y+wd-1)*2//3
        d_x,d_y= ( x+ht-1), (y+wd-1)*2//3
        sum_middle = int_image[d_x,d_y]+int_image[a_x,a_y]-int_image[b_x,b_y]-int_image[c_x,c_y]
        a_x, a_y = b_x, b_y
        c_x, c_y = d_x, d_y
        b_x, b_y = x-1, y+wd-1
        d_x,d_y= ( x+ht-1), (y+wd-1)
        sum_right = int_image[d_x,d_y]+int_image[a_x,a_y]-int_image[b_x,b_y]-int_image[c_x,c_y]
        value = sum_middle - sum_right - sum_left
    else:
        curr_feat = feature_kernel_sizes[feat_type]
        ht, wd = curr_feat[0]*scale_x, curr_feat[1]*scale_y
        a_x, a_y = x-1,y-1
        d_x,d_y= ( x+ht-1)//2, (y+wd-1)//2
        b_x, b_y = x-1, (y+wd-1)//2
        c_x, c_y = (x+ht-1)//2, (y-1)
        tl = int_image[d_x,d_y]+int_image[a_x,a_y]-int_image[b_x,b_y]-int_image[c_x,c_y]
        a_x, a_y = b_x, b_y
        c_x, c_y = d_x, d_y
        b_x, b_y = (x-1), y+wd-1
        d_x,d_y= ( x+ht-1)//2, (y+wd-1)
        tr = int_image[d_x,d_y]+int_image[a_x,a_y]-int_image[b_x,b_y]-int_image[c_x,c_y]
        a_x, a_y = (x+ht-1)//2,(y-1)
        d_x,d_y= ( x+ht-1)//2, (y+wd-1)
        b_x, b_y = (x+ht-1)//2, (y+wd-1)//2
        c_x, c_y = (x+ht-1), (y-1)
        bl = int_image[d_x,d_y]+int_image[a_x,a_y]-int_image[b_x,b_y]-int_image[c_x,c_y]
        a_x, a_y = b_x, b_y
        c_x, c_y = d_x, d_y
        b_x, b_y = (x+ht-1)//2, (y+wd-1)
        d_x,d_y= ( x+ht-1), (y+wd-1)
        br = int_image[d_x,d_y]+int_image[a_x,a_y]-int_image[b_x,b_y]-int_image[c_x,c_y]
        value = tr + bl - tl - br
    return value

def getFeatures(feature_kernel_sizes, img):
    img_shape = img.shape
    
    feat=[]
    for feat_type in range(5):
        features = []
        curr_feat = feature_kernel_sizes[feat_type]
        for ix in range(1,img_shape[0]):
            for iy in range(1,img_shape[1]):
                scale_y=1
                scale_x=1
                ht, wd = curr_feat[0]*scale_x, curr_feat[1]*scale_y
                while(ix+ht<img_shape[1]):
                    scale_y=1
                    ht, wd = curr_feat[0]*scale_x, curr_feat[1]*scale_y
                    while(iy+wd<img_shape[0]):
                        ht, wd = curr_feat[0]*scale_x, curr_feat[1]*scale_y
                        try:
                            features = [getFeatureValue(feature_kernel_sizes,img, feat_type, ix, iy, scale_x, scale_y),\
                                        ix,iy,scale_x,scale_y, feat_type]
                            feat.append(features)
                        except:
                            pass
                        scale_y+=1
                    scale_x+=1
    # ft = np.array()
    # return np.array(feat) #, (ix, iy), (scale_x, scale_y), feat_type   
    return feat #, (ix, iy), (scale_x, scale_y), feat_type

@delayed
def extract_feature_image(img, feature_type, feature_coord=None):
    """Extract the haar feature for the current image"""
    ii = integral_image(img)
    return haar_like_feature(ii, 0, 0, ii.shape[0], ii.shape[1],
                             feature_type=feature_type,
                             feature_coord=feature_coord)
def convToPlusMinus(h):
    return  [int(x)-1 if x == False else int(x) for x in h]

def drawFeat(feature_kernel_sizes,image, feat_type, x, y, scale_x, scale_y):
    image = np.pad(image, pad_width=1, mode='constant', constant_values=0)
    if feat_type==0:
        curr_feat = feature_kernel_sizes[feat_type]
        ht, wd = curr_feat[0]*scale_x, curr_feat[1]*scale_y
        a_x, a_y = x-1,y-1
        # d_x,d_y= ( x+ht-1)//2, (y+wd-1)
        # b_x, b_y = x-1, y+wd-1
        # c_x, c_y = (x+ht-1)//2, y-1
        # for i in range(a_x, c_x):
        #     for j in range(a_y, b_y):
        #         image[i,j] = 0
        # a_x, a_y = c_x, c_y
        # b_x, b_y = d_x, d_y
        # d_x,d_y= ( x+ht-1), (y+wd-1)
        # c_x, c_y = x+ht-1, y-1
        # for i in range(a_x, c_x):
        #     for j in range(a_y, b_y):
        #         image[i,j] = 1
        for i in range(a_x, a_x+ht):
            for j in range(a_y,a_y+wd):
                image[i,j]=0
        for i in range(a_x+(ht//2), a_x+ht):
            for j in range(a_y,a_y+wd):
                image[i,j]=1
                
    elif feat_type==1:
        curr_feat = feature_kernel_sizes[feat_type]
        ht, wd = curr_feat[0]*scale_x, curr_feat[1]*scale_y
        a_x, a_y = x-1,y-1
        # d_x,d_y= ( x+ht-1), (y+wd-1)//2
        # b_x, b_y = (x+ht-1), y-1
        # c_x, c_y = x-1, (y+wd-1)//2
        # for i in range(a_x, c_x):
        #     for j in range(a_y, b_y):
        #         image[i,j] = 1
        # a_x, a_y = b_x, b_y
        # b_x, b_y = (x+ht-1), y-1
        # c_x, c_y = d_x, d_y
        # d_x,d_y= ( x+ht-1), (y+wd-1)
        # for i in range(a_x, c_x):
        #     for j in range(a_y, b_y):
        #         image[i,j] = 0
        for i in range(a_x, a_x+ht):
            for j in range(a_y,a_y+wd):
                image[i,j]=0
        for i in range(a_x, a_x+ht):
            for j in range(a_y+wd//2,a_y+wd):
                image[i,j]=1
    elif feat_type==2:
        curr_feat = feature_kernel_sizes[feat_type]
        ht, wd = curr_feat[0]*scale_x, curr_feat[1]*scale_y
        a_x, a_y = x-1,y-1
        d_x,d_y= ( x+ht-1)//3, (y+wd-1)
        b_x, b_y = x-1, y+wd-1
        c_x, c_y = (x+ht-1)//3, y-1
        for i in range(a_x, c_x):
            for j in range(a_y, b_y):
                image[i,j] = 1
        a_x, a_y = c_x, c_y
        b_x, b_y = d_x, d_y
        d_x,d_y= ( x+ht-1)*2//3, (y+wd-1)
        c_x, c_y = (x+ht-1)*2//3, y-1
        for i in range(a_x, c_x):
            for j in range(a_y, b_y):
                image[i,j] = 0
        a_x, a_y = c_x, c_y
        b_x, b_y = d_x, d_y
        d_x,d_y= ( x+ht-1), (y+wd-1)
        c_x, c_y = x-1, (y+wd-1)
        for i in range(a_x, c_x):
            for j in range(a_y, b_y):
                image[i,j] = 1
    elif feat_type==3:        
        curr_feat = feature_kernel_sizes[feat_type]
        ht, wd = curr_feat[0]*scale_x, curr_feat[1]*scale_y
        a_x, a_y = x-1,y-1
        # d_x,d_y= ( x+ht-1), (y+wd-1)//3
        # b_x, b_y = x-1, (y+wd-1)//3
        # c_x, c_y = x+ht-1, y-1
        # for i in range(a_x, c_x):
        #     for j in range(a_y, b_y):
        #         image[i,j] = 1
        # a_x, a_y = b_x, b_y
        # c_x, c_y = d_x, d_y
        # b_x, b_y = (x-1), (y+wd-1)*2//3
        # d_x,d_y= ( x+ht-1), (y+wd-1)*2//3
        # for i in range(a_x, c_x):
        #     for j in range(a_y, b_y):
        #         image[i,j] = 0
        # a_x, a_y = b_x, b_y
        # c_x, c_y = d_x, d_y
        # b_x, b_y = x-1, y+wd-1
        # d_x,d_y= ( x+ht-1), (y+wd-1)
        # for i in range(a_x, c_x):
        #     for j in range(a_y, b_y):
        #         image[i,j] = 1
        for i in range(a_x, a_x+ht):
            for j in range(a_y,a_y+wd):
                image[i,j]=1
        for i in range(a_x+(ht//3), a_x+(ht*2//3)):
            for j in range(a_y,a_y+wd):
                image[i,j]=0
        
    else:
        curr_feat = feature_kernel_sizes[feat_type]
        ht, wd = curr_feat[0]*scale_x, curr_feat[1]*scale_y
        a_x, a_y = x-1,y-1
        d_x,d_y= ( x+ht-1)//2, (y+wd-1)//2
        b_x, b_y = x-1, (y+wd-1)//2
        c_x, c_y = (x+ht-1)//2, (y-1)
        for i in range(a_x, c_x):
            for j in range(a_y, b_y):
                image[i,j] = 1
        a_x, a_y = b_x, b_y
        c_x, c_y = d_x, d_y
        b_x, b_y = (x-1), y+wd-1
        d_x,d_y= ( x+ht-1)//2, (y+wd-1)
        for i in range(a_x, c_x):
            for j in range(a_y, b_y):
                image[i,j] = 0
        a_x, a_y = (x+ht-1)//2,(y-1)
        d_x,d_y= ( x+ht-1)//2, (y+wd-1)
        b_x, b_y = (x+ht-1)//2, (y+wd-1)//2
        c_x, c_y = (x+ht-1), (y-1)
        for i in range(a_x, c_x):
            for j in range(a_y, b_y):
                image[i,j] = 1
        a_x, a_y = b_x, b_y
        c_x, c_y = d_x, d_y
        b_x, b_y = (x+ht-1)//2, (y+wd-1)
        d_x,d_y= ( x+ht-1), (y+wd-1)
        for i in range(a_x, c_x):
            for j in range(a_y, b_y):
                image[i,j] = 0
    return image[1:-1,1:-1]


if __name__ == '__main__':
    # dump_data()
    trainFace, trainNonFace, testFace, testNonFace = loadImageArraysFromDumps() 
    trainFace, trainNonFace, testFace, testNonFace = normalize_dataset(trainFace, trainNonFace, testFace, testNonFace)
    #get integral images 
    trainFaceInt = np.array([getIntegralImage(trainFace[i]) for i in range(trainFace.shape[0])])
    trainNonFaceInt = np.array([getIntegralImage(trainNonFace[i]) for i in range(trainNonFace.shape[0])])
    testFaceInt = np.array([getIntegralImage(testFace[i]) for i in range(testFace.shape[0])])
    testNonFaceInt = np.array([getIntegralImage(testNonFace[i]) for i in range(testNonFace.shape[0])])
    # feature_types = ['type-2-x', 'type-2-y', 'type-3-x', 'type-3-y', 'type-4']
    # images = np.vstack((trainFace[0:200],trainNonFace[0:200]))
    images = np.vstack((trainFace,trainNonFace))
    # test_images = np.vstack((testFace[100:150],testNonFace[100:150]))
    test_images = np.vstack((testFace,testNonFace))
    # # Build a computation graph using Dask. This allows the use of multiple
    # # CPU cores later during the actual computation
    # X = delayed(extract_feature_image(img, feature_types) for img in images)
    # # Compute the result
    # t_start = time()
    # X = np.array(X.compute(scheduler='threads'))
    # time_full_feature_comp = time() - t_start
    
    # feats, feat_cords, feat_scales, feat_types = getFeatures(feature_kernel_sizes, X)
    # y = np.array([1] * 100 + [0] * 100)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=150,
    #                                                 random_state=0,
    #                                                 stratify=y)
    # feature_coord, feature_type = \
    # haar_like_feature_coord(width=images.shape[2], height=images.shape[1],
    #                         feature_type=feature_types)
     
    # clf = RandomForestClassifier(n_estimators=1000, max_depth=None,
    #                          max_features=100, n_jobs=-1, random_state=0)
    # t_start = time()
    # clf.fit(X_train, y_train)
    # time_full_train = time() - t_start
    # auc_full_features = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    
    # # Sort features in order of importance and plot the six most significant
    # idx_sorted = np.argsort(clf.feature_importances_)[::-1]
    
    # fig, axes = plt.subplots(5, 2)
    # for idx, ax in enumerate(axes.ravel()):
    #     image = images[0]
    #     image = draw_haar_like_feature(image, 0, 0,
    #                                    images.shape[2],
    #                                    images.shape[1],
    #                                    [feature_coord[idx_sorted[idx]]])
    #     ax.imshow(image)
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    
    # _ = fig.suptitle('The most important features')
    
    # cdf_feature_importances = np.cumsum(clf.feature_importances_[idx_sorted])
    
    # cdf_feature_importances /= cdf_feature_importances[-1]  # divide by max value
    # sig_feature_count = np.count_nonzero(cdf_feature_importances < 0.7)
    # sig_feature_percent = round(sig_feature_count /
    #                             len(cdf_feature_importances) * 100, 1)
    # print(('{} features, or {}%, account for 70% of branch points in the '
    #        'random forest.').format(sig_feature_count, sig_feature_percent))
    
    # # Select the determined number of most informative features
    # feature_coord_sel = feature_coord[idx_sorted[:sig_feature_count]]
    # feature_type_sel = feature_type[idx_sorted[:sig_feature_count]]
    # # Note: it is also possible to select the features directly from the matrix X,
    # # but we would like to emphasize the usage of `feature_coord` and `feature_type`
    # # to recompute a subset of desired features.
    
    # # Build the computational graph using Dask
    # X = delayed(extract_feature_image(img, feature_type_sel, feature_coord_sel)
    #             for img in images)
    # # Compute the result
    # t_start = time()
    # X = np.array(X.compute(scheduler='threads'))
    # time_subs_feature_comp = time() - t_start
    
    # y = np.array([1] * 100 + [0] * 100)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=150,
    #                                                     random_state=0,
    #                                                     stratify=y)
    
    # t_start = time()
    # clf.fit(X_train, y_train)
    # time_subs_train = time() - t_start
    
    # auc_subs_features = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    
    # summary = (('Computing the full feature set took {:.3f}s, plus {:.3f}s '
    #             'training, for an AUC of {:.2f}. Computing the restricted '
    #             'feature set took {:.3f}s, plus {:.3f}s training, '
    #             'for an AUC of {:.2f}.')
    #            .format(time_full_feature_comp, time_full_train,
    #                    auc_full_features, time_subs_feature_comp,
    #                    time_subs_train, auc_subs_features))
    
    # print(summary)
    # plt.show()
    
    #get haar-like features 
    feature_kernel_sizes = [(2,1), (1,2), (3,1), (1,3), (2,2)]
    # feats, feat_cords, feat_scales, feat_types = getFeatures(feature_kernel_sizes, images)
    feats = []
    for i in tqdm(range(images.shape[0])):
        feats.append(getFeatures(feature_kernel_sizes, images[i]))
    feats = np.array(feats)
    y = np.array([1] * trainFace.shape[0] + [-1] * trainNonFace.shape[0])
    # X_train, X_test, y_train, y_test = train_test_split(feats, y, train_size=150,
                                                    # random_state=0,
                                                    # stratify=y)
    X_train=feats
    threshold = np.zeros(X_train.shape[1])
    counts_all = []
    ths=[]
    for i in tqdm(range(X_train.shape[1])):
        count=[]   
        min_t = X_train[:,i,0].min() 
        # print(min_t)
        max_t = X_train[:,i,0].max()
        thresh_range = np.linspace(min_t, max_t, 50)
        # count = np.zeros((X_train.shape[0],X_train.shape[1]))
        for th,val_th in enumerate(thresh_range):
            ct = np.sum(X_train[:trainFace.shape[0],i,0]<val_th)+np.sum(X_train[trainFace.shape[0]:,i,0]>val_th)
            count.append(ct)
        max_thresh_index = np.argmax(count)
        counts_all.append(count)
        ths.append(thresh_range[max_thresh_index])
    tprs=np.array(counts_all)
    best_performers = np.amax(tprs,axis=1)
    best_10classifiers = np.argsort(best_performers)[-10:][::-1]
    #return threshold and count for best features and index 
    final_thresholds = np.array(ths)[best_10classifiers]
    
    best10 = [best_10classifiers, final_thresholds,best_performers[best_10classifiers]]
    best10= np.array(best10)
    # h=np.zeros((best10.shape[0], X_train.shape[0]))
    #draw best10 - write the function today 
    bfs = feats[1,best_10classifiers,:]
    bfs = np.array(bfs)
    fig, axes = plt.subplots(5, 2)
    for idx, ax in enumerate(axes.ravel()):
        image = trainFace[1]
        test = drawFeat(feature_kernel_sizes,image,  int(bfs[idx,5]), int(bfs[idx,1]),\
                        int(bfs[idx,2]), int(bfs[idx,3]), int(bfs[idx,4]))
        # plt.imshow(test, cmap='gray')
        ax.imshow(test,cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
    
    _ = fig.suptitle('The most important features')
    #------------------------------------------------------------------------#
    # adaboost #
    #------------------------------------------------------------------------#
    
    # h_i_x shape 10x200 - calculate using the best classifiers. 
    # find the classifiers hi(x) +-1 
    
    weights = (1/images.shape[0])*np.ones((1,trainFace.shape[0]+trainNonFace.shape[0])) #10x200
    err = np.zeros((10,1)) #10x200
    alpha_t=np.zeros((10,1))
    #pseudo code 
    for itr in range(10):    
        for i in range(10):
            hi = []
            for j in range(bfs.shape[0]):
                # h_temp=[X_train[:100,int(best10[0,j]), 0]<best10[1,j], X_train[100:,int(best10[0,j]), 0]>best10[1,j]]
                h_temp=[X_train[:,int(best10[0,j]),0]<final_thresholds[j]]
                h = np.array([convToPlusMinus(ll) for ll in h_temp]).reshape(trainFace.shape[0]+trainNonFace.shape[0],)
                hi.append(h)
            hi=np.array(hi)      
            
            indicator = np.array(np.where(hi[i]!=y))
            for ix,j in enumerate(indicator): 
                # err[i]= np.sum(weights[0,j]*hi[i,j])
                err[i]= np.sum(weights[0,j])
            chosen_h_idx,min_err = np.argmin(err), np.amin(err)
            print(err)    
        alpha_t[chosen_h_idx]=0.5*np.log((1-min_err)/(min_err))
        print(alpha_t)
        weights = weights*np.exp(-y*alpha_t[chosen_h_idx]*hi[chosen_h_idx,:])
        print(weights)
        weights = weights/np.sum(weights)
        # weights = weights/np.sum(weights) #normalize weights 
        
    #------------------------------------------------------------------------#
    # testing
    #------------------------------------------------------------------------#
    vals=[]
    for i in range(bfs.shape[0]):
        val=[]
        for test_image in test_images:
            ftval = getFeatureValue(feature_kernel_sizes,test_image,int(bfs[i,5]), \
                                  int(bfs[i,1]), int(bfs[i,2]), int(bfs[i,3]), \
                                      int(bfs[i,4]))
            val.append(ftval)
        vals.append(val)
    vals = np.array(vals)
    # for itr in range(10):    
    errtest=np.zeros((10,1))
    for i in range(10):
        hitest = []
        for j in range(bfs.shape[0]):
            # h_testtemp=[test_images[:50,int(best10[0,j]), 0]<best10[1,j], test_images[50:,int(best10[0,j]), 0]>best10[1,j]]
            # h_testtemp=[test_images[:,int(best10[0,j]), 0]<best10[1,j]]
            h_testtemp=[vals[j,:]<best10[1,j]]
            htest = np.array([convToPlusMinus(ll) for ll in h_testtemp]).reshape(testFace.shape[0]+testNonFace.shape[0],)
            hitest.append(htest)
        hitest=np.array(hitest)                                     
        
    #strong classifier
    res = np.sign(np.sum((alpha_t*hitest),axis=0))
    # res = np.sum((alpha_t*hitest),axis=0)>=0.5*np.sum(alpha_t)
    res1 = convToPlusMinus(res)
    # y = np.array([1] * 50 + [-1] * 50)
    y = np.array([1] * testFace.shape[0] + [-1] * testNonFace.shape[0])
    print(np.sum(y==res1))
#     for i in range(bfs.shape[0]):
#         h_temp=[X_train[:,int(best10[0,i]), 0]<best10[1,i]]
#         h = np.array([convToPlusMinus(ll) for ll in h_temp]).reshape(200,)        
#         hi.append(h)
#     hi=np.array(hi)      
    # err = np.zeros((10,1))
    # for j in range(10):
    #     err[i] = np.multiply(weights[i],)
    #     err=np.multiply(np.array(hi!=y).sum(axis=1),weights[i])
    