# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 09:31:36 2020

@author: sarde

"""
from time import time
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
from skimage.transform import integral_image

root = 'D:/00_NCSU/Spring2020/ECE763_ComputerVision/Project/project02/'

def dump_data():
    '''
    Read and dump data into .p files 
    Returns
    -------
    None.
    '''
    size=(16,16)
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

@delayed
def extract_feature_image(img, feature_type, feature_coord=None):
    """Extract the haar feature for the current image"""
    ii = integral_image(img)
    return haar_like_feature(ii, 0, 0, ii.shape[0], ii.shape[1],\
                             feature_type=feature_type,\
                             feature_coord=feature_coord)
def convToPlusMinus(h):
    '''
    Convert to \pm 1 format

    Parameters
    ----------
    h : Boolean array
        Boolean array for preds.
    Returns
    -------
    list of ints
        \pm 1 format
    '''
    return  [int(x)-1 if x == False else int(x) for x in h]

def confusion_matrix(true, predictions):
    '''
    Helper function to plot ROC

    Parameters
    ----------
    true : np array 
        true labels.
    predictions : np array
        predictions.
    Returns
    -------
    TP : list
        True positives
    FP : list
        False positives
    TN : list
        True negatives
    FN : list
        False negatives

    '''
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for t, p in zip(true, predictions):
        if t == 1 and p == 1: 
            TP += 1
        elif t == -1 and p == 1:
            FP += 1
        elif t == 1 and p == -1:  
            FN += 1
        else: 
            TN += 1
    return TP, FP, TN, FN

def roc_curve(alpha_t, hitest, testFace, testNonFace, test_images):
    '''
    Plot ROC for the strong classifier by changing thresholds. 

    Parameters
    ----------
    alpha_t : list
        weights for the weak classifiers. 
    hitest : np array
        weak classifier outputs.
    testFace : np array
        used to get shape of the testFace array.
    testNonFace : np array
        used to get shape of testNonFace array.
    test_images : np array
        used to get shape of test_images array.
    Returns
    -------
    None.
    '''
    x = []
    y = []
    min_t = np.sum(alpha_t*hitest,axis=0).min()
    max_t = np.sum(alpha_t*hitest,axis=0).max()
    trange = np.linspace(min_t, max_t, 1000)
    true = np.array([1] * testFace.shape[0] + [-1] * testNonFace.shape[0])
    for thresh in trange: 
        res = np.sign(np.sum((alpha_t*hitest),axis=0)-thresh)
        res1 = convToPlusMinus(res)
        TP, FP, TN, FN = confusion_matrix(true, res1)
        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)
        x.append(FPR)
        y.append(TPR)
    plt.title('Receiver Operating Characteristic')
    plt.plot(x, y, 'b')
    plt.plot([0, 1], [0, 1],'r--')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    
def plotHaarFeatures(idx_sorted, images,feature_coord):
    '''
    Plot top 10 Haar features for given indices. 

    Parameters
    idx_sorted : list
        sorted array of indices with max importance.
    images : np array
        image array used to show any image.
    feature_coord : np array
        haar feature coordinates.
    '''
    fig, axes = plt.subplots(5, 2)
    for idx, ax in enumerate(axes.ravel()):
        image = images[0]
        image = draw_haar_like_feature(image, 0, 0,\
                                        images.shape[2],\
                                        images.shape[1],\
                                        [feature_coord[idx_sorted[idx]]])
        ax.imshow(image)
        ax.set_xticks([])
        ax.set_yticks([])
    
    _ = fig.suptitle('The most important features')
    
    
def best10BeforeBoosting(X, trainFace, feature_coord, images):
    '''
    Find and plot best 10 features before boosting. 

    Parameters
    ----------
    X : np array 
        Haar Features.
    trainFace : np array
        face images array used to get shape.
    feature_coord : np array
        list of haar coordinates.
    images : np array
        images.
    Returns
    -------
    all_thresholds : np array 
        thresholds for weak classifiers. 
    '''
    counts_all = []
    ths=[]
    for i in tqdm(range(X.shape[1])):
        count=[]   
        min_t = X[:,i].min() 
        max_t = X[:,i].max()
        thresh_range = np.linspace(min_t, max_t, 50)
        #traverse through range of thresholds to select optimum 
        for th,val_th in enumerate(thresh_range): 
            ct = np.sum(X[:trainFace.shape[0],i]<val_th)+np.sum(X[trainFace.shape[0]:,i]>val_th)
            count.append(ct)
        max_thresh_index = np.argmax(count)
        counts_all.append(count)
        ths.append(thresh_range[max_thresh_index])
    tps=np.array(counts_all)
    best_performers = np.amax(tps,axis=1)
    best_10classifiers = np.argsort(best_performers)[-10:][::-1]
    #return threshold and count for best features and index 
    all_thresholds = np.array(ths)
    plotHaarFeatures(best_10classifiers,images,feature_coord)
    return all_thresholds

def adaboost(no_weak_clfs, images, trainFace, trainNonFace, X, y,all_thresholds):
    '''
    Adaboost Algorithm. Used to get top features for specified no_weak_classifiers. 

    Parameters
    ----------
    no_weak_clfs : int
        number of weak classifiers to consider.
    images : np array
        training data - used to get shape
    trainFace : np array
        face training data - used to get shape
    trainNonFace : np array
        non face training data - used to get shape
    X : np array
        haar features
    all_thresholds : np array
        thresholds for weak classifiers.
    Returns
    -------
    alpha_t : list
        weights for weak classifiers
    classifier_indices : list
        indices of chosen classifiers
    classifier_errors : list
        errors for chosen classifiers.
    '''
    # weights initialized as 1/n 
    # weights = (1/images.shape[0])*np.ones((1,trainFace.shape[0]+trainNonFace.shape[0])) 
    weights = np.hstack((((1/trainFace.shape[0])*np.ones((1,trainFace.shape[0]))),\
                                             ((1/trainNonFace.shape[0])*np.ones((1,trainNonFace.shape[0])))))
    err = np.zeros((X.shape[1],1)) #error initialization as zero 
    alpha_t=np.zeros((no_weak_clfs,1))  #alphas initialized as a np array
    classifier_indices = []
    classifier_errors=[]
    #no of passes equal to no of weak clfs
    for itr in tqdm(range(no_weak_clfs)):    
        #normalize weights 
        weights = weights/np.sum(weights)
        hi = []
        for i in range(X.shape[1]):
            #weak classifier acts as one node decision stump
            #predictions - from weak classifiers. - paritylogic implemented here
            h_temp=X[:,i]<all_thresholds[i]
            # h = np.array([convToPlusMinus(ll) for ll in h_temp]).reshape(trainFace.shape[0]+trainNonFace.shape[0],)
            h = convToPlusMinus(h_temp)
            hi.append(h)    #form an array of all the predictions for all examples
            #find locations of errors     
            indicator = np.array(np.where(h!=y))    #indicator func - pred!=true
            #find error at ith feature
            err[i] = np.sum(weights[0,indicator[0]])
        hi=np.array(hi) 
        #chose weak classifier with minimum error 
        chosen_h_idx,min_err = np.argmin(err), np.amin(err)
        classifier_indices.append(chosen_h_idx)
        classifier_errors.append(min_err)
        #find alpha_t
        alpha_t[itr]=0.5*np.log((1-min_err)/(min_err))
        #update weights
        weights = weights*np.exp(-y*alpha_t[itr]*hi[chosen_h_idx,:])
        #find preds for training accuracy - by combination of weak clfs till itr
        res = np.sign(np.sum((alpha_t[0:itr+1]*hi[classifier_indices]),axis=0))
        res1 = convToPlusMinus(res)
        print(f'Training Accuracy with {itr+1} weak classifier(s)', np.sum(y==res1)/images.shape[0])
    return alpha_t,classifier_indices,classifier_errors

def testing(classifier_indices, X_test, testFace, testNonFace, test_images, alpha_t, all_thresholds):
    '''
    Testing on test images. 

    Parameters
    ----------
    classifier_indices : list
        indices of chosen classifiers
    X_test : np array
        features for test images. 
    images : np array
        training data - used to get shape
    trainFace : np array
        face training data - used to get shape
    trainNonFace : np array
        non face training data - used to get shape
    alpha_t : list
        weights for weak classifiers.

    Returns
    -------
    None.

    '''
    #predictions using the best weak classifiers found from adaboost 
    hitest = []
    for j in classifier_indices:
        h_testtemp=X_test[:,j]<all_thresholds[j]
        # htest = np.array([convToPlusMinus(ll) for ll in h_testtemp]).reshape(testFace.shape[0]+testNonFace.shape[0],)
        htest= convToPlusMinus(h_testtemp)    
        hitest.append(htest)
    hitest=np.array(hitest)                                     
        
    #strong classifier 
    res = np.sign(np.sum((alpha_t*hitest),axis=0))
    res1 = convToPlusMinus(res)
    y = np.array([1] * testFace.shape[0] + [-1] * testNonFace.shape[0])
    
    # tp = np.sum(y[:testFace.shape[0]]==res1[:testFace.shape[0]])
    # tn = np.sum(y[testFace.shape[0]:]==res1[testFace.shape[0]:])
    print('Accuracy', np.sum(y==res1)/test_images.shape[0])
    return hitest
    
if __name__ == '__main__':
    # dump_data()
    trainFace, trainNonFace, testFace, testNonFace = loadImageArraysFromDumps() 
    trainFace, trainNonFace, testFace, testNonFace = normalize_dataset(trainFace, trainNonFace, testFace, testNonFace)
    feature_types = ['type-2-x', 'type-2-y', 'type-3-x', 'type-3-y', 'type-4']
    trainFace = trainFace[:200]
    trainNonFace = trainNonFace[:200]
    images = np.vstack((trainFace,trainNonFace))
    testFace = testFace[100:150]
    testNonFace = testNonFace[100:150]
    test_images = np.vstack((testFace,testNonFace))
    # # Build a computation graph using Dask. This allows the use of multiple
    # # CPU cores later during the actual computation
    X = delayed(extract_feature_image(img, feature_types) for img in images)
    # Compute the result
    t_start = time()
    X = np.array(X.compute(scheduler='threads'))
    time_full_feature_comp = time() - t_start
    
    no_weak_clfs = 10
    # feats, feat_cords, feat_scales, feat_types = getFeatures(feature_kernel_sizes, X)
    y = np.array([1] * trainFace.shape[0] + [-1] * trainNonFace.shape[0])

    feature_coord, feature_type = \
    haar_like_feature_coord(width=images.shape[2], height=images.shape[1],
                            feature_type=feature_types)

    all_thresholds=best10BeforeBoosting(X, trainFace, feature_coord, images)
    #------------------------------------------------------------------------#
    # adaboost 
    #------------------------------------------------------------------------#
    alpha_t,classifier_indices,classifier_errors \
        = adaboost(no_weak_clfs, images, trainFace, trainNonFace, X, y,all_thresholds)
    #------------------------------------------------------------------------#
    # testing
    #------------------------------------------------------------------------#
    X_test = delayed(extract_feature_image(img, feature_types) for img in test_images)
    # Compute the result
    t_start = time()
    X_test = np.array(X_test.compute(scheduler='threads'))
    time_full_feature_comp = time() - t_start
    #preds for test images
    hitest=testing(classifier_indices, X_test, testFace, testNonFace, test_images, alpha_t,all_thresholds)
    plt.figure()
    roc_curve(alpha_t, hitest, testFace, testNonFace, test_images)
    plotHaarFeatures(classifier_indices,images,feature_coord)