# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 14:00:32 2020

@author: sarde
"""
#T-Distribution

import numpy as np
import matplotlib.pyplot as plt 
# import scipy as sp
from simpleGaussian import plotROC
from sklearn.preprocessing import MinMaxScaler
from scipy.special import gamma, gammaln, digamma
from scipy.optimize import fminbound
import pickle

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
    trainFace=pickle.load(open('Data_dumps/trainFace10.p', 'rb'))
    trainNonFace=pickle.load(open('Data_dumps/trainNonFace10.p', 'rb'))
    testFace=pickle.load(open('Data_dumps/testFace10.p', 'rb'))
    testNonFace=pickle.load(open('Data_dumps/testNonFace10.p', 'rb'))
    return trainFace, trainNonFace, testFace, testNonFace
       
def normalizeData(trainFace, trainNonFace, testFace, testNonFace):
    '''
    Normalize data from 0 to 1 using MinMaxScaler

    Parameters
    ----------
    trainFace : np array
        train set for face.
    trainNonFace : np array
        train set for non face.
    testFace : np array
        test set for face.
    testNonFace : np array
        test set for non face.

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
    scale = MinMaxScaler()
    scale.fit(trainFace)
    trainFace=scale.transform(trainFace)
    scale.fit(trainNonFace)
    trainNonFace=scale.transform(trainNonFace)
    scale.fit(testFace)
    testFace=scale.transform(testFace)
    scale.fit(testNonFace)
    testNonFace=scale.transform(testNonFace)
    return trainFace, trainNonFace, testFace, testNonFace

def visualizeMeanAndCov(mean,diag_cov):
    '''
    Visualize the mean and covariance from the data. 

    Parameters
    ----------
    mean : np array
        learned mean from the data.

    diag_cov : np array
        diagonal covariance

    Returns
    -------
    None.

    '''
    plt.figure()
    plt.imshow(mean.reshape(10,10), cmap='gray')
    plt.show()
    plt.figure()
    plt.imshow(diag_cov.reshape(10,10), cmap='gray')
    plt.show()
    
def create_t_dist(x,mean, covar, v):
    '''
    Create t-distribution function

    Parameters
    ----------
    x : np array
        data.
    mean : np arary
        mean.
    covar : np array
        covariance.
    v : float
        degree of freedom

    Returns
    -------
    pdf : np array (1x1)
        likelihood for the given data.

    '''
    D = x.shape[0]
    fac1 = gamma((v+D)/2)/(((v*np.pi)**(D/2))*np.sqrt(np.linalg.det(covar))*gamma(v/2))
    fac2 = 1 + np.matmul(np.matmul((x-mean.reshape(-1,1)).T, np.linalg.inv\
                                   (covar)),(x-mean.reshape(-1,1)))/2
    pdf = fac1*(fac2**(-(v+D)/2))
    return pdf

def InitParameters():
    '''
    Initialize Parameters for the t dist

    Returns
    -------
   mean : np arary
        mean.
    covar : np array
        covariance.
    v : float
        degree of freedom

    '''
    v = 10
    means=np.zeros((100,1))
    covars= np.random.uniform(low=0.0, high=1.0, size=(100,100)) * np.identity(100)
    # covars=np.matmul(covars,covars.T)
    return v,means, np.diag(np.diag(covars))

def costFunctionV(v,x, mean, cov, exph, explh):
    d = mean.shape[0]
    cons=0.5*v*np.log(0.5*v)-(gammaln(0.5*v))
    cost=0
    for i in range(1000):   
        cost+=cons+((0.5*v-1)*explh[i]-(0.5*v*exph[i]))+((d*explh[i]-d*np.log(np.pi*2)\
                      -np.log(np.abs(np.linalg.det(cov)))-\
                      np.dot((x[:,i].reshape(-1,1)-mean.reshape(-1,1)).T\
                             , np.dot(np.linalg.inv(cov), \
                                      (x[:,i].reshape(-1,1)-mean.reshape(-1,1))\
                                          ))*exph[i])/2)
        return -cost
    
def EMForT(x,mean, cov, v):
    '''
    Expectation Maxmimization for T Dist

    Parameters
    ----------
    x : np array
        data.
    mean : np arary
        mean.
    covar : np array
        covariance.
    v : float
        degree of freedom

    Returns
    -------
    mean : np arary
        mean.
    cov : np array
        covariance.
    v : float
        degree of freedom
    '''
    D = mean.shape[0]
    exph=np.zeros((1000,1))
    explnh=np.zeros((1000,1))
    #expectation step - E Step 
    for i in range(1000):
        exph[i] = (v+D)/ (v +np.dot(np.dot((x[:,i].reshape(-1,1)-\
                                            mean.reshape(-1,1)).T,np.linalg.inv\
                                           (cov)),(x[:,i].reshape(-1,1)-\
                                                   mean.reshape(-1,1))))
        explnh[i]= digamma((v +D)*0.5)-np.log((v + np.dot\
                                               (np.dot((x[:,i].reshape(-1,1)-\
                                                        mean.reshape(-1,1)).T, \
                                                                np.linalg.inv(cov))\
                                                ,(x[:,i].reshape(-1,1)-\
                                                  mean.reshape(-1,1))))/2) 
    #maximization step - M Step
    #find means
    # mean = np.sum(np.multiply(x.T,exph), axis=0) / np.sum(exph)
    num=0
    den=0
    for i in range(1000):
        num+=exph[i]*x[:,i]
        den+=exph[i]
    mean=num/den
    #find covs
    num=0
    for i in range(1000):
        num+= exph[i]*(np.dot((x[:,i].reshape(-1,1)-mean.reshape(-1,1)),\
                              (x[:,i].reshape(-1,1)-mean.reshape(-1,1)).T))
    cov=num/den
    cov = np.diag(np.diag(cov))
    #find v - dof 
    v=fminbound(costFunctionV, 6, 10, args=(x, mean, cov, exph, explnh)) 
    return mean, cov, v

def tDistModel(testFace, testNonFace, face_mean, non_face_mean, face_cov, \
               non_face_cov, v1, v2):
    '''
    T dist model

    Parameters
    ----------
     testFace : np array
        test set for face.
    testNonFace : np array
        test set for non face.
    face_mean : np array
        learned mean for face
    non_face_mean : np array 
        learned mean for non face
    face_cov : np array
        learned cov for face
    non_face_cov : np array
        learned cov for non face
    v1 : float
        dof for face
    v2 : float
        dof for non face

    Returns
    -------
    pnf_f : list
        posterior for test data.
    pf_f : list
        posterior for test data
    pf_nf : list
        posterior for test data
    pnf_nf : list
        posterior for test data

    '''
    face_face=0
    face_nonface=0
    nonface_face=0
    nonface_nonface=0
    pf_f=[]
    pnf_f=[]
    pf_nf=[]
    pnf_nf=[]
    for i in range(testFace.shape[1]):
        face_face=create_t_dist(testFace.T[:,i].reshape(-1,1),\
                                        face_mean, face_cov, v1)
        nonface_face=create_t_dist(testFace.T[:,i].reshape(-1,1),\
                                        non_face_mean, non_face_cov, v2)
        face_nonface=create_t_dist(testNonFace.T[:,i].reshape(-1,1),\
                                        face_mean, face_cov, v1)
        nonface_nonface=create_t_dist(testNonFace.T[:,i].reshape(-1,1),\
                                        non_face_mean, non_face_cov, v2)
        pf_f.append(face_face/(face_face+face_nonface))
        pf_nf.append(face_nonface/(face_nonface+face_face))
        pnf_f.append(nonface_face/(nonface_nonface+nonface_face))
        pnf_nf.append(nonface_nonface/(nonface_nonface+nonface_face))    
    return pnf_f, pf_f, pf_nf, pnf_nf

if __name__ == "__main__":
    np.random.seed(10)
    #load images
    trainFace, trainNonFace, testFace, testNonFace=loadImageArraysFromDumps()
    #reshape to col vectors
    trainFace=trainFace.reshape(trainFace.shape[0],-1).T
    trainNonFace=trainNonFace.reshape(trainNonFace.shape[0], -1).T
    testFace=testFace.reshape(testFace.shape[0],-1).T
    testNonFace=testNonFace.reshape(testNonFace.shape[0],-1).T
    #normalize data
    trainFace, trainNonFace, testFace, testNonFace=normalizeData(trainFace, trainNonFace, testFace, testNonFace)
    #initparams
    v1,face_mean, face_cov = InitParameters()
    v2,non_face_mean, non_face_cov = InitParameters()
    # print(v1, face_mean, face_cov, v2, non_face_mean, non_face_cov, sep='\n')
    # #EMforT
    for i in range(5):
        face_mean, face_cov, v1 = EMForT(trainFace, face_mean, face_cov, v1)
        non_face_mean, non_face_cov, v2 = EMForT(trainNonFace, non_face_mean, non_face_cov, v2)
    # print(v1, face_mean, face_cov, v2, non_face_mean, non_face_cov, sep='\n')
    # visualizeMeanAndCov(face_means,np.diag(face_covars))
        # print(v1,v2, sep =' ')
    pnf_f, pf_f, pf_nf, pnf_nf=tDistModel(testFace, testNonFace, face_mean, non_face_mean, face_cov, non_face_cov, v1, v2)
    
    tp = (np.asarray(pf_f)>0.5).sum()
    tn = (np.asarray(pnf_nf)>0.5).sum()
    fp = (np.asarray(pf_nf)>0.5).sum()
    fn = (np.asarray(pnf_f)>0.5).sum()
    print('False Positive Rate', fp/100)
    print('False Negative Rate', fn/100)
    print('Misclassification Rate',(fp+fn)/200)
    print('Accuracy', (tp+tn)/200)
    visualizeMeanAndCov(face_mean,np.diag(face_cov))
    visualizeMeanAndCov(non_face_mean,np.diag(non_face_cov))
    # pnf_f, pf_f = tDistModel()
    preds = np.append(pf_nf,pf_f)
    actual=np.append([0]*100, [1]*100)
    plotROC(actual, preds)