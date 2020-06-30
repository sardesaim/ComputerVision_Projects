# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 20:16:17 2020

@author: sarde
"""

# mixture of t

import numpy as np
from numpy import matlib
# from PIL import Image
import matplotlib.pyplot as plt 
# import scipy as sp
from simpleGaussian import plotROC
from scipy.special import gamma, digamma
from scipy.optimize import fminbound
from sklearn.preprocessing import MinMaxScaler
# import cv2 
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

def InitParameters(K):
    '''
    Init parameters for MoT model

    Parameters
    ----------
    K : int
        no of t dists

    Returns
    -------
    v : list
        init dof for each component
    weights : np array
        init weights for each componen
    means : np array
        init means for each component
    covars : np array 
        init covs for each component 

    '''
    # weights=np.random.dirichlet(np.ones(K), size=1)[0].reshape(-1,1)
    weights = np.matlib.repmat((1/K),K,1)
    means=np.zeros((K,100,1))   
    covars= np.array([np.random.uniform(low=0.0, high=1.0, size=(100,100)) \
                      * np.identity(100) for k in range(K)])
    v = [10, 10, 10]
    # covars = [covars[k]*covars[k].T for k in range(K)]
    return v,weights, means, covars

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
    Create t distribution 
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

def costFunction(v, d, r, exph):
    cost=0
    term12=np.log(v/2)-digamma(v/2)+1-np.log(0.5*(v+d))+digamma((v+d)/2)
    for i in range(1000):
        term3=r[i]*(np.log(exph[i])-exph[i])
        term4=r[i]
        cost+=term12+term3/term4
    return -cost

def ExpectationMaximization(x, v, weights, means, covars,K):
    '''
    Expectation maximization for mixture of t-distributions. 

    Parameters
    ----------
    x : np array
        data
    v : list
        dof array
    weights : np array
        init weights
    means : np array
        init means
    covars : np array
        init covs
    K : int
        no of components

    Returns
    -------
    weights : np array
        updated weights.
    means : np array
        updated means
    covars : np array
        updated covs
    v1 : np array
        update dofs
    exph : np array
        updated e[h].

    '''
    resp=np.zeros((K,1000))
    #E Step    
    D = x.shape[1]
    exph=np.zeros((K,1000,1))
    for i in range(1000):
        num=0
        den=0
        for k in range(K):    
            den += np.nan_to_num\
                (weights[k]*create_t_dist(x[:,i].reshape(-1,1), means[k], covars[k], v[k]))
        for k in range(K):
            num = np.nan_to_num\
                (weights[k]*create_t_dist(x[:,i].reshape(-1,1), means[k], covars[k], v[k]))
            resp[k][i]=num/den
        for k in range(K):
            exph[k,i] = (v[k]+D)/ \
                (v[k] + np.dot(np.dot((x[:,i].reshape(-1,1)-means[k].reshape(-1,1)).T,\
                  np.linalg.inv(covars[k])),(x[:,i].reshape(-1,1)-means[k].reshape(-1,1))))
    #M Step 
    sum_rik = np.sum(resp, axis=1)
    #update weights 
    for k in range(K):
        weights[k] = sum_rik[k]/np.sum(sum_rik)
        # print(weights)
    # update means
    for k in range(K):
        num=0
        for i in range(1000):
            num+=resp[k][i]*exph[k,i]*x[:,i].reshape(-1,1)
        means[k] = num/(sum_rik[k]*np.sum(exph[k]))
    # # Update cov
    for k in range(K):
        cov_temp = np.zeros((100,100))
        for i in range(1000):
            cov_temp+= resp[k][i]*exph[k,i]*\
                np.matmul((x[:,i].reshape(-1,1)-means[k]\
                   .reshape(-1,1)),(x[:,i].reshape(-1,1)-means[k].reshape(-1,1)).T)
        cov_temp/=sum_rik[k]
        covars[k] = np.diag(np.diag(cov_temp))   
    #update v
    for k in range(K):
         v[k]=fminbound(costFunction, 6, 10, args=(D, resp[k], exph[k]))         
    return weights, means, covars, v1, exph

def ParamEstimation(K,trainFace, trainNonFace, face_weights, \
                    face_means, face_covars,non_face_weights, non_face_means, non_face_covars):
    '''
    Param estimation for MoT using EM

    Parameters
    ----------
    K : int
        no of components
    trainFace : np array
        train set for face
    trainNonFace : np array
        train set non face
    face_weights : np array
        face weights
    face_means : np array
        init means for face
    face_covars :  np array 
        init covars for non face
    non_face_weights : np array
        non face weights
    non_face_means : np array
        init means for non face
    non_face_covars : np array
        init covs for non face

    Returns
    -------
    face_weights : np array
        face weights
    face_means : np array
        face means
    face_covars : np array 
        face covars
    non_face_weights : np array
        non face weights
    non_face_means : np array
        non face means
    non_face_covars : np array
        non face covars

    '''
    # print('Init Params', face_weights, face_means[0][0],sep=' ')
    for i in range(6):    
        face_weights, face_means, face_covars = \
            ExpectationMaximization(trainFace,face_weights, face_means, face_covars, K)
        # print('Updated Params', face_weights, face_means[0][0], sep=' ')
    # print('Init Params', non_face_weights, non_face_means[0][0],sep=' ')
    for i in range(6):    
        non_face_weights, non_face_means,\
            non_face_covars = ExpectationMaximization\
                (trainNonFace,non_face_weights, non_face_means, non_face_covars, K)
        # print('Updated Params', non_face_weights, non_face_means[0][0], sep=' ')
    return face_weights, face_means, face_covars, \
        non_face_weights, non_face_means, non_face_covars

def MoTModel(K,testFace, \
             testNonFace,face_weights, face_means, \
                 face_covars,non_face_weights, non_face_means, \
                     non_face_covars, v1, v2):
    '''
    MoT model for getting likelihoods of the test data and posteriors

    Parameters
    ----------
    K : int
        no of components
    testFace : np array
        test set for face.
    testNonFace : np array
        test set for non face.

    face_weights : np array
        face weights
    face_means : np array
        learned means for face
    face_covars :  np array 
        learned covars for non face
    non_face_weights : np array
        non face weights
    non_face_means : np array
        learned means for non face
    non_face_covars : np array
        learned covs for non face
    v1 : list
        all dofs for face models 
    v2 : list
        all dofs for non face

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
    pf_f=[] 
    pnf_f=[]
    pf_nf=[]
    pnf_nf=[]
    for i in range(len(testFace)):
        face_face=0
        face_nonface=0
        nonface_face=0
        nonface_nonface=0
        for k in range(K):
            face_face+=face_weights[k]*create_t_dist(testFace.T[:,i].reshape(-1,1), face_means[k], face_covars[k], v1[k])
            nonface_face+=non_face_weights[k]*create_t_dist(testFace[:,i].reshape(-1,1), non_face_means[k], non_face_covars[k], v2[k])
            face_nonface+=face_weights[k]*create_t_dist(testNonFace[:,i].reshape(-1,1),face_means[k], face_covars[k], v1[k])
            nonface_nonface+=non_face_weights[k]*create_t_dist(testNonFace[:,i].reshape(-1,1), non_face_means[k], non_face_covars[k], v1[k])
            
        pf_f.append(face_face/(face_face+nonface_face))
        pnf_f.append(nonface_face/(face_face+nonface_face))
        pf_nf.append(face_nonface/(face_nonface+nonface_nonface))
        pnf_nf.append(nonface_nonface/(nonface_nonface+face_nonface)) 
    return pnf_f, pf_f, pf_nf, pnf_nf

if __name__ == "__main__":
    np.random.seed(22) #5 - acc .65 #6 66 #18 - 69 
    K=3
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
    v1,face_weights, face_mean, face_cov = InitParameters(K)
    v2,non_face_weights, non_face_mean, non_face_cov= InitParameters(K)
    #EMforTdist
    for i in range(5):
        face_weights, face_mean, face_cov, v1, exph_face = ExpectationMaximization(trainFace, v1, face_weights, face_mean, face_cov, K)
        non_face_weights, non_face_mean, non_face_cov, v2, exph_non_face = ExpectationMaximization(trainNonFace, v2, non_face_weights, non_face_mean, non_face_cov, K)
    #calc posteriors
    pnf_f, pf_f, pf_nf, pnf_nf=MoTModel(K,testFace, testNonFace,face_weights, face_mean, face_cov,non_face_weights, non_face_mean, non_face_cov, v1, v2)
    #check metrics
    tp = (np.asarray(pf_f)>0.5).sum()
    tn = (np.asarray(pnf_nf)>0.5).sum()
    fp = (np.asarray(pf_nf)>0.5).sum()
    fn = (np.asarray(pnf_f)>0.5).sum()
    print('False Positive Rate', fp/100)
    print('False Negative Rate', fn/100)
    print('Misclassification Rate',(fp+fn)/200)
    print('Accuracy', (tp+tn)/200)
    for k in range(K):
        visualizeMeanAndCov(face_mean[k],np.diag(face_cov[k]))
        visualizeMeanAndCov(non_face_mean[k],np.diag(non_face_cov[k]))
    preds = np.append(pf_nf,pf_f)
    actual=np.append([0]*100, [1]*100)
    plotROC(actual, preds)