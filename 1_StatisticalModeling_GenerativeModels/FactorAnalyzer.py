# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 22:22:37 2020

@author: sarde
"""

import numpy as np
import matplotlib.pyplot as plt 
# import scipy as sp
from simpleGaussian import plotROC
from sklearn.preprocessing import MinMaxScaler
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

def create_gauss_pdf(x,mean, covar):
    '''
    Generate Multivariate Gaussian pdf. 

    Parameters
    ----------
    x : np array 
        data.
    mean : np array
        learned mean from the data.
    cov : np array
        learned covariance from the data

    Returns
    -------
    pdf : np array (1x1)
        likelihood for the given data.

    '''
    exponent=-0.5*np.matmul(np.matmul((x-mean).T,np.linalg.inv(covar)), (x-mean))
    # pdf = exponent-0.5*np.log(np.abs(np.linalg.det(covar)))#-np.log((2*np.pi))*x.shape[0]*0.5
    den = np.sqrt(np.abs((np.linalg.det(covar)))*(2*np.pi)**x.shape[0])
    pdf = np.exp(exponent)/den
    return pdf

def InitParameters(x,k,d):
    '''
    Parmater initialization for Factor Analyzer model 

    Parameters
    ----------
    x : np array
        data
    k : int
        number of components
    d : int
        dimensions

    Returns
    -------
    E_h : np array
        e[h]
    E_hht : np array
        e[h]e[h].T
    phi : np array
        phi subspace.
    mean : np array
        mean.
    cov : np array
        covariance.

    '''
    # E_h = np.zeros((1000,k,d))
    E_h = np.zeros((1000,k,1))
    E_hht = np.zeros((1000,k,k))
    phi = np.random.rand(d,k)
    cov = np.random.uniform(0,1.0,(d,d))
    cov=np.diag(np.diag(cov))
    mean = np.mean(x, axis = 0).reshape(-1,1)
    return E_h, E_hht, phi, mean, cov

def ExpectationMaximization(k,x,E_h, E_hht, phi, mean, cov):
    '''
    Expecation maximization for the Factor Analyzer

    Parameters
    ----------
    k : int
        no of components
    x : np array
        data
     E_h : np array
        e[h]
    E_hht : np array
        e[h]e[h].T
    phi : np array
        phi subspace.
    mean : np array
        mean.
    cov : np array
        cov.

    Returns
    -------
    mean : np array
        learned mean.
    cov : np array
        learned cov.
    phi : np array
        learned phi.

    '''
    # Expectation step
    for i in range(1000):
        eht1 = np.linalg.inv((np.matmul\
                              (np.matmul(phi.T, np.linalg.inv(cov)),phi))+np.eye(k))
        eht2 = np.matmul(phi.T, np.linalg.inv(cov))
        eht3 = x[:,i].reshape(-1,1) - mean.reshape(-1,1)
        E_h[i] = np.matmul(np.matmul(eht1, eht2), eht3)
        E_hht[i] = eht1 + np.matmul(E_h[i], E_h[i].T)
    
    #Maximization Step
    #mean
    mean = mean 
    #phi
    temp=0
    for i in range(1000):
        temp+=np.matmul((x[:,i].reshape(-1,1)-mean.reshape(-1,1)),E_h[i].T)
    temp1 = np.sum(E_hht, axis=0)
    phi = np.matmul(temp, np.linalg.inv(temp1))
    #cov 
    t1=0
    for i in range(1000):
        t1+=np.matmul((x[:,i].reshape(-1,1)-\
                       mean.reshape(-1,1)),(x[:,i].reshape(-1,1)\
                                            -mean.reshape(-1,1)).T)\
            -np.matmul(phi, np.matmul(E_h[i],\
                                      (x[:,i].reshape(-1,1)-\
                                       mean.reshape(-1,1)).T))
    t1/=1000 
    cov = np.diag(np.diag(t1))
    return mean, cov, phi

def FactorAnalyzer(testFace, testNonFace, face_mean, \
                  non_face_mean, face_cov, non_face_cov):
    '''
    Factor Analyzer for the data. 
    Compare likelihoods 

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
        face_face=create_gauss_pdf(testFace.T[:,i].reshape(-1,1),\
                                        face_mean, face_cov)
        face_nonface=create_gauss_pdf(testFace.T[:,i].reshape(-1,1),\
                                        non_face_mean, non_face_cov)
        nonface_face=create_gauss_pdf(testNonFace.T[:,i].reshape(-1,1),\
                                        face_mean, face_cov)
        nonface_nonface=create_gauss_pdf(testNonFace.T[:,i].reshape(-1,1),\
                                        non_face_mean, non_face_cov)
        pf_f.append(face_face/(face_face+nonface_face))
        pnf_f.append(nonface_face/(face_face+nonface_face))
        pf_nf.append(face_nonface/(face_nonface+nonface_nonface))
        pnf_nf.append(nonface_nonface/(nonface_nonface+face_nonface))
    
    return pnf_f, pf_f, pf_nf, pnf_nf

def visualizeMeanAndCov(mean,diag_cov, phi):
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
    cov = np.diag(diag_cov)- np.matmul(phi, phi.T)
    diag_cov = np.diag(cov)
    plt.figure()
    plt.imshow(mean.reshape(10,10), cmap='gray')
    plt.show()
    plt.figure()
    plt.imshow(diag_cov.reshape(10,10), cmap='gray')
    plt.show()

if __name__ == "__main__":
    np.random.seed(5) #1-3  
    K=40
    D=100
    #load images
    trainFace, trainNonFace, testFace, testNonFace=loadImageArraysFromDumps()
    #reshape to col vectors
    trainFace=trainFace.reshape(trainFace.shape[0],-1).T
    trainNonFace=trainNonFace.reshape(trainNonFace.shape[0], -1).T
    testFace=testFace.reshape(testFace.shape[0],-1).T
    testNonFace=testNonFace.reshape(testNonFace.shape[0],-1).T
    #normalize data
    trainFace, trainNonFace, testFace, testNonFace=normalizeData(trainFace, trainNonFace, testFace, testNonFace)
    #init
    f_E_h, f_E_hht, face_phi, face_mean, face_cov=InitParameters(trainFace.T,K,D)
    nf_E_h, nf_E_hht, non_face_phi, non_face_mean, non_face_cov=InitParameters(testFace.T,K,D)
    #EM
    for i in range(11): #10 iterations - convergence
        face_mean, face_cov, face_phi=ExpectationMaximization(K, trainFace, f_E_h, f_E_hht, face_phi, face_mean, face_cov)
        non_face_mean, non_face_cov, face_phi=ExpectationMaximization(K, trainNonFace, nf_E_h, nf_E_hht, non_face_phi, non_face_mean, non_face_cov)
    face_cov+= np.matmul(face_phi, face_phi.T)
    non_face_cov+= np.matmul(non_face_phi, non_face_phi.T)
    # face_cov=np.diag(np.diag(face_cov))
    # non_face_cov=np.diag(np.diag(non_face_cov))
    visualizeMeanAndCov(face_mean, np.diag(face_cov), face_phi)
    visualizeMeanAndCov(non_face_mean, np.diag(non_face_cov), non_face_phi)
    nf_f,f_f, f_nf, nf_nf = FactorAnalyzer(testFace, testNonFace, face_mean, non_face_mean, face_cov, non_face_cov)
    tp = (np.asarray(f_f)>0.5).sum()
    tn = (np.asarray(nf_nf)>0.5).sum()
    fp = (np.asarray(f_nf)>0.5).sum()
    fn = (np.asarray(nf_f)>0.5).sum()
    print('False Positive Rate', fp/100)
    print('False Negative Rate', fn/100)
    print('Misclassification Rate',(fp+fn)/200)
    print('Accuracy', (tp+tn)/200)
    preds = np.append(f_nf,f_f)
    actual=np.append([0]*100, [1]*100)
    plotROC(actual, preds)