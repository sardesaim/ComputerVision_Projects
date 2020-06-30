# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 11:21:46 2020

@author: sarde
"""
import numpy as np
from numpy import matlib
# from PIL import Image
import matplotlib.pyplot as plt 
from sklearn.preprocessing import MinMaxScaler
from simpleGaussian import plotROC
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
    
def create_gauss_pdf(x,mean, covar):
    '''
    Generate Multivariate Gaussian pdf. 
    Log likelihood used to avoid overflow errors.
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
    # den = np.sqrt(np.abs(np.linalg.det(covar)))#*(2*np.pi)**x.shape[0])
    # pdf = np.nan_to_num(np.exp(exponent)/den)
    den = np.abs(np.linalg.det(covar))#*(2*np.pi)**x.shape[0])
    pdf = exponent-0.5*np.log(den)
    return pdf

def InitParameters(K):
    '''
    Initialize Parameters for EM Algorithm. 

    Parameters
    ----------
    K : int
        Number of components in the Mixture

    Returns
    -------
    weights : np array
       Weights of each component
    means : np array
        Means of all components
    covars : np array
        covars of all components

    '''
    # weights=np.random.dirichlet(np.ones(K), size=1)[0].reshape(-1,1)
    weights = np.matlib.repmat((1/K),K,1)
    means=np.zeros((K,100,1))   
    covars= np.array([np.random.uniform(low=0.0, high=1.0, size=(100,100)) \
                      * np.identity(100) for k in range(K)])
    return weights, means, covars

def ExpectationMaximization(x, weights, means, covars,K):
    '''
    EM Algorithm for Mixture Models. 

    Parameters
    ----------
    x : np array
        Data
    weights : np array
       Weights of each component
    means : np array
        Means of all components
    covars : np array
        covars of all components
     K : int
        Number of components in the Mixture

    Returns
    -------
    weights : np array
       Weights of each component
    means : np array
        Means of all components
    covars : np array
        covars of all components

    '''
    #responsbility 
    resp=np.zeros((K,1000))
    #E Step    
    for i in range(1000):
        num=0
        den=0
        for k in range(K):    
            den += np.nan_to_num(weights[k]*create_gauss_pdf\
                                 (x[:,i].reshape(-1,1), means[k], covars[k]))
        for k in range(K):
            num = np.nan_to_num(weights[k]*create_gauss_pdf\
                                (x[:,i].reshape(-1,1), means[k], covars[k]))
            resp[k][i]=num/den
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
            num+=resp[k][i]*x[:,i].reshape(-1,1)
        means[k] = num/sum_rik[k]
    # Update cov
    for k in range(K):
        cov_temp = np.zeros((100,100))
        for i in range(1000):
            cov_temp+= resp[k][i]*np.matmul((x[:,i].reshape(-1,1)-means[k].\
                                             reshape(-1,1)),(x[:,i].reshape\
                                                             (-1,1)-means[k]\
                                                                 .reshape(-1,1)).T)
        cov_temp/=sum_rik[k]
        covars[k] = np.diag(np.diag(cov_temp))            
    return weights, means, covars    

def ParamEstimation(K,face_weights, face_means, face_covars,\
                    non_face_weights, non_face_means, non_face_covars):
    '''
    Parameter estimation using EM Algorithm 

    Parameters
    ----------
    K : int 
        no of components
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
        learned means for face
    face_covars :  np array 
        learned covars for non face
    non_face_weights : np array
        non face weights
    non_face_means : np array
        learned means for non face
    non_face_covars : np array
        learned covs for non face

    '''
    # print('Init Params', face_weights, face_means[0][0],sep=' ')
    for i in range(6):    
        face_weights, face_means, face_covars = \
            ExpectationMaximization(trainFace,face_weights, face_means, \
                                    face_covars, K)
        # print('Updated Params', face_weights, face_means[0][0], sep=' ')
    # print('Init Params', non_face_weights, non_face_means[0][0],sep=' ')
    for i in range(6):    
        non_face_weights, non_face_means, non_face_covars =\
            ExpectationMaximization(trainNonFace,non_face_weights, \
                                    non_face_means, non_face_covars, K)
        # print('Updated Params', non_face_weights, non_face_means[0][0], sep=' ')
    return face_weights, face_means, face_covars, non_face_weights, \
        non_face_means, non_face_covars

def MixtureModel(K,testFace, \
                 testNonFace,face_weights, face_means, \
                     face_covars,non_face_weights, non_face_means,\
                         non_face_covars):
    '''
    Mixture of Gaussian model for the data. 
    Compare likelihoods 


    Parameters
    ----------
    K : int
        number of components.
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
        #calculate posterior by weighted sum of each component
        for k in range(K):
            face_face+=face_weights[k]*create_gauss_pdf(testFace.T[:,i].reshape(-1,1), face_means[k], face_covars[k])
            nonface_face+=non_face_weights[k]*create_gauss_pdf(testFace[:,i].reshape(-1,1), non_face_means[k], non_face_covars[k])
            face_nonface+=face_weights[k]*create_gauss_pdf(testNonFace[:,i].reshape(-1,1),face_means[k], face_covars[k])
            nonface_nonface+=non_face_weights[k]*create_gauss_pdf(testNonFace[:,i].reshape(-1,1), non_face_means[k], non_face_covars[k])
            
        pf_f.append(face_face/(face_face+nonface_face))
        pnf_f.append(nonface_face/(face_face+nonface_face))
        pf_nf.append(face_nonface/(face_nonface+nonface_nonface))
        pnf_nf.append(nonface_nonface/(nonface_nonface+face_nonface)) 
    return pnf_f, pf_f, pf_nf, pnf_nf
        
if __name__ == "__main__":
    np.random.seed(6)
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
    face_weights, face_means, face_covars = InitParameters(K)
    non_face_weights, non_face_means, non_face_covars = InitParameters(K)
    #estimate parameters using EM 
    face_weights, face_means, face_covars, non_face_weights, non_face_means, \
        non_face_covars = ParamEstimation(K,face_weights, face_means, face_covars,non_face_weights, non_face_means, non_face_covars)
    #mixture  model
    pnf_f, pf_f,pf_nf, pnf_nf = MixtureModel(K,testFace, testNonFace,face_weights, face_means,\
                               face_covars, non_face_weights, non_face_means, non_face_covars)
    #find metrics
    tp = (np.asarray(pf_f)>0.5).sum()
    tn = (np.asarray(pnf_nf)>0.5).sum()
    fp = (np.asarray(pf_nf)>0.5).sum()
    fn = (np.asarray(pnf_f)>0.5).sum()
    print('False Positive Rate', fp/100)
    print('False Negative Rate', fn/100)
    print('Misclassification Rate',(fp+fn)/200)
    print('Accuracy', (tp+tn)/200)
    for k in range(K):
        visualizeMeanAndCov(face_means[k],np.diag(face_covars[k]))
        visualizeMeanAndCov(non_face_means[k],np.diag(non_face_covars[k]))
    preds = np.append(pf_nf,pf_f)
    actual=np.append([0]*100, [1]*100)
    plotROC(actual, preds)