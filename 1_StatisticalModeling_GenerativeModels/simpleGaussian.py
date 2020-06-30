# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 22:09:21 2020

@author: sarde
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt 
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import pickle


size=(10,10)

def readDumpImages():
    '''
    Read Images using PIL and Dump in .p object using pickle

    Returns
    -------
    None.

    '''
    root='D:/00_NCSU/Spring2020/ECE763_ComputerVision/Project/project01/'
    trainFace=np.array([np.array(Image.open(root+'pos_train/face'+str(i)+\
                                            '.jpg').convert('LA'))[:,:,0]\
                        for i in range(1000)])
    trainNonFace=np.array([np.array(Image.open(root+'neg_train/non_face'+str(i)+\
                                            '.jpg').convert('LA'))[:,:,0] \
                           for i in range(1000)])
    testFace=np.array([np.array(Image.open(root+'pos_test/face'+str(i)+\
                                            '.jpg').convert('LA'))[:,:,0] \
                       for i in range(1000,1100)])
    testNonFace=np.array([np.array(Image.open(root+'neg_test/non_face'+str(i)+\
                                            '.jpg').convert('LA'))[:,:,0] \
                          for i in range(1000,1100)])
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
    trainFace=pickle.load(open('Data_dumps/trainFace10.p', 'rb'))
    trainNonFace=pickle.load(open('Data_dumps/trainNonFace10.p', 'rb'))
    testFace=pickle.load(open('Data_dumps/testFace10.p', 'rb'))
    testNonFace=pickle.load(open('Data_dumps/testNonFace10.p', 'rb'))
    return trainFace, trainNonFace, testFace, testNonFace

# def trainPCA(trainFace, trainNonFace, testFace, testNonFace):
#     '''
#     Perform PCA for high dimensional Data.

#     Parameters
#     ----------
#     trainFace : np array
#         train set for face.
#     trainNonFace : np array
#         train set for non face.
#     testFace : np array
#         test set for face.
#     testNonFace : np array
#         test set for non face.

#     Returns
#     -------
#     pca_f : PCA Object
#         PCA components for face training data.
#     pca_nf : PCA object
#         PCA components for nonface data. 
#     pca_f_test : PCA Object
#         PCA components for face testing data.
#     pca_nf_test : PCA object
#         PCA components for non face testing data.
#     trainFacePCA : np array
#         train set for face.
#     trainNonFacePCA : np array
#         train set for non face.
#     testFacePCA : np array
#         test set for face.
#     testNonFacePCA : np array
#         test set for non face.

#     '''
#     pca_f = PCA(n_components=100)
#     pca_f.fit(trainFace)
#     trainFacePCA=pca_f.transform(trainFace)
#     pca_nf = PCA(n_components=100)
#     pca_nf.fit(trainNonFace)
#     trainNonFacePCA=pca_nf.transform(trainNonFace)
#     pca_f_test = PCA(n_components=100)
#     pca_f_test.fit(testNonFace)
#     testFacePCA=pca_f_test.transform(testNonFace)
#     pca_nf_test = PCA(n_components=100)
#     pca_nf_test.fit(testNonFace)
#     testNonFacePCA=pca_nf_test.transform(testNonFace)
#     return pca_f, pca_nf, pca_f_test, pca_nf_test,\
#         trainFacePCA, trainNonFacePCA, testFacePCA, testNonFacePCA
       

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

def findMeanAndCov(data):
    '''
    Learn Mean and Covariance from the data

    Parameters
    ----------
    data : np array
        Training data set (face/ non-face).

    Returns
    -------
    mean : np array
        learned mean from the data.
    cov : np array
        learned covariance from the data
    diag_cov : np array
        diagonal covariance

    '''
    #mean and cov 
    mean=np.mean(data, axis=1).reshape(1,-1).T
    cov = np.cov(data)
    # cov+=np.eye(cov.shape[0])*np.exp(-1)
    cov=np.diag(np.diag(cov))
    diag_cov=np.diag(cov).reshape(-1,1)
    return mean, cov, diag_cov

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
    plt.imshow(mean.reshape(size), cmap='gray')
    plt.show()
    plt.figure()
    plt.imshow(diag_cov.reshape(size), cmap='gray')
    plt.show()
    
# def visualizeMeanAndCovPCA(pca_component, means, pca_mean, cov, pca_cov):
    # mean_img=np.dot(pca_component.T,means[:,0])+pca_mean
    # mean_img = np.array(mean_img).astype('uint8')
    # mean_img = np.reshape(mean_img,size)
    # plt.imshow(mean_img)
    # plt.show()
    # cov_img = np.dot(pca_component.T, np.diag(cov))+np.diag(pca_cov)
    # cov_img = np.array(cov_img).astype('uint8')
    # cov_img = np.reshape(cov_img, size)
    # plt.imshow(cov_img)
    # plt.show()
    
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
    #can use log likelihood as well. 
    # pdf = exponent-0.5*np.log(np.abs(np.linalg.det(covar)))
    den = np.sqrt(np.abs(np.linalg.det(covar))*(2*np.pi)**x.shape[0])
    pdf = np.exp(exponent)/den
    return pdf

def plotROC(labels,P_Roc):
    '''
    Plot ROC for the given data. 

    Parameters
    ----------
    labels : array
        true labels.
    P_Roc : array
        predictions.
    pos : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    fpr, tpr, thresholds = roc_curve(labels,P_Roc)
    roc_auc = auc(labels,P_Roc)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    
def GaussianModel(testFace, testNonFace, face_mean, non_face_mean, face_cov, non_face_cov):
    '''
    Simple Gaussian model for the data. 
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
                                        face_mean, face_cov).squeeze()
        nonface_face=create_gauss_pdf(testFace.T[:,i].reshape(-1,1),\
                                        non_face_mean, non_face_cov).squeeze()
        face_nonface=create_gauss_pdf(testNonFace.T[:,i].reshape(-1,1),\
                                        face_mean, face_cov).squeeze()
        nonface_nonface=create_gauss_pdf(testNonFace.T[:,i].reshape(-1,1),\
                                        non_face_mean, non_face_cov).squeeze()
        pf_f.append(face_face/(face_face+nonface_face))
        pnf_f.append(nonface_face/(face_face+nonface_face))
        pf_nf.append(face_nonface/(face_nonface+nonface_nonface))
        pnf_nf.append(nonface_nonface/(nonface_nonface+face_nonface))    
    return pnf_f, pf_f, pf_nf, pnf_nf
            
if __name__ =='__main__':
    #read images
    # readDumpImages()
    #load images from dumps and flatten them to column vectors
    trainFace, trainNonFace, testFace, testNonFace = loadImageArraysFromDumps()
    trainFace=trainFace.reshape(trainFace.shape[0],-1)
    trainNonFace=trainNonFace.reshape(trainNonFace.shape[0], -1)
    testFace=testFace.reshape(testFace.shape[0],-1)
    testNonFace=testNonFace.reshape(testNonFace.shape[0],-1)
    #normalizeData
    trainFace, trainNonFace, testFace, testNonFace =\
        normalizeData(trainFace, trainNonFace, testFace, testNonFace)
    # findMeansAndCovs
    face_mean, face_cov, face_cov_diag = findMeanAndCov(trainFace.T)
    non_face_mean, non_face_cov, non_face_cov_diag=findMeanAndCov(trainNonFace.T)
    # visualize_means and cov
    visualizeMeanAndCov(face_mean, face_cov_diag)
    visualizeMeanAndCov(non_face_mean, non_face_cov_diag)
    #PCA to reduce dimensionality
    # pca_f, pca_nf, _ , _ , train_f_pca, train_nf_pca, test_f_pca, test_nf_pca\
        # = trainPCA(trainFace, trainNonFace, testFace, testNonFace)
    
    #normalizeData
    # train_f_pca, train_nf_pca, test_f_pca, test_nf_pca=\
    #     normalizeData(train_f_pca, train_nf_pca, test_f_pca, test_nf_pca)  
        
    # face_mean_pca, face_cov_pca, face_cov_diag_pca = findMeanAndCov(train_f_pca.T)
    # non_face_mean_pca, non_face_cov_pca, non_face_cov_diag_pca = \
    # findMeanAndCov(train_nf_pca.T)
    
    # visualizeMeanAndCovPCA(pca_f.components_, face_mean_pca, pca_f.mean_,\
    # face_cov_pca, pca_f.get_covariance())
    # visualizeMeanAndCovPCA(pca_nf.components_, non_face_mean_pca, \
    # pca_nf.mean_, non_face_cov_pca, pca_nf.get_covariance())
    # face_cov = np.nan_to_num(np.log(face_cov))
    
     # nf_f, f_f, f_nf,nf_nf=GaussianModel(\
                                          # test_f_pca.T, test_nf_pca.T, \
                                          # face_mean_pca, non_face_mean_pca, \
                                              # face_cov_pca, non_face_cov_pca)
    
    #fit test data on the Gaussian Model
    nf_f,f_f, f_nf, nf_nf = GaussianModel(testFace, testNonFace, face_mean, \
                                           non_face_mean, face_cov, non_face_cov)
    #finding out accuracy and metrics
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