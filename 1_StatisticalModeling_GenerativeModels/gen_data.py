# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 22:31:18 2020

@author: sarde
"""
import numpy as np 
import os
import cv2
import matplotlib.pyplot as plt

rng = np.random.default_rng()
size=(20,20)
def getPaths():
    # fh = open('FDDB-fold-01-ellipseList.txt')
    fh = open('wider_face_train_bbx_gt.txt')
    #read line
    count=0
    paths=[]
    boxes=[]
    lines = fh.readlines()
    for line in lines:
        if line=='1\n':
            filename = lines[count-1].strip('\n')
            bounding_box = lines[count+1]
            num = bounding_box.split(' ')[:4]
            num = [int(float(i)) for i in num]
            paths.append(filename)
            boxes.append(num)
        count+=1
    fh.close()
    return paths, boxes

def IoU(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = max(boxA[2], boxB[2]), max(boxA[3], boxB[3])
    areaOfIntersection=(xB-xA+1)*(yB-yA+1)
    boxAArea=(boxA[2]-boxA[0]+1)*(boxA[3]-boxA[1]+1)
    boxBArea=(boxB[2]-boxB[0]+1)*(boxB[3]-boxB[1]+1)
    iou = areaOfIntersection/float(boxAArea+boxBArea-areaOfIntersection)
    return iou

def extractImages():    
    FaceImgs=[]
    setFace = []
    setNonFace = []
    paths, boxes = getPaths()
    os.chdir('D:/00_NCSU/00_Resources/00_Datasets/CV/Proj1/WiderFace/WIDER_train/images/')
    for box,path in zip(boxes[:1100],paths[:1100]):
        FaceImgs.append(cv2.imread(path))    
    os.chdir('D:/00_NCSU/Spring2020/ECE763_ComputerVision/Project/project01')
    for i,img in enumerate(FaceImgs):
        max_x, max_y, dims = img.shape
        bbx,bby,bbw,bbh=boxes[i][0], boxes[i][1], boxes[i][2],boxes[i][3]
        imgCropped = img[bby:bby+bbh, bbx:bbx+bbw, :]
        imgCropped = cv2.resize(imgCropped, size, interpolation = cv2.INTER_CUBIC)
        plt.imshow(imgCropped)
        setFace.append(imgCropped)
        if i<1000:
            try:
                os.mkdir('pos_train')
                os.mkdir('neg_train')
            except:
                pass
            pos_path='D:/00_NCSU/Spring2020/ECE763_ComputerVision/Project/project01/pos_train/'
            neg_path='D:/00_NCSU/Spring2020/ECE763_ComputerVision/Project/project01/neg_train/'
        else:
            try:
                os.mkdir('pos_test')
                os.mkdir('neg_test')
            except:
                pass
            pos_path='D:/00_NCSU/Spring2020/ECE763_ComputerVision/Project/project01/pos_test/'
            neg_path='D:/00_NCSU/Spring2020/ECE763_ComputerVision/Project/project01/neg_test/'
        filename = pos_path+'face'+str(i)+'.jpg'
        cv2.imwrite(filename,imgCropped)
        
        boxA = boxes[i]
        iou=100
        nf_shape=(0,0,0)
        while iou>=0.4 or nf_shape!=(size[0],size[0],3):
            x1, y1 = np.random.randint(0,max_x-60), np.random.randint(0,max_y-60)
            boxB = [x1,y1, x1+size[0], y1+size[1]]
            iou=IoU(boxA, boxB)
            nf_img = img[y1:y1+size[0], x1:x1+size[1],:]
            nf_shape = nf_img.shape
        setNonFace.append(nf_img)
        # nfChoice = np.random.randint(1,4)
        # if nfChoice==1:
        #     nfImg = img[boxes[i][1]:boxes[i][1]-boxes[i][3]//2+boxes[i][3],\
        #                 boxes[i][0]:boxes[i][0]-boxes[i][2]//2+boxes[i][2], :]
        # elif nfChoice==2:
        #     nfImg = img[boxes[i][1]:boxes[i][1]+boxes[i][3]//2+boxes[i][3],\
        #                 boxes[i][0]:boxes[i][0]-boxes[i][2]//2+boxes[i][2], :]
        # elif nfChoice==3:
        #     nfImg = img[boxes[i][1]:boxes[i][1]-boxes[i][3]//2+boxes[i][3],\
        #                 boxes[i][0]:boxes[i][0]+boxes[i][2]//2+boxes[i][2], :]
        # elif nfChoice==4:
        #     nfImg = img[boxes[i][1]:boxes[i][1]+boxes[i][3]//2+boxes[i][3],\
        #                 boxes[i][0]:boxes[i][0]+boxes[i][2]//2+boxes[i][2], :]
        # nfImg = cv2.resize(nfImg, (20,20), interpolation = cv2.INTER_CUBIC)
        # plt.imshow(nfImg)
        # setNonFace.append(nfImg)
        # try:
        #     os.mkdir('neg_train')
        # except:
        #     pass
        filename = neg_path+'non_face'+str(i)+'.jpg'
        cv2.imwrite(filename,nf_img)
    os.chdir('D:/00_NCSU/00_Resources/00_Datasets/CV/Proj1/WiderFace/WIDER_train/images/')
extractImages()