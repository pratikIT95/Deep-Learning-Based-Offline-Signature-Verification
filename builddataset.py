import cv2

import os

import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

import math

from scipy import ndimage

from scipy.ndimage import interpolation as inter

def preprocess(img):
    
    #Resizing the image

    rimg = cv2.resize(img,(500,200))

    #Threshloding the image to binary form with threshold value range (220,255)

    ret,thresh1 = cv2.threshold(rimg,220,255,cv2.THRESH_BINARY)

    #Eroding the thresholded image

    kernel = np.ones((2,2),np.uint8)

    dilated = cv2.dilate(thresh1,kernel,iterations = 2)

    erosion=cv2.bitwise_not(dilated)

    image=erosion

    #cv2.imshow("Image",image)
    
    return image

def extractfeatures(flag):

    testingset="C:\\Users\\HP\\Desktop\\Project Stuff\\Signature Dataset\\Testdata_SigComp2011\\SigComp11-Offlinetestset\\Dutch\\Lol"

    files=os.listdir(testingset)

    print(len(files))

    features=np.zeros((len(files),7),dtype=np.float16)

    for outer in range(0,len(files)):
        
        img=cv2.imread(testingset+"\\"+files[outer],0)
        print(files[outer])

        im=preprocess(img)
        
        npim=np.array(im)

        np1d=np.ndarray.flatten(im)

        #Calculate the center of gravity of image

        cog=ndimage.measurements.center_of_mass(npim)

        #Calculate the entropy of the image

        def entropy(signal):
                '''
                function returns entropy of a signal
                signal must be a 1-D numpy array
                '''
                lensig=signal.size
                symset=list(set(signal))
                numsym=len(symset)
                propab=[np.size(signal[signal==i])/(1.0*lensig) for i in symset]
                ent=np.sum([p*np.log2(1.0/p) for p in propab])
                return ent

        entr=entropy(np1d)


        #Finding contours of the image

        image, contours, hierarchy = cv2.findContours(im,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        #The first contour

        con=contours[0]

        #Finding the rightmost contour point

        maxi=0
        maxj=0

        for i in range(1,len(contours)):
            c=contours[i].reshape(-1)
            c=c.flatten()
            #print(len(c))
            for j in range(0,len(c)):
                if(j%2==0):
                    if(c[j]>maxi):
                        maxi=c[j]
                        maxj=c[j+1]

        #Finding the bottom most contour point

        bottompointy=0
        bottompointx=0

        highestpointy=200

        for i in range(0,len(contours)):
            c=contours[i].reshape(-1)
            c=c.flatten()
            #print(len(c))
            for j in range(0,len(c)):
                if(j%2==1):
                    if(c[j]>bottompointy):
                        bottompointy=c[j]
                        bottompointx=c[j-1]
                    if(c[j]<highestpointy):
                        highestpointy=c[j]
        
        con=con.reshape(-1)
        con=con.flatten()
        mini=con[0]
        minj=con[1]
        
        height=bottompointy-highestpointy
        width=maxi-mini
        

        #Finding the aspect ratio of the image

        aspectratio=width/height

        #Finding the slope of the image

        if(maxi-mini==0):

            slope=90

        else:
        
            slope=math.degrees(math.atan((maxj-minj)/(maxi-mini)))

        #Finding the skewness of the image

        def find_score(arr, angle):
            data = inter.rotate(arr, angle, reshape=False, order=0)
            hist = np.sum(data, axis=1)
            score = np.sum((hist[1:] - hist[:-1]) ** 2)
            return hist, score


        delta = 1
        limit = 5
        angles = np.arange(-limit, limit+delta, delta)
        scores = []
        for angle in angles:
            hist, score = find_score(npim, angle)
            scores.append(score)

        best_score = max(scores)
        best_angle = angles[scores.index(best_score)]
        #print('Skew Angle:',best_angle)
        features[outer][0]=best_angle
        #print("Slope Angle=",slope)
        features[outer][1]=slope
        #print("Center of mass=",cog)
        features[outer][2]=cog[0]
        features[outer][3]=cog[1]
        #print("Aspect ratio=",aspectratio)
        features[outer][4]=aspectratio
        #print("Entropy= " +str(entr))
        features[outer][5]=entr
        if(len(files[outer])>10):
            flag=0
        else:
            flag=1
        print(flag)
        features[outer][6]=flag

        #print(features[outer])

    return features

features=extractfeatures(1)

df=pd.DataFrame(features,columns=['Skew Angle','Slope Angle','Center of Mass x','Center of Mass y','Aspect Ratio','Entropy','Genuine'])

df.to_csv("SignatureDataset.csv")
