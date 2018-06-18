import numpy as np
import cv2
import math
from scipy import ndimage
from scipy.ndimage import interpolation as inter
im = cv2.imread('binary3.png',0)
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
#print("Lowest signature point=",bottompointx," , ", bottompointy)
#print("Leftmost signature point=",maxi," , ",maxj)
con=con.reshape(-1)
con=con.flatten()
mini=con[0]
minj=con[1]
#print("Min point ",mini," , ", minj)
height=bottompointy-highestpointy
width=maxi-mini
#print("Height=", height )
#print("Width=", width )

#Finding the aspect ratio of the image

aspectratio=width/height

#Finding the slope of the image

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
print('Best angle:',best_angle)
print("Slope Angle=",slope)
print("Center of mass=",cog)
print("Aspect ratio=",aspectratio)
print("Entropy= " +str(entr))
