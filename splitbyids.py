##import keras
##from keras.models import Sequential
from sklearn.model_selection import train_test_split
#from keras.layers import Dense
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import numpy
import pandas as pd
def savetocsv(X,currid):
    directory="C:\\Users\\HP\\Desktop\\Project Stuff\\Code\\Test\\Dataset_IDs\\"
    newlist=[]
    for i in range(0,2294):
        if(X[i,7]==currid):
            #print(currid)
            newlist.append(X[i])
    #print(newlist[0])
    df=pd.DataFrame(newlist,columns=['Skew Angle','Slope Angle','Center of Mass x','Center of Mass y','Aspect Ratio','Entropy','Genuine','ID'])
    filename=str(directory)+str("ID_")+str(currid)+str(".csv")
    df.to_csv(filename)
    return
seed=7
numpy.random.seed(seed)
#Load the trainingset
dataset=pd.read_csv("SignatureDatasetWithID.csv",delimiter=",")
#print(dataset)
setofids=[]
X=dataset.iloc[:,1:9].values
for i in range(0,2294):
    #print(X[i,8])
    if X[i,7] not in setofids:
        setofids.append(X[i,7])
print(setofids)
for i in range(len(setofids)):
    savetocsv(X,setofids[i])    
