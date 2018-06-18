import keras
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.layers import Dense
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import numpy
import pandas as pd
import os
import matplotlib.pyplot as plt
gen=0
forged=0
testsize=0.5
chart=numpy.zeros((64,2))
globcount=0
falseaccept=0
falsereject=0
far=0
frr=0
#print(arr)
def runnetwork(dataset,i):
    global arr
    global globcount
    global falseaccept
    global falsereject
    global gen
    global forged
    global far
    global frr
    seed=7
    numpy.random.seed(seed)
    X = dataset.iloc[:,1:7].values
    #X_test = testset.iloc[:,1:7].values
    #print(X)
    Y = dataset.iloc[:,7].values
    ID = dataset.iloc[:,8].values
    #Y_test = testset.iloc[:,7].values
    #print(Y)
    # Splitting the dataset into the Training set and Test set
    global testsize
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = testsize)
    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    for x in Y_test:
        if x==0.0:
            gen+=1
        else:
            forged+=1
    ### Define and Compile
    classifier = Sequential()
    # Adding the input layer and the first hidden layer
    classifier.add(Dense(activation="relu", input_dim=6, units=16, kernel_initializer="uniform"))
    # Adding the second hidden layer
    classifier.add(Dense(activation="relu", units=8, kernel_initializer="uniform"))
    # Adding the third hidden layer
    #classifier.add(Dense(activation="relu", units=2, kernel_initializer="uniform"))
    # Adding the output layer
    classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    # Fitting our model 
    classifier.fit(X_train, Y_train, batch_size = 10, epochs = 120,verbose=0)

    ### Evaluate the model
    ##scores = model.evaluate(X, Y)
    ##
    ##print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    y_pred = classifier.predict(X_test)
    ##for i in range(0,len(y_pred)):
    ##    print(Y_test[i],y_pred[i])
    #y_pred = (y_pred > 0.5)
    for i in range(0,len(y_pred)):
        if(y_pred[i]>0.5):
            y_pred[i]=1.0
            if(Y_test[i]==0.0):
                falseaccept+=1
        else:
            y_pred[i]=0.0
            if(Y_test[i]==1.0):
                falsereject+=1
        #print(Y_test[i],y_pred[i])                
    cm = confusion_matrix(Y_test, y_pred)
    #print(cm)
    scores=cm[0][0]+cm[1][1]
    chart[globcount][0]=ID[i]
    chart[globcount][1]=scores/len(y_pred)*100.0
    globcount+=1
    #print(scores/len(y_pred)*100.0)
#Load the trainingset
filename="C:\\Users\\HP\\Desktop\\Project Stuff\\Code\\Test\\Dataset_IDs"
files=os.listdir(filename)
for i in range(0,len(files)):
    dataset=pd.read_csv(filename+"\\"+files[i],delimiter=",")
    runnetwork(dataset,i)
directory="C:\\Users\\HP\\Desktop\\Project Stuff\\Code\\Test\\Insights"
##l=len(os.listdir(directory))
##filename=directory+"//"+"insights_writerdependent"+str(l+1)+".txt"
##numpy.savetxt(filename,chart,fmt="%10.2f")
filename="C:\\Users\\HP\\Desktop\\Project Stuff\\Code\\Test\\Insights\\insights_writerdependent.txt"
far=falseaccept/forged
frr=falsereject/gen
f=open(filename,"a")
s="Accuracy="+str(numpy.average(chart[:,1]))+" Test Size="+str(testsize)+" Total Number of Layers=3 FAR="+str(far*100)+" FRR="+str(frr*100)+" Epochs=120\n"
print(s)
f.write(s)
f.close()
