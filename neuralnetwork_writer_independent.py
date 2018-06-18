import keras
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.layers import Dense
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import numpy
import pandas as pd
seed=7
numpy.random.seed(seed)
def neuralnetwork(testsize):
    #Load the trainingset
    dataset=pd.read_csv("SignatureDataset.csv",delimiter=",")
    #testset=pd.read_csv("testing.csv",delimiter=",")
    X = dataset.iloc[:,1:7].values
    Y = dataset.iloc[:,9].values
    # Splitting the dataset into the Training set and Test set
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = testsize)
    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    gen=0
    forged=0
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
    #classifier.add(Dense(activation = "sigmoid",units = 2, kernel_initializer = 'uniform'))
    # Adding the output layer
    classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    # Fitting our model 
    classifier.fit(X_train, Y_train, batch_size = 10, epochs = 115,verbose=0)

    y_pred = classifier.predict(X_test)
    falseaccept=0
    falsereject=0
    for i in range(0,len(y_pred)):
        if(y_pred[i]>0.5):
            y_pred[i]=1.0
            if(Y_test[i]==0.0):
                falseaccept+=1
        else:
            y_pred[i]=0.0
            if(Y_test[i]==1.0):
                falsereject+=1
    
    scores=0
    ##for i in range(0,len(y_pred)):
    ##    if(y_pred[i]-Y_test[i]==0.0):
    ##        scores+=1              
    cm = confusion_matrix(Y_test, y_pred)
    print(cm)
    scores=cm[0][0]+cm[1][1]
    far=falseaccept/forged
    frr=falsereject/gen
    print(scores/len(y_pred)*100.0)
    return scores/len(y_pred)*100.0,far*100.0,frr*100.0
count=0.2
f=open("writerindependent4.txt","w")
s=""
while(count<=0.5):
    score,far,frr=neuralnetwork(count)
    s=s+"Test Set Size="+str(count)+" Accuracy="+str(score)+" FAR="+str(far)+" FRR="+str(frr)+" Epochs=115"
    s+="\n"
    count+=0.05
f.write(s)
f.close()
