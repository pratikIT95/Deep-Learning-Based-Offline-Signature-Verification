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
#Load the trainingset
dataset=pd.read_csv("SignatureDataset.csv",delimiter=",")
#testset=pd.read_csv("testing.csv",delimiter=",")
X = dataset.iloc[:,1:9].values
#X_test = testset.iloc[:,1:7].values
##print(X)
Y = dataset.iloc[:,9].values
#Y_test = testset.iloc[:,7].values
##print(Y)
# Splitting the dataset into the Training set and Test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.4)
# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
### Define and Compile
classifier = Sequential()
# Adding the input layer and the first hidden layer
classifier.add(Dense(activation="relu", input_dim=8, units=16, kernel_initializer="uniform"))
# Adding the second hidden layer
classifier.add(Dense(activation="relu", units=8, kernel_initializer="uniform"))
# Adding the third hidden layer
classifier.add(Dense(activation="sigmoid", units=2, kernel_initializer="uniform"))
# Adding the output layer
classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting our model 
classifier.fit(X_train, Y_train, batch_size = 10, epochs = 100,verbose=0)

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
    else:
        y_pred[i]=0.0
scores=0
for i in range(0,len(y_pred)):
    if(y_pred[i]-Y_test[i]==0.0):
        scores+=1              
cm = confusion_matrix(Y_test, y_pred)
print(cm)
print(scores/len(y_pred)*100.0)
