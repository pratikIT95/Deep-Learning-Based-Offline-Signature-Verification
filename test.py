from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy
import pandas as pd
seed=7
numpy.random.seed(seed)
#Load the trainingset
testsize=0.25
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
print(gen,forged)

