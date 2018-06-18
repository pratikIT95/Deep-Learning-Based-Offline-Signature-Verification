import pandas as pd

import numpy as np
#Read red wine data

red=pd.read_csv("winequality-red.csv",sep=";")

#Read white wine data
white=pd.read_csv("winequality-white.csv",sep=";")

print(white.info())

print(red.info())

import matplotlib.pyplot as plt
##
##fig, ax = plt.subplots(1, 2)
##
##ax[0].hist(red.alcohol, 10, facecolor='red', alpha=0.5, label="Red wine")
##ax[1].hist(white.alcohol, 10, facecolor='white', ec="black", lw=0.5, alpha=0.5, label="White wine")
##
##fig.subplots_adjust(left=0, right=1, bottom=0, top=0.5, hspace=0.05, wspace=1)
##ax[0].set_ylim([0, 1000])
##ax[0].set_xlabel("Alcohol in % Vol")
##ax[0].set_ylabel("Frequency")
##ax[1].set_xlabel("Alcohol in % Vol")
##ax[1].set_ylabel("Frequency")
###ax[0].legend(loc='best')
###ax[1].legend(loc='best')
##fig.suptitle("Distribution of Alcohol in % Vol")
##
##plt.show()

# Add `type` column to `red` with value 1
red['type'] = 1

# Add `type` column to `white` with value 0
white['type'] = 0

# Append `white` to `red`
wines = red.append(white, ignore_index=True)

# Import `train_test_split` from `sklearn.model_selection`
from sklearn.model_selection import train_test_split

# Specify the data 
X=wines.ix[:,0:11]

# Specify the target labels and flatten the array 
y=np.ravel(wines.type)

# Split the data up in train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Import `StandardScaler` from `sklearn.preprocessing`
from sklearn.preprocessing import StandardScaler

# Define the scaler 
scaler = StandardScaler().fit(X_train)

# Scale the train set
X_train = scaler.transform(X_train)

# Scale the test set
X_test = scaler.transform(X_test)

# Import `Sequential` from `keras.models`
from keras.models import Sequential

# Import `Dense` from `keras.layers`
from keras.layers import Dense

# Initialize the constructor
model = Sequential()

# Add an input layer 
model.add(Dense(12, activation='relu', input_shape=(11,)))

# Add one hidden layer 
model.add(Dense(8, activation='relu'))

# Add an output layer 
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
                   
model.fit(X_train, y_train,epochs=20, batch_size=1, verbose=1)

y_pred = model.predict(X_test)

score = model.evaluate(X_test, y_test,verbose=1)

print(score)
