# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
# init - Weight....
# uniform - Weights are initialized to small uniformly random values between 0 and 0.05. 
classifier.add(Dense(input_dim = 11, output_dim = 6, init = 'uniform', activation = 'relu'))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
# The optimizer is the search technique used to update weights in your model.
# SGD: stochastic gradient descent, with support for momentum.
# RMSprop: adaptive learning rate optimization method proposed by Geoff Hinton.
# Adam: Adaptive Moment Estimation (Adam) that also uses adaptive learning rates.

# The loss function, also called the objective function is the evaluation of the model
# used by the optimizer to navigate the weight space.
# 'mse': for mean squared error.
# 'binary_crossentropy': for binary logarithmic loss (logloss).
# 'categorical_crossentropy': for multi-class logarithmic loss (logloss).

# Metrics are evaluated by the model during training.
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
# Epochs (nb_epoch) is the number of times that the model is exposed to the training
# dataset.
# Batch Size (batch_size) is the number of training instances shown to the model before
# a weight update is performed.
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 8)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
