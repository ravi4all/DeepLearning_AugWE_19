import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from keras.utils import np_utils

text = open('data.txt').read()
text = text.lower()

characters = sorted(list(set(text)))

n_to_char = {n:char for n,char in enumerate(characters)}
char_to_n = {char:n for n,char in enumerate(characters)}

X = []
Y = []

length = len(text)
seq_length = 100

for i in range(0,length-seq_length):
    seq = text[i:i+seq_length]
    label = text[i+seq_length]
    X.append([char_to_n[char] for char in seq])
    Y.append(char_to_n[label])

X_train = np.reshape(X,(len(X), seq_length, 1))
X_train = X_train / len(characters)
Y_train = np_utils.to_categorical(Y)
    
model = Sequential()
model.add(LSTM(500,input_shape=(X_train.shape[1], X_train.shape[2]),
               return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(400))
model.add(Dropout(0.2))
model.add(Dense(Y_train.shape[1],activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam')
model.fit(X_train, Y_train, epochs=100, batch_size=50)
model.save('text_gen.h5')

str_map = X[4]
full_str = [n_to_char[value] for value in str_map]

#1. Start a for loop till seq_length
#2. Reshape str_map
#3. Normalize it
#4. Find prediction (prediction_index)
#5. Find Sequence of str_map
#6. append the char of prediction_index in full_str
#7. append prediction_index to str_map
#8. slice str_map from 1:length of str_map
#9. End Loop
#10. Join the full_str using another loop

for i in range(seq_length):
    x = np.reshape(str_map,(1,len(str_map), 1))
    x = x / len(characters)
    prediction_index = np.argmax(model.predict(x))
    seq = [n_to_char[val] for val in str_map]
    full_str.append(n_to_char[prediction_index])
    str_map.append(prediction_index)
    str_map = str_map[1:len(str_map)]
    
pred_text = ""
for char in full_str:
    pred_text += char

print(pred_text)








    