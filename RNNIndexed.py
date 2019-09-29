import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN

import Support

def D3Index(D2Indexed):
    return [[word] for word in [sentence for sentence in D2Indexed]]

for i in range(15,16):
    data = Support.Indexed('data/trainingset.csv','data/testset.csv', i)
    X = np.array(D3Index(data.X))
    y = data.npy()
    inputLayer = i
    secondLayer = i // 2
    outputLayer = secondLayer // 2
    outputs = data.targetSize()

    RNN = Sequential()
    RNN.add(SimpleRNN(1))
    RNN.add(Dense(outputs, activation='sigmoid'))
    RNN.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    epochs = 5
    batch_size = 1

    RNN.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0)
    predictions = RNN.predict(np.array(D3Index(data.X_test)))
    #print(predictions)
    print(data.decodePredictions(predictions))