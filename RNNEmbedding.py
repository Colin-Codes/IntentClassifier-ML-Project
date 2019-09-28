import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Embedding, SimpleRNN

import Support

for i in range(15,16):
    data = Support.Indexed('data/trainingset.csv','data/testset.csv', i)
    X = data.npX()
    y = data.npy()
    inputLayer = i
    secondLayer = i // 2
    outputLayer = secondLayer // 2
    vocabSize = data.vocabSize()
    embeddedDims = 300
    outputs = data.targetSize()

    RNN = Sequential()
    RNN.add(Embedding(vocabSize, embeddedDims, input_length=inputLayer))
    RNN.add(SimpleRNN(embeddedDims))
    RNN.add(Dense(outputs, activation='sigmoid'))
    RNN.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    epochs = 10
    batch_size = 1

    RNN.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0)
    predictions = RNN.predict(data.npX_test())
    print(predictions)
    print(data.decodePredictions(predictions))