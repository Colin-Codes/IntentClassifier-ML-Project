import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Embedding, GRU

import Support

for i in range(30,31):
    data = Support.Indexed('data/trainingset.csv','data/testset.csv', i)
    X = data.npX()
    y = data.npy()
    inputLayer = i
    secondLayer = i // 2
    outputLayer = secondLayer // 2
    vocabSize = data.vocabSize()
    embeddedDims = 300
    outputs = data.targetSize()

    LSTMClassifier = Sequential()
    LSTMClassifier.add(Embedding(vocabSize, embeddedDims, input_length=inputLayer))
    LSTMClassifier.add(GRU(embeddedDims))
    LSTMClassifier.add(Dense(outputs, activation='sigmoid'))
    LSTMClassifier.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    epochs = 15
    batch_size = 698

    LSTMClassifier.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0)
    predictions = LSTMClassifier.predict(data.npX_test())
    print(predictions)
    print(data.decodePredictions(predictions))