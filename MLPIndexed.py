import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

import Support

for i in range(15,16):
    data = Support.Indexed('data/trainingset.csv','data/testset.csv', i)
    X = data.npX()
    y = data.npy()
    inputLayer = i
    secondLayer = i // 2
    outputLayer = secondLayer // 2
    outputs = data.targetSize()

    MLP = Sequential()
    MLP.add(Dense(secondLayer, input_dim=inputLayer))
    MLP.add(Dense(outputLayer))
    MLP.add(Dense(outputs, activation='sigmoid'))
    MLP.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    epochs = 15
    batch_size = 1

    MLP.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0)
    predictions = MLP.predict(data.npX_test())
    print(predictions)
    print(data.decodePredictions(predictions))