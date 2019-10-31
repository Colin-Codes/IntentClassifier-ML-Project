import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold
from keras.models import Sequential
from keras.layers import Dense, Embedding, SimpleRNN, LSTM, GRU

import Support

def buildModel(modelType, embeddedDims, vocabSize, inputs, outputs):
    Model = Sequential()
    Model.add(Embedding(vocabSize, embeddedDims, input_length=inputs))
    if modelType == "RNN":
        Model.add(SimpleRNN(embeddedDims))
    elif modelType == "LSTM":
        Model.add(LSTM(embeddedDims))
    else:
        Model.add(GRU(embeddedDims))
    Model.add(Dense(outputs, activation='sigmoid'))
    Model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return Model

def modelTrainer(modelType, embeddedDims, split, epochs, batchSize, start, end):
    models = []
    for i in range(start, end + 1):

        data = Support.Indexed('data/trainingset_augmented.csv','data/testset.csv', i)
        X = data.npX()
        y = data.npy()
        
        if split > 1:
            split = StratifiedKFold(n_splits=split, shuffle=True, random_state=1)
            for iTrain, iVal in split.split(X, data.decodePredictions(y)):
                model = buildModel(modelType, embeddedDims, data.vocabSize(), i, data.targetSize())
                model.fit(X[iTrain], y[iTrain], validation_data=(X[iVal], y[iVal]), epochs=epochs, batch_size=batchSize)
                models.append(model)
        else:
            model = buildModel(modelType, embeddedDims, data.vocabSize(), i, data.targetSize())
            model.fit(X, y, validation_split=split, epochs=epochs, batch_size=batchSize)
            models.append(model)
    return models