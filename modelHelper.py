import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer
from keras.models import Sequential
from keras.layers import Dense, Embedding, SimpleRNN, LSTM, GRU
from dataHelper import Data

def buildDeepModel(modelType, embeddedDims, vocabSize, inputs, outputs):
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

def trainDeepModel(modelType, split, embeddedDims, epochs, batchSize, sentenceSize, trainFilePath, testFilePath):
    models = []

    data = Data(trainFilePath, testFilePath, sentenceSize)
    X = data.npX()
    y = data.npy()
    
    if split > 1:
        split = StratifiedKFold(n_splits=split, shuffle=False, random_state=1)
        for iTrain, iVal in split.split(X, data.decodePredictions(y)):
            model = buildDeepModel(modelType, embeddedDims, data.vocabSize(), sentenceSize, data.targetSize())
            model.fit(X[iTrain], y[iTrain], validation_data=(X[iVal], y[iVal]), epochs=epochs, batch_size=batchSize)
            models.append([model, data])
    else:
        model = buildDeepModel(modelType, embeddedDims, data.vocabSize(), sentenceSize, data.targetSize())
        model.fit(X, y, validation_split=split, epochs=epochs, batch_size=batchSize)
        models.append([model, data])

    return models[0]

def buildShallowModel(modelType, k):
    if modelType == "KNN":
        model = KNeighborsClassifier(n_neighbors=k)
    return model

def trainShallowModel(modelType, split, k, trainFilePath, testFilePath):

    data = Data(trainFilePath,testFilePath, -1)
    X_BoW = data.x_BagOfWords()
    y = data.y_labels()

    # Model creation and validation
    model = buildShallowModel(modelType, k)
    if split > 1:
        folds = StratifiedKFold(n_splits=split, shuffle=True, random_state=1)
        print(cross_val_score(model, X_BoW, y, cv=folds, verbose=10))
    model.fit(X_BoW, y)

    return [model, data]