import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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

def trainDeepModel(_modelType, _split, _embeddedDims, _epochs, _batchSize, _sentenceSize, _shuffleData, _randomSeed, _trainFilePath, _testFilePath):
    models = []

    data = Data(_trainFilePath, _testFilePath, _sentenceSize, _shuffle=_shuffleData)
    X = data.npX()
    y = data.npy()
    
    if _split > 1:
        stratSplit = StratifiedKFold(n_splits=_split, shuffle=_shuffleData, random_state=_randomSeed)
        for iTrain, iVal in stratSplit.split(X, data.decodePredictions(y)):
            model = buildDeepModel(_modelType, _embeddedDims, data.vocabSize(), _sentenceSize, data.targetSize())
            model.fit(X[iTrain], y[iTrain], validation_data=(X[iVal], y[iVal]), epochs=_epochs, batch_size=_batchSize)
            models.append([model, data])
    elif _split == 1:
        model = buildDeepModel(_modelType, _embeddedDims, data.vocabSize(), _sentenceSize, data.targetSize())
        model.fit(X, y, epochs=_epochs, batch_size=_batchSize)
        models.append([model, data])
    else:
        model = buildDeepModel(_modelType, _embeddedDims, data.vocabSize(), _sentenceSize, data.targetSize())
        model.fit(X, y, validation_split=_split, epochs=_epochs, batch_size=_batchSize)
        models.append([model, data])

    return models[0]

def buildShallowModel(_modelType, _k):
    if _modelType == "KNN":
        model = KNeighborsClassifier(n_neighbors=_k)
    return model

def trainShallowModel(_modelType, _split, _k, _embeddingMode, _shuffleData, _randomSeed, _trainFilePath, _testFilePath):

    data = Data(_trainFilePath,_testFilePath, _shuffle=_shuffleData)
    if _embeddingMode == "BoW":
        X = data.x_BagOfWords()
    else:    
        X = data.x_TFIDF()
    y = data.y_labels()

    # Model creation and validation
    model = buildShallowModel(_modelType, _k)
    if _split > 1:
        folds = StratifiedKFold(n_splits=_split, shuffle=_shuffleData, random_state=_randomSeed)
        print(cross_val_score(model, X, y, cv=folds, verbose=10))
    elif _split == 1:
        model.fit(X, y)
    else:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=_split, random_state=_randomSeed)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        print('Validation accuracy: ' + str(accuracy_score(y_val, y_pred)))

    model.fit(X, y)
    return [model, data]