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

def buildDeepModel(modelName, embeddedDims, vocabSize, inputs, outputs):
    Model = Sequential()
    Model.add(Embedding(vocabSize, embeddedDims, input_length=inputs))
    if modelName == "RNN":
        Model.add(SimpleRNN(embeddedDims))
    elif modelName == "LSTM":
        Model.add(LSTM(embeddedDims))
    else:
        Model.add(GRU(embeddedDims))
    Model.add(Dense(outputs, activation='sigmoid'))
    Model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return Model

def trainDeepModel(Parameters):
    models = []

    _modelName = Parameters.modelName
    _split = Parameters.kFolds
    _embeddedDims = Parameters.embeddedDims
    _sentenceSize = Parameters.sentenceSize
    _epochs = Parameters.epochs
    _batchSize = Parameters.batchSize
    _shuffleData = Parameters.shuffleData
    _randomSeed = Parameters.randomSeed

    data = Data(Parameters)
    X = data.npX()
    y = data.npy()

    if _split > 1:
        stratSplit = StratifiedKFold(n_splits=_split, shuffle=_shuffleData, random_state=_randomSeed)
        for iTrain, iVal in stratSplit.split(X, data.decodePredictions(y)):
            model = buildDeepModel(_modelName, _embeddedDims, data.vocabSize(), _sentenceSize, data.targetSize())
            model.fit(X[iTrain], y[iTrain], validation_data=(X[iVal], y[iVal]), epochs=_epochs, batch_size=_batchSize, shuffle=_shuffleData)
            models.append([model, data])
    elif _split == 1:
        model = buildDeepModel(_modelName, _embeddedDims, data.vocabSize(), _sentenceSize, data.targetSize())
        model.fit(X, y, epochs=_epochs, batch_size=_batchSize, shuffle=_shuffleData)
        models.append([model, data])
    else:
        model = buildDeepModel(_modelName, _embeddedDims, data.vocabSize(), _sentenceSize, data.targetSize())
        model.fit(X, y, validation_split=_split, epochs=_epochs, batch_size=_batchSize, shuffle=_shuffleData, )
        models.append([model, data])

    return models[0]

def buildShallowModel(_modelName, _k):
    if _modelName == "KNN":
        model = KNeighborsClassifier(n_neighbors=_k)
    return model

def trainShallowModel(Parameters):

    _modelName = Parameters.modelName
    _embeddingMode = Parameters.embeddingMode
    _kFolds = Parameters.kFolds
    _split = Parameters.split
    _shuffleData = Parameters.shuffleData
    _randomSeed = Parameters.randomSeed

    data = Data(Parameters)
    if _embeddingMode == "BoW":
        X = data.x_BagOfWords()
    else:    
        X = data.x_TFIDF()
    y = data.y_labels()

    # Model creation and validation
    model = buildShallowModel(_modelName, _kFolds)
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