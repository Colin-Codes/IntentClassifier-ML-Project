from modelHelper import trainDeepModel, trainShallowModel
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from dataHelper import Data
import numpy as np


class Experiments:
    def __init__(self, _experiments):
        self.experiments = _experiments
        self.models = []

    def run(self):
        for Parameters in self.experiments:
            if Parameters.modelType == "KNN":
                self.models.append(trainShallowModel(Parameters.modelType, Parameters.kFolds, Parameters.nNeighbours, Parameters.trainFilePath, Parameters.testFilePath))
            else:
                self.models.append(trainDeepModel(Parameters.modelType, Parameters.kFolds, Parameters.embeddedDims, Parameters.epochs, Parameters.batchSize, Parameters.sentenceSize, Parameters.trainFilePath, Parameters.testFilePath))

    def evaluate(self):
        # Evaluation
        for modelData in self.models:
            model = modelData[0]
            data = modelData[1]
            if data.modelType() == "Shallow":
                y_pred = model.predict(data.x_test_BagOfWords())
                y = data.y_test_labels()
            else:
                y_pred = data.decodePredictions(model.predict(data.npX_test()))
                y = data.decodePredictions(data.npy_test())
            #print(y_pred)
            #print(y)
            print('Confusion Matrix:')
            print(confusion_matrix(y, y_pred))
            print('Classification Report:')
            print(classification_report(y, y_pred))
            print('Accuracy Score:')
            print(accuracy_score(y, y_pred))

class Experiment:
    def __init__(self, _modelType, _kFolds=5, _nNeighbours=5, _embeddedDims=300, _epochs=1, _batchSize = 32, _sentenceSize=20, _trainFilePath='data/trainingset_augmented.csv', _testFilePath='data/testset.csv'):
        self.modelType = _modelType
        self.kFolds = _kFolds
        self.nNeighbours = _nNeighbours
        self.embeddedDims = _embeddedDims
        self.epochs = _epochs
        self.batchSize = _batchSize
        self.sentenceSize = _sentenceSize
        self.trainFilePath = _trainFilePath
        self.testFilePath = _testFilePath
