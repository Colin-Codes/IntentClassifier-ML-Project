from modelHelper import trainDeepModel, trainShallowModel
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from dataHelper import Data
from datetime import datetime
import numpy as np
import pandas as pd
import pickle
import random


class Experiments:
    def __init__(self, _experiments):
        self.experiments = _experiments
        self.models = []

    def run(self):
        for Parameters in self.experiments:
            if Parameters.modelName == "KNN":
                self.models.append(trainShallowModel(Parameters))
            else:
                self.models.append(trainDeepModel(Parameters))

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

    def printResults(self, outputFileName=''):
        # Recording
        dateTimeNow = datetime.now()
        for modelData in self.models:
            model = modelData[0]
            data = modelData[1]
            Parameters = data.parameters
            if outputFileName == '':         
                dateTimeString = str(dateTimeNow.strftime("%Y%m%d_%H%M%S"))   
                if data.modelType() == 'Shallow':
                    outputFileName = Parameters.modelName + '_' + str(Parameters.kFolds) + '_' + str(Parameters.nNeighbours) + '_' + str(Parameters.embeddingMode) + '_' + dateTimeString + '.csv'
                else:
                    outputFileName = Parameters.modelName + '_' + str(Parameters.kFolds) + '_' + str(Parameters.embeddedDims) + '_' + str(Parameters.epochs) + '_' + str(Parameters.batchSize) + '_' + str(Parameters.sentenceSize) + '_' + dateTimeString + '.csv'
            if data.modelType() == "Shallow":
                y_pred = model.predict(data.x_test_BagOfWords())
                y = data.y_test_labels()
            else:
                y_pred = data.decodePredictions(model.predict(data.npX_test()))
                y = data.decodePredictions(data.npy_test())
            X = data.Xtext_test()
            trainingSet = pd.DataFrame(list(zip(y_pred, y, X)), columns=['Predicted Class','True Class','Email'])
            trainingSet.to_csv('results/' + outputFileName, index = None, header=True)            
            classReport = classification_report(y, y_pred, output_dict=True)
            classReport['modelName'] = Parameters.modelName
            classReport['modelkFolds'] = Parameters.kFolds
            classReport['modelnNeighbours'] = Parameters.nNeighbours
            classReport['modelEmbeddingMode'] = Parameters.embeddingMode
            classReport['modelEmbeddedDims'] = Parameters.embeddedDims
            classReport['modelEpochs'] = Parameters.epochs
            classReport['modelbatchSize'] = Parameters.batchSize
            classReport['modelsentenceSize'] = Parameters.sentenceSize

            with open('results/CR_' + outputFileName.replace('.csv','.pickle'), 'wb') as pikl:
                pickle.dump(classReport, pikl)

    def showResults(self, _fileName):
        with open('results/' + _fileName, 'rb') as pikl:
            classReportDict = pickle.load(pikl)
        print(classReportDict)

class Experiment:
    def __init__(self, _modelName, _kFolds=5, _nNeighbours=5,_embeddingMode='BoW', _embeddedDims=300, _epochs=1, _batchSize = 32, _sentenceSize=-1, _shuffleData=True, _randomSeed=-1, _trainFilePath='data/trainingset_augmented.csv', _testFilePath='data/testset.csv'):
        dateTimeNow = datetime.now()
        random.seed(int(dateTimeNow.strftime('%f')))
        self.modelName = _modelName
        self.kFolds = _kFolds
        self.nNeighbours = _nNeighbours
        self.embeddingMode = _embeddingMode
        self.embeddedDims = _embeddedDims
        self.epochs = _epochs
        self.batchSize = _batchSize
        self.sentenceSize = _sentenceSize
        self.trainFilePath = _trainFilePath
        self.testFilePath = _testFilePath
        self.shuffleData = _shuffleData
        if _randomSeed == -1:
            self.randomSeed = random.random() * int(dateTimeNow.strftime("%f"))
        else:
            self.randomSeed = _randomSeed
