from modelHelper import trainDeepModel, trainShallowModel
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from dataHelper import Data
from datetime import datetime
from matplotlib import cm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import random
import os


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
            thresholdIndex = data.indexPrediction('Threshold')
            thresholdValue = data.parameters.threshold
            if data.modelType() == "Shallow":
                Predictions = np.insert(model.predict_proba(data.x_test_BagOfWords()), thresholdIndex, thresholdValue, axis=1)
                y_pred = data.decodePredictions(Predictions)
                y = data.y_test_labels()
            else:
                Predictions = np.insert(model.predict(data.npX_test()), thresholdIndex, thresholdValue, axis=1)
                y_pred = data.decodePredictions(Predictions)
                y = data.decodePredictions(data.npy_test())
            print('Confusion Matrix:')
            print(confusion_matrix(y, y_pred))
            print('Classification Report:')
            print(classification_report(y, y_pred))
            print('Accuracy Score:')
            print(accuracy_score(y, y_pred))

    def printClassResults(self, saveFolder='results/'):
        # Recording
        dateTimeNow = datetime.now()
        for modelData in self.models:
            model = modelData[0]
            data = modelData[1]
            Parameters = data.parameters
            createFilePath = saveFolder
            if not os.path.exists(createFilePath):
                os.mkdir(createFilePath)
            createFilePath += Parameters.modelName + '/'
            if not os.path.exists(createFilePath):
                os.mkdir(createFilePath)
            createFilePath += str(Parameters.kFolds) + '_Folds/'
            if not os.path.exists(createFilePath):
                os.mkdir(createFilePath)

            dateTimeString = str(dateTimeNow.strftime("%Y%m%d_%H%M%S"))   
            if data.modelType() == 'Shallow':
                createFilePath += str(Parameters.nNeighbours) + '_Neighbours/'
                if not os.path.exists(createFilePath):
                    os.mkdir(createFilePath)
                createFilePath += str(Parameters.embeddingMode) + '/'
                if not os.path.exists(createFilePath):
                    os.mkdir(createFilePath)
                outputFileName = createFilePath + dateTimeString + '.csv'
            else:
                createFilePath += str(Parameters.embeddedDims) + '_Dimensions/'
                if not os.path.exists(createFilePath):
                    os.mkdir(createFilePath)
                createFilePath += str(Parameters.epochs) + '_Epochs/'
                if not os.path.exists(createFilePath):
                    os.mkdir(createFilePath)
                createFilePath += str(Parameters.batchSize) + '_Batch/'
                if not os.path.exists(createFilePath):
                    os.mkdir(createFilePath)
                createFilePath += str(Parameters.sentenceSize) + '_Sentence/'
                if not os.path.exists(createFilePath):
                    os.mkdir(createFilePath)
                outputFileName = createFilePath + dateTimeString + '.csv'
            thresholdIndex = data.indexPrediction('Threshold')
            thresholdValue = data.parameters.threshold
            if data.modelType() == "Shallow":                
                Predictions = np.insert(model.predict_proba(data.x_test_BagOfWords()), thresholdIndex, thresholdValue, axis=1)
                y_pred = data.decodePredictions(Predictions)
                y = data.y_test_labels()
            else:
                Predictions = np.insert(model.predict(data.npX_test()), thresholdIndex, thresholdValue, axis=1)
                y_pred = data.decodePredictions(Predictions)
                y = data.decodePredictions(data.npy_test())
            X = data.Xtext_test()
            trainingSet = pd.DataFrame(list(zip(y_pred, y, X)), columns=['Predicted Class','True Class','Email'])
            trainingSet.to_csv(outputFileName, index = None, header=True)            
            classReport = classification_report(y, y_pred, output_dict=True)
            classReport['modelmodelName'] = Parameters.modelName
            classReport['modelkFolds'] = Parameters.kFolds
            classReport['modelnNeighbours'] = Parameters.nNeighbours
            classReport['modelembeddingMode'] = Parameters.embeddingMode
            classReport['modelembeddedDims'] = Parameters.embeddedDims
            classReport['modelepochs'] = Parameters.epochs
            classReport['modelbatchSize'] = Parameters.batchSize
            classReport['modelsentenceSize'] = Parameters.sentenceSize
            classReport['modelthreshold'] = Parameters.threshold

            with open(outputFileName.replace('.csv','.pickle'), 'wb') as pikl:
                pickle.dump(classReport, pikl)

    def showClassResults(self, _fileName):
        with open(_fileName, 'rb') as pikl:
            classReportDict = pickle.load(pikl)
        print(classReportDict)
    
    def printMultiClassReport(self, Model, Parameter, Bounds = [], saveFolder = 'results/'):

        # Get pickled results Dicts for the specified model, limit using Bounds
        Results = []
        rootdir = saveFolder + Model + '/1_Folds'
        for root, dirs, files in os.walk(rootdir):
            if Model == 'KNN':
                for nNeighbourDir in dirs:
                    n = nNeighbourDir.split('_')[0]
                    if Parameter != 'nNeighbour' or len(Bounds) == 0 or (Bounds[0] <= n and n < Bounds[1]):
                        if Parameter != 'embeddingMode' or len(Bounds) == 0 or 'BoW' in Bounds:
                            self.appendResults(Results, rootdir + '/' + nNeighbourDir + '/BoW')
                        if Parameter != 'embeddingMode' or len(Bounds) == 0 or 'TFIDF' in Bounds:
                            self.appendResults(Results, rootdir + '/' + nNeighbourDir + '/TFIDF')
            else:
                for embeddedDimsDir in dirs:
                    embeddedDimsRoot = rootdir + '/' + embeddedDimsDir
                    embeddedDims = embeddedDimsDir.split('_')[0]
                    if Parameter != 'embeddedDims' or len(Bounds) == 0 or (Bounds[0] <= embeddedDims and embeddedDims < Bounds[1]):
                        for root, dirs, files in os.walk(embeddedDimsRoot):
                            for epochDir in dirs:
                                epochRoot = embeddedDimsRoot + '/' + epochDir
                                epochs = epochDir.split('_')[0]
                                if Parameter != 'epochs' or len(Bounds) == 0 or (Bounds[0] <= epochs and epochs < Bounds[1]):
                                    for root, dirs, files in os.walk(epochRoot):
                                        for batchDir in dirs:
                                            batchRoot = epochRoot + '/' + batchDir
                                            batchSize = batchDir.split('_')[0]
                                            if Parameter != 'batchSize' or len(Bounds) == 0 or (Bounds[0] <= batchSize and batchSize < Bounds[1]):
                                                for root, dirs, files in os.walk(batchRoot):
                                                    for sentenceDir in dirs:
                                                        sentenceRoot = batchRoot + '/' + sentenceDir
                                                        sentenceSize = batchDir.split('_')[0]
                                                        if Parameter != 'sentenceSize' or len(Bounds) == 0 or (Bounds[0] <= sentenceSize and sentenceSize < Bounds[1]):
                                                            for root, dirs, files in os.walk(sentenceRoot):
                                                                self.appendResults(Results, sentenceRoot)
    
        # Generate a list of labels from the first Result
        labelsAvailable = [label for label in Results[0].keys() if 'model' not in label and label not in ['micro avg','macro avg','weighted avg']]
        if 'Threshold' not in labelsAvailable:
            labelsAvailable.append('Threshold')

        # Generate a list of evenly distributed colours
        cmap = cm.get_cmap('gist_rainbow')
        colourMapping = cmap(np.linspace(0, 1, len(labelsAvailable)))
        
        # Iterate over each class in each dict, append to the class index as a tuple: the parameter value (0) and the Precision/Recall (1) 
        x = []
        yPrecision = []
        yRecall = []
        labels = []
        colours = []

        for result in Results:
            for key in result.keys():
                if key in labelsAvailable:
                    x.append(result['model' + Parameter])
                    yPrecision.append(result[key]['precision'])
                    yRecall.append(result[key]['recall'])
                    labels.append(key)    
                    colours.append(labelsAvailable.index(key)) 

        # Create precision graph
        plt.scatter(x, yPrecision, c=colours, label=labels, s=5)
        plt.title('Precision against ' + Parameter + ' (' + Model + ')')
        plt.legend(loc=2)
        if not os.path.exists(saveFolder):
            os.mkdir(saveFolder)
        createFilePath = saveFolder + 'graphs/'
        if not os.path.exists(createFilePath):
            os.mkdir(createFilePath)
        createFilePath += 'precision/'
        if not os.path.exists(createFilePath):
            os.mkdir(createFilePath)
        saveRoot = createFilePath + Model + '_' + Parameter + '_precision'
        if len(Bounds) == 0:
            plt.savefig(saveRoot + '.png')
        elif len(Bounds) == 1:
            plt.savefig(saveRoot + '_' + str(Bounds[0]) + '.png')
        else:
            plt.savefig(saveRoot + '_' + str(Bounds[0]) + '_' + str(Bounds[1]) + '.png')
        
        # Create recall graph
        plt.scatter(x, yRecall, c=colours, label=labels, s=5)
        plt.title('Recall against ' + Parameter + ' (' + Model + ')')
        plt.legend(loc=2)            
        if not os.path.exists(saveFolder):
            os.mkdir(saveFolder)
        createFilePath = saveFolder + 'graphs/'
        if not os.path.exists(createFilePath):
            os.mkdir(createFilePath)
        createFilePath += 'recall/'
        if not os.path.exists(createFilePath):
            os.mkdir(createFilePath)
        saveRoot = createFilePath + Model + '_' + Parameter + '_recall'
        if len(Bounds) == 0:
            plt.savefig(saveRoot + '.png')
        elif len(Bounds) == 1:
            plt.savefig(saveRoot + '_' + str(Bounds[0]) + '.png')
        else:
            plt.savefig(saveRoot + '_' + str(Bounds[0]) + '_' + str(Bounds[1]) + '.png')

        return

    def showClassreport(self, _filename):
        # Show matplotlib
        return ''
    
    def appendResults(self, ResultsList, directoryPath):
        for root, dirs, files in os.walk(directoryPath):
            for pickleFile in files:
                if '.pickle' in pickleFile:
                    with open(directoryPath + '/' + pickleFile, 'rb') as pikl:
                        ResultsList.append(pickle.load(pikl))

class Experiment:
    def __init__(self, _modelName, _kFolds=5, _nNeighbours=5,_embeddingMode='BoW', _embeddedDims=300, _epochs=1, _batchSize = 32, _sentenceSize=-1, threshold=0.0, _shuffleData=True, _randomSeed=-1, _trainFilePath='data/trainingset_augmented.csv', _testFilePath='data/testset.csv'):
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
        self.threshold = threshold
        self.shuffleData = _shuffleData
        if _randomSeed == -1:
            self.randomSeed = random.random() * int(dateTimeNow.strftime("%f"))
        else:
            self.randomSeed = _randomSeed
        self.trainFilePath = _trainFilePath
        self.testFilePath = _testFilePath
