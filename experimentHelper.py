from modelHelper import trainDeepModel, trainShallowModel
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from dataHelper import Data
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import math
import random
import os
import csv


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
        if saveFolder[-1] != '/':
            saveFolder += '/'

        # Recording
        dateTimeNow = datetime.now()
        for modelData in self.models:
            model = modelData[0]
            data = modelData[1]
            Parameters = data.parameters
            createFilePath = ''

            # If there are multiple new dirs in saveFolder, iterate over each one
            for folder in saveFolder.split('/'):
                if folder != '':
                    createFilePath += folder + '/'
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
                if Parameters.embeddingMode == "BoW":           
                    Predictions = np.insert(model.predict_proba(data.x_test_BagOfWords()), thresholdIndex, thresholdValue, axis=1)
                else:
                    Predictions = np.insert(model.predict_proba(data.x_test_TFIDF()), thresholdIndex, thresholdValue, axis=1)
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

    def printSentenceLengths(self):
        
        if not os.path.exists('data'):
            os.mkdir('data')
        
        if not os.path.exists('data/graphs'):
            os.mkdir('data/graphs')

        for _, _, files in os.walk('data/'):
            for csvFile in files:
                if '.csv' in csvFile:
                    with open('data/' + csvFile, "r") as importTrainCSV:
                        samples = [i[1:2] for i in list(csv.reader(importTrainCSV))[1:]]
                        counts = [len(i[0].split(' ')) for i in samples]
                        plt.figure()
                        plt.hist(counts, facecolor='blue', alpha=0.5, bins=30)
                        plt.xlabel('Sentence lengths')
                        plt.ylabel('Occurence')
                        plt.title('Histogram of sentence lengths in dataset: ' + csvFile)
                        plt.savefig('data/graphs/' + csvFile.replace('.csv','.png'))

        for i in plt.get_fignums():
            plt.close(i)
        return

    def printCompareParameterClasses(self, Model, classList=[], parameter='threshold', parameterValues=[], saveFolder='results/Global/'):
        ### This function is useful for comparing the effect of changes in a parameter across all classes

        # Ensure folder format is correct
        if saveFolder[-1] != '/':
            saveFolder += '/'

        # Get pickled results Dicts for the specified model(s)
        ExperimentResults = []
        rootdir = saveFolder + Model + '/1_Folds'
        for _, dirs, _ in os.walk(rootdir):
            if Model == 'KNN':
                for nNeighbourDir in dirs:
                    self.appendResults(ExperimentResults, rootdir + '/' + nNeighbourDir + '/BoW')
                    self.appendResults(ExperimentResults, rootdir + '/' + nNeighbourDir + '/TFIDF')
            else:
                for embeddedDimsDir in dirs:
                    embeddedDimsRoot = rootdir + '/' + embeddedDimsDir
                    for _, dirs, _ in os.walk(embeddedDimsRoot):
                        for epochDir in dirs:
                            epochRoot = embeddedDimsRoot + '/' + epochDir
                            for _, dirs, _ in os.walk(epochRoot):
                                for batchDir in dirs:
                                    batchRoot = epochRoot + '/' + batchDir
                                    for _, dirs, _ in os.walk(batchRoot):
                                        for sentenceDir in dirs:
                                            sentenceRoot = batchRoot + '/' + sentenceDir
                                            self.appendResults(ExperimentResults, sentenceRoot)
    
        # Generate a list of labels from the first Result
        labelsAvailable = [label for label in ExperimentResults[0].keys() if 'model' not in label and label not in ['micro avg','macro avg','weighted avg']]
        parameterRange = sorted(set([Results['model' + parameter] for Results in ExperimentResults]))
        if len(parameterValues) > 0:
            parameterRange = [i for i in parameterRange if i in parameterValues]

        # Iterate over each class in each dict, append to the class index as a tuple: the parameter value (0) and the Precision/Recall (1) 
        xLabels = [[j for j in labelsAvailable] for i in parameterRange]
        yPrecision = [[[] for j in labelsAvailable] for i in parameterRange]
        yRecall = [[[] for j in labelsAvailable] for i in parameterRange]
        aggregatePrecision = [[] for j in labelsAvailable]
        aggregateRecall = [[] for j in labelsAvailable]
        for i in parameterRange:
            for Result in ExperimentResults:
                if Result['model' + parameter] == i:
                    for key in Result.keys():
                        if key in labelsAvailable:
                            aggregatePrecision[labelsAvailable.index(key)].append(Result[key]['precision'])
                            aggregateRecall[labelsAvailable.index(key)].append(Result[key]['recall'])
                            yPrecision[parameterRange.index(i)][labelsAvailable.index(key)].append(Result[key]['precision'])
                            yRecall[parameterRange.index(i)][labelsAvailable.index(key)].append(Result[key]['recall'])    
        MeanPrecision = [np.mean(i) for i in aggregatePrecision]
        MeanRecall = [np.mean(i) for i in aggregateRecall]

        # Average all results within each class   
        xPrecisionLabels = [[i for _,i in sorted(zip(MeanPrecision,j))] for j in xLabels]
        xRecallLabels = [[i for _,i in sorted(zip(MeanRecall,j))] for j in xLabels]
        yPrecisionMean = [[np.mean(i) * 100 for _,i in sorted(zip(MeanPrecision,j))] for j in yPrecision]
        yRecallMean = [[np.mean(i) * 100 for _,i in sorted(zip(MeanRecall,j))] for j in yRecall]
        yPrecisionErr = [[np.std(i) * 100 for _,i in sorted(zip(MeanPrecision,j))] for j in yPrecision]
        yRecallErr = [[np.std(i) * 100 for _,i in sorted(zip(MeanRecall,j))] for j in yRecall]

        precisionSaveRoot = saveFolder + 'graphs/precision/' + Model + '/comparison_' + parameter
        recallSaveRoot = saveFolder + 'graphs/recall/' + Model + '/comparison_' + parameter
        titleExtension = ', split by ' + parameter + ' (' + Model + ')'
        for value in parameterValues:
            precisionSaveRoot += '_' + str(value)
            recallSaveRoot += '_' + str(value)

        # Create precision graph
        self.createPlots(xPlots=xPrecisionLabels, yPlots=yPrecisionMean, yErrList = yPrecisionErr, plotLabels=parameterRange, xAxisName = 'Class', yAxisName = 'Precision (%)', legendTitle=parameter.capitalize(), title = 'Precision against Class' + titleExtension, fileName = precisionSaveRoot + '_' + Model + '_precision.png', scatter=True)
           
        # Create recall graph
        self.createPlots(xPlots=xRecallLabels, yPlots=yRecallMean, yErrList = yRecallErr, plotLabels=parameterRange, xAxisName = 'Class', yAxisName = 'Recall (%)', legendTitle=parameter.capitalize(), title = 'Recall against Class' + titleExtension, fileName = recallSaveRoot + '_' + Model + '_recall.png', scatter=True)

        if len(classList) > 0:

            # Get data
            xFilteredLabels = [[j for j in classList] for i in parameterRange]
            yFilteredPrecision = [[[] for j in classList] for i in parameterRange]
            yFilteredRecall = [[[] for j in classList] for i in parameterRange]
            aggregateFilteredPrecision = [[] for j in classList]
            aggregateFilteredRecall = [[] for j in classList]
            for i in parameterRange:
                for Result in ExperimentResults[i]:
                    if Result['model' + parameter] == i:
                        for key in Result.keys():
                            if key in classList:
                                aggregateFilteredPrecision[classList.index(key)].append(Result[key]['precision'])
                                aggregateFilteredRecall[classList.index(key)].append(Result[key]['recall'])
                                yFilteredPrecision[parameterRange.index(i)][classList.index(key)].append(Result[key]['precision'])
                                yFilteredRecall[parameterRange.index(i)][classList.index(key)].append(Result[key]['recall'])
            MeanFilteredPrecision = [np.mean(i) for i in aggregateFilteredPrecision]
            MeanFilteredRecall = [np.mean(i) for i in aggregateFilteredRecall]
            
            #Average data
            xFilteredPrecisionLabels = [[i for _,i in sorted(zip(MeanFilteredPrecision,j))] for j in xFilteredLabels]
            xFilteredRecallLabels = [[i for _,i in sorted(zip(MeanFilteredRecall,j))] for j in xFilteredLabels]
            yFilteredPrecisionMean = [[np.mean(i) * 100 for _,i in sorted(zip(MeanFilteredPrecision,j))] for j in yFilteredPrecision]
            yFilteredRecallMean = [[np.mean(i) * 100 for _,i in sorted(zip(MeanFilteredRecall,j))] for j in yFilteredRecall]
            yFilteredPrecisionErr = [[np.std(i) * 100 for _,i in sorted(zip(MeanFilteredPrecision,j))] for j in yFilteredPrecision]
            yFilteredRecallErr = [[np.std(i) * 100 for _,i in sorted(zip(MeanFilteredRecall,j))] for j in yFilteredRecall]

            # Create filtered precision graph 
            self.createPlots(xPlots=xFilteredPrecisionLabels, yPlots=yFilteredPrecisionMean, yErrList = yFilteredPrecisionErr, plotLabels=parameterRange, xAxisName = 'Class', yAxisName = 'Precision (%)', legendTitle=parameter.capitalize(), title = 'Precision against Class (filtered)' + titleExtension, fileName = precisionSaveRoot + '_' + Model + '_precision_filtered.png', scatter=True)
            
            # Create filtered recall graph
            self.createPlots(xPlots=xFilteredRecallLabels, yPlots=yFilteredRecallMean, yErrList = yFilteredRecallErr, plotLabels=parameterRange, xAxisName = 'Class', yAxisName = 'Recall (%)', legendTitle=parameter.capitalize(), title = 'Recall against Class (filtered)' + titleExtension, fileName = recallSaveRoot + '_' + Model + '_recall_filtered.png', scatter=True)
        return

    def printCompareExperimentsClasses(self, ModelsList=[], classList=[], loadFolders = ['results/'], saveFolder='results/Global/'):
        ### This function is useful for comparing the results of two or more experiments, eg. different datasets

        # Ensure folder format is correct
        if saveFolder[-1] != '/':
            saveFolder += '/'
        loadFolders = [i + '/' if i[-1] !='/' else i for i in loadFolders]

        # Get plot characteristic from models or folders
        byModel = False
        byFolder = False

        for Models in ModelsList:
            for Model in ModelsList[0]:
                if Model not in Models:
                    byModel = True
                    break
                    
        for loadFolder in loadFolders:
            firstFolder = loadFolders[0]
            if loadFolder != firstFolder:
                byFolder = True
                break

        # Get pickled results Dicts for the specified model(s)
        ExperimentResults = [[] for i in loadFolders]
        plotLabels = []
        for i in range(0,len(loadFolders)):
            if byFolder == True and byModel == True:
                plotLabels.append('-'.join(ModelsList[i]) + '-' + loadFolders[i].split('/')[-2])
            elif byModel == True:
                plotLabels.append(ModelsList[i](0))
            elif byFolder == True:
                plotLabels.append(loadFolders[i].split('/')[-2])
            else:
                print('"printCompareExperimentsClasses" is for comparing different models and or experiments, please use "printClassParameterReports" or "printCompareClasses" instead.')
                return

            loadFolder = loadFolders[i]
            if loadFolder[-1] != '/':
                loadFolder += '/'
            for model in ModelsList[i]:
                rootdir = loadFolder + model + '/1_Folds'
                for _, dirs, _ in os.walk(rootdir):
                    if model == 'KNN':
                        for nNeighbourDir in dirs:
                            self.appendResults(ExperimentResults[i], rootdir + '/' + nNeighbourDir + '/BoW')
                            self.appendResults(ExperimentResults[i], rootdir + '/' + nNeighbourDir + '/TFIDF')
                    else:
                        for embeddedDimsDir in dirs:
                            embeddedDimsRoot = rootdir + '/' + embeddedDimsDir
                            for _, dirs, _ in os.walk(embeddedDimsRoot):
                                for epochDir in dirs:
                                    epochRoot = embeddedDimsRoot + '/' + epochDir
                                    for _, dirs, _ in os.walk(epochRoot):
                                        for batchDir in dirs:
                                            batchRoot = epochRoot + '/' + batchDir
                                            for _, dirs, _ in os.walk(batchRoot):
                                                for sentenceDir in dirs:
                                                    sentenceRoot = batchRoot + '/' + sentenceDir
                                                    self.appendResults(ExperimentResults[i], sentenceRoot)
    
        # Generate a list of labels from the first Result
        labelsAvailable = [label for label in ExperimentResults[0][0].keys() if 'model' not in label and label not in ['micro avg','macro avg','weighted avg']]
        
        # Iterate over each class in each dict, append to the class index as a tuple: the parameter value (0) and the Precision/Recall (1) 
        xLabels = [[j for j in labelsAvailable] for i in loadFolders]
        yPrecision = [[[] for j in labelsAvailable] for i in loadFolders]
        yRecall = [[[] for j in labelsAvailable] for i in loadFolders]
        for i in range(0,len(loadFolders)):
            Results = loadFolders[i]
            for Result in Results:
                for key in Result.keys():
                    if key in labelsAvailable:
                        yPrecision[i][labelsAvailable.index(key)].append(Result[key]['precision'])
                        yRecall[i][labelsAvailable.index(key)].append(Result[key]['recall'])

        # Average all results within each class        
        yPrecisionMean = [[np.mean(i) * 100 for i in j] for j in yPrecision]
        yRecallMean = [[np.mean(i) * 100 for i in j] for j in yRecall]
        yPrecisionErr = [[np.std(i) * 100 for i in j] for j in yPrecision]
        yRecallErr = [[np.std(i) * 100 for i in j] for j in yRecall]

        precisionSaveRoot  = ''
        recallSaveRoot  = ''
        titleExtension = ''
        if byFolder == True and byModel == True:
            precisionSaveRoot = saveFolder + 'graphs/precision/comparison_multiplemodels_multiplefolders'
            recallSaveRoot = saveFolder + 'graphs/recall/comparison_multiplemodels_multiplefolders'
        elif byModel == True:
            precisionSaveRoot = saveFolder + 'graphs/precision/' + loadFolders[0].split('/')[-2] + '/comparison_' + loadFolders[0].split('/')[-2]
            recallSaveRoot = saveFolder + 'graphs/recall/' + loadFolders[0].split('/')[-2] + '/comparison_' + loadFolders[0].split('/')[-2]
            titleExtension = 'by model type'
        elif byFolder == True:
            precisionSaveRoot = saveFolder + 'graphs/precision/' + ModelsList[0][0] + '/comparison_' + ModelsList[0][0]
            recallSaveRoot = saveFolder + 'graphs/recall/' + ModelsList[0][0] + '/comparison_' + ModelsList[0][0]
            titleExtension = 'by dataset'

        # Create precision graph
        self.createPlots(xPlots=xLabels, yPlots=yPrecisionMean, yErrList = yPrecisionErr, plotLabels=plotLabels, xAxisName = 'Class', yAxisName = 'Precision (%)', legendTitle=titleExtension.replace('by ',''), title = 'Precision against Class ' + titleExtension, fileName = precisionSaveRoot + '_precision.png')
           
        # Create recall graph
        self.createPlots(xPlots=xLabels, yPlots=yRecallMean, yErrList = yRecallErr, plotLabels=plotLabels, xAxisName = 'Class', yAxisName = 'Recall (%)', legendTitle=titleExtension.replace('by ',''), title = 'Recall against Class ' + titleExtension, fileName = recallSaveRoot + '_recall.png')

        if len(classList) > 0:  

            # Get data
            xFilteredLabels = [[j for j in classList[i]] for i in loadFolders]
            yFilteredPrecision = [[[] for j in classList[i]] for i in loadFolders]
            yFilteredRecall = [[[] for j in classList[i]] for i in loadFolders]   
            for i in range(0,len(loadFolders)):
                Results = loadFolders[i]
                for Result in Results:
                    for key in Result.keys():
                        if key in classList[i]:
                            yFilteredPrecision[i][classList[i].index(key)].append(Result[key]['precision'])
                            yFilteredRecall[i][classList[i].index(key)].append(Result[key]['recall'])
        
            # Average data
            yFilteredPrecisionMean = [[np.mean(i) * 100 for i in j] for j in yFilteredPrecision]
            yFilteredRecallMean = [[np.mean(i) * 100 for i in j] for j in yFilteredRecall]
            yFilteredPrecisionErr = [[np.std(i) * 100 for i in j] for j in yFilteredPrecision]
            yFilteredRecallErr = [[np.std(i) * 100 for i in j] for j in yFilteredRecall]

            # Create filtered precision graph 
            self.createPlots(xPlots=xFilteredLabels, yPlots=yFilteredPrecisionMean, yErrList = yFilteredPrecisionErr, plotLabels=plotLabels, xAxisName = 'Class', yAxisName = 'Precision (%)', legendTitle=titleExtension.replace('by ',''), title = 'Precision against Class (filtered) ' + titleExtension, fileName = precisionSaveRoot + '_precision_filtered.png')
            
            # Create filtered recall graph
            self.createPlots(xPlots=xFilteredLabels, yPlots=yFilteredRecallMean, yErrList = yFilteredRecallErr, plotLabels=plotLabels, xAxisName = 'Class', yAxisName = 'Recall (%)', legendTitle=titleExtension.replace('by ',''), title = 'Recall against Class (filtered) ' + titleExtension, fileName = recallSaveRoot + '_recall_filtered.png')
        return

    def printCompareClasses(self, Model, minimum=50, loadFolders = ['results/'], saveFolder='results/Global/'):
        if saveFolder[-1] != '/':
            saveFolder += '/'

        # Get pickled results Dicts for the specified model, limit using Bounds
        Results = []
        for loadFolder in loadFolders:
            if loadFolder[-1] != '/':
                loadFolder += '/'
            rootdir = loadFolder + Model + '/1_Folds'
            for _, dirs, _ in os.walk(rootdir):
                if Model == 'KNN':
                    for nNeighbourDir in dirs:
                        self.appendResults(Results, rootdir + '/' + nNeighbourDir + '/BoW')
                        self.appendResults(Results, rootdir + '/' + nNeighbourDir + '/TFIDF')
                else:
                    for embeddedDimsDir in dirs:
                        embeddedDimsRoot = rootdir + '/' + embeddedDimsDir
                        for _, dirs, _ in os.walk(embeddedDimsRoot):
                            for epochDir in dirs:
                                epochRoot = embeddedDimsRoot + '/' + epochDir
                                for _, dirs, _ in os.walk(epochRoot):
                                    for batchDir in dirs:
                                        batchRoot = epochRoot + '/' + batchDir
                                        for _, dirs, _ in os.walk(batchRoot):
                                            for sentenceDir in dirs:
                                                sentenceRoot = batchRoot + '/' + sentenceDir
                                                self.appendResults(Results, sentenceRoot)
    
        # Generate a list of labels from the first Result
        labelsAvailable = [label for label in Results[0].keys() if 'model' not in label and label not in ['micro avg','macro avg','weighted avg']]
        
        x = []
        yPrecision = []
        yRecall = []
        yPrecisionMean = []
        yRecallMean = []
        yPrecisionErr = []
        yRecallErr = []

        # Iterate over each class in each dict, append to the class index as a tuple: the parameter value (0) and the Precision/Recall (1) 
        for result in Results:
            for label in labelsAvailable: 
                if label not in x:
                    x.append(label)
                    yPrecision.append([])
                    yRecall.append([])
                if label in result.keys():
                    yPrecision[x.index(label)].append(result[label]['precision'])
                    yRecall[x.index(label)].append(result[label]['recall'])
        
        yPrecisionMean = [np.mean(i) * 100 for i in yPrecision]
        yRecallMean = [np.mean(i) * 100 for i in yRecall]
        yPrecisionErr = [np.std(i) * 100 for i in yPrecision]
        yRecallErr = [np.std(i) * 100 for i in yRecall]

        xPrecision = [xClass for _,xClass in sorted(zip(yPrecisionMean,x))]
        xRecall = [xClass for _,xClass in sorted(zip(yRecallMean,x))]

        ySortedPrecisionErr = [yErr for _,yErr in sorted(zip(yPrecisionMean,yPrecisionErr))]
        ySortedRecallErr = [yErr for _,yErr in sorted(zip(yRecallMean,yRecallErr))]

        ySortedPrecisionMean = sorted(yPrecisionMean)
        ySortedRecallMean = sorted(yRecallMean)

        precisionSaveRoot = saveFolder + 'graphs/precision/' + Model + '/compare_classes' + Model
        recallSaveRoot = saveFolder + 'graphs/recall/' + Model + '/compare_classes' + Model

        # Create precision graph
        self.createPlots(xPlots=[xPrecision], yPlots=[ySortedPrecisionMean], yErrList = [ySortedPrecisionErr], xAxisName = 'Class', yAxisName = 'Precision (%)', title = 'Precision against Class (' + Model + ')', fileName = precisionSaveRoot + '_precision.png')
               
        # Create recall graph
        self.createPlots(xPlots=[xRecall], yPlots=[ySortedRecallMean], yErrList = [ySortedRecallErr], xAxisName = 'Class', yAxisName = 'Recall (%)', title = 'Recall against Class (' + Model + ')', fileName = recallSaveRoot + '_recall.png')
        
        if minimum > 0:
            # Create filtered precision graph
            xFiltered = []
            yFilteredPrecisionMean = []
            yFilteredPrecisionErr = []

            for i in range(0,len(x)):
                if (ySortedPrecisionMean[i] + ySortedPrecisionErr[i]) > minimum:
                    xFiltered.append(xPrecision[i])
                    yFilteredPrecisionMean.append(ySortedPrecisionMean[i])
                    yFilteredPrecisionErr.append(ySortedPrecisionErr[i])
            self.createPlots(xPlots=[xFiltered], yPlots=[yFilteredPrecisionMean], yErrList = [yFilteredPrecisionErr], xAxisName = 'Class', yAxisName = 'Precision (%)', title = 'Precision (filtered) against Class (' + Model + ')', fileName = precisionSaveRoot + '_precision_Filtered.png')
        
            # Create filtered recall graph
            xFiltered = []
            yFilteredRecallMean = []
            yFilteredRecallErr = []

            for i in range(0,len(x)):
                if (ySortedRecallMean[i] + ySortedRecallErr[i]) > minimum:
                    xFiltered.append(xRecall[i])
                    yFilteredRecallMean.append(ySortedRecallMean[i])
                    yFilteredRecallErr.append(ySortedRecallErr[i])
            self.createPlots(xPlots=[xFiltered], yPlots=[yFilteredRecallMean], yErrList = [yFilteredRecallErr], xAxisName = 'Class', yAxisName = 'Recall (%)', title = 'Recall (filtered) against Class (' + Model + ')', fileName = recallSaveRoot + '_recall_Filtered.png')
        return
    
    def printParameterReports(self, Model, Parameter, Bounds = [], loadFolder = '', saveFolder = 'results/', classList = []):
        if loadFolder == '':
            loadFolder = saveFolder
            
        if loadFolder[-1] != '/':
            loadFolder += '/'
        
        if saveFolder[-1] != '/':
            saveFolder += '/'

        # Get pickled results Dicts for the specified model, limit using Bounds
        Results = []
        rootdir = loadFolder + Model + '/1_Folds'
        for _, dirs, _ in os.walk(rootdir):
            if Model == 'KNN':
                for nNeighbourDir in dirs:
                    n = nNeighbourDir.split('_')[0]
                    if Parameter != 'nNeighbour' or len(Bounds) == 0 or (Bounds[0] <= int(n) and int(n) < Bounds[1]):
                        if Parameter != 'embeddingMode' or len(Bounds) == 0 or 'BoW' in Bounds:
                            self.appendResults(Results, rootdir + '/' + nNeighbourDir + '/BoW')
                        if Parameter != 'embeddingMode' or len(Bounds) == 0 or 'TFIDF' in Bounds:
                            self.appendResults(Results, rootdir + '/' + nNeighbourDir + '/TFIDF')
            else:
                for embeddedDimsDir in dirs:
                    embeddedDimsRoot = rootdir + '/' + embeddedDimsDir
                    embeddedDims = embeddedDimsDir.split('_')[0]
                    if Parameter != 'embeddedDims' or len(Bounds) == 0 or (Bounds[0] <= int(embeddedDims) and int(embeddedDims) < Bounds[1]):
                        for _, dirs, _ in os.walk(embeddedDimsRoot):
                            for epochDir in dirs:
                                epochRoot = embeddedDimsRoot + '/' + epochDir
                                epochs = epochDir.split('_')[0]
                                if Parameter != 'epochs' or len(Bounds) == 0 or (Bounds[0] <= int(epochs) and int(epochs) < Bounds[1]):
                                    for _, dirs, _ in os.walk(epochRoot):
                                        for batchDir in dirs:
                                            batchRoot = epochRoot + '/' + batchDir
                                            batchSize = batchDir.split('_')[0]
                                            if Parameter != 'batchSize' or len(Bounds) == 0 or (Bounds[0] <= int(batchSize) and int(batchSize) < Bounds[1]):
                                                for _, dirs, _ in os.walk(batchRoot):
                                                    for sentenceDir in dirs:
                                                        sentenceRoot = batchRoot + '/' + sentenceDir
                                                        sentenceSize = batchDir.split('_')[0]
                                                        if Parameter != 'sentenceSize' or len(Bounds) == 0 or (Bounds[0] <= int(sentenceSize) and int(sentenceSize) < Bounds[1]):
                                                            self.appendResults(Results, sentenceRoot)
    
        # Generate a list of labels from the first Result
        labelsAvailable = [label for label in Results[0].keys() if 'model' not in label and label not in ['micro avg','macro avg','weighted avg']]

        x = []
        xFiltered = []
        yPrecision = []
        yRecall = []
        yFilteredPrecision = []
        yFilteredRecall = []
        yPrecisionMean = []
        yRecallMean = []
        yPrecisionErr = []
        yRecallErr = []
        yFilteredPrecisionMean = []
        yFilteredRecallMean = []
        yFilteredPrecisionErr = []
        yFilteredRecallErr = []

        # Iterate over each class in each dict, append to the class index as a tuple: the parameter value (0) and the Precision/Recall (1) 
        for result in Results:
            search = result['model' + Parameter]
            if search not in x:
                x.append(search)
                yPrecision.append([])
                yRecall.append([])
            if search not in xFiltered:
                xFiltered.append(search)
                yFilteredPrecision.append([])
                yFilteredRecall.append([])
            for key in result.keys():
                if key in labelsAvailable:
                    yPrecision[x.index(search)].append(result[key]['precision'])
                    yRecall[x.index(search)].append(result[key]['recall'])
                if key in classList:
                    yFilteredPrecision[xFiltered.index(search)].append(result[key]['precision'])
                    yFilteredRecall[xFiltered.index(search)].append(result[key]['recall'])
        
        yPrecisionMean = [np.mean(i) * 100 for i in yPrecision]
        yRecallMean = [np.mean(i) * 100 for i in yRecall]
        yPrecisionErr = [np.std(i) * 100 for i in yPrecision]
        yRecallErr = [np.std(i) * 100 for i in yRecall]
        
        yFilteredPrecisionMean = [np.mean(i) * 100 for i in yFilteredPrecision]
        yFilteredRecallMean = [np.mean(i) * 100 for i in yFilteredRecall]
        yFilteredPrecisionErr = [np.std(i) * 100 for i in yFilteredPrecision]
        yFilteredRecallErr = [np.std(i) * 100 for i in yFilteredRecall]

        precisionSaveRoot = saveFolder + 'graphs/precision/' + Model + '/parameterReport_' + Model + '_' + Parameter
        recallSaveRoot = saveFolder + 'graphs/recall/' + Model + '/parameterReport_' + Model + '_' + Parameter
        for Bound in Bounds:
            precisionSaveRoot += '_' + Bound
            recallSaveRoot += '_' + Bound

        # Create precision graph
        self.createPlots(xPlots=[x], yPlots=[yPrecisionMean], yErrList = [yPrecisionErr], xAxisName = Parameter, yAxisName = 'Precision (%)', title = 'Precision against ' + Parameter + ' (' + Model + ')', fileName = precisionSaveRoot + '_precision.png')
            
        # Create recall graph
        self.createPlots(xPlots=[x], yPlots=[yRecallMean], yErrList = [yRecallErr], xAxisName = Parameter, yAxisName = 'Recall (%)', title = 'Recall against ' + Parameter + ' (' + Model + ')', fileName = recallSaveRoot + '_recall.png')
            
        if len(classList) > 0:     
            # Create filtered precision graph 
            self.createPlots(xPlots=[xFiltered], yPlots=[yFilteredPrecisionMean], yErrList = [yFilteredPrecisionErr], xAxisName = Parameter, yAxisName = 'Precision (%)', title = 'Precision (filtered) against ' + Parameter + ' (' + Model + ')', fileName = precisionSaveRoot + '_precision_filtered.png')
            
            # Create filtered recall graph
            self.createPlots(xPlots=[xFiltered], yPlots=[yFilteredRecallMean], yErrList = [yFilteredRecallErr], xAxisName = Parameter, yAxisName = 'Recall (%)', title = 'Recall (filtered) against ' + Parameter + ' (' + Model + ')', fileName = recallSaveRoot + '_recall_filtered.png')
            
        return
    
    def appendResults(self, ResultsList, directoryPath):
        for _, _, files in os.walk(directoryPath):
            for pickleFile in files:
                if '.pickle' in pickleFile:
                    with open(directoryPath + '/' + pickleFile, 'rb') as pikl:
                        ResultsList.append(pickle.load(pikl))
    
    def createPlots(self, xPlots, yPlots, yErrList = [], plotLabels=[], xAxisName = '', yAxisName = '', legendTitle='', title = '', fileName = '', scatter=False):
        
        rotation = 'horizontal'
        logscale = False
        xText = False

        for xPlot in xPlots:
            if len(xPlot) > 10:
                rotation = 'vertical'
                break

        for xPlot in xPlots:
            if type(xPlot[0]) is not str and len(xPlots) == 1 and min(xPlot) > 0:
                if max(xPlot) / min(xPlot) >= 100:
                    logscale = True
                    break
            else:
                xText = True
                break     

        scatterShapes = ['o','s','D','v','^','h','p','d','<','>']
        colors = {} 
        colors['Maroon'] = '#800000'
        colors['Red'] = '#e6194B'
        colors['Orange'] = '#f58231'
        colors['Yellow'] = '#ffe119'
        colors['Green'] = '#bfef45'
        colors['Mint'] = '#aaffc3'
        colors['Cyan'] = '#42d4f4'
        colors['Blue'] = '#4363d8'
        colors['Navy'] = '#000075'
        colors['Black'] = '#000000'
        edgecolors = ['#000000']
        if len(plotLabels) == 0:
            edgecolors = [colors['Blue']]
        elif len(plotLabels) == 2:
            edgecolors = [colors['Red'],colors['Blue']]
        elif len(plotLabels) == 3:
            edgecolors = [colors['Red'],colors['Yellow'],colors['Blue']]
        elif len(plotLabels) == 4:
            edgecolors = [colors['Red'],colors['Yellow'],colors['Green'],colors['Blue']]
        elif len(plotLabels) == 5:
            edgecolors = [colors['Red'],colors['Orange'],colors['Yellow'],colors['Green'],colors['Blue']]
        elif len(plotLabels) == 6:
            edgecolors = [colors['Red'],colors['Orange'],colors['Yellow'],colors['Green'],colors['Mint'],colors['Blue']]
        elif len(plotLabels) == 7:
            edgecolors = [colors['Red'],colors['Orange'],colors['Yellow'],colors['Green'],colors['Mint'],colors['Cyan'],colors['Blue']]
        elif len(plotLabels) == 8:
            edgecolors = [colors['Maroon'],colors['Red'],colors['Orange'],colors['Yellow'],colors['Green'],colors['Mint'],colors['Cyan'],colors['Blue']]
        elif len(plotLabels) == 9:
            edgecolors = [colors['Maroon'],colors['Red'],colors['Orange'],colors['Yellow'],colors['Green'],colors['Mint'],colors['Cyan'],colors['Blue'],colors['Navy']]
        elif len(plotLabels) == 10:
            edgecolors = [colors['Maroon'],colors['Red'],colors['Orange'],colors['Yellow'],colors['Green'],colors['Mint'],colors['Cyan'],colors['Blue'],colors['Navy'],colors['Black']]
        else:
            print('There must be between two and ten plot labels')
            return

        plt.figure()   
        plt.ylim(0,105)
        if len(plotLabels) == 0:
            if scatter == True:
                if logscale == True:
                    plt.scatter([math.log(i) for i in xPlots[0]], yPlots[0], alpha=0.6, marker=scatterShapes[0])
                else:
                    plt.scatter(xPlots[0], yPlots[0], alpha=0.6, marker=scatterShapes[0])
            else:
                if logscale == True:
                    plt.errorbar([math.log(i) for i in xPlots[0]], yPlots[0], yerr=yErrList[0], fmt='bo', capsize=5)
                else:
                    plt.errorbar(xPlots[0], yPlots[0], yerr=yErrList[0], fmt='bo', capsize=5)
        else:
            for i in range(0,len(plotLabels)):
                if scatter == True:
                    plt.scatter(xPlots[i], yPlots[i], label=plotLabels[i], marker=scatterShapes[i], edgecolors=edgecolors[i], facecolors='none',alpha=0.6)
                else:
                    plt.errorbar(xPlots[i], yPlots[i], yerr=yErrList[i], fmt='bo', capsize=5, label=plotLabels[i])
            plt.legend(title=legendTitle)
        if logscale == True:
            plt.xticks(ticks=[math.log(i) for i in xPlots[0]],labels=xPlots[0],rotation=rotation)
        elif xText == True:     
            plt.xticks(rotation='vertical', fontsize=8)
            plt.subplots_adjust(bottom=0.3)
        else:
            plt.xticks(ticks=xPlots[0],labels=xPlots[0],rotation=rotation)
        if rotation == 'vertical':
            plt.subplots_adjust(bottom=0.3)
        plt.xlabel(xAxisName)
        plt.ylabel(yAxisName)  
        plt.grid(b=True, which='major', color='#666666', linestyle='-')
        plt.title(title)

        createFilePath = ''
        for folder in fileName.split('/'):
            if folder != '' and '.png' not in folder:
                createFilePath += folder + '/'
                if not os.path.exists(createFilePath):
                    os.mkdir(createFilePath)

        plt.savefig(fileName)

        for i in plt.get_fignums():
            plt.close(i)
        return

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
