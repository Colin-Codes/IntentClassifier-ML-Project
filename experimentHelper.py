from modelHelper import trainDeepModel, trainShallowModel
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from dataHelper import Data
from datetime import datetime
from matplotlib import cm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import math
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

    def printCompareParameterClasses(self, Model, classList=[], parameter='threshold', saveFolder='results/Global/'):
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
        parameterRange = [Results['model' + parameter] for Results in ExperimentResults]

        # Iterate over each class in each dict, append to the class index as a tuple: the parameter value (0) and the Precision/Recall (1) 
        xLabels = [[[] for j in labelsAvailable] for i in parameterRange]
        yPrecision = [[[] for j in labelsAvailable] for i in parameterRange]
        yRecall = [[[] for j in labelsAvailable] for i in parameterRange]
        xFilteredLabels = [[j for j in classList[i]] for i in parameterRange]
        yFilteredPrecision = [[[] for j in classList[i]] for i in parameterRange]
        yFilteredRecall = [[[] for j in classList[i]] for i in parameterRange]
        for i in parameterRange:
            for Results in ExperimentResults[i]:
                for Result in Results:
                    if Result['model' + parameter] == i:
                        for key in Result.keys():
                            if key in labelsAvailable:
                                yPrecision[i][labelsAvailable.index(key)].append(Result[key]['precision'])
                                yRecall[i][labelsAvailable.index(key)].append(Result[key]['recall'])
                            if key in classList[i]:
                                yFilteredPrecision[i][classList[i].index(key)].append(Result[key]['precision'])
                                yFilteredRecall[i][classList[i].index(key)].append(Result[key]['recall'])

        # Average all results within each class
        yPrecisionMean = []
        yRecallMean = []
        yPrecisionErr = []
        yRecallErr = []
        
        yPrecisionMean = [[np.mean(i) for i in j] for j in yPrecision]
        yRecallMean = [[np.mean(i) for i in j] for j in yRecall]
        yPrecisionErr = [[np.std(i) for i in j] for j in yPrecision]
        yRecallErr = [[np.std(i) for i in j] for j in yRecall]
        
        yFilteredPrecisionMean = [[np.mean(i) for i in j] for j in yFilteredPrecision]
        yFilteredRecallMean = [[np.mean(i) for i in j] for j in yFilteredRecall]
        yFilteredPrecisionErr = [[np.std(i) for i in j] for j in yFilteredPrecision]
        yFilteredRecallErr = [[np.std(i) for i in j] for j in yFilteredRecall]

        saveRoot  = ''
        titleExtension = ''
        saveRoot = saveFolder + 'graphs/comparison_' + parameter
        titleExtension = ' split by ' + parameter

        # Create precision graph
        self.createErrorPlots(xPlots=xLabels, yPlots=yPrecisionMean, yErrList = yPrecisionErr, xAxisName = 'Class', yAxisName = 'Precision (%)', title = 'Precision against Class ' + titleExtension, FileName = saveRoot + '_precision.png')
           
        # Create recall graph
        self.createErrorPlots(xPlots=xLabels, yPlots=yRecallMean, yErrList = yRecallErr, xAxisName = 'Class', yAxisName = 'Recall (%)', title = 'Recall against Class ' + titleExtension, FileName = saveRoot + '_recall.png')

        if len(classList) > 0:     
            # Create filtered precision graph 
            self.createErrorPlots(xPlots=xFilteredLabels, yPlots=yFilteredPrecisionMean, yErrList = yFilteredPrecisionErr, xAxisName = 'Class', yAxisName = 'Precision (%)', title = 'Precision against Class (filtered) ' + titleExtension, FileName = saveRoot + '_precision_filtered.png')
            
            # Create filtered recall graph
            self.createErrorPlots(xPlots=xFilteredLabels, yPlots=yFilteredRecallMean, yErrList = yFilteredRecallErr, xAxisName = 'Class', yAxisName = 'Recall (%)', title = 'Recall against Class (filtered) ' + titleExtension, FileName = saveRoot + '_recall_filtered.png')
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
            if len(ModelsList) > 0:
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
                plotLabels.append(Models[i[0]] + '-' + loadFolders[i].split('/')[-2])
            elif byModel == True:
                plotLabels.append(Models[i[0]])
            elif byFolder == True:
                plotLabels.append(loadFolders[i].split('/')[-2])
            else:
                print('"printCompareExperimentsClasses" is for comparing different models and or experiments, please use "printClassParameterReports" or "printCompareClasses" instead.')
                return

            loadFolder = loadFolders[i]
            if loadFolder[-1] != '/':
                loadFolder += '/'
            for model in Models[i]:
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
        xFilteredLabels = [[j for j in classList[i]] for i in loadFolders]
        yFilteredPrecision = [[[] for j in classList[i]] for i in loadFolders]
        yFilteredRecall = [[[] for j in classList[i]] for i in loadFolders]
        for i in range(0,len(loadFolders)):
            Results = loadFolders[i]
            for Result in Results:
                for key in Result.keys():
                    if key in labelsAvailable:
                        xLabels[i][labelsAvailable.index(key)].append(key)
                        yPrecision[i][labelsAvailable.index(key)].append(Result[key]['precision'])
                        yRecall[i][labelsAvailable.index(key)].append(Result[key]['recall'])
                    if key in classList[i]:
                        xFilteredLabels[i][classList[i].index(key)].append(key)
                        yFilteredPrecision[i][classList[i].index(key)].append(Result[key]['precision'])
                        yFilteredRecall[i][classList[i].index(key)].append(Result[key]['recall'])

        # Average all results within each class
        yPrecisionMean = []
        yRecallMean = []
        yPrecisionErr = []
        yRecallErr = []
        
        yPrecisionMean = [[np.mean(i) for i in j] for j in yPrecision]
        yRecallMean = [[np.mean(i) for i in j] for j in yRecall]
        yPrecisionErr = [[np.std(i) for i in j] for j in yPrecision]
        yRecallErr = [[np.std(i) for i in j] for j in yRecall]
        
        yFilteredPrecisionMean = [[np.mean(i) for i in j] for j in yFilteredPrecision]
        yFilteredRecallMean = [[np.mean(i) for i in j] for j in yFilteredRecall]
        yFilteredPrecisionErr = [[np.std(i) for i in j] for j in yFilteredPrecision]
        yFilteredRecallErr = [[np.std(i) for i in j] for j in yFilteredRecall]

        saveRoot  = ''
        titleExtension = ''
        if byFolder == True and byModel == True:
            saveRoot = saveFolder + 'graphs/comparison_multiplemodels_multiplefolders'
        elif byModel == True:
            saveRoot = saveFolder + 'graphs/comparison_' + loadFolders[0].split('/')[-2]
            titleExtension = 'by model type'
        elif byFolder == True:
            saveRoot = saveFolder + 'graphs/comparison_' + Models[0][0]
            titleExtension = 'by dataset'

        # Create precision graph
        self.createErrorPlots(xPlots=xLabels, yPlots=yPrecisionMean, yErrList = yPrecisionErr, xAxisName = 'Class', yAxisName = 'Precision (%)', title = 'Precision against Class ' + titleExtension, FileName = saveRoot + '_precision.png')
           
        # Create recall graph
        self.createErrorPlots(xPlots=xLabels, yPlots=yRecallMean, yErrList = yRecallErr, xAxisName = 'Class', yAxisName = 'Recall (%)', title = 'Recall against Class ' + titleExtension, FileName = saveRoot + '_recall.png')

        if len(classList) > 0:     
            # Create filtered precision graph 
            self.createErrorPlots(xPlots=xFilteredLabels, yPlots=yFilteredPrecisionMean, yErrList = yFilteredPrecisionErr, xAxisName = 'Class', yAxisName = 'Precision (%)', title = 'Precision against Class (filtered) ' + titleExtension, FileName = saveRoot + '_precision_filtered.png')
            
            # Create filtered recall graph
            self.createErrorPlots(xPlots=xFilteredLabels, yPlots=yFilteredRecallMean, yErrList = yFilteredRecallErr, xAxisName = 'Class', yAxisName = 'Recall (%)', title = 'Recall against Class (filtered) ' + titleExtension, FileName = saveRoot + '_recall_filtered.png')
        return

    def printCompareClasses(self, Model, minimum=0.5, loadFolders = ['results/'], saveFolder='results/Global/'):
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

        saveRoot = saveFolder + 'graphs/compare_classes' + Model

        # Create precision graph
        self.createErrorPlots(xPlots=[xPrecision], yPlots=[ySortedPrecisionMean], yErrList = [ySortedPrecisionErr], xAxisName = 'Class', yAxisName = 'Precision (%)', title = 'Precision against Class (' + Model + ')', FileName = saveRoot + '_precision.png')
               
        # Create recall graph
        self.createErrorPlots(xPlots=[xRecall], yPlots=[ySortedRecallMean], yErrList = [ySortedRecallErr], xAxisName = 'Class', yAxisName = 'Recall (%)', title = 'Recall against Class (' + Model + ')', FileName = saveRoot + '_recall.png')
        
        if minimum > 0:
            # Create filtered precision graph
            xFiltered = []
            yFilteredPrecisionMean = []
            yFilteredPrecisionErr = []

            for i in range(0,len(x)):
                if (ySortedPrecisionMean[i] + ySortedPrecisionErr[i]) > minimum:
                    xFiltered.append(x[i])
                    yFilteredPrecisionMean.append(ySortedPrecisionMean[i])
                    yFilteredPrecisionErr.append(ySortedPrecisionErr[i])
            self.createErrorPlots(xPlots=[xFiltered], yPlots=[yFilteredPrecisionMean], yErrList = [yFilteredPrecisionErr], xAxisName = 'Class', yAxisName = 'Precision (%)', title = 'Precision against Class (' + Model + ')', FileName = saveRoot + '_precision_Filtered.png')
        
            # Create filtered recall graph
            xFiltered = []
            yFilteredRecallMean = []
            yFilteredRecallErr = []

            for i in range(0,len(x)):
                if (ySortedRecallMean[i] + ySortedRecallErr[i]) > minimum:
                    xFiltered.append(x[i])
                    yFilteredRecallMean.append(ySortedRecallMean[i])
                    yFilteredRecallErr.append(ySortedRecallErr[i])
            self.createErrorPlots(xPlots=[xFiltered], yPlots=[yFilteredRecallMean], yErrList = [yFilteredRecallErr], xAxisName = 'Class', yAxisName = 'Recall (%)', title = 'Recall against Class (' + Model + ')', FileName = saveRoot + '_recall_Filtered.png')
        return
    
    def printClassParameterReports(self, Model, Parameter, Bounds = [], loadFolder = '', saveFolder = 'results/', classList = []):
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

        saveRoot = saveFolder + 'graphs/class_parameter_' + Model + '_' + Parameter
        for Bound in Bounds:
            saveRoot += '_' + Bound

        # Create precision graph
        self.createErrorPlots(xPlots=[x], yPlots=[yPrecisionMean], yErrList = [yPrecisionErr], xAxisName = Parameter, yAxisName = 'Precision (%)', title = 'Precision against ' + Parameter + ' (' + Model + ')', FileName = saveRoot + '_precision.png')
            
        # Create recall graph
        self.createErrorPlots(xPlots=[x], yPlots=[yRecallMean], yErrList = [yRecallErr], xAxisName = Parameter, yAxisName = 'Recall (%)', title = 'Recall against ' + Parameter + ' (' + Model + ')', FileName = saveRoot + '_recall.png')
            
        if len(classList) > 0:     
            # Create filtered precision graph 
            self.createErrorPlots(xPlots=[xFiltered], yPlots=[yFilteredPrecisionMean], yErrList = [yFilteredPrecisionErr], xAxisName = Parameter, yAxisName = 'Precision (%)', title = 'Precision (filtered) against ' + Parameter + ' (' + Model + ')', FileName = saveRoot + '_precision_filtered.png')
            
            # Create filtered recall graph
            self.createErrorPlots(xPlots=[xFiltered], yPlots=[yFilteredRecallMean], yErrList = [yFilteredRecallErr], xAxisName = Parameter, yAxisName = 'Recall (%)', title = 'Recall (filtered) against ' + Parameter + ' (' + Model + ')', FileName = saveRoot + '_recall_filtered.png')
            
        return
    
    def appendResults(self, ResultsList, directoryPath):
        for _, _, files in os.walk(directoryPath):
            for pickleFile in files:
                if '.pickle' in pickleFile:
                    with open(directoryPath + '/' + pickleFile, 'rb') as pikl:
                        ResultsList.append(pickle.load(pikl))
    
    def createErrorPlots(self, xPlots, yPlots, xLabelsList = [], yErrList = [], plotLabels=[], xAxisName = '', yAxisName = '', title = '', FileName = ''):
        
        rotation = 'horizontal'
        logscale = False
        xText = False

        for xPlot in xPlots:
            if len(xPlot) > 10:
                rotation = 'vertical'
                break

        for xPlot in xPlots:
            if type(xPlot[0]) is not str:
                if max(xPlot) / min(min(xPlot),1) >= 100:
                    logscale = True
                    break
            else:
                xText = True
                break            

        plt.figure()   
        if len(plotLabels) == 0:
            plt.errorbar(xPlots[0], yPlots[0], yerr=yErrList[0], fmt='bo', capsize=5)
        else:
            for i in range(0,len(plotLabels)):
                plt.errorbar(xPlots[i], yPlots[i], yerr=yErrList[i], fmt='bo', capsize=5, label=plotLabels[i])
            plt.legend()
        if logscale == True:
            xLabels = plt.xticks()
            xTicks = [math.log(i) for i in xTicks]
            plt.xticks(ticks=xTicks, labels=xLabels, rotation=rotation)
        elif xText == True:             
            plt.xticks(rotation='vertical', fontsize=8)
            plt.subplots_adjust(bottom=0.2)
        else:
            plt.xticks(rotation=rotation)
        plt.xlabel(xAxisName)
        plt.ylabel(yAxisName)        
        plt.grid(b=True, which='major', color='#666666', linestyle='-')
        plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.3)
        plt.title(title)

        createFilePath = ''
        for folder in FileName.split('/'):
            if folder != '' and '.png' not in folder:
                createFilePath += folder + '/'
                if not os.path.exists(createFilePath):
                    os.mkdir(createFilePath)

        plt.savefig(FileName)

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
