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

    def printCompareExperiments(self, Models=[''], classList=[], loadFolders = ['results/'], experimentLabels=['results'], saveFolder='results/Global/'):
        ### This function is useful for comparing the results of two or more experiments, eg. different datasets

        if saveFolder[-1] != '/':
            saveFolder += '/'
        
        if len(loadFolders) != len(experimentLabels):
            print('Insufficient labels provided!')
            return

        # Get pickled results Dicts for the specified model, limit using Bounds
        ExperimentResults = [[] for i in loadFolders]
        for i in range(0,len(loadFolders)):
            loadFolder = loadFolders[i]
            if loadFolder[-1] != '/':
                loadFolder += '/'
            for model in Models:
                rootdir = loadFolder + model + '/1_Folds'
                for root, dirs, files in os.walk(rootdir):
                    if model == 'KNN':
                        for nNeighbourDir in dirs:
                            self.appendResults(ExperimentResults[i], rootdir + '/' + nNeighbourDir + '/BoW')
                            self.appendResults(ExperimentResults[i], rootdir + '/' + nNeighbourDir + '/TFIDF')
                    else:
                        for embeddedDimsDir in dirs:
                            embeddedDimsRoot = rootdir + '/' + embeddedDimsDir
                            for root, dirs, files in os.walk(embeddedDimsRoot):
                                for epochDir in dirs:
                                    epochRoot = embeddedDimsRoot + '/' + epochDir
                                    for root, dirs, files in os.walk(epochRoot):
                                        for batchDir in dirs:
                                            batchRoot = epochRoot + '/' + batchDir
                                            for root, dirs, files in os.walk(batchRoot):
                                                for sentenceDir in dirs:
                                                    sentenceRoot = batchRoot + '/' + sentenceDir
                                                    for root, dirs, files in os.walk(sentenceRoot):
                                                        self.appendResults(ExperimentResults[i], sentenceRoot)
    
        # Generate a list of labels from the first Result
        labelsAvailable = [label for label in ExperimentResults[0][0].keys() if 'model' not in label and label not in ['micro avg','macro avg','weighted avg']]
        
        x = []
        yPrecision = [[] for i in loadFolders]
        yRecall = [[] for i in loadFolders]
        yFilteredPrecision = [[] for i in loadFolders]
        yFilteredRecall = [[] for i in loadFolders]
        yPrecisionMean = []
        yRecallMean = []
        yPrecisionErr = []
        yRecallErr = []

        # Iterate over each class in each dict, append to the class index as a tuple: the parameter value (0) and the Precision/Recall (1) 
        for i in range(0,len(loadFolders)):
            Results = loadFolders[i]
            for Result in Results:
                for key in Result.keys():
                    if key in labelsAvailable:
                        yPrecision[i].append(Result[key]['precision'])
                        yRecall[i].append(Result[key]['recall'])
                    if key in classList:
                        yFilteredPrecision[i].append(Result[key]['precision'])
                        yFilteredRecall[i].append(Result[key]['recall'])
        
        yPrecisionMean = [np.mean(i) for i in yPrecision]
        yRecallMean = [np.mean(i) for i in yRecall]
        yPrecisionErr = [np.std(i) for i in yPrecision]
        yRecallErr = [np.std(i) for i in yRecall]
        
        yFilteredPrecisionMean = [np.mean(i) for i in yFilteredPrecision]
        yFilteredRecallMean = [np.mean(i) for i in yFilteredRecall]
        yFilteredPrecisionErr = [np.std(i) for i in yFilteredPrecision]
        yFilteredRecallErr = [np.std(i) for i in yFilteredRecall]

        rotation=''
        if len(experimentLabels) > 4:
            rotation = 'vertical'
        else:
            rotation = 'horizontal'

        modelTitle = ''
        for model in Models:
            modelTitle += model + ', '
        modelTitle = modelTitle[:-2]

        # Create precision graph
        plt.figure()
        plt.errorbar(experimentLabels, yPrecisionMean, yerr=yPrecisionErr, fmt='bo', capsize=5)
        plt.xticks(rotation=rotation)
        plt.title('Precision between experiments (' + modelTitle + ')')
        if not os.path.exists(saveFolder):
            os.mkdir(saveFolder)
        createFilePath = saveFolder + 'graphs/'
        if not os.path.exists(createFilePath):
            os.mkdir(createFilePath)
        createFilePath += 'precision/'
        if not os.path.exists(createFilePath):
            os.mkdir(createFilePath)
        saveRoot = createFilePath + '_Compare' + '_' + modelTitle.replace(', ','_') + '_recall'
        plt.savefig(saveRoot + '.png')
        
        # Create recall graph
        plt.figure()
        plt.errorbar(experimentLabels, yRecallMean, yerr=yRecallErr, fmt='bo', capsize=5)
        plt.xticks(rotation=rotation)
        plt.title('Recall between experiments (' + modelTitle + ')')
        if not os.path.exists(saveFolder):
            os.mkdir(saveFolder)
        createFilePath = saveFolder + 'graphs/'
        if not os.path.exists(createFilePath):
            os.mkdir(createFilePath)
        createFilePath += 'recall/'
        if not os.path.exists(createFilePath):
            os.mkdir(createFilePath)
        saveRoot = createFilePath + '_Compare' + '_' + modelTitle.replace(', ','_') + '_recall'
        plt.savefig(saveRoot + '.png')

        if len(classList) > 0:     
            # Create filtered precision graph 
            plt.figure()
            plt.errorbar(experimentLabels, yFilteredPrecisionMean, yerr=yFilteredPrecisionErr, fmt='bo', capsize=5)
            plt.xticks(rotation=rotation)
            plt.title('Precision (filtered) between experiments (' + modelTitle + ')')
            if not os.path.exists(saveFolder):
                os.mkdir(saveFolder)
            createFilePath = saveFolder + 'graphs/'
            if not os.path.exists(createFilePath):
                os.mkdir(createFilePath)
            createFilePath += 'precision/'
            if not os.path.exists(createFilePath):
                os.mkdir(createFilePath)
            saveRoot = createFilePath + '_Compare' + '_' + modelTitle.replace(', ','_') + '_recall' + '_Filtered'
            plt.savefig(saveRoot + '.png')
            
            # Create filtered recall graph
            plt.figure()
            plt.errorbar(experimentLabels, yFilteredRecallMean, yerr=yFilteredRecallErr, fmt='bo', capsize=5)
            plt.xticks(rotation=rotation)
            plt.title('Recall (filtered) between experiments (' + modelTitle + ')')
            if not os.path.exists(saveFolder):
                os.mkdir(saveFolder)
            createFilePath = saveFolder + 'graphs/'
            if not os.path.exists(createFilePath):
                os.mkdir(createFilePath)
            createFilePath += 'recall/'
            if not os.path.exists(createFilePath):
                os.mkdir(createFilePath)
            saveRoot = createFilePath + '_Compare' + '_' + modelTitle.replace(', ','_') + '_recall' + '_Filtered'
            plt.savefig(saveRoot + '.png')

        for i in plt.get_fignums():
            plt.close(i)
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
            for root, dirs, files in os.walk(rootdir):
                if Model == 'KNN':
                    for nNeighbourDir in dirs:
                        n = nNeighbourDir.split('_')[0]
                        self.appendResults(Results, rootdir + '/' + nNeighbourDir + '/BoW')
                        self.appendResults(Results, rootdir + '/' + nNeighbourDir + '/TFIDF')
                else:
                    for embeddedDimsDir in dirs:
                        embeddedDimsRoot = rootdir + '/' + embeddedDimsDir
                        for root, dirs, files in os.walk(embeddedDimsRoot):
                            for epochDir in dirs:
                                epochRoot = embeddedDimsRoot + '/' + epochDir
                                for root, dirs, files in os.walk(epochRoot):
                                    for batchDir in dirs:
                                        batchRoot = epochRoot + '/' + batchDir
                                        for root, dirs, files in os.walk(batchRoot):
                                            for sentenceDir in dirs:
                                                sentenceRoot = batchRoot + '/' + sentenceDir
                                                for root, dirs, files in os.walk(sentenceRoot):
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
            precision = []
            recall = []
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

        # Create precision graph
        saveRoot = saveFolder + 'graphs/' + Model + '_precision_classes'
        self.createErrorPlots(xPlots=[xPrecision], yPlots=[ySortedPrecisionMean], yErrList = [ySortedPrecisionErr], xAxisName = 'Class', yAxisName = 'Precision (%)', title = 'Precision against Class for ' + Model, FileName = saveRoot + '.png')
               
        # Create recall graph
        saveRoot = saveFolder + 'graphs/' + Model + '_recall_classes'
        self.createErrorPlots(xPlots=[xRecall], yPlots=[ySortedRecallMean], yErrList = [ySortedRecallErr], xAxisName = 'Class', yAxisName = 'Recall (%)', title = 'Recall against Class for ' + Model, FileName = saveRoot + '.png')
        
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

            saveRoot = saveFolder + 'graphs/' + Model + '_precision_classes_Filtered'
            self.createErrorPlots(xPlots=[xFiltered], yPlots=[yFilteredPrecisionMean], yErrList = [yFilteredPrecisionErr], xAxisName = 'Class', yAxisName = 'Precision (%)', title = 'Precision against Class for ' + Model, FileName = saveRoot + '.png')
        
            # Create filtered recall graph
            xFiltered = []
            yFilteredRecallMean = []
            yFilteredRecallErr = []

            for i in range(0,len(x)):
                if (ySortedRecallMean[i] + ySortedRecallErr[i]) > minimum:
                    xFiltered.append(x[i])
                    yFilteredRecallMean.append(ySortedRecallMean[i])
                    yFilteredRecallErr.append(ySortedRecallErr[i])
                    
            saveRoot = saveFolder + 'graphs/' + Model + '_recall_classes_Filtered'
            self.createErrorPlots(xPlots=[xFiltered], yPlots=[yFilteredRecallMean], yErrList = [yFilteredRecallErr], xAxisName = 'Class', yAxisName = 'Recall (%)', title = 'Recall against Class for ' + Model, FileName = saveRoot + '.png')
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
        for root, dirs, files in os.walk(rootdir):
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
                        for root, dirs, files in os.walk(embeddedDimsRoot):
                            for epochDir in dirs:
                                epochRoot = embeddedDimsRoot + '/' + epochDir
                                epochs = epochDir.split('_')[0]
                                if Parameter != 'epochs' or len(Bounds) == 0 or (Bounds[0] <= int(epochs) and int(epochs) < Bounds[1]):
                                    for root, dirs, files in os.walk(epochRoot):
                                        for batchDir in dirs:
                                            batchRoot = epochRoot + '/' + batchDir
                                            batchSize = batchDir.split('_')[0]
                                            if Parameter != 'batchSize' or len(Bounds) == 0 or (Bounds[0] <= int(batchSize) and int(batchSize) < Bounds[1]):
                                                for root, dirs, files in os.walk(batchRoot):
                                                    for sentenceDir in dirs:
                                                        sentenceRoot = batchRoot + '/' + sentenceDir
                                                        sentenceSize = batchDir.split('_')[0]
                                                        if Parameter != 'sentenceSize' or len(Bounds) == 0 or (Bounds[0] <= int(sentenceSize) and int(sentenceSize) < Bounds[1]):
                                                            for root, dirs, files in os.walk(sentenceRoot):
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

        # Create precision graph
        saveRoot = saveFolder + 'graphs/precision_' + Model + '_' + Parameter
        for Bound in Bounds:
            saveRoot += '_' + Bound
        self.createErrorPlots(xPlots=[x], yPlots=[yPrecisionMean], yErrList = [yPrecisionErr], xAxisName = Parameter, yAxisName = 'Precision (%)', title = 'Precision against ' + Parameter + '(' + Model + ')', FileName = saveRoot + '.png')
            
        # Create recall graph
        saveRoot = saveFolder + 'graphs/recall_' + Model + '_' + Parameter
        for Bound in Bounds:
            saveRoot += '_' + Bound
        self.createErrorPlots(xPlots=[x], yPlots=[yRecallMean], yErrList = [yRecallErr], xAxisName = Parameter, yAxisName = 'Recall (%)', title = 'Recall against ' + Parameter + '(' + Model + ')', FileName = saveRoot + '.png')
            
        if len(classList) > 0:     
            # Create filtered precision graph 
            saveRoot = saveFolder + 'graphs/precision_' + Model + '_' + Parameter + '_Filtered'
            for Bound in Bounds:
                saveRoot += '_' + Bound
            self.createErrorPlots(xPlots=[xFiltered], yPlots=[yFilteredPrecisionMean], yErrList = [yFilteredPrecisionErr], xAxisName = Parameter, yAxisName = 'Precision (%)', title = 'Precision (filtered) against ' + Parameter + '(' + Model + ')', FileName = saveRoot + '.png')
            
            # Create filtered recall graph
            saveRoot = saveFolder + 'graphs/recall_' + Model + '_' + Parameter + '_Filtered'
            for Bound in Bounds:
                saveRoot += '_' + Bound
            self.createErrorPlots(xPlots=[xFiltered], yPlots=[yFilteredRecallMean], yErrList = [yFilteredRecallErr], xAxisName = Parameter, yAxisName = 'Recall (%)', title = 'Recall (filtered) against ' + Parameter + '(' + Model + ')', FileName = saveRoot + '.png')
            
        return
    
    def appendResults(self, ResultsList, directoryPath):
        for root, dirs, files in os.walk(directoryPath):
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
