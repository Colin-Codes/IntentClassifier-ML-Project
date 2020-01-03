from experimentHelper import Experiment
from experimentHelper import Experiments
import copy

experiments = []
# Examples
# experiments.append(Experiment(_modelName="RNN", _kFolds=1, _embeddedDims=300, _epochs=1, _batchSize = 32, _sentenceSize=30, threshold = 0.0, _trainFilePath='data/trainingSet_augmented.csv', _testFilePath='data/testSet.csv'))
# experiments.append(Experiment(_modelName="GRU", _kFolds=1, _embeddedDims=300, _epochs=1, _batchSize = 32, _sentenceSize=30, threshold = 0.0, _trainFilePath='data/trainingSet_augmented.csv', _testFilePath='data/testSet.csv'))
# experiments.append(Experiment(_modelName="LSTM", _kFolds=1, _embeddedDims=300, _epochs=1, _batchSize = 32, _sentenceSize=20, threshold = 0.0, _trainFilePath='data/trainingSet_augmented.csv', _testFilePath='data/testSet.csv'))
# experiments.append(Experiment(_modelName="KNN", _kFolds=1, _nNeighbours=5, _embeddingMode='TFIDF', threshold = 0.0, _trainFilePath='data/trainingSet_augmented.csv', _testFilePath='data/testSet.csv'))

# testRun = Experiments(experiments)
# testRun.showClassResults('KNN/1_Folds/2_Neighbours/BoW/20191228_120212.pickle')

# augmentedSaveFolder='results/Augmented/'

# # KNN general experiment
# for i in range(0, 30):
#     print('Experiment: ' + str(i + 1))
#     experiments = []
#     for n in range(1, 16):
#         experiments.append(Experiment(_modelName="KNN", _kFolds=1, _nNeighbours=n, _embeddingMode='TFIDF', threshold = 0.0, _trainFilePath='data/trainingSet_augmented.csv', _testFilePath='data/testSet.csv'))
#         experiments.append(Experiment(_modelName="KNN", _kFolds=1, _nNeighbours=n, _embeddingMode='BoW', threshold = 0.0, _trainFilePath='data/trainingSet_augmented.csv', _testFilePath='data/testSet.csv'))
#     testRun = Experiments(experiments)
#     testRun.run()
#     testRun.printClassResults(saveFolder='results/Augmented/KNN')
# print('Experiment complete')

# KNN general results
# testRun = Experiments(experiments)
# for i in ('BoW','TFIDF'):
#     testRun.printClassParameterReports('KNN', 'nNeighbours',saveFolder='results/Augmented/KNN/',Bounds=[i])

# # Neural Network general experiment
# for i in range(0, 30):
#     print('Experiment: ' + str(i + 1))
#     experiments = []
#     for Model in ('RNN','LSTM','GRU'):
#         for Dims in range(100, 1001, 100):
#             for SentenceSize in (10, 15, 20, 25):
#                 for Epochs in range(1, 6):
#                     for Batch in (32,64,128,37250):
#                         experiments = []
#                         print('Experiment: ' + str(i + 1) + ' Model: ' + Model + ' Dimensions: ' + str(Dims) + ' Epochs: ' + str(Epochs) + ' Batch: ' + str(Batch) + ' Sentence Size: ' + str(SentenceSize))
#                         experiments.append(Experiment(_modelName=Model, _kFolds=1, _embeddedDims=Dims, _epochs=Epochs, _batchSize = Batch, _sentenceSize=SentenceSize, threshold = 0.0, _trainFilePath='data/trainingSet_augmented.csv', _testFilePath='data/testSet.csv'))
#                         testRun = Experiments(experiments)
#                         testRun.run()
#                         testRun.printClassResults(saveFolder='results/Augmented/Exploratory')
# print('Experiment complete')

# # Neural Network Epochs - Batch Size experiment
# for i in range(0, 1):
#     print('Experiment: ' + str(i + 1))
#     experiments = []
#     for Model in ('RNN','LSTM','GRU'):
#         #for Dims in range(100, 1001, 100):
#         Dims = 300
#         #for SentenceSize in (10, 15, 20, 25):
#         SentenceSize = 20
#         for Epochs in (1,3,5):
#             for Batch in (32,64,128,37250):
#                 experiments = []
#                 print('Experiment: ' + str(i + 1) + ' Model: ' + Model + ' Dimensions: ' + str(Dims) + ' Epochs: ' + str(Epochs) + ' Batch: ' + str(Batch) + ' Sentence Size: ' + str(SentenceSize))
#                 experiments.append(Experiment(_modelName=Model, _kFolds=1, _embeddedDims=Dims, _epochs=Epochs, _batchSize = Batch, _sentenceSize=SentenceSize, threshold = 0.0, _trainFilePath='data/trainingSet_augmented.csv', _testFilePath='data/testSet.csv'))
#                 testRun = Experiments(experiments)
#                 testRun.run()
#                 testRun.printClassResults(saveFolder='results/Augmented/BatchEpoch')
# print('Experiment complete')

# # Neural Network Epochs - Batch Size results
# testRun = Experiments(experiments)
# for i in ('RNN','GRU','LSTM'):
#     for j in ('epochs','batchSize'):
#         testRun.printClassParameterReports(i, j, saveFolder='results/Augmented/BatchEpoch')

# # Neural Network Dimensions experiment
# for i in range(0, 1):
#     print('Experiment: ' + str(i + 1))
#     experiments = []
#     for Model in ('RNN','LSTM','GRU'):
#         SentenceSize = 20
#         Epochs = 3
#         Batch = 32
#         for Dims in range(100, 1001, 100):
#             experiments = []
#             print('Experiment: ' + str(i + 1) + ' Model: ' + Model + ' Dimensions: ' + str(Dims) + ' Epochs: ' + str(Epochs) + ' Batch: ' + str(Batch) + ' Sentence Size: ' + str(SentenceSize))
#             experiments.append(Experiment(_modelName=Model, _kFolds=1, _embeddedDims=Dims, _epochs=Epochs, _batchSize = Batch, _sentenceSize=SentenceSize, threshold = 0.0, _trainFilePath='data/trainingSet_augmented.csv', _testFilePath='data/testSet.csv'))
#             testRun = Experiments(experiments)
#             testRun.run()
#             testRun.printClassResults(saveFolder='results/Augmented/Dimensions/')
# print('Experiment complete')

# # Neural Network Dimensions results
# testRun = Experiments(experiments)
# for i in ('RNN','GRU','LSTM'):
#     testRun.printClassParameterReports(i, 'embeddedDims', saveFolder='results/Augmented/Dimensions/')

# # Neural Network Sentence Size experiment (increments of 5)
# for i in range(0, 1):
#     print('Experiment: ' + str(i + 1))
#     experiments = []
#     for Model in ('RNN','LSTM','GRU'):
#         Epochs = 3
#         Batch = 32
#         Dims = 100
#         for SentenceSize in (10, 15, 20, 25):
#             experiments = []
#             print('Experiment: ' + str(i + 1) + ' Model: ' + Model + ' Dimensions: ' + str(Dims) + ' Epochs: ' + str(Epochs) + ' Batch: ' + str(Batch) + ' Sentence Size: ' + str(SentenceSize))
#             experiments.append(Experiment(_modelName=Model, _kFolds=1, _embeddedDims=Dims, _epochs=Epochs, _batchSize = Batch, _sentenceSize=SentenceSize, threshold = 0.0, _trainFilePath='data/trainingSet_augmented.csv', _testFilePath='data/testSet.csv'))
#             testRun = Experiments(experiments)
#             testRun.run()
#             testRun.printClassResults(saveFolder='results/Augmented/SentenceSize/')
# print('Experiment complete')

# # Neural Network Sentence Size results
# testRun = Experiments(experiments)
# for i in ('RNN','GRU','LSTM'):
#     testRun.printClassParameterReports(i, 'sentenceSize', saveFolder='results/Augmented/SentenceSize/')

# # Neural Network Sentence Size experiment 2 (increments of 1, 5-25)
# for i in range(0, 1):
#     print('Experiment: ' + str(i + 1))
#     experiments = []
#     for Model in ('RNN','LSTM','GRU'):
#         Epochs = 3
#         Batch = 32
#         Dims = 100
#         for SentenceSize in range(5,26):
#             experiments = []
#             print('Experiment: ' + str(i + 1) + ' Model: ' + Model + ' Dimensions: ' + str(Dims) + ' Epochs: ' + str(Epochs) + ' Batch: ' + str(Batch) + ' Sentence Size: ' + str(SentenceSize))
#             experiments.append(Experiment(_modelName=Model, _kFolds=1, _embeddedDims=Dims, _epochs=Epochs, _batchSize = Batch, _sentenceSize=SentenceSize, threshold = 0.0, _trainFilePath='data/trainingSet_augmented.csv', _testFilePath='data/testSet.csv'))
#             testRun = Experiments(experiments)
#             testRun.run()
#             testRun.printClassResults(saveFolder='results/Augmented/SentenceSize2/')
# print('Experiment complete')

# # Neural Network Sentence Size results 2
# testRun = Experiments(experiments)
# for i in ('RNN','GRU','LSTM'):
#     testRun.printClassParameterReports(i, 'sentenceSize', saveFolder='results/Augmented/SentenceSize2/')

# # Neural Network Epochs - Batch Size experiment (Do batch size 1)
# for i in range(0, 1):
#     print('Experiment: ' + str(i + 1))
#     experiments = []
#     for Model in ('RNN','LSTM','GRU'):
#         Dims = 300
#         SentenceSize = 20
#         Batch = 1
#         for Epochs in (1,3,5):
#             experiments = []
#             print('Experiment: ' + str(i + 1) + ' Model: ' + Model + ' Dimensions: ' + str(Dims) + ' Epochs: ' + str(Epochs) + ' Batch: ' + str(Batch) + ' Sentence Size: ' + str(SentenceSize))
#             experiments.append(Experiment(_modelName=Model, _kFolds=1, _embeddedDims=Dims, _epochs=Epochs, _batchSize = Batch, _sentenceSize=SentenceSize, threshold = 0.0, _trainFilePath='data/trainingSet_augmented.csv', _testFilePath='data/testSet.csv'))
#             testRun = Experiments(experiments)
#             testRun.run()
#             testRun.printClassResults(saveFolder='results/Augmented/EpochsBatchSize/')
# print('Experiment complete')

# # Neural Network Epochs - Batch Size results
# testRun = Experiments(experiments)
# for i in ('RNN','GRU','LSTM'):
#     for j in ('epochs','batchSize'):
#         testRun.printClassParameterReports(i, j,saveFolder='results/Augmented/EpochsBatchSize/')

# # Neural Network Dimensions experiment 2 - increments of 10, 10-100
# for i in range(0, 1):
#     print('Experiment: ' + str(i + 1))
#     experiments = []
#     for Model in ('RNN','LSTM','GRU'):
#         SentenceSize = 20
#         Epochs = 3
#         Batch = 32
#         for Dims in range(10, 91, 10):
#             experiments = []
#             print('Experiment: ' + str(i + 1) + ' Model: ' + Model + ' Dimensions: ' + str(Dims) + ' Epochs: ' + str(Epochs) + ' Batch: ' + str(Batch) + ' Sentence Size: ' + str(SentenceSize))
#             experiments.append(Experiment(_modelName=Model, _kFolds=1, _embeddedDims=Dims, _epochs=Epochs, _batchSize = Batch, _sentenceSize=SentenceSize, threshold = 0.0, _trainFilePath='data/trainingSet_augmented.csv', _testFilePath='data/testSet.csv'))
#             testRun = Experiments(experiments)
#             testRun.run()
#             testRun.printClassResults(saveFolder='results/Augmented/Dimensions/')
# print('Experiment complete')

# # Neural Network Dimensions results
# testRun = Experiments(experiments)
# for i in ('RNN','GRU','LSTM'):
#     testRun.printClassParameterReports(i, 'embeddedDims', saveFolder='results/Augmented/Dimensions/', Bounds=(10,101))

# # KNN general experiment 2 fixed decoding
# for i in range(0, 1):
#     print('Experiment: ' + str(i + 1))
#     experiments = []
#     for n in range(1, 16):
#         experiments.append(Experiment(_modelName="KNN", _kFolds=1, _nNeighbours=n, _embeddingMode='TFIDF', threshold = 0.0, _trainFilePath='data/trainingSet_augmented.csv', _testFilePath='data/testSet.csv'))
#         experiments.append(Experiment(_modelName="KNN", _kFolds=1, _nNeighbours=n, _embeddingMode='BoW', threshold = 0.0, _trainFilePath='data/trainingSet_augmented.csv', _testFilePath='data/testSet.csv'))
#     testRun = Experiments(experiments)
#     testRun.run()
#     testRun.printClassResults(saveFolder='results/Augmented/KNN')
# print('Experiment complete')

# # KNN general results 2
# testRun = Experiments(experiments)
# for i in ('BoW','TFIDF'):
#     testRun.printClassParameterReports('KNN', 'nNeighbours',saveFolder='results/Augmented/KNN/',Bounds=[i])

# # Compare class results to determine dead classes
# testRun = Experiments(experiments)
# for i in ('RNN','GRU','LSTM'):
#     testRun.printCompareClasses(i, loadFolders=['results/Augmented/Dimensions','results/Augmented/EpochsBatchSize','results/Augmented/SentenceSize2'], saveFolder='results/Augmented/CompareClasses')
# testRun.printCompareClasses('KNN',loadFolders=['results/Augmented/KNN'])

KNNLiveClasses = ['Callback','Error','Information','Leaver','Pricing','Project','Reminder','Report','Status','Template','Weight']
RNNLiveClasses = ['Forward','Gables','Leaver','Pricing','Project','Reminder','Report','Status','Template','Weight']
LSTMLiveClasses = ['Delivery','Documents','EqualGlass','Error','Feedback','Forward','Gables','Information','Leaver','Logo','Pricing','Project','Reminder','Report','Status','Template','Weight']
GRULiveClasses = ['Documents','EqualGlass','Error','Feedback','Forward','Gables','Information','Leaver','Logo','Pricing','Project','Reminder','Report','Status','Template','Weight']

## Regenerate results to include filtered versions

# # KNN general results 2
# testRun = Experiments(experiments)
# for i in ('BoW','TFIDF'):
#     testRun.printClassParameterReports('KNN', 'nNeighbours',saveFolder='results/Augmented/KNN/',Bounds=[i], classList=KNNLiveClasses)

# # Neural Network Dimensions results
# testRun = Experiments(experiments)
# testRun.printClassParameterReports('RNN', 'embeddedDims', saveFolder='results/Augmented/Dimensions/', classList=RNNLiveClasses)
# testRun.printClassParameterReports('LSTM', 'embeddedDims', saveFolder='results/Augmented/Dimensions/', classList=LSTMLiveClasses)
# testRun.printClassParameterReports('GRU', 'embeddedDims', saveFolder='results/Augmented/Dimensions/', classList=GRULiveClasses)

# # Neural Network Epochs - Batch Size results
# testRun = Experiments(experiments)
# for j in ('epochs','batchSize'):
#     testRun.printClassParameterReports('RNN', j,saveFolder='results/Augmented/EpochsBatchSize/',classList=RNNLiveClasses)
#     testRun.printClassParameterReports('LSTM', j,saveFolder='results/Augmented/EpochsBatchSize/',classList=LSTMLiveClasses)
#     testRun.printClassParameterReports('GRU', j,saveFolder='results/Augmented/EpochsBatchSize/',classList=GRULiveClasses)

# # Neural Network Sentence Size results
# testRun = Experiments(experiments)
# testRun.printClassParameterReports('RNN', 'sentenceSize', saveFolder='results/Augmented/SentenceSize2/', classList=RNNLiveClasses)
# testRun.printClassParameterReports('LSTM', 'sentenceSize', saveFolder='results/Augmented/SentenceSize2/', classList=LSTMLiveClasses)
# testRun.printClassParameterReports('GRU', 'sentenceSize', saveFolder='results/Augmented/SentenceSize2/', classList=GRULiveClasses)

RNNParams = Experiment(_modelName='RNN', _kFolds=1, _embeddedDims=600, _epochs=3, _batchSize = 32, _sentenceSize=19, threshold = 0.0, _trainFilePath='data/trainingSet_augmented.csv', _testFilePath='data/testSet.csv')
LSTMParams = Experiment(_modelName='LSTM', _kFolds=1, _embeddedDims=300, _epochs=3, _batchSize = 1, _sentenceSize=18, threshold = 0.0, _trainFilePath='data/trainingSet_augmented.csv', _testFilePath='data/testSet.csv')
GRUParams = Experiment(_modelName='GRU', _kFolds=1, _embeddedDims=80, _epochs=3, _batchSize = 1, _sentenceSize=21, threshold = 0.0, _trainFilePath='data/trainingSet_augmented.csv', _testFilePath='data/testSet.csv')

# saveFolder='results/Augmented/Threshold/'
# # KNN threshold and n neighbours choice
# for i in range(0, 1):
#     print('Experiment: ' + str(i + 1))
#     for j in range(0, 10):
#         threshold = j / 10
#         for n in range(1, 16):
#             experiments = []
#             experiments.append(Experiment(_modelName="KNN", _kFolds=1, _nNeighbours=n, _embeddingMode='TFIDF', threshold = threshold, _trainFilePath='data/trainingSet_augmented.csv', _testFilePath='data/testSet.csv'))
#             experiments.append(Experiment(_modelName="KNN", _kFolds=1, _nNeighbours=n, _embeddingMode='BoW', threshold = threshold, _trainFilePath='data/trainingSet_augmented.csv', _testFilePath='data/testSet.csv'))
#             testRun = Experiments(experiments)
#             testRun.run()
#             testRun.printClassResults(saveFolder=saveFolder)
#         for n in range(20, 101, 10):
#             experiments = []
#             experiments.append(Experiment(_modelName="KNN", _kFolds=1, _nNeighbours=n, _embeddingMode='TFIDF', threshold = threshold, _trainFilePath='data/trainingSet_augmented.csv', _testFilePath='data/testSet.csv'))
#             experiments.append(Experiment(_modelName="KNN", _kFolds=1, _nNeighbours=n, _embeddingMode='BoW', threshold = threshold, _trainFilePath='data/trainingSet_augmented.csv', _testFilePath='data/testSet.csv'))
#             testRun = Experiments(experiments)
#             testRun.run()
#             testRun.printClassResults(saveFolder=saveFolder)
#         for n in range(200, 1001, 100):
#             experiments = []
#             experiments.append(Experiment(_modelName="KNN", _kFolds=1, _nNeighbours=n, _embeddingMode='TFIDF', threshold = threshold, _trainFilePath='data/trainingSet_augmented.csv', _testFilePath='data/testSet.csv'))
#             experiments.append(Experiment(_modelName="KNN", _kFolds=1, _nNeighbours=n, _embeddingMode='BoW', threshold = threshold, _trainFilePath='data/trainingSet_augmented.csv', _testFilePath='data/testSet.csv'))
#             testRun = Experiments(experiments)
#             testRun.run()
#             testRun.printClassResults(saveFolder=saveFolder)
# print('Experiment complete')
# for i in ('BoW','TFIDF'):
#     testRun.printClassParameterReports('KNN', 'nNeighbours',saveFolder=saveFolder,Bounds=[i])
#     testRun.printClassParameterReports('KNN', 'threshold',saveFolder=saveFolder,Bounds=[i])

# # RNN threshold choice (also confirm sentence size and dimensions)
# for i in range(0, 1):
#     print('Experiment: ' + str(i + 1))
#     experiments = []
#     params = copy.deepcopy(RNNParams)
#     for j in range(0, 10):
#         threshold = j / 10
#         for sentenceSize in (6,19):
#             params.sentenceSize = sentenceSize
#             for dims in (100,600):
#                 experiments = []    
#                 params.embeddedDims = dims
#                 print('Experiment: ' + str(i + 1) + ' Model: ' + params.modelName + ' Dimensions: ' + str(params.embeddedDims) + ' Epochs: ' + str(params.epochs) + ' Batch: ' + str(params.batchSize) + ' Sentence Size: ' + str(params.sentenceSize))
#                 experiments.append(Experiment(_modelName=params.modelName, _kFolds=1, _embeddedDims=params.embeddedDims, _epochs=params.epochs, _batchSize = params.batchSize, _sentenceSize=params.sentenceSize, threshold = threshold, _trainFilePath='data/trainingSet_augmented.csv', _testFilePath='data/testSet.csv'))
#                 testRun = Experiments(experiments)
#                 testRun.run()
#                 testRun.printClassResults(saveFolder=saveFolder)
#     testRun = Experiments(experiments)
#     testRun.printClassParameterReports(params.modelName, 'threshold', saveFolder=saveFolder)
#     testRun.printClassParameterReports(params.modelName, 'sentenceSize', saveFolder=saveFolder)
#     testRun.printClassParameterReports(params.modelName, 'embeddedDims', saveFolder=saveFolder)
# print('Experiment complete')

# # LSTM threshold choice (also confirm sentence size and dimensions)
# for i in range(0, 1):

#     print('Experiment: ' + str(i + 1))
#     params = copy.deepcopy(LSTMParams)
#     for j in range(0, 10):
#         threshold = j / 10
#         experiments = []
#         print('Experiment: ' + str(i + 1) + ' Model: ' + params.modelName + ' Dimensions: ' + str(params.embeddedDims) + ' Epochs: ' + str(params.epochs) + ' Batch: ' + str(params.batchSize) + ' Sentence Size: ' + str(params.sentenceSize))
#         experiments.append(Experiment(_modelName=params.modelName, _kFolds=1, _embeddedDims=params.embeddedDims, _epochs=params.epochs, _batchSize = params.batchSize, _sentenceSize=params.sentenceSize, threshold = threshold, _trainFilePath='data/trainingSet_augmented.csv', _testFilePath='data/testSet.csv'))
#         testRun = Experiments(experiments)
#         testRun.run()
#         testRun.printClassResults(saveFolder=saveFolder)
#     testRun = Experiments(experiments)
#     testRun.printClassParameterReports(params.modelName, 'threshold', saveFolder=saveFolder)
# print('Experiment complete')

# # GRU threshold choice (also confirm sentence size and dimensions)
# for i in range(0, 1):

#     print('Experiment: ' + str(i + 1))
#     params = copy.deepcopy(GRUParams)
#     for j in range(0, 10):
#         threshold = j / 10
#         experiments = []
#         print('Experiment: ' + str(i + 1) + ' Model: ' + params.modelName + ' Dimensions: ' + str(params.embeddedDims) + ' Epochs: ' + str(params.epochs) + ' Batch: ' + str(params.batchSize) + ' Sentence Size: ' + str(params.sentenceSize))
#         experiments.append(Experiment(_modelName=params.modelName, _kFolds=1, _embeddedDims=params.embeddedDims, _epochs=params.epochs, _batchSize = params.batchSize, _sentenceSize=params.sentenceSize, threshold = threshold, _trainFilePath='data/trainingSet_augmented.csv', _testFilePath='data/testSet.csv'))
#         testRun = Experiments(experiments)
#         testRun.run()
#         testRun.printClassResults(saveFolder=saveFolder)
#     testRun = Experiments(experiments)
#     testRun.printClassParameterReports(params.modelName, 'threshold', saveFolder=saveFolder)
# print('Experiment complete')

########################
### Balanced dataset ###
########################

balancedSaveFolder='results/Balanced/'
balancedTrainFilePath='data/trainingSet_balanced.csv'
saveFolder = balancedSaveFolder + 'Threshold/'
# KNN threshold and n neighbours choice
# for i in range(0, 1):
#     print('Experiment: ' + str(i + 1))
#     for j in range(0, 10):
#         threshold = j / 10
#         for n in range(1, 16):
#             experiments = []
#             experiments.append(Experiment(_modelName="KNN", _kFolds=1, _nNeighbours=n, _embeddingMode='TFIDF', threshold = threshold, _trainFilePath=balancedTrainFilePath, _testFilePath='data/testSet.csv'))
#             experiments.append(Experiment(_modelName="KNN", _kFolds=1, _nNeighbours=n, _embeddingMode='BoW', threshold = threshold, _trainFilePath=balancedTrainFilePath, _testFilePath='data/testSet.csv'))
#             testRun = Experiments(experiments)
#             testRun.run()
#             testRun.printClassResults(saveFolder=balancedSaveFolder + 'Threshold/')
#         for n in range(20, 101, 10):
#             experiments = []
#             experiments.append(Experiment(_modelName="KNN", _kFolds=1, _nNeighbours=n, _embeddingMode='TFIDF', threshold = threshold, _trainFilePath=balancedTrainFilePath, _testFilePath='data/testSet.csv'))
#             experiments.append(Experiment(_modelName="KNN", _kFolds=1, _nNeighbours=n, _embeddingMode='BoW', threshold = threshold, _trainFilePath=balancedTrainFilePath, _testFilePath='data/testSet.csv'))
#             testRun = Experiments(experiments)
#             testRun.run()
#             testRun.printClassResults(saveFolder=saveFolder)
#         for n in range(200, 1001, 100):
#             experiments = []
#             experiments.append(Experiment(_modelName="KNN", _kFolds=1, _nNeighbours=n, _embeddingMode='TFIDF', threshold = threshold, _trainFilePath=balancedTrainFilePath, _testFilePath='data/testSet.csv'))
#             experiments.append(Experiment(_modelName="KNN", _kFolds=1, _nNeighbours=n, _embeddingMode='BoW', threshold = threshold, _trainFilePath=balancedTrainFilePath, _testFilePath='data/testSet.csv'))
#             testRun = Experiments(experiments)
#             testRun.run()
#             testRun.printClassResults(saveFolder=saveFolder)
# print('Experiment complete')
# for i in ('BoW','TFIDF'):
#     testRun.printClassParameterReports('KNN', 'nNeighbours',saveFolder=saveFolder,Bounds=[i])
#     testRun.printClassParameterReports('KNN', 'threshold',saveFolder=saveFolder,Bounds=[i])

saveFolder= balancedSaveFolder + 'EpochsBatchSize/'
# Neural Network Epochs - Batch Size experiment
for i in range(0, 1):
    print('Experiment: ' + str(i + 1))
    experiments = []
    for Model in ('RNN','LSTM','GRU'):
        Dims = 300
        SentenceSize = 20
        #for Epochs in (1,3,5):
        for Epochs in (7,10):
            for Batch in (1,32,64,128,7450):
                experiments = []
                print('Experiment: ' + str(i + 1) + ' Model: ' + Model + ' Dimensions: ' + str(Dims) + ' Epochs: ' + str(Epochs) + ' Batch: ' + str(Batch) + ' Sentence Size: ' + str(SentenceSize))
                experiments.append(Experiment(_modelName=Model, _kFolds=1, _embeddedDims=Dims, _epochs=Epochs, _batchSize = Batch, _sentenceSize=SentenceSize, threshold = 0.0, _trainFilePath=balancedTrainFilePath, _testFilePath='data/testSet.csv'))
                testRun = Experiments(experiments)
                testRun.run()
                testRun.printClassResults(saveFolder=saveFolder)
    for j in ('epochs','batchSize'):
        if Model == 'RNN':
            testRun.printClassParameterReports(Model, j, saveFolder=saveFolder, classList=RNNLiveClasses)
        if Model == 'LSTM':
            testRun.printClassParameterReports(Model, j, saveFolder=saveFolder, classList=LSTMLiveClasses)
        else:
            testRun.printClassParameterReports(Model, j, saveFolder=saveFolder, classList=GRULiveClasses)

saveFolder= balancedSaveFolder + 'Dimensions/'
# Neural Network Dimensions experiment 2 - increments of 10, 10-100
for i in range(0, 1):
    print('Experiment: ' + str(i + 1))
    experiments = []
    for Model in ('RNN','LSTM','GRU'):
        SentenceSize = 20
        Epochs = 5
        Batch = 1
        for Dims in range(10, 91, 10):
            experiments = []
            print('Experiment: ' + str(i + 1) + ' Model: ' + Model + ' Dimensions: ' + str(Dims) + ' Epochs: ' + str(Epochs) + ' Batch: ' + str(Batch) + ' Sentence Size: ' + str(SentenceSize))
            experiments.append(Experiment(_modelName=Model, _kFolds=1, _embeddedDims=Dims, _epochs=Epochs, _batchSize = Batch, _sentenceSize=SentenceSize, threshold = 0.0, _trainFilePath=balancedTrainFilePath, _testFilePath='data/testSet.csv'))
            testRun = Experiments(experiments)
            testRun.run()
            testRun.printClassResults(saveFolder=saveFolder)
        for Dims in range(100, 1001, 100):
            experiments = []
            print('Experiment: ' + str(i + 1) + ' Model: ' + Model + ' Dimensions: ' + str(Dims) + ' Epochs: ' + str(Epochs) + ' Batch: ' + str(Batch) + ' Sentence Size: ' + str(SentenceSize))
            experiments.append(Experiment(_modelName=Model, _kFolds=1, _embeddedDims=Dims, _epochs=Epochs, _batchSize = Batch, _sentenceSize=SentenceSize, threshold = 0.0, _trainFilePath=balancedTrainFilePath, _testFilePath='data/testSet.csv'))
            testRun = Experiments(experiments)
            testRun.run()
            testRun.printClassResults(saveFolder=saveFolder)
        if Model == 'RNN':
            testRun.printClassParameterReports(Model, 'embeddedDims', saveFolder=saveFolder)
        if Model == 'LSTM':
            testRun.printClassParameterReports(Model, 'embeddedDims', saveFolder=saveFolder)
        else:
            testRun.printClassParameterReports(Model, 'embeddedDims', saveFolder=saveFolder)

# # Revisit augmented KNN - Embedding fixed
# KNN threshold and n neighbours choice
# saveFolder = 'results/Augmented/Threshold'
# for i in range(0, 1):
#     print('Experiment: ' + str(i + 1))
#     for j in range(0, 10):
#         threshold = j / 10
#         for n in range(1, 16):
#             experiments = []
#             experiments.append(Experiment(_modelName="KNN", _kFolds=1, _nNeighbours=n, _embeddingMode='TFIDF', threshold = threshold, _trainFilePath='data/trainingSet_augmented.csv', _testFilePath='data/testSet.csv'))
#             experiments.append(Experiment(_modelName="KNN", _kFolds=1, _nNeighbours=n, _embeddingMode='BoW', threshold = threshold, _trainFilePath='data/trainingSet_augmented.csv', _testFilePath='data/testSet.csv'))
#             testRun = Experiments(experiments)
#             testRun.run()
#             testRun.printClassResults(saveFolder=balancedSaveFolder + 'Threshold/')
#         for n in range(20, 101, 10):
#             experiments = []
#             experiments.append(Experiment(_modelName="KNN", _kFolds=1, _nNeighbours=n, _embeddingMode='TFIDF', threshold = threshold, _trainFilePath='data/trainingSet_augmented.csv', _testFilePath='data/testSet.csv'))
#             experiments.append(Experiment(_modelName="KNN", _kFolds=1, _nNeighbours=n, _embeddingMode='BoW', threshold = threshold, _trainFilePath='data/trainingSet_augmented.csv', _testFilePath='data/testSet.csv'))
#             testRun = Experiments(experiments)
#             testRun.run()
#             testRun.printClassResults(saveFolder=saveFolder)
#         for n in range(200, 1001, 100):
#             experiments = []
#             experiments.append(Experiment(_modelName="KNN", _kFolds=1, _nNeighbours=n, _embeddingMode='TFIDF', threshold = threshold, _trainFilePath='data/trainingSet_augmented.csv', _testFilePath='data/testSet.csv'))
#             experiments.append(Experiment(_modelName="KNN", _kFolds=1, _nNeighbours=n, _embeddingMode='BoW', threshold = threshold, _trainFilePath='data/trainingSet_augmented.csv'c, _testFilePath='data/testSet.csv'))
#             testRun = Experiments(experiments)
#             testRun.run()
#             testRun.printClassResults(saveFolder=saveFolder)
#     testRun.printClassParameterReports('KNN', 'nNeighbours',saveFolder=saveFolder,Bounds=['BoW'])
#     testRun.printClassParameterReports('KNN', 'threshold',saveFolder=saveFolder,Bounds=['TFIDF'])
#     testRun.printClassParameterReports('KNN', 'nNeighbours',saveFolder=saveFolder,Bounds=['BoW'])
#     testRun.printClassParameterReports('KNN', 'threshold',saveFolder=saveFolder,Bounds=['TFIDF'])

# Reprint augmented graphs and some balanced
# # Neural Network Dimensions results
# testRun = Experiments(experiments)
# testRun.printClassParameterReports('RNN', 'embeddedDims', saveFolder='results/Augmented/Dimensions/', classList=RNNLiveClasses)
# testRun.printClassParameterReports('LSTM', 'embeddedDims', saveFolder='results/Augmented/Dimensions/', classList=LSTMLiveClasses)
# testRun.printClassParameterReports('GRU', 'embeddedDims', saveFolder='results/Augmented/Dimensions/', classList=GRULiveClasses)

# # Neural Network Epochs - Batch Size results
# testRun = Experiments(experiments)
# for j in ('epochs','batchSize'):
#     testRun.printClassParameterReports('RNN', j,saveFolder='results/Augmented/EpochsBatchSize/',classList=RNNLiveClasses)
#     testRun.printClassParameterReports('LSTM', j,saveFolder='results/Augmented/EpochsBatchSize/',classList=LSTMLiveClasses)
#     testRun.printClassParameterReports('GRU', j,saveFolder='results/Augmented/EpochsBatchSize/',classList=GRULiveClasses)

# # Neural Network Sentence Size results
# testRun = Experiments(experiments)
# testRun.printClassParameterReports('RNN', 'sentenceSize', saveFolder='results/Augmented/SentenceSize2/', classList=RNNLiveClasses)
# testRun.printClassParameterReports('LSTM', 'sentenceSize', saveFolder='results/Augmented/SentenceSize2/', classList=LSTMLiveClasses)
# testRun.printClassParameterReports('GRU', 'sentenceSize', saveFolder='results/Augmented/SentenceSize2/', classList=GRULiveClasses)

# # Threshold results
# for Model in ['KNN','RNN','LSTM','GRU']:
#     testRun.printClassParameterReports(params.modelName, 'threshold', saveFolder='results/Augmented/Threshold')
# for i in ('BoW','TFIDF'):
#     testRun.printClassParameterReports('KNN', 'nNeighbours',saveFolder='results/Augmented/Threshold',Bounds=[i])

# # KNN results
# for i in ('BoW','TFIDF'):
#     testRun.printClassParameterReports('KNN', 'nNeighbours',saveFolder='results/Augmented/Threshold',Bounds=[i])

# # Compare class results to determine dead classes
# testRun = Experiments(experiments)
# for i in ('RNN','GRU','LSTM'):
#     testRun.printCompareClasses(i, loadFolders=['results/Augmented/Dimensions','results/Augmented/EpochsBatchSize','results/Augmented/SentenceSize'], saveFolder='results/Augmented/CompareClasses')
# testRun.printCompareClasses('KNN',loadFolders=['results/Augmented/KNN'])

# # Neural Network Dimensions results
# testRun = Experiments(experiments)
# testRun.printClassParameterReports('RNN', 'embeddedDims', saveFolder='results/Balanced/Dimensions/')
# testRun.printClassParameterReports('LSTM', 'embeddedDims', saveFolder='results/Balanced/Dimensions/')
# testRun.printClassParameterReports('GRU', 'embeddedDims', saveFolder='results/Balanced/Dimensions/')

# # Neural Network Epochs - Batch Size results
# testRun = Experiments(experiments)
# for j in ('epochs','batchSize'):
#     testRun.printClassParameterReports('RNN', j,saveFolder='results/Balanced/EpochsBatchSize/')
#     testRun.printClassParameterReports('LSTM', j,saveFolder='results/Balanced/EpochsBatchSize/')
#     testRun.printClassParameterReports('GRU', j,saveFolder='results/Balanced/EpochsBatchSize/')

# saveFolder= balancedSaveFolder + 'KNN/'
# # KNN balanced experiment
# for i in range(0, 1):
#     print('Experiment: ' + str(i + 1))
#     experiments = []
#     for n in range(1, 16):
#         experiments.append(Experiment(_modelName="KNN", _kFolds=1, _nNeighbours=n, _embeddingMode='TFIDF', threshold = 0.0, _trainFilePath=balancedTrainFilePath, _testFilePath='data/testSet.csv'))
#         experiments.append(Experiment(_modelName="KNN", _kFolds=1, _nNeighbours=n, _embeddingMode='BoW', threshold = 0.0, _trainFilePath=balancedTrainFilePath, _testFilePath='data/testSet.csv'))
#     testRun = Experiments(experiments)
#     testRun.run()
#     testRun.printClassResults(saveFolder=saveFolder)
#     for i in ('BoW','TFIDF'):
#       testRun.printClassParameterReports('KNN', 'nNeighbours',saveFolder=saveFolder,Bounds=[i])

# saveFolder= balancedSaveFolder + 'SentenceSize/'
# # Neural Network Dimensions experiment 2 - increments of 10, 10-100
# Decide on Dimensions
# for i in range(0, 1):
#     print('Experiment: ' + str(i + 1))
#     experiments = []
#     for Model in ('RNN','LSTM','GRU'):
#         Epochs = 5
#         Batch = 1
#         Dims = ?
#         for SentenceSize in range(5,26):
#             experiments = []
#             print('Experiment: ' + str(i + 1) + ' Model: ' + Model + ' Dimensions: ' + str(Dims) + ' Epochs: ' + str(Epochs) + ' Batch: ' + str(Batch) + ' Sentence Size: ' + str(SentenceSize))
#             experiments.append(Experiment(_modelName=Model, _kFolds=1, _embeddedDims=Dims, _epochs=Epochs, _batchSize = Batch, _sentenceSize=SentenceSize, threshold = 0.0, _trainFilePath=balancedTrainFilePath, _testFilePath='data/testSet.csv'))
#             testRun = Experiments(experiments)
#             testRun.run()
#             testRun.printClassResults(saveFolder=saveFolder)
#         if Model == 'RNN':
#             testRun.printClassParameterReports(Model, 'embeddedDims', saveFolder=saveFolder, classList=RNNLiveClasses)
#         if Model == 'LSTM':
#             testRun.printClassParameterReports(Model, 'embeddedDims', saveFolder=saveFolder, classList=LSTMLiveClasses)
#         else:
#             testRun.printClassParameterReports(Model, 'embeddedDims', saveFolder=saveFolder, classList=GRULiveClasses)

# # Any sentence size follow up?

# # Check KNN results fixed - should be a difference between BoW and TFIDF results

# # Decide which classes to filter:

# # Compare class results to determine dead classes
# testRun = Experiments(experiments)
# for i in ('RNN','GRU','LSTM'):
#     testRun.printCompareClasses(i, loadFolders=['results/Balanced/Dimensions','results/Balanced/EpochsBatchSize','results/Balanced/SentenceSize2'], saveFolder='results/Balanced/CompareClasses')
# testRun.printCompareClasses('KNN',loadFolders=['results/Balanced/KNN'])

# KNNBalancedClasses = []
# RNNBalancedClasses = []
# LSTMBalancedClasses = []
# GRUBalancedClasses = []

## Regenerate results to include new filtering

# Experiments = []

# # KNN general results 2
# testRun = Experiments(experiments)
# for i in ('BoW','TFIDF'):
#     testRun.printClassParameterReports('KNN', 'nNeighbours',saveFolder='results/Balanced/KNN/',Bounds=[i], classList=KNNBalancedClasses)

# # Neural Network Dimensions results
# testRun = Experiments(experiments)
# testRun.printClassParameterReports('RNN', 'embeddedDims', saveFolder='results/Balanced/Dimensions/', classList=RNNBalancedClasses)
# testRun.printClassParameterReports('LSTM', 'embeddedDims', saveFolder='results/Balanced/Dimensions/', classList=LSTMBalancedClasses)
# testRun.printClassParameterReports('GRU', 'embeddedDims', saveFolder='results/Balanced/Dimensions/', classList=GRUBalancedClasses)

# # Neural Network Epochs - Batch Size results
# testRun = Experiments(experiments)
# for j in ('epochs','batchSize'):
#     testRun.printClassParameterReports('RNN', j,saveFolder='results/Balanced/EpochsBatchSize/',classList=RNNBalancedClasses)
#     testRun.printClassParameterReports('LSTM', j,saveFolder='results/Balanced/EpochsBatchSize/',classList=LSTMBalancedClasses)
#     testRun.printClassParameterReports('GRU', j,saveFolder='results/Balanced/EpochsBatchSize/',classList=GRUBalancedClasses)

# # Neural Network Sentence Size results
# testRun = Experiments(experiments)
# testRun.printClassParameterReports('RNN', 'sentenceSize', saveFolder='results/Balanced/SentenceSize/', classList=RNNBalancedClasses)
# testRun.printClassParameterReports('LSTM', 'sentenceSize', saveFolder='results/Balanced/SentenceSize/', classList=LSTMBalancedClasses)
# testRun.printClassParameterReports('GRU', 'sentenceSize', saveFolder='results/Balanced/SentenceSize/', classList=GRUBalancedClasses)

# Decide on optimised parameters
# RNNBalancedParams = Experiment(_modelName='RNN', _kFolds=1, _embeddedDims=600, _epochs=3, _batchSize = 32, _sentenceSize=19, threshold = 0.0, _trainFilePath='data/trainingSet_augmented.csv', _testFilePath='data/testSet.csv')
# LSTMBalancedParams = Experiment(_modelName='LSTM', _kFolds=1, _embeddedDims=300, _epochs=3, _batchSize = 1, _sentenceSize=18, threshold = 0.0, _trainFilePath='data/trainingSet_augmented.csv', _testFilePath='data/testSet.csv')
# GRUBalancedParams = Experiment(_modelName='GRU', _kFolds=1, _embeddedDims=80, _epochs=3, _batchSize = 1, _sentenceSize=21, threshold = 0.0, _trainFilePath='data/trainingSet_augmented.csv', _testFilePath='data/testSet.csv')

# saveFolder= balancedSaveFolder + 'Threshold/'
# # KNN threshold and n neighbours choice - doing it a second time as word embeddings now fixed.
# for i in range(0, 1):
#     print('Experiment: ' + str(i + 1))
#     for j in range(0, 10):
#         threshold = j / 10
#         for n in range(1, 16):
#             experiments = []
#             experiments.append(Experiment(_modelName="KNN", _kFolds=1, _nNeighbours=n, _embeddingMode='TFIDF', threshold = threshold, _trainFilePath='data/trainingSet_augmented.csv', _testFilePath='data/testSet.csv'))
#             experiments.append(Experiment(_modelName="KNN", _kFolds=1, _nNeighbours=n, _embeddingMode='BoW', threshold = threshold, _trainFilePath='data/trainingSet_augmented.csv', _testFilePath='data/testSet.csv'))
#             testRun = Experiments(experiments)
#             testRun.run()
#             testRun.printClassResults(saveFolder=saveFolder)
#         for n in range(20, 101, 10):
#             experiments = []
#             experiments.append(Experiment(_modelName="KNN", _kFolds=1, _nNeighbours=n, _embeddingMode='TFIDF', threshold = threshold, _trainFilePath='data/trainingSet_augmented.csv', _testFilePath='data/testSet.csv'))
#             experiments.append(Experiment(_modelName="KNN", _kFolds=1, _nNeighbours=n, _embeddingMode='BoW', threshold = threshold, _trainFilePath='data/trainingSet_augmented.csv', _testFilePath='data/testSet.csv'))
#             testRun = Experiments(experiments)
#             testRun.run()
#             testRun.printClassResults(saveFolder=saveFolder)
#         for n in range(200, 1001, 100):
#             experiments = []
#             experiments.append(Experiment(_modelName="KNN", _kFolds=1, _nNeighbours=n, _embeddingMode='TFIDF', threshold = threshold, _trainFilePath='data/trainingSet_augmented.csv', _testFilePath='data/testSet.csv'))
#             experiments.append(Experiment(_modelName="KNN", _kFolds=1, _nNeighbours=n, _embeddingMode='BoW', threshold = threshold, _trainFilePath='data/trainingSet_augmented.csv', _testFilePath='data/testSet.csv'))
#             testRun = Experiments(experiments)
#             testRun.run()
#             testRun.printClassResults(saveFolder=saveFolder)
# print('Experiment complete')
# for i in ('BoW','TFIDF'):
#     testRun.printClassParameterReports('KNN', 'nNeighbours',saveFolder=saveFolder,Bounds=[i])
#     testRun.printClassParameterReports('KNN', 'threshold',saveFolder=saveFolder,Bounds=[i])

# # RNN threshold choice (also confirm sentence size and dimensions)
# for i in range(0, 1):
#     print('Experiment: ' + str(i + 1))
#     experiments = []
#     params = copy.deepcopy(RNNParams)
#     for j in range(0, 10):
#         threshold = j / 10
#         for sentenceSize in (6,19):
#             params.sentenceSize = sentenceSize
#             for dims in (100,600):
#                 experiments = []    
#                 params.embeddedDims = dims
#                 print('Experiment: ' + str(i + 1) + ' Model: ' + params.modelName + ' Dimensions: ' + str(params.embeddedDims) + ' Epochs: ' + str(params.epochs) + ' Batch: ' + str(params.batchSize) + ' Sentence Size: ' + str(params.sentenceSize))
#                 experiments.append(Experiment(_modelName=params.modelName, _kFolds=1, _embeddedDims=params.embeddedDims, _epochs=params.epochs, _batchSize = params.batchSize, _sentenceSize=params.sentenceSize, threshold = threshold, _trainFilePath='data/trainingSet_augmented.csv', _testFilePath='data/testSet.csv'))
#                 testRun = Experiments(experiments)
#                 testRun.run()
#                 testRun.printClassResults(saveFolder=saveFolder)
#     testRun = Experiments(experiments)
#     testRun.printClassParameterReports(params.modelName, 'threshold', saveFolder=saveFolder)
#     testRun.printClassParameterReports(params.modelName, 'sentenceSize', saveFolder=saveFolder)
#     testRun.printClassParameterReports(params.modelName, 'embeddedDims', saveFolder=saveFolder)
# print('Experiment complete')

# # LSTM threshold choice (also confirm sentence size and dimensions)
# for i in range(0, 1):

#     print('Experiment: ' + str(i + 1))
#     params = copy.deepcopy(LSTMParams)
#     for j in range(0, 10):
#         threshold = j / 10
#         experiments = []
#         print('Experiment: ' + str(i + 1) + ' Model: ' + params.modelName + ' Dimensions: ' + str(params.embeddedDims) + ' Epochs: ' + str(params.epochs) + ' Batch: ' + str(params.batchSize) + ' Sentence Size: ' + str(params.sentenceSize))
#         experiments.append(Experiment(_modelName=params.modelName, _kFolds=1, _embeddedDims=params.embeddedDims, _epochs=params.epochs, _batchSize = params.batchSize, _sentenceSize=params.sentenceSize, threshold = threshold, _trainFilePath='data/trainingSet_augmented.csv', _testFilePath='data/testSet.csv'))
#         testRun = Experiments(experiments)
#         testRun.run()
#         testRun.printClassResults(saveFolder=saveFolder)
#     testRun = Experiments(experiments)
#     testRun.printClassParameterReports(params.modelName, 'threshold', saveFolder=saveFolder)
# print('Experiment complete')

# # GRU threshold choice (also confirm sentence size and dimensions)
# for i in range(0, 1):

#     print('Experiment: ' + str(i + 1))
#     params = copy.deepcopy(GRUParams)
#     for j in range(0, 10):
#         threshold = j / 10
#         experiments = []
#         print('Experiment: ' + str(i + 1) + ' Model: ' + params.modelName + ' Dimensions: ' + str(params.embeddedDims) + ' Epochs: ' + str(params.epochs) + ' Batch: ' + str(params.batchSize) + ' Sentence Size: ' + str(params.sentenceSize))
#         experiments.append(Experiment(_modelName=params.modelName, _kFolds=1, _embeddedDims=params.embeddedDims, _epochs=params.epochs, _batchSize = params.batchSize, _sentenceSize=params.sentenceSize, threshold = threshold, _trainFilePath='data/trainingSet_augmented.csv', _testFilePath='data/testSet.csv'))
#         testRun = Experiments(experiments)
#         testRun.run()
#         testRun.printClassResults(saveFolder=saveFolder)
#     testRun = Experiments(experiments)
#     testRun.printClassParameterReports(params.modelName, 'threshold', saveFolder=saveFolder)
# print('Experiment complete')

########################
### Original dataset ###
########################

# originalSaveFolder='results/Balanced/'
# originalTrainFilePath='data/trainingSet_balanced.csv'

# saveFolder= originalSaveFolder + 'EpochsBatchSize/'
# # Neural Network Epochs - Batch Size experiment
# for i in range(0, 1):
#     print('Experiment: ' + str(i + 1))
#     experiments = []
#     for Model in ('RNN','LSTM','GRU'):
#         Dims = 300
#         SentenceSize = 20
#         for Epochs in (1,3,5):
#             for Batch in (1,32,64,128,7450):
#                 experiments = []
#                 print('Experiment: ' + str(i + 1) + ' Model: ' + Model + ' Dimensions: ' + str(Dims) + ' Epochs: ' + str(Epochs) + ' Batch: ' + str(Batch) + ' Sentence Size: ' + str(SentenceSize))
#                 experiments.append(Experiment(_modelName=Model, _kFolds=1, _embeddedDims=Dims, _epochs=Epochs, _batchSize = Batch, _sentenceSize=SentenceSize, threshold = 0.0, _trainFilePath=originalTrainFilePath, _testFilePath='data/testSet.csv'))
#                 testRun = Experiments(experiments)
#                 testRun.run()
#                 testRun.printClassResults(saveFolder=saveFolder)
#     for j in ('epochs','batchSize'):
#         if Model == 'RNN':
#             testRun.printClassParameterReports(Model, j, saveFolder=saveFolder)
#         if Model == 'LSTM':
#             testRun.printClassParameterReports(Model, j, saveFolder=saveFolder)
#         else:
#             testRun.printClassParameterReports(Model, j, saveFolder=saveFolder)

# Based on results, do more exploration?

# saveFolder= originalSaveFolder + 'Dimensions/'
# # Neural Network Dimensions experiment 2 - increments of 10, 10-100
# for i in range(0, 1):
#     print('Experiment: ' + str(i + 1))
#     experiments = []
#     for Model in ('RNN','LSTM','GRU'):
#         SentenceSize = 20
#         Epochs = 5
#         Batch = 1
#         for Dims in range(10, 91, 10):
#             experiments = []
#             print('Experiment: ' + str(i + 1) + ' Model: ' + Model + ' Dimensions: ' + str(Dims) + ' Epochs: ' + str(Epochs) + ' Batch: ' + str(Batch) + ' Sentence Size: ' + str(SentenceSize))
#             experiments.append(Experiment(_modelName=Model, _kFolds=1, _embeddedDims=Dims, _epochs=Epochs, _batchSize = Batch, _sentenceSize=SentenceSize, threshold = 0.0, _trainFilePath=originalTrainFilePath, _testFilePath='data/testSet.csv'))
#             testRun = Experiments(experiments)
#             testRun.run()
#             testRun.printClassResults(saveFolder=saveFolder)
#         for Dims in range(100, 1001, 100):
#             experiments = []
#             print('Experiment: ' + str(i + 1) + ' Model: ' + Model + ' Dimensions: ' + str(Dims) + ' Epochs: ' + str(Epochs) + ' Batch: ' + str(Batch) + ' Sentence Size: ' + str(SentenceSize))
#             experiments.append(Experiment(_modelName=Model, _kFolds=1, _embeddedDims=Dims, _epochs=Epochs, _batchSize = Batch, _sentenceSize=SentenceSize, threshold = 0.0, _trainFilePath=originalTrainFilePath, _testFilePath='data/testSet.csv'))
#             testRun = Experiments(experiments)
#             testRun.run()
#             testRun.printClassResults(saveFolder=saveFolder)
#         if Model == 'RNN':
#             testRun.printClassParameterReports(Model, 'embeddedDims', saveFolder=saveFolder)
#         if Model == 'LSTM':
#             testRun.printClassParameterReports(Model, 'embeddedDims', saveFolder=saveFolder)
#         else:
#             testRun.printClassParameterReports(Model, 'embeddedDims', saveFolder=saveFolder)

# Based on results, do more exploration? Any optimisation to implement below?

# saveFolder= balancedSaveFolder + 'SentenceSize/'
# # Neural Network Dimensions experiment 2 - increments of 10, 10-100
# Decide on Dimensions
# for i in range(0, 1):
#     print('Experiment: ' + str(i + 1))
#     experiments = []
#     for Model in ('RNN','LSTM','GRU'):
#         Epochs = 5
#         Batch = 1
#         Dims = ?
#         for SentenceSize in range(5,26):
#             experiments = []
#             print('Experiment: ' + str(i + 1) + ' Model: ' + Model + ' Dimensions: ' + str(Dims) + ' Epochs: ' + str(Epochs) + ' Batch: ' + str(Batch) + ' Sentence Size: ' + str(SentenceSize))
#             experiments.append(Experiment(_modelName=Model, _kFolds=1, _embeddedDims=Dims, _epochs=Epochs, _batchSize = Batch, _sentenceSize=SentenceSize, threshold = 0.0, _trainFilePath=originalTrainFilePath, _testFilePath='data/testSet.csv'))
#             testRun = Experiments(experiments)
#             testRun.run()
#             testRun.printClassResults(saveFolder=saveFolder)
#         if Model == 'RNN':
#             testRun.printClassParameterReports(Model, 'embeddedDims', saveFolder=saveFolder, classList=RNNLiveClasses)
#         if Model == 'LSTM':
#             testRun.printClassParameterReports(Model, 'embeddedDims', saveFolder=saveFolder, classList=LSTMLiveClasses)
#         else:
#             testRun.printClassParameterReports(Model, 'embeddedDims', saveFolder=saveFolder, classList=GRULiveClasses)

# Based on results, do more exploration? Any optimisation to implement below?

# saveFolder= balancedSaveFolder + 'KNN/'
# # KNN balanced experiment
# for i in range(0, 1):
#     print('Experiment: ' + str(i + 1))
#     experiments = []
#     for n in range(1, 16):
#         experiments.append(Experiment(_modelName="KNN", _kFolds=1, _nNeighbours=n, _embeddingMode='TFIDF', threshold = 0.0, _trainFilePath=originalTrainFilePath, _testFilePath='data/testSet.csv'))
#         experiments.append(Experiment(_modelName="KNN", _kFolds=1, _nNeighbours=n, _embeddingMode='BoW', threshold = 0.0, _trainFilePath=originalTrainFilePath, _testFilePath='data/testSet.csv'))
#     testRun = Experiments(experiments)
#     testRun.run()
#     testRun.printClassResults(saveFolder=saveFolder)
#     for i in ('BoW','TFIDF'):
#       testRun.printClassParameterReports('KNN', 'nNeighbours',saveFolder=saveFolder,Bounds=[i])

# Based on results, do more exploration? Any optimisation to implement below?

# # Decide which classes to filter:

# # Compare class results to determine dead classes
# testRun = Experiments(experiments)
# for i in ('RNN','GRU','LSTM'):
#     testRun.printCompareClasses(i, loadFolders=['results/Balanced/Dimensions','results/Balanced/EpochsBatchSize','results/Balanced/SentenceSize2'], saveFolder='results/Balanced/CompareClasses')
# testRun.printCompareClasses('KNN',loadFolders=['results/Balanced/KNN'])

# KNNBalancedClasses = []
# RNNBalancedClasses = []
# LSTMBalancedClasses = []
# GRUBalancedClasses = []

## Regenerate results to include new filtering

# Experiments = []

# # KNN general results 2
# testRun = Experiments(experiments)
# for i in ('BoW','TFIDF'):
#     testRun.printClassParameterReports('KNN', 'nNeighbours',saveFolder='results/Balanced/KNN/',Bounds=[i], classList=KNNBalancedClasses)

# # Neural Network Dimensions results
# testRun = Experiments(experiments)
# testRun.printClassParameterReports('RNN', 'embeddedDims', saveFolder='results/Balanced/Dimensions/', classList=RNNBalancedClasses)
# testRun.printClassParameterReports('LSTM', 'embeddedDims', saveFolder='results/Balanced/Dimensions/', classList=LSTMBalancedClasses)
# testRun.printClassParameterReports('GRU', 'embeddedDims', saveFolder='results/Balanced/Dimensions/', classList=GRUBalancedClasses)

# # Neural Network Epochs - Batch Size results
# testRun = Experiments(experiments)
# for j in ('epochs','batchSize'):
#     testRun.printClassParameterReports('RNN', j,saveFolder='results/Balanced/EpochsBatchSize/',classList=RNNBalancedClasses)
#     testRun.printClassParameterReports('LSTM', j,saveFolder='results/Balanced/EpochsBatchSize/',classList=LSTMBalancedClasses)
#     testRun.printClassParameterReports('GRU', j,saveFolder='results/Balanced/EpochsBatchSize/',classList=GRUBalancedClasses)

# # Neural Network Sentence Size results 2
# testRun = Experiments(experiments)
# testRun.printClassParameterReports('RNN', 'sentenceSize', saveFolder='results/Balanced/SentenceSize/', classList=RNNBalancedClasses)
# testRun.printClassParameterReports('LSTM', 'sentenceSize', saveFolder='results/Balanced/SentenceSize/', classList=LSTMBalancedClasses)
# testRun.printClassParameterReports('GRU', 'sentenceSize', saveFolder='results/Balanced/SentenceSize/', classList=GRUBalancedClasses)

# Decide on optimised parameters
# RNNOriginalParams = Experiment(_modelName='RNN', _kFolds=1, _embeddedDims=600, _epochs=3, _batchSize = 32, _sentenceSize=19, threshold = 0.0, _trainFilePath=originalTrainFilePath, _testFilePath='data/testSet.csv')
# LSTMOriginalParams = Experiment(_modelName='LSTM', _kFolds=1, _embeddedDims=300, _epochs=3, _batchSize = 1, _sentenceSize=18, threshold = 0.0, _trainFilePath=originalTrainFilePath, _testFilePath='data/testSet.csv')
# GRUOriginalParams = Experiment(_modelName='GRU', _kFolds=1, _embeddedDims=80, _epochs=3, _batchSize = 1, _sentenceSize=21, threshold = 0.0, _trainFilePath=originalTrainFilePath, _testFilePath='data/testSet.csv')

# saveFolder= originalSaveFolder + 'Threshold/'
# # KNN threshold and n neighbours choice
# for i in range(0, 1):
#     print('Experiment: ' + str(i + 1))
#     for j in range(0, 10):
#         threshold = j / 10
#         for n in range(1, 16):
#             experiments = []
#             experiments.append(Experiment(_modelName="KNN", _kFolds=1, _nNeighbours=n, _embeddingMode='TFIDF', threshold = threshold, _trainFilePath=originalTrainFilePath, _testFilePath='data/testSet.csv'))
#             experiments.append(Experiment(_modelName="KNN", _kFolds=1, _nNeighbours=n, _embeddingMode='BoW', threshold = threshold, _trainFilePath=originalTrainFilePath, _testFilePath='data/testSet.csv'))
#             testRun = Experiments(experiments)
#             testRun.run()
#             testRun.printClassResults(saveFolder=saveFolder)
#         for n in range(20, 101, 10):
#             experiments = []
#             experiments.append(Experiment(_modelName="KNN", _kFolds=1, _nNeighbours=n, _embeddingMode='TFIDF', threshold = threshold, _trainFilePath=originalTrainFilePath, _testFilePath='data/testSet.csv'))
#             experiments.append(Experiment(_modelName="KNN", _kFolds=1, _nNeighbours=n, _embeddingMode='BoW', threshold = threshold, _trainFilePath=originalTrainFilePath, _testFilePath='data/testSet.csv'))
#             testRun = Experiments(experiments)
#             testRun.run()
#             testRun.printClassResults(saveFolder=saveFolder)
#         for n in range(200, 1001, 100):
#             experiments = []
#             experiments.append(Experiment(_modelName="KNN", _kFolds=1, _nNeighbours=n, _embeddingMode='TFIDF', threshold = threshold, _trainFilePath=originalTrainFilePath, _testFilePath='data/testSet.csv'))
#             experiments.append(Experiment(_modelName="KNN", _kFolds=1, _nNeighbours=n, _embeddingMode='BoW', threshold = threshold, _trainFilePath=originalTrainFilePath, _testFilePath='data/testSet.csv'))
#             testRun = Experiments(experiments)
#             testRun.run()
#             testRun.printClassResults(saveFolder=saveFolder)
# for i in ('BoW','TFIDF'):
#     testRun.printClassParameterReports('KNN', 'nNeighbours',saveFolder=saveFolder,Bounds=[i])
#     testRun.printClassParameterReports('KNN', 'threshold',saveFolder=saveFolder,Bounds=[i])

# # RNN threshold choice (also confirm sentence size and dimensions)
# for i in range(0, 1):
#     print('Experiment: ' + str(i + 1))
#     experiments = []
#     params = copy.deepcopy(RNNParams)
#     for j in range(0, 10):
#         threshold = j / 10
#         for sentenceSize in (6,19):
#             params.sentenceSize = sentenceSize
#             for dims in (100,600):
#                 experiments = []    
#                 params.embeddedDims = dims
#                 print('Experiment: ' + str(i + 1) + ' Model: ' + params.modelName + ' Dimensions: ' + str(params.embeddedDims) + ' Epochs: ' + str(params.epochs) + ' Batch: ' + str(params.batchSize) + ' Sentence Size: ' + str(params.sentenceSize))
#                 experiments.append(Experiment(_modelName=params.modelName, _kFolds=1, _embeddedDims=params.embeddedDims, _epochs=params.epochs, _batchSize = params.batchSize, _sentenceSize=params.sentenceSize, threshold = threshold, _trainFilePath=originalTrainFilePath, _testFilePath='data/testSet.csv'))
#                 testRun = Experiments(experiments)
#                 testRun.run()
#                 testRun.printClassResults(saveFolder=saveFolder)
#     testRun = Experiments(experiments)
#     testRun.printClassParameterReports(params.modelName, 'threshold', saveFolder=saveFolder)
#     testRun.printClassParameterReports(params.modelName, 'sentenceSize', saveFolder=saveFolder)
#     testRun.printClassParameterReports(params.modelName, 'embeddedDims', saveFolder=saveFolder)

# # LSTM threshold choice (also confirm sentence size and dimensions)
# for i in range(0, 1):

#     print('Experiment: ' + str(i + 1))
#     params = copy.deepcopy(LSTMParams)
#     for j in range(0, 10):
#         threshold = j / 10
#         experiments = []
#         print('Experiment: ' + str(i + 1) + ' Model: ' + params.modelName + ' Dimensions: ' + str(params.embeddedDims) + ' Epochs: ' + str(params.epochs) + ' Batch: ' + str(params.batchSize) + ' Sentence Size: ' + str(params.sentenceSize))
#         experiments.append(Experiment(_modelName=params.modelName, _kFolds=1, _embeddedDims=params.embeddedDims, _epochs=params.epochs, _batchSize = params.batchSize, _sentenceSize=params.sentenceSize, threshold = threshold, _trainFilePath=originalTrainFilePath, _testFilePath='data/testSet.csv'))
#         testRun = Experiments(experiments)
#         testRun.run()
#         testRun.printClassResults(saveFolder=saveFolder)
#     testRun = Experiments(experiments)
#     testRun.printClassParameterReports(params.modelName, 'threshold', saveFolder=saveFolder)

# # GRU threshold choice (also confirm sentence size and dimensions)
# for i in range(0, 1):

#     print('Experiment: ' + str(i + 1))
#     params = copy.deepcopy(GRUParams)
#     for j in range(0, 10):
#         threshold = j / 10
#         experiments = []
#         print('Experiment: ' + str(i + 1) + ' Model: ' + params.modelName + ' Dimensions: ' + str(params.embeddedDims) + ' Epochs: ' + str(params.epochs) + ' Batch: ' + str(params.batchSize) + ' Sentence Size: ' + str(params.sentenceSize))
#         experiments.append(Experiment(_modelName=params.modelName, _kFolds=1, _embeddedDims=params.embeddedDims, _epochs=params.epochs, _batchSize = params.batchSize, _sentenceSize=params.sentenceSize, threshold = threshold, _trainFilePath=originalTrainFilePath, _testFilePath='data/testSet.csv'))
#         testRun = Experiments(experiments)
#         testRun.run()
#         testRun.printClassResults(saveFolder=saveFolder)
#     testRun = Experiments(experiments)
#     testRun.printClassParameterReports(params.modelName, 'threshold', saveFolder=saveFolder)

# Graph precision/recall vs class with plot for each threshold just to be sure
# experiments = []
# testRun = Experiments(experiments)
# testRun.printCompareParameterClasses(self, 'KNN', classList=augmentedClassKNNList, parameter='threshold', saveFolder='results/Augmented/Threshold')
# testRun.printCompareParameterClasses(self, 'RNN', classList=augmentedClassRNNList, parameter='threshold', saveFolder='results/Augmented/Threshold')
# testRun.printCompareParameterClasses(self, 'LSTM', classList=augmentedClassRNNList, parameter='threshold', saveFolder='results/Augmented/Threshold')
# testRun.printCompareParameterClasses(self, 'GRU', classList=augmentedClassRNNList, parameter='threshold', saveFolder='results/Augmented/Threshold')

# testRun.printCompareParameterClasses(self, 'KNN', classList=balancedClassKNNList, parameter='threshold', saveFolder='results/Balanced/Threshold')
# testRun.printCompareParameterClasses(self, 'RNN', classList=balancedClassRNNList, parameter='threshold', saveFolder='results/Balanced/Threshold')
# testRun.printCompareParameterClasses(self, 'LSTM', classList=balancedClassRNNList, parameter='threshold', saveFolder='results/Balanced/Threshold')
# testRun.printCompareParameterClasses(self, 'GRU', classList=balancedClassRNNList, parameter='threshold', saveFolder='results/Balanced/Threshold')

# testRun.printCompareParameterClasses(self, 'KNN', classList=originalClassKNNList, parameter='threshold', saveFolder='results/Original/Threshold')
# testRun.printCompareParameterClasses(self, 'RNN', classList=originalClassRNNList, parameter='threshold', saveFolder='results/Original/Threshold')
# testRun.printCompareParameterClasses(self, 'LSTM', classList=originalClassRNNList, parameter='threshold', saveFolder='results/Original/Threshold')
# testRun.printCompareParameterClasses(self, 'GRU', classList=originalClassRNNList, parameter='threshold', saveFolder='results/Original/Threshold')

###############################
### Optimised final results ###
###############################

# Double check optimised values, fill in params lists below, check q referenced throughout

# Run through once to gauge running time, then run through multiple times

# saveFolder= [originalSaveFolder + 'Optimal/',balancedSaveFolder + 'Optimal/',augmentedSaveFolder + 'Optimal/']
# trainFilePath=[originalTrainFilePath,balancedTrainFilePath,augmentedTrainFilePath]
# RNNParams = []
# LSTMParams = []
# GRUParams = []

# for q in range(0,len(saveFolder))
    # for i in range(0, 1):
    # # KNN 
    #     print('Experiment: ' + str(i + 1))
    #     threshold = ?
    #     n = ?
    #     experiments = []
    #     experiments.append(Experiment(_modelName="KNN", _kFolds=1, _nNeighbours=n, _embeddingMode='TFIDF', threshold = threshold, _trainFilePath=trainFilePath[q], _testFilePath='data/testSet.csv'))
    #     experiments.append(Experiment(_modelName="KNN", _kFolds=1, _nNeighbours=n, _embeddingMode='BoW', threshold = threshold, _trainFilePath=rainFilePath[q], _testFilePath='data/testSet.csv'))
    #     testRun = Experiments(experiments)
    #     testRun.run()
    #     testRun.printClassResults(saveFolder=saveFolder[q])
    #     testRun.printClassParameterReports('KNN', 'nNeighbours',saveFolder=saveFolder[q],Bounds=[i])
    #     testRun.printClassParameterReports('KNN', 'threshold',saveFolder=saveFolder[q],Bounds=[i])

    # # RNN 
    #     print('Experiment: ' + str(i + 1))
    #     params = copy.deepcopy(RNNParams[q])
    #     threshold = ?   
    #     experiments = []
    #     print('Experiment: ' + str(i + 1) + ' Model: ' + params.modelName + ' Dimensions: ' + str(params.embeddedDims) + ' Epochs: ' + str(params.epochs) + ' Batch: ' + str(params.batchSize) + ' Sentence Size: ' + str(params.sentenceSize))
    #     experiments.append(Experiment(_modelName=params.modelName, _kFolds=1, _embeddedDims=params.embeddedDims, _epochs=params.epochs, _batchSize = params.batchSize, _sentenceSize=params.sentenceSize, threshold = threshold, _trainFilePath=trainFilePath[q], _testFilePath='data/testSet.csv'))
    #     testRun = Experiments(experiments)
    #     testRun.run()
    #     testRun.printClassResults(saveFolder=saveFolder[q])
    #     testRun = Experiments(experiments)

    # # LSTM threshold choice (also confirm sentence size and dimensions)
    # for i in range(0, 1):

    #     print('Experiment: ' + str(i + 1))
    #     params = copy.deepcopy(LSTMParams[q])
    #     threshold = ?
    #     experiments = []
    #     print('Experiment: ' + str(i + 1) + ' Model: ' + params.modelName + ' Dimensions: ' + str(params.embeddedDims) + ' Epochs: ' + str(params.epochs) + ' Batch: ' + str(params.batchSize) + ' Sentence Size: ' + str(params.sentenceSize))
    #     experiments.append(Experiment(_modelName=params.modelName, _kFolds=1, _embeddedDims=params.embeddedDims, _epochs=params.epochs, _batchSize = params.batchSize, _sentenceSize=params.sentenceSize, threshold = threshold, _trainFilePath=trainFilePath[q], _testFilePath='data/testSet.csv'))
    #     testRun = Experiments(experiments)
    #     testRun.run()
    #     testRun.printClassResults(saveFolder=saveFolder[q])
    #     testRun = Experiments(experiments)

    # # GRU threshold choice (also confirm sentence size and dimensions)
    # for i in range(0, 1):

    #     print('Experiment: ' + str(i + 1))
    #     params = copy.deepcopy(GRUParams[q])
    #     threshold = ?
    #     experiments = []
    #     print('Experiment: ' + str(i + 1) + ' Model: ' + params.modelName + ' Dimensions: ' + str(params.embeddedDims) + ' Epochs: ' + str(params.epochs) + ' Batch: ' + str(params.batchSize) + ' Sentence Size: ' + str(params.sentenceSize))
    #     experiments.append(Experiment(_modelName=params.modelName, _kFolds=1, _embeddedDims=params.embeddedDims, _epochs=params.epochs, _batchSize = params.batchSize, _sentenceSize=params.sentenceSize, threshold = threshold, _trainFilePath=trainFilePath[q], _testFilePath='data/testSet.csv'))
    #     testRun = Experiments(experiments)
    #     testRun.run()
    #     testRun.printClassResults(saveFolder=saveFolder[q])
    #     testRun = Experiments(experiments)

    # Graph augmented models against eachother
    # experiments = []
    # testRun = Experiments(experiments)
    # augmentedLoadFolder = 'results/Augmented/Optimal'
    # augmentedLoadFolders = [augmentedLoadFolder,augmentedLoadFolder,augmentedLoadFolder,augmentedLoadFolder]
    # augmentedModelList = ['KNN','RNN','LSTM','GRU']
    # augmentedClassList = [augmentedKNNClassList,augmentedRNNClassList,augmentedLSTMClassList,augmentedGRUClassList]
    # testRun.printCompareExperimentsClasses(self, ModelsList=augmentedModelList, classList=augmentedClassList, loadFolders=augmentedLoadFolders, saveFolder=augmentedLoadFolder)

    # Graph balanced models against eachother
    # experiments = []
    # testRun = Experiments(experiments)
    # balancedLoadFolder = 'results/Balanced/Optimal'
    # balancedLoadFolders = [balancedLoadFolder,balancedLoadFolder,balancedLoadFolder,balancedLoadFolder]
    # balancedModelList = ['KNN','RNN','LSTM','GRU']
    # balancedClassList = [balancedKNNClassList,balancedRNNClassList,balancedLSTMClassList,balancedGRUClassList]
    # testRun.printCompareExperimentsClasses(self, ModelsList=balancedModelList, classList=balancedClassList, loadFolders=balancedLoadFolders, saveFolder=balancedLoadFolder)

    # Graph original models against eachother
    # experiments = []
    # testRun = Experiments(experiments)
    # originalLoadFolder = 'results/Original/Optimal'
    # originalLoadFolders = [originalLoadFolder,originalLoadFolder,originalLoadFolder,originalLoadFolder]
    # originalModelList = ['KNN','RNN','LSTM','GRU']
    # originalClassList = [originalKNNClassList,originalRNNClassList,originalLSTMClassList,originalGRUClassList]
    # testRun.printCompareExperimentsClasses(self, ModelsList=originalModelList, classList=originalClassList, loadFolders=originalLoadFolders, saveFolder=originalLoadFolder)

    # Graph dataset aggregates against eachother
    # experiments = []
    # testRun = Experiments(experiments)
    # aggregateLoadFolders = [augmentedLoadFolder,balancedLoadFolder,originalLoadFolder]
    # aggregateModelList = [augmentedModelList,originalModelList,originalModelList]
    # aggregateClassList = [augmentedClassList,originalClassList,originalClassList]
    # testRun.printCompareExperimentsClasses(self, ModelsList=aggregateModelList, classList=aggregateClassList, loadFolders=aggregateLoadFolders, saveFolder=aggregateLoadFolder)



    # Decide on a dataset, decide on a model / threshold

