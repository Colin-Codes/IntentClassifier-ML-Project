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

# # Neural Network Sentence Size results 2
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
#     for j in range(1, 10):
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

### Balanced dataset ###

balancedSaveFolder='results/Balanced/'
balancedTrainFilePath='data/trainingSet_balanced.csv'
saveFolder = balancedSaveFolder + 'Threshold/'
# KNN threshold and n neighbours choice
# for i in range(0, 1):
#     print('Experiment: ' + str(i + 1))
#     for j in range(1, 10):
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
            testRun.printClassParameterReports(Model, 'embeddedDims', saveFolder=saveFolder, classList=RNNLiveClasses)
        if Model == 'LSTM':
            testRun.printClassParameterReports(Model, 'embeddedDims', saveFolder=saveFolder, classList=LSTMLiveClasses)
        else:
            testRun.printClassParameterReports(Model, 'embeddedDims', saveFolder=saveFolder, classList=GRULiveClasses)