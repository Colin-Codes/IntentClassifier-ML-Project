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

# testRun = Experiments(experiments)
# testRun.printSentenceLengths()

# # Neural Network exploratory experiment
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
#                         testRun.printClassResults(saveFolder='results/Exploratory')


##########################
### Augmented dataset ###
##########################

augmentedSaveFolder='results/Augmented/'
augmentedTrainFilePath='data/trainingSet_augmented.csv'

#region Augmented

#region KNN

# saveFolder='results/Augmented/KNN/'
# # KNN n neighbours and embedding choice
# for Mode in ('BoW','TFIDF'):
#     for n in range(1, 16):
#         print('Model: KNN, Dataset: Augmented, nNeighbours: ' + str(n) + ', Vectorization mode: ' + Mode)
#         experiments = []
#         experiments.append(Experiment(_modelName="KNN", _kFolds=1, _nNeighbours=n, _embeddingMode=Mode, threshold = 0.0, _trainFilePath=augmentedTrainFilePath, _testFilePath='data/testSet.csv'))
#         testRun = Experiments(experiments)
#         testRun.run()
#         testRun.printClassResults(saveFolder=saveFolder)
#     for n in range(20, 101, 10):
#         print('Model: KNN, Dataset: Augmented, nNeighbours: ' + str(n) + ', Vectorization mode: ' + Mode)
#         experiments = []
#         experiments.append(Experiment(_modelName="KNN", _kFolds=1, _nNeighbours=n, _embeddingMode=Mode, threshold = 0.0, _trainFilePath=augmentedTrainFilePath, _testFilePath='data/testSet.csv'))
#         testRun = Experiments(experiments)
#         testRun.run()
#         testRun.printClassResults(saveFolder=saveFolder)
#     for n in range(200, 1001, 100):
#         print('Model: KNN, Dataset: Augmented, nNeighbours: ' + str(n) + ', Vectorization mode: ' + Mode)
#         experiments = []
#         experiments.append(Experiment(_modelName="KNN", _kFolds=1, _nNeighbours=n, _embeddingMode=Mode, threshold = 0.0, _trainFilePath=augmentedTrainFilePath, _testFilePath='data/testSet.csv'))
#         testRun = Experiments(experiments)
#         testRun.run()
#         testRun.printClassResults(saveFolder=saveFolder)
#     testRun.printParameterReports('KNN', 'nNeighbours',saveFolder=saveFolder,Bounds=[Mode])
# testRun.printCompareParameterClasses('KNN', parameter='embeddingMode', saveFolder=saveFolder)
# for variation in [[1,5,10,50,100,500,1000],[1,2,3,4,5,6,7,8,9],[10,11,12,13,14,15],[10,20,30,40,50],[60,70,80,90,100],[100,200,300,400,500],[600,700,800,900,1000]]:
#     testRun.printCompareParameterClasses('KNN', parameter='nNeighbours', parameterValues=variation, saveFolder=saveFolder)

# endregion

#region Epochs / Batch Size

# saveFolder = augmentedSaveFolder + 'EpochsBatchSize/'
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
#             for Batch in (1,32,64,128,37250):
#                 experiments = []
#                 print('Experiment: ' + str(i + 1) + ' Model: ' + Model + ' Dimensions: ' + str(Dims) + ' Epochs: ' + str(Epochs) + ' Batch: ' + str(Batch) + ' Sentence Size: ' + str(SentenceSize))
#                 experiments.append(Experiment(_modelName=Model, _kFolds=1, _embeddedDims=Dims, _epochs=Epochs, _batchSize = Batch, _sentenceSize=SentenceSize, threshold = 0.0, _trainFilePath=augmentedTrainFilePath, _testFilePath='data/testSet.csv'))
#                 testRun = Experiments(experiments)
#                 testRun.run()
#                 testRun.printClassResults(saveFolder=saveFolder)
#         for parameter in ['epochs','batchSize']:
#             testRun.printCompareParameterClasses(Model, parameter=parameter, saveFolder=saveFolder)
#             testRun.printParameterReports(Model, parameter=parameter, saveFolder=saveFolder)

#endregion

#region Dimensions

# saveFolder = augmentedSaveFolder + 'Dimensions/'
# # Neural Network Dimensions experiment
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
#             experiments.append(Experiment(_modelName=Model, _kFolds=1, _embeddedDims=Dims, _epochs=Epochs, _batchSize = Batch, _sentenceSize=SentenceSize, threshold = 0.0, _trainFilePath=augmentedTrainFilePath, _testFilePath='data/testSet.csv'))
#             testRun = Experiments(experiments)
#             testRun.run()
#             testRun.printClassResults(saveFolder=saveFolder)
#         for Dims in range(100, 1001, 100):
#             experiments = []
#             print('Experiment: ' + str(i + 1) + ' Model: ' + Model + ' Dimensions: ' + str(Dims) + ' Epochs: ' + str(Epochs) + ' Batch: ' + str(Batch) + ' Sentence Size: ' + str(SentenceSize))
#             experiments.append(Experiment(_modelName=Model, _kFolds=1, _embeddedDims=Dims, _epochs=Epochs, _batchSize = Batch, _sentenceSize=SentenceSize, threshold = 0.0, _trainFilePath=augmentedTrainFilePath, _testFilePath='data/testSet.csv'))
#             testRun = Experiments(experiments)
#             testRun.run()
#             testRun.printClassResults(saveFolder=saveFolder)
#         for variation in [[10,50,100,300,600,800],[10,20,30,40,50],[60,70,80,90,100],[100,200,300,400,500],[500,600,700,800]]:
#             testRun.printCompareParameterClasses(Model, parameter='embeddedDims', parameterValues=variation, saveFolder=saveFolder)
# testRun.printParameterReports('RNN', 'embeddedDims', saveFolder=saveFolder, classList=RNNAugmentedLiveClasses)
# testRun.printParameterReports('LSTM', 'embeddedDims', saveFolder=saveFolder, classList=LSTMAugmentedLiveClasses)
# testRun.printParameterReports('GRU', 'embeddedDims', saveFolder=saveFolder, classList=GRUAugmentedLiveClasses)

#endregion

#region Sentence Size

saveFolder = augmentedSaveFolder + 'SentenceSize/'
# # Neural Network Sentence Size experiment 
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
#             experiments.append(Experiment(_modelName=Model, _kFolds=1, _embeddedDims=Dims, _epochs=Epochs, _batchSize = Batch, _sentenceSize=SentenceSize, threshold = 0.0, _trainFilePath=augmentedTrainFilePath, _testFilePath='data/testSet.csv'))
#             testRun = Experiments(experiments)
#             testRun.run()
#             testRun.printClassResults(saveFolder=saveFolder)
#     testRun.printParameterReports(Model, 'sentenceSize', saveFolder=saveFolder)
#     for variation in [[5,15,20,25],[5,6,7,8,9],[10,11,12,13,14],[15,16,17,18,19],[20,21,22,23,24,25]]:
#         testRun.printCompareParameterClasses(Model, parameter='sentenceSize', parameterValues=variation, saveFolder=saveFolder)

#endregion

#region Compare classes

# Compare class results to determine dead classes
# testRun = Experiments(experiments)
# for i in ('RNN','GRU','LSTM'):
#     testRun.printCompareClasses(i, loadFolders=['results/Augmented/Dimensions','results/Augmented/EpochsBatchSize','results/Augmented/SentenceSize'], saveFolder='results/Augmented/CompareClasses')
# testRun.printCompareClasses('KNN',loadFolders=['results/Augmented/KNN'], saveFolder='results/Augmented/CompareClasses')

#endregion

#region Define live classes

KNNAugmentedLiveClasses = ['Callback','Error','Information','Leaver','Pricing','Project','Reminder','Report','Status','Template','Weight']
RNNAugmentedLiveClasses = ['Forward','Gables','Leaver','Pricing','Project','Reminder','Report','Status','Template','Weight']
LSTMAugmentedLiveClasses = ['Delivery','Documents','EqualGlass','Error','Feedback','Forward','Gables','Information','Leaver','Logo','Pricing','Project','Reminder','Report','Status','Template','Weight']
GRUAugmentedLiveClasses = ['Documents','EqualGlass','Error','Feedback','Forward','Gables','Information','Leaver','Logo','Pricing','Project','Reminder','Report','Status','Template','Weight']

#endregion

#region Reprint graphs with filtering

## Regenerate results to include filtered versions

# # KNN general results
# testRun = Experiments(experiments)
# for i in ('BoW','TFIDF'):
#     testRun.printParameterReports('KNN', 'nNeighbours',saveFolder='results/Augmented/KNN/',Bounds=[i], classList=KNNLiveClasses)

# # Neural Network Dimensions results
# testRun = Experiments(experiments)
# testRun.printParameterReports('RNN', 'embeddedDims', saveFolder='results/Augmented/Dimensions/', classList=RNNLiveClasses)
# testRun.printParameterReports('LSTM', 'embeddedDims', saveFolder='results/Augmented/Dimensions/', classList=LSTMLiveClasses)
# testRun.printParameterReports('GRU', 'embeddedDims', saveFolder='results/Augmented/Dimensions/', classList=GRULiveClasses)

# # Neural Network Epochs - Batch Size results
# testRun = Experiments(experiments)
# for j in ('epochs','batchSize'):
#     testRun.printParameterReports('RNN', j,saveFolder='results/Augmented/EpochsBatchSize/',classList=RNNLiveClasses)
#     testRun.printParameterReports('LSTM', j,saveFolder='results/Augmented/EpochsBatchSize/',classList=LSTMLiveClasses)
#     testRun.printParameterReports('GRU', j,saveFolder='results/Augmented/EpochsBatchSize/',classList=GRULiveClasses)

# # Neural Network Sentence Size results
# testRun = Experiments(experiments)
# testRun.printParameterReports('RNN', 'sentenceSize', saveFolder='results/Augmented/SentenceSize/', classList=RNNLiveClasses)
# testRun.printParameterReports('LSTM', 'sentenceSize', saveFolder='results/Augmented/SentenceSize/', classList=LSTMLiveClasses)
# testRun.printParameterReports('GRU', 'sentenceSize', saveFolder='results/Augmented/SentenceSize/', classList=GRULiveClasses)

#endregion

#region Optimised parameters

RNNAugmentedParams = Experiment(_modelName='RNN', _kFolds=1, _embeddedDims=600, _epochs=3, _batchSize = 32, _sentenceSize=19, threshold = 0.0, _trainFilePath='data/trainingSet_augmented.csv', _testFilePath='data/testSet.csv')
LSTMAugmentedParams = Experiment(_modelName='LSTM', _kFolds=1, _embeddedDims=300, _epochs=3, _batchSize = 1, _sentenceSize=18, threshold = 0.0, _trainFilePath='data/trainingSet_augmented.csv', _testFilePath='data/testSet.csv')
GRUAugmentedParams = Experiment(_modelName='GRU', _kFolds=1, _embeddedDims=80, _epochs=3, _batchSize = 1, _sentenceSize=21, threshold = 0.0, _trainFilePath='data/trainingSet_augmented.csv', _testFilePath='data/testSet.csv')

#endregion

#region Threshold

# saveFolder='results/Augmented/Threshold/'
# # KNN threshold and n neighbours choice
# for j in range(0, 10):
#     threshold = j / 10
#     Mode = 'TFIDF'
#     neighbours = 10
#     print('Model: KNN, Dataset: Augmented, nNeighbours: ' + str(neighbours) + ', Vectorization mode: ' + Mode + ', Threshold: ' + str(threshold))
#     experiments = []
#     experiments.append(Experiment(_modelName="KNN", _kFolds=1, _nNeighbours=neighbours, _embeddingMode=Mode, threshold = threshold, _trainFilePath=augmentedTrainFilePath, _testFilePath='data/testSet.csv'))
#     testRun = Experiments(experiments)
#     testRun.run()
#     testRun.printClassResults(saveFolder=saveFolder)
# testRun.printParameterReports('KNN', 'threshold',saveFolder=saveFolder)
# testRun.printCompareParameterClasses('KNN', parameter='threshold', saveFolder=saveFolder)

# RNN threshold choice (also confirm sentence size and dimensions)
# experiments = []
# params = copy.deepcopy(RNNAugmentedParams)
# for j in range(0, 10):
#     threshold = j / 10
#     for sentenceSize in (6,19):
#         params.sentenceSize = sentenceSize
#         for dims in (100,600):
#             experiments = []    
#             params.embeddedDims = dims
#             print('Experiment: ' + str(i + 1) + ' Model: ' + params.modelName + ' Dimensions: ' + str(params.embeddedDims) + ' Epochs: ' + str(params.epochs) + ' Batch: ' + str(params.batchSize) + ' Sentence Size: ' + str(params.sentenceSize))
#             experiments.append(Experiment(_modelName=params.modelName, _kFolds=1, _embeddedDims=params.embeddedDims, _epochs=params.epochs, _batchSize = params.batchSize, _sentenceSize=params.sentenceSize, threshold = threshold, _trainFilePath='data/trainingSet_augmented.csv', _testFilePath='data/testSet.csv'))
#             testRun = Experiments(experiments)
#             testRun.run()
#             testRun.printClassResults(saveFolder=saveFolder)
#testRun.printParameterReports(params.modelName, 'threshold', classList=RNNAugmentedLiveClasses, saveFolder=saveFolder)
#estRun.printCompareParameterClasses(params.modelName, parameter='threshold', saveFolder=saveFolder)

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
#     testRun.printParameterReports(params.modelName, 'threshold', saveFolder=saveFolder)
#     testRun.printCompareParameterClasses(params.modelName, parameter='threshold', saveFolder=saveFolder)

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
#     testRun.printParameterReports(params.modelName, 'threshold', saveFolder=saveFolder)
#     testRun.printCompareParameterClasses(params.modelName, parameter='threshold', saveFolder=saveFolder)

#endregion

#endregion

########################
### Balanced dataset ###
########################

balancedSaveFolder='results/Balanced/'
balancedTrainFilePath='data/trainingSet_balanced.csv'

#region Balanced

#region KNN

# saveFolder='results/Balanced/KNN/'
# # KNN n neighbours and embedding choice
# for Mode in ('BoW','TFIDF'):
#     for n in range(1, 16):
#         print('Model: KNN, Dataset: Balanced, nNeighbours: ' + str(n) + ', Vectorization mode: ' + Mode)
#         experiments = []
#         experiments.append(Experiment(_modelName="KNN", _kFolds=1, _nNeighbours=n, _embeddingMode=Mode, threshold = 0.0, _trainFilePath=balancedTrainFilePath, _testFilePath='data/testSet.csv'))
#         testRun = Experiments(experiments)
#         testRun.run()
#         testRun.printClassResults(saveFolder=saveFolder)
#     for n in range(20, 101, 10):
#         print('Model: KNN, Dataset: Balanced, nNeighbours: ' + str(n) + ', Vectorization mode: ' + Mode)
#         experiments = []
#         experiments.append(Experiment(_modelName="KNN", _kFolds=1, _nNeighbours=n, _embeddingMode=Mode, threshold = 0.0, _trainFilePath=balancedTrainFilePath, _testFilePath='data/testSet.csv'))
#         testRun = Experiments(experiments)
#         testRun.run()
#         testRun.printClassResults(saveFolder=saveFolder)
#     for n in range(200, 1001, 100):
#         print('Model: KNN, Dataset: Balanced, nNeighbours: ' + str(n) + ', Vectorization mode: ' + Mode)
#         experiments = []
#         experiments.append(Experiment(_modelName="KNN", _kFolds=1, _nNeighbours=n, _embeddingMode=Mode, threshold = 0.0, _trainFilePath=balancedTrainFilePath, _testFilePath='data/testSet.csv'))
#         testRun = Experiments(experiments)
#         testRun.run()
#         testRun.printClassResults(saveFolder=saveFolder)
#     testRun.printParameterReports('KNN', 'nNeighbours',saveFolder=saveFolder,Bounds=[Mode])
# testRun.printCompareParameterClasses('KNN', parameter='embeddingMode', saveFolder=saveFolder)
# for variation in [[1,5,10,50,100,500,1000],[1,2,3,4,5,6,7,8,9],[10,11,12,13,14,15],[10,20,30,40,50],[60,70,80,90,100],[100,200,300,400,500],[600,700,800,900,1000]]:
#     testRun.printCompareParameterClasses('KNN', parameter='nNeighbours', parameterValues=variation, saveFolder=saveFolder)

# endregion

#region Epochs / Batch Size

# saveFolder= balancedSaveFolder + 'EpochsBatchSize/'
# # Neural Network Epochs - Batch Size experiment
# for i in range(0, 1):
#     print('Experiment: ' + str(i + 1))
#     experiments = []
#     for Model in ('RNN','LSTM','GRU'):
#         Dims = 300
#         SentenceSize = 20
#         #for Epochs in (1,3,5):
#         for Epochs in (7,10):
#             for Batch in (1,32,64,128,7450):
#                 experiments = []
#                 print('Experiment: ' + str(i + 1) + ' Model: ' + Model + ' Dimensions: ' + str(Dims) + ' Epochs: ' + str(Epochs) + ' Batch: ' + str(Batch) + ' Sentence Size: ' + str(SentenceSize))
#                 experiments.append(Experiment(_modelName=Model, _kFolds=1, _embeddedDims=Dims, _epochs=Epochs, _batchSize = Batch, _sentenceSize=SentenceSize, threshold = 0.0, _trainFilePath=balancedTrainFilePath, _testFilePath='data/testSet.csv'))
#                 testRun = Experiments(experiments)
#                 testRun.run()
#                 testRun.printClassResults(saveFolder=saveFolder)
#         for parameter in ['epochs','batchSize']:
#             testRun.printCompareParameterClasses(Model, parameter=parameter, saveFolder=saveFolder)
#             testRun.printParameterReports(Model, parameter, saveFolder=saveFolder)

#endregion

#region Dimensions

# saveFolder= balancedSaveFolder + 'Dimensions/'
# # Neural Network Dimensions experiment
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
#             experiments.append(Experiment(_modelName=Model, _kFolds=1, _embeddedDims=Dims, _epochs=Epochs, _batchSize = Batch, _sentenceSize=SentenceSize, threshold = 0.0, _trainFilePath=balancedTrainFilePath, _testFilePath='data/testSet.csv'))
#             testRun = Experiments(experiments)
#             testRun.run()
#             testRun.printClassResults(saveFolder=saveFolder)
#         for Dims in range(100, 801, 100):
#             experiments = []
#             print('Experiment: ' + str(i + 1) + ' Model: ' + Model + ' Dimensions: ' + str(Dims) + ' Epochs: ' + str(Epochs) + ' Batch: ' + str(Batch) + ' Sentence Size: ' + str(SentenceSize))
#             experiments.append(Experiment(_modelName=Model, _kFolds=1, _embeddedDims=Dims, _epochs=Epochs, _batchSize = Batch, _sentenceSize=SentenceSize, threshold = 0.0, _trainFilePath=balancedTrainFilePath, _testFilePath='data/testSet.csv'))
#             testRun = Experiments(experiments)
#             testRun.run()
#             testRun.printClassResults(saveFolder=saveFolder)
#         testRun.printParameterReports(Model, 'embeddedDims', saveFolder=saveFolder)
#         for variation in [[10,50,100,300,600,800],[10,20,30,40,50],[60,70,80,90,100],[100,200,300,400,500],[500,600,700,800]]:
#             testRun.printCompareParameterClasses(Model, parameter='embeddedDims', parameterValues=variation, saveFolder=saveFolder)

#endregion

#region Sentence Size

saveFolder= balancedSaveFolder + 'SentenceSize/'
# Neural Network Sentence Size
#     experiments = []
#     for Model in ('RNN','LSTM','GRU'):
#         Epochs = 5
#         Batch = 1
#         Dims = 100
#         for SentenceSize in range(5,26):
#             experiments = []
#             print('Experiment: ' + str(i + 1) + ' Model: ' + Model + ' Dimensions: ' + str(Dims) + ' Epochs: ' + str(Epochs) + ' Batch: ' + str(Batch) + ' Sentence Size: ' + str(SentenceSize))
#             experiments.append(Experiment(_modelName=Model, _kFolds=1, _embeddedDims=Dims, _epochs=Epochs, _batchSize = Batch, _sentenceSize=SentenceSize, threshold = 0.0, _trainFilePath=balancedTrainFilePath, _testFilePath='data/testSet.csv'))
#             testRun = Experiments(experiments)
#             testRun.run()
#             testRun.printClassResults(saveFolder=saveFolder)
#         if Model == 'RNN':
#             testRun.printParameterReports(Model, 'sentenceSize', saveFolder=saveFolder)#, classList=RNNAugmentedLiveClasses
#         if Model == 'LSTM':
#             testRun.printParameterReports(Model, 'sentenceSize', saveFolder=saveFolder)#, classList=LSTMAugmentedLiveClasses
#         else:
#             testRun.printParameterReports(Model, 'sentenceSize', saveFolder=saveFolder)#, classList=GRUAugmentedLiveClasses
#     for variation in [[5,15,20,25],[5,6,7,8,9],[10,11,12,13,14],[15,16,17,18,19],[20,21,22,23,24,25]]:
#         testRun.printCompareParameterClasses(Model, parameter='sentenceSize', parameterValues=variation, saveFolder=saveFolder)

#endregion

#region Compare classes

# Compare class results to determine dead classes
# testRun = Experiments(experiments)
# for i in ('RNN','GRU','LSTM'):
#     testRun.printCompareClasses(i, loadFolders=['results/Balanced/Dimensions','results/Balanced/EpochsBatchSize','results/Balanced/SentenceSize'], saveFolder='results/Balanced/CompareClasses')
# testRun.printCompareClasses('KNN',loadFolders=['results/Balanced/KNN'], saveFolder='results/Balanced/CompareClasses')

#endregion

#region Define live classes

# KNNBalancedLiveClasses = ['Callback','Error','Information','Leaver','Pricing','Project','Reminder','Report','Status','Template','Weight']
# RNNBalancedLiveClasses = ['Forward','Gables','Leaver','Pricing','Project','Reminder','Report','Status','Template','Weight']
# LSTMBalancedLiveClasses = ['Delivery','Documents','EqualGlass','Error','Feedback','Forward','Gables','Information','Leaver','Logo','Pricing','Project','Reminder','Report','Status','Template','Weight']
# GRUBalancedLiveClasses = ['Documents','EqualGlass','Error','Feedback','Forward','Gables','Information','Leaver','Logo','Pricing','Project','Reminder','Report','Status','Template','Weight']

#endregion

#region Reprint graphs with filtering

## Regenerate results to include filtered versions

# # KNN general results
# testRun = Experiments(experiments)
# for i in ('BoW','TFIDF'):
#     testRun.printParameterReports('KNN', 'nNeighbours',saveFolder='results/Balanced/KNN/',Bounds=[i], classList=KNNBalancedLiveClasses)

# # Neural Network Dimensions results
# testRun = Experiments(experiments)
# testRun.printParameterReports('RNN', 'embeddedDims', saveFolder='results/Balanced/Dimensions/', classList=RNNBalancedLiveClasses)
# testRun.printParameterReports('LSTM', 'embeddedDims', saveFolder='results/Balanced/Dimensions/', classList=LSTMBalancedLiveClasses)
# testRun.printParameterReports('GRU', 'embeddedDims', saveFolder='results/Balanced/Dimensions/', classList=GRUBalancedLiveClasses)

# # Neural Network Epochs - Batch Size results
# testRun = Experiments(experiments)
# for j in ('epochs','batchSize'):
#     testRun.printParameterReports('RNN', j,saveFolder='results/Balanced/EpochsBatchSize/',classList=RNNBalancedLiveClasses)
#     testRun.printParameterReports('LSTM', j,saveFolder='results/Balanced/EpochsBatchSize/',classList=LSTMBalancedLiveClasses)
#     testRun.printParameterReports('GRU', j,saveFolder='results/Balanced/EpochsBatchSize/',classList=GRUBalancedLiveClasses)

# # Neural Network Sentence Size results
# testRun = Experiments(experiments)
# testRun.printParameterReports('RNN', 'sentenceSize', saveFolder='results/Balanced/SentenceSize/', classList=RNNBalancedLiveClasses)
# testRun.printParameterReports('LSTM', 'sentenceSize', saveFolder='results/Balanced/SentenceSize/', classList=LSTMBalancedLiveClasses)
# testRun.printParameterReports('GRU', 'sentenceSize', saveFolder='results/Balanced/SentenceSize/', classList=GRUBalancedLiveClasses)

#endregion

#region Optimised parameters

RNNBalancedParams = Experiment(_modelName='RNN', _kFolds=1, _embeddedDims=100, _epochs=5, _batchSize = 32, _sentenceSize=5, threshold = 0.0, _trainFilePath=balancedTrainFilePath, _testFilePath='data/testSet.csv')
LSTMBalancedParams = Experiment(_modelName='LSTM', _kFolds=1, _embeddedDims=60, _epochs=10, _batchSize = 32, _sentenceSize=18, threshold = 0.0, _trainFilePath=balancedTrainFilePath, _testFilePath='data/testSet.csv')
GRUBalancedParams = Experiment(_modelName='GRU', _kFolds=1, _embeddedDims=300, _epochs=10, _batchSize = 1, _sentenceSize=14, threshold = 0.0, _trainFilePath=balancedTrainFilePath, _testFilePath='data/testSet.csv')

# #endregion

#region Threshold

# saveFolder='results/Balanced/Threshold/'
# # KNN threshold and n neighbours choice
# for j in range(0, 10):
#     threshold = j / 10
#     Mode = 'TFIDF'
#     neighbours = 10
#     print('Model: KNN, Dataset: Balanced, nNeighbours: ' + str(neighbours) + ', Vectorization mode: ' + Mode + ', Threshold: ' + str(threshold))
#     experiments = []
#     experiments.append(Experiment(_modelName="KNN", _kFolds=1, _nNeighbours=neighbours, _embeddingMode=Mode, threshold = threshold, _trainFilePath=balancedTrainFilePath, _testFilePath='data/testSet.csv'))
#     testRun = Experiments(experiments)
#     testRun.run()
#     testRun.printClassResults(saveFolder=saveFolder)
# testRun.printParameterReports('KNN', 'threshold',saveFolder=saveFolder)
# testRun.printCompareParameterClasses('KNN', parameter='threshold', saveFolder=saveFolder)

# RNN threshold choice (also confirm sentence size and dimensions)
# for i in range(0, 1):
#     print('Experiment: ' + str(i + 1))
#     experiments = []
#     params = copy.deepcopy(RNNBalancedParams)
#     for j in range(0, 10):
#         threshold = j / 10
#         experiments = []    
#         print('Experiment: ' + str(i + 1) + ' Model: ' + params.modelName + ' Dimensions: ' + str(params.embeddedDims) + ' Epochs: ' + str(params.epochs) + ' Batch: ' + str(params.batchSize) + ' Sentence Size: ' + str(params.sentenceSize))
#         experiments.append(Experiment(_modelName=params.modelName, _kFolds=1, _embeddedDims=params.embeddedDims, _epochs=params.epochs, _batchSize = params.batchSize, _sentenceSize=params.sentenceSize, threshold = threshold, _trainFilePath=balancedTrainFilePath, _testFilePath='data/testSet.csv'))
#         testRun = Experiments(experiments)
#         testRun.run()
#         testRun.printClassResults(saveFolder=saveFolder)
#     testRun = Experiments(experiments)
#     testRun.printParameterReports(params.modelName, 'threshold', saveFolder=saveFolder)
#     testRun.printParameterReports(params.modelName, 'sentenceSize', saveFolder=saveFolder)
#     testRun.printParameterReports(params.modelName, 'embeddedDims', saveFolder=saveFolder)
#     testRun.printCompareParameterClasses(params.modelName, parameter='threshold', saveFolder=saveFolder)

# # LSTM threshold choice (also confirm sentence size and dimensions)
# for i in range(0, 1):

#     print('Experiment: ' + str(i + 1))
#     params = copy.deepcopy(LSTMBalancedParams)
#     for j in range(0, 10):
#         threshold = j / 10
#         experiments = []
#         print('Experiment: ' + str(i + 1) + ' Model: ' + params.modelName + ' Dimensions: ' + str(params.embeddedDims) + ' Epochs: ' + str(params.epochs) + ' Batch: ' + str(params.batchSize) + ' Sentence Size: ' + str(params.sentenceSize))
#         experiments.append(Experiment(_modelName=params.modelName, _kFolds=1, _embeddedDims=params.embeddedDims, _epochs=params.epochs, _batchSize = params.batchSize, _sentenceSize=params.sentenceSize, threshold = threshold, _trainFilePath=balancedTrainFilePath, _testFilePath='data/testSet.csv'))
#         testRun = Experiments(experiments)
#         testRun.run()
#         testRun.printClassResults(saveFolder=saveFolder)
#     testRun = Experiments(experiments)
#     testRun.printParameterReports(params.modelName, 'threshold', saveFolder=saveFolder)
#     testRun.printCompareParameterClasses(params.modelName, parameter='threshold', saveFolder=saveFolder)

# # GRU threshold choice (also confirm sentence size and dimensions)
# for i in range(0, 1):

#     print('Experiment: ' + str(i + 1))
#     params = copy.deepcopy(GRUBalancedParams)
#     for j in range(0, 10):
#         threshold = j / 10
#         experiments = []
#         print('Experiment: ' + str(i + 1) + ' Model: ' + params.modelName + ' Dimensions: ' + str(params.embeddedDims) + ' Epochs: ' + str(params.epochs) + ' Batch: ' + str(params.batchSize) + ' Sentence Size: ' + str(params.sentenceSize))
#         experiments.append(Experiment(_modelName=params.modelName, _kFolds=1, _embeddedDims=params.embeddedDims, _epochs=params.epochs, _batchSize = params.batchSize, _sentenceSize=params.sentenceSize, threshold = threshold, _trainFilePath=balancedTrainFilePath, _testFilePath='data/testSet.csv'))
#         testRun = Experiments(experiments)
#         testRun.run()
#         testRun.printClassResults(saveFolder=saveFolder)
#     testRun = Experiments(experiments)
#     testRun.printParameterReports(params.modelName, 'threshold', saveFolder=saveFolder)
#     testRun.printCompareParameterClasses(params.modelName, parameter='threshold', saveFolder=saveFolder)

#endregion

#endregion

########################
### Original dataset ###
########################

originalSaveFolder='results/Original/'
originalTrainFilePath='data/trainingSet.csv'

#region Original

#region KNN

# saveFolder='results/Original/KNN/'
# # KNN n neighbours and embedding choice
# for Mode in ('BoW','TFIDF'):
#     for n in range(1, 16):
#         print('Model: KNN, Dataset: Original, nNeighbours: ' + str(n) + ', Vectorization mode: ' + Mode)
#         experiments = []
#         experiments.append(Experiment(_modelName="KNN", _kFolds=1, _nNeighbours=n, _embeddingMode=Mode, threshold = 0.0, _trainFilePath=originalTrainFilePath, _testFilePath='data/testSet.csv'))
#         testRun = Experiments(experiments)
#         testRun.run()
#         testRun.printClassResults(saveFolder=saveFolder)
#     for n in range(20, 101, 10):
#         print('Model: KNN, Dataset: Original, nNeighbours: ' + str(n) + ', Vectorization mode: ' + Mode)
#         experiments = []
#         experiments.append(Experiment(_modelName="KNN", _kFolds=1, _nNeighbours=n, _embeddingMode=Mode, threshold = 0.0, _trainFilePath=originalTrainFilePath, _testFilePath='data/testSet.csv'))
#         testRun = Experiments(experiments)
#         testRun.run()
#         testRun.printClassResults(saveFolder=saveFolder)
#     for n in range(200, 701, 100):
#         print('Model: KNN, Dataset: Original, nNeighbours: ' + str(n) + ', Vectorization mode: ' + Mode)
#         experiments = []
#         experiments.append(Experiment(_modelName="KNN", _kFolds=1, _nNeighbours=n, _embeddingMode=Mode, threshold = 0.0, _trainFilePath=originalTrainFilePath, _testFilePath='data/testSet.csv'))
#         testRun = Experiments(experiments)
#         testRun.run()
#         testRun.printClassResults(saveFolder=saveFolder)
#     testRun.printParameterReports('KNN', 'nNeighbours',saveFolder=saveFolder,Bounds=[Mode])
# testRun.printCompareParameterClasses('KNN', parameter='embeddingMode', saveFolder=saveFolder)
# for variation in [[1,5,10,50,100,500,1000],[1,2,3,4,5,6,7,8,9],[10,11,12,13,14,15],[10,20,30,40,50],[60,70,80,90,100],[100,200,300,400,500,600,700]]:
#     testRun.printCompareParameterClasses('KNN', parameter='nNeighbours', parameterValues=variation, saveFolder=saveFolder)

# endregion

#region Epochs / Batch Size

# saveFolder= originalSaveFolder + 'EpochsBatchSize/'
# # Neural Network Epochs - Batch Size experiment
# for i in range(0, 1):
#     print('Experiment: ' + str(i + 1))
#     experiments = []
#     #for Model in ('RNN','LSTM','GRU'):
#     for Model in ['GRU']:
#         Dims = 300
#         SentenceSize = 20
#         #for Epochs in (1,3,5,7,10,15,20,30,40,50,60,70,80,90,100):
#         for Epochs in (1,3,5,7,10,15,20,30,50,70):
#             for Batch in (1,32,64,128,768):
#                 experiments = []
#                 print('Experiment: ' + str(i + 1) + ' Model: ' + Model + ' Dimensions: ' + str(Dims) + ' Epochs: ' + str(Epochs) + ' Batch: ' + str(Batch) + ' Sentence Size: ' + str(SentenceSize))
#                 experiments.append(Experiment(_modelName=Model, _kFolds=1, _embeddedDims=Dims, _epochs=Epochs, _batchSize = Batch, _sentenceSize=SentenceSize, threshold = 0.0, _trainFilePath=originalTrainFilePath, _testFilePath='data/testSet.csv'))
#                 testRun = Experiments(experiments)
#                 testRun.run()
#                 testRun.printClassResults(saveFolder=saveFolder)
#         for parameter in ['epochs','batchSize']:
#             testRun.printCompareParameterClasses(Model, parameter=parameter, saveFolder=saveFolder)
#             testRun.printParameterReports(Model, parameter, saveFolder=saveFolder)

# Based on results, do more exploration?

#endregion

#region Dimensions

# saveFolder= originalSaveFolder + 'Dimensions/'
# # Neural Network Dimensions experiment 2 - increments of 10, 10-100
# for i in range(0, 1):
#     print('Experiment: ' + str(i + 1))
#     experiments = []
#     for Model in ('RNN','LSTM','GRU'):
#         SentenceSize = 20
#         Epochs = 50
#         Batch = 1
#         for Dims in range(10, 91, 10):
#             experiments = []
#             print('Experiment: ' + str(i + 1) + ' Model: ' + Model + ' Dimensions: ' + str(Dims) + ' Epochs: ' + str(Epochs) + ' Batch: ' + str(Batch) + ' Sentence Size: ' + str(SentenceSize))
#             experiments.append(Experiment(_modelName=Model, _kFolds=1, _embeddedDims=Dims, _epochs=Epochs, _batchSize = Batch, _sentenceSize=SentenceSize, threshold = 0.0, _trainFilePath=originalTrainFilePath, _testFilePath='data/testSet.csv'))
#             testRun = Experiments(experiments)
#             testRun.run()
#             testRun.printClassResults(saveFolder=saveFolder)
#         for Dims in range(100, 801, 100):
#             experiments = []
#             print('Experiment: ' + str(i + 1) + ' Model: ' + Model + ' Dimensions: ' + str(Dims) + ' Epochs: ' + str(Epochs) + ' Batch: ' + str(Batch) + ' Sentence Size: ' + str(SentenceSize))
#             experiments.append(Experiment(_modelName=Model, _kFolds=1, _embeddedDims=Dims, _epochs=Epochs, _batchSize = Batch, _sentenceSize=SentenceSize, threshold = 0.0, _trainFilePath=originalTrainFilePath, _testFilePath='data/testSet.csv'))
#             testRun = Experiments(experiments)
#             testRun.run()
#             testRun.printClassResults(saveFolder=saveFolder)
#         testRun.printParameterReports(Model, 'embeddedDims', saveFolder=saveFolder)
#         for variation in [[10,50,100,300,600,800],[10,20,30,40,50],[60,70,80,90,100],[100,200,300,400,500],[500,600,700,800]]:
#             testRun.printCompareParameterClasses(Model, parameter='embeddedDims', parameterValues=variation, saveFolder=saveFolder)


# Based on results, do more exploration? Any optimisation to implement below?

#endregion

#region Sentence Size

# saveFolder= originalSaveFolder + 'SentenceSize/'
# experiments = []
# for Model in ('RNN','LSTM','GRU'):
#     Epochs = 50
#     Batch = 1
#     Dims = 80
#     for SentenceSize in range(5,26):
#         experiments = []
#         print('Experiment: ' + str(i + 1) + ' Model: ' + Model + ' Dimensions: ' + str(Dims) + ' Epochs: ' + str(Epochs) + ' Batch: ' + str(Batch) + ' Sentence Size: ' + str(SentenceSize))
#         experiments.append(Experiment(_modelName=Model, _kFolds=1, _embeddedDims=Dims, _epochs=Epochs, _batchSize = Batch, _sentenceSize=SentenceSize, threshold = 0.0, _trainFilePath=originalTrainFilePath, _testFilePath='data/testSet.csv'))
    # testRun = Experiments(experiments)
#         testRun.run()
#         testRun.printClassResults(saveFolder=saveFolder)
#     testRun.printParameterReports(Model, 'sentenceSize', saveFolder=saveFolder)
    # for variation in [[5,15,20,25],[5,6,7,8,9],[10,11,12,13,14],[15,16,17,18,19],[20,21,22,23,24,25]]:
    #     testRun.printCompareParameterClasses(Model, parameter='sentenceSize', parameterValues=variation, saveFolder=saveFolder)

# Based on results, do more exploration? Any optimisation to implement below?

#endregion

#region Compare classes

# Compare class results to determine dead classes
# testRun = Experiments(experiments)
# for i in ('RNN','GRU','LSTM'):
#     testRun.printCompareClasses(i, loadFolders=['results/Original/Dimensions','results/Original/EpochsBatchSize','results/Original/SentenceSize'], saveFolder='results/Original/CompareClasses')
# testRun.printCompareClasses('KNN',loadFolders=['results/Original/KNN'], saveFolder='results/Original/CompareClasses')

#endregion

#region Define live classes

KNNOriginalLiveClasses = ['Callback','Error','Information','Leaver','Pricing','Project','Reminder','Report','Status','Template','Weight']
RNNOriginalLiveClasses = ['Forward','Gables','Leaver','Pricing','Project','Reminder','Report','Status','Template','Weight']
LSTMOriginalLiveClasses = ['Delivery','Documents','EqualGlass','Error','Feedback','Forward','Gables','Information','Leaver','Logo','Pricing','Project','Reminder','Report','Status','Template','Weight']
GRUOriginalLiveClasses = ['Documents','EqualGlass','Error','Feedback','Forward','Gables','Information','Leaver','Logo','Pricing','Project','Reminder','Report','Status','Template','Weight']

#endregion

#region Reprint graphs with filtering

## Regenerate results to include filtered versions

# # KNN general results
# testRun = Experiments(experiments)
# for i in ('BoW','TFIDF'):
#     testRun.printParameterReports('KNN', 'nNeighbours',saveFolder='results/Original/KNN/',Bounds=[i], classList=KNNOriginalLiveClasses)

# # Neural Network Dimensions results
# testRun = Experiments(experiments)
# testRun.printParameterReports('RNN', 'embeddedDims', saveFolder='results/Original/Dimensions/', classList=RNNOriginalLiveClasses)
# testRun.printParameterReports('LSTM', 'embeddedDims', saveFolder='results/Original/Dimensions/', classList=LSTMOriginalLiveClasses)
# testRun.printParameterReports('GRU', 'embeddedDims', saveFolder='results/Original/Dimensions/', classList=GRUOriginalLiveClasses)

# # Neural Network Epochs - Batch Size results
# testRun = Experiments(experiments)
# for j in ('epochs','batchSize'):
#     testRun.printParameterReports('RNN', j,saveFolder='results/Original/EpochsBatchSize/',classList=RNNOriginalLiveClasses)
#     testRun.printParameterReports('LSTM', j,saveFolder='results/Original/EpochsBatchSize/',classList=LSTMOriginalLiveClasses)
#     testRun.printParameterReports('GRU', j,saveFolder='results/Original/EpochsBatchSize/',classList=GRUOriginalLiveClasses)

# # Neural Network Sentence Size results
# testRun = Experiments(experiments)
# testRun.printParameterReports('RNN', 'sentenceSize', saveFolder='results/Original/SentenceSize/', classList=RNNOriginalLiveClasses)
# testRun.printParameterReports('LSTM', 'sentenceSize', saveFolder='results/Original/SentenceSize/', classList=LSTMOriginalLiveClasses)
# testRun.printParameterReports('GRU', 'sentenceSize', saveFolder='results/Original/SentenceSize/', classList=GRUOriginalLiveClasses)

#endregion

#region Optimised parameters

RNNOriginalParams = Experiment(_modelName='RNN', _kFolds=1, _embeddedDims=30, _epochs=50, _batchSize = 32, _sentenceSize=5, threshold = 0.0, _trainFilePath=originalTrainFilePath, _testFilePath='data/testSet.csv')
LSTMOriginalParams = Experiment(_modelName='LSTM', _kFolds=1, _embeddedDims=30, _epochs=70, _batchSize = 1, _sentenceSize=16, threshold = 0.0, _trainFilePath=originalTrainFilePath, _testFilePath='data/testSet.csv')
GRUOriginalParams = Experiment(_modelName='GRU', _kFolds=1, _embeddedDims=60, _epochs=50, _batchSize = 1, _sentenceSize=21, threshold = 0.0, _trainFilePath=originalTrainFilePath, _testFilePath='data/testSet.csv')

# #endregion
#region Threshold

# saveFolder='results/Original/Threshold/'
# # KNN threshold and n neighbours choice
# for j in range(0, 10):
#     threshold = j / 10
#     Mode = 'TFIDF'
#     neighbours = 9
#     print('Model: KNN, Dataset: Original, nNeighbours: ' + str(neighbours) + ', Vectorization mode: ' + Mode + ', Threshold: ' + str(threshold))
#     experiments = []
#     experiments.append(Experiment(_modelName="KNN", _kFolds=1, _nNeighbours=neighbours, _embeddingMode=Mode, threshold = threshold, _trainFilePath=originalTrainFilePath, _testFilePath='data/testSet.csv'))
#     testRun = Experiments(experiments)
#     testRun.run()
#     testRun.printClassResults(saveFolder=saveFolder)
# testRun.printParameterReports('KNN', 'threshold',saveFolder=saveFolder)
# testRun.printCompareParameterClasses('KNN', parameter='threshold', saveFolder=saveFolder)

# RNN threshold choice
# experiments = []
# params = copy.deepcopy(RNNOriginalParams)
# for j in range(0, 10):
#     threshold = j / 10
#     experiments = []    
#     print('Model: ' + params.modelName + ' Dimensions: ' + str(params.embeddedDims) + ' Epochs: ' + str(params.epochs) + ' Batch: ' + str(params.batchSize) + ' Sentence Size: ' + str(params.sentenceSize) + ', threshold: ' + str(threshold))
#     experiments.append(Experiment(_modelName=params.modelName, _kFolds=1, _embeddedDims=params.embeddedDims, _epochs=params.epochs, _batchSize = params.batchSize, _sentenceSize=params.sentenceSize, threshold = threshold, _trainFilePath=originalTrainFilePath, _testFilePath='data/testSet.csv'))
#     testRun = Experiments(experiments)
#     testRun.run()
#     testRun.printClassResults(saveFolder=saveFolder)
# testRun = Experiments(experiments)
# testRun.printParameterReports(params.modelName, 'threshold', saveFolder=saveFolder)
# testRun.printCompareParameterClasses(params.modelName, parameter='threshold', saveFolder=saveFolder)

# # LSTM threshold choice (also confirm sentence size and dimensions)
# params = copy.deepcopy(LSTMOriginalParams)
# for j in range(0, 10):
#     threshold = j / 10
#     experiments = []
#     print('Model: ' + params.modelName + ' Dimensions: ' + str(params.embeddedDims) + ' Epochs: ' + str(params.epochs) + ' Batch: ' + str(params.batchSize) + ' Sentence Size: ' + str(params.sentenceSize) + ', threshold: ' + str(threshold))
#     experiments.append(Experiment(_modelName=params.modelName, _kFolds=1, _embeddedDims=params.embeddedDims, _epochs=params.epochs, _batchSize = params.batchSize, _sentenceSize=params.sentenceSize, threshold = threshold, _trainFilePath=originalTrainFilePath, _testFilePath='data/testSet.csv'))
#     testRun = Experiments(experiments)
#     testRun.run()
#     testRun.printClassResults(saveFolder=saveFolder)
# testRun = Experiments(experiments)
# testRun.printParameterReports(params.modelName, 'threshold', saveFolder=saveFolder)
# testRun.printCompareParameterClasses(params.modelName, parameter='threshold', saveFolder=saveFolder)

# # GRU threshold choice (also confirm sentence size and dimensions)
# params = copy.deepcopy(GRUOriginalParams)
# for j in range(0, 10):
#     threshold = j / 10
#     experiments = []
#     print('Model: ' + params.modelName + ' Dimensions: ' + str(params.embeddedDims) + ' Epochs: ' + str(params.epochs) + ' Batch: ' + str(params.batchSize) + ' Sentence Size: ' + str(params.sentenceSize) + ', threshold: ' + str(threshold))
#     experiments.append(Experiment(_modelName=params.modelName, _kFolds=1, _embeddedDims=params.embeddedDims, _epochs=params.epochs, _batchSize = params.batchSize, _sentenceSize=params.sentenceSize, threshold = threshold, _trainFilePath=originalTrainFilePath, _testFilePath='data/testSet.csv'))
#     testRun = Experiments(experiments)
#     testRun.run()
#     testRun.printClassResults(saveFolder=saveFolder)
# testRun = Experiments(experiments)
# testRun.printParameterReports(params.modelName, 'threshold', saveFolder=saveFolder)
# testRun.printCompareParameterClasses(params.modelName, parameter='threshold', saveFolder=saveFolder)

#endregion

#endregion

###############################
### Optimised final results ###
###############################

#region Re-run filtered threshold reports
# for folder in ['results/Augmented/Threshold','results/Balanced/Threshold','results/Original/Threshold']:
#     for Model in [['RNN',RNNAugmentedLiveClasses],['LSTM',LSTMAugmentedLiveClasses],['GRU',GRUAugmentedLiveClasses]]:
#         testRun.printParameterReports(Model[0], 'threshold', classList=Model[1], saveFolder=folder)
#         testRun.printCompareParameterClasses(Model[0], parameter='threshold', saveFolder=folder)

#endregion

#region Check and respecify optimal parameters



#endregion

# Double check optimised values, fill in params lists below, check q referenced throughout

# Run through once to gauge running time, then run through multiple times
RNNAugmentedParams = Experiment(_modelName='RNN', _kFolds=1, _embeddedDims=100, _epochs=3, _batchSize = 64, _sentenceSize=6, threshold = 0.9, _trainFilePath=augmentedTrainFilePath, _testFilePath='data/testSet.csv')
LSTMAugmentedParams = Experiment(_modelName='LSTM', _kFolds=1, _embeddedDims=300, _epochs=3, _batchSize = 64, _sentenceSize=18, threshold = 0.9, _trainFilePath=augmentedTrainFilePath, _testFilePath='data/testSet.csv')
GRUAugmentedParams = Experiment(_modelName='GRU', _kFolds=1, _embeddedDims=100, _epochs=3, _batchSize = 64, _sentenceSize=21, threshold = 0.9, _trainFilePath=augmentedTrainFilePath, _testFilePath='data/testSet.csv')
RNNBalancedParams = Experiment(_modelName='RNN', _kFolds=1, _embeddedDims=100, _epochs=5, _batchSize = 32, _sentenceSize=5, threshold = 0.3, _trainFilePath=balancedTrainFilePath, _testFilePath='data/testSet.csv')
LSTMBalancedParams = Experiment(_modelName='LSTM', _kFolds=1, _embeddedDims=60, _epochs=10, _batchSize = 1, _sentenceSize=19, threshold = 0.2, _trainFilePath=balancedTrainFilePath, _testFilePath='data/testSet.csv')
GRUBalancedParams = Experiment(_modelName='GRU', _kFolds=1, _embeddedDims=700, _epochs=10, _batchSize = 1, _sentenceSize=14, threshold = 0.9, _trainFilePath=balancedTrainFilePath, _testFilePath='data/testSet.csv')
RNNOriginalParams = Experiment(_modelName='RNN', _kFolds=1, _embeddedDims=60, _epochs=70, _batchSize = 1, _sentenceSize=5, threshold = 0.3, _trainFilePath=originalTrainFilePath, _testFilePath='data/testSet.csv')
LSTMOriginalParams = Experiment(_modelName='LSTM', _kFolds=1, _embeddedDims=100, _epochs=70, _batchSize = 1, _sentenceSize=11, threshold = 0.9, _trainFilePath=originalTrainFilePath, _testFilePath='data/testSet.csv')
GRUOriginalParams = Experiment(_modelName='GRU', _kFolds=1, _embeddedDims=80, _epochs=50, _batchSize = 1, _sentenceSize=19, threshold = 0.9, _trainFilePath=originalTrainFilePath, _testFilePath='data/testSet.csv')

saveFolder= [originalSaveFolder + 'Optimal/',balancedSaveFolder + 'Optimal/',augmentedSaveFolder + 'Optimal/']
trainFilePath=[originalTrainFilePath,balancedTrainFilePath,augmentedTrainFilePath]
RNNParams = [RNNOriginalParams, RNNBalancedParams, RNNAugmentedParams]
LSTMParams = [LSTMOriginalParams, LSTMBalancedParams, LSTMAugmentedParams]
GRUParams = [GRUOriginalParams, GRUBalancedParams, GRUAugmentedParams]
ParamsList = [RNNParams,LSTMParams,GRUParams]

for i in range(0, 30):
    for q in range(0,len(saveFolder)):
        for Params in ParamsList:
            modelParams = copy.deepcopy(Params[q])
            experiments = []
            print('Experiment: ' + str(i + 1) + ' Model: ' + modelParams.modelName + ' Dimensions: ' + str(modelParams.embeddedDims) + ' Epochs: ' + str(modelParams.epochs) + ' Batch: ' + str(modelParams.batchSize) + ' Sentence Size: ' + str(modelParams.sentenceSize) + ' Threshold: ' + str(modelParams.threshold))
            experiments.append(Experiment(_modelName=modelParams.modelName, _kFolds=1, _embeddedDims=modelParams.embeddedDims, _epochs=modelParams.epochs, _batchSize = modelParams.batchSize, _sentenceSize=modelParams.sentenceSize, threshold = modelParams.threshold, _trainFilePath=trainFilePath[q], _testFilePath='data/testSet.csv'))
            testRun = Experiments(experiments)
            testRun.run()
            testRun.printClassResults(saveFolder=saveFolder[q])
            testRun = Experiments(experiments)

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
    #     testRun.printParameterReports('KNN', 'nNeighbours',saveFolder=saveFolder[q],Bounds=[i])
    #     testRun.printParameterReports('KNN', 'threshold',saveFolder=saveFolder[q],Bounds=[i])

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