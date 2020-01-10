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

testRun = Experiments(experiments)
testRun.printClassDistributions()

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
#         print('Model: KNN, Dataset: Augmented, nNeighbours: ' + str(n) + ', Vectorization mode: ' + Mode + ', Dataset: ' + augmentedTrainFilePath)
#         experiments = []
#         experiments.append(Experiment(_modelName="KNN", _kFolds=1, _nNeighbours=n, _embeddingMode=Mode, threshold = 0.0, _trainFilePath=augmentedTrainFilePath, _testFilePath='data/testSet.csv'))
#         testRun = Experiments(experiments)
#         testRun.run()
#         testRun.printClassResults(saveFolder=saveFolder)
#     for n in range(20, 101, 10):
#         print('Model: KNN, Dataset: Augmented, nNeighbours: ' + str(n) + ', Vectorization mode: ' + Mode + ', Dataset: ' + augmentedTrainFilePath)
#         experiments = []
#         experiments.append(Experiment(_modelName="KNN", _kFolds=1, _nNeighbours=n, _embeddingMode=Mode, threshold = 0.0, _trainFilePath=augmentedTrainFilePath, _testFilePath='data/testSet.csv'))
#         testRun = Experiments(experiments)
#         testRun.run()
#         testRun.printClassResults(saveFolder=saveFolder)
#     for n in range(200, 1001, 100):
#         print('Model: KNN, Dataset: Augmented, nNeighbours: ' + str(n) + ', Vectorization mode: ' + Mode + ', Dataset: ' + augmentedTrainFilePath)
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
#                 print('Experiment: ' + str(i + 1) + ' Model: ' + Model + ' Dimensions: ' + str(Dims) + ' Epochs: ' + str(Epochs) + ' Batch: ' + str(Batch) + ' Sentence Size: ' + str(SentenceSize) + ', Dataset: ' + augmentedTrainFilePath)
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
#             print('Experiment: ' + str(i + 1) + ' Model: ' + Model + ' Dimensions: ' + str(Dims) + ' Epochs: ' + str(Epochs) + ' Batch: ' + str(Batch) + ' Sentence Size: ' + str(SentenceSize) + ', Dataset: ' + augmentedTrainFilePath)
#             experiments.append(Experiment(_modelName=Model, _kFolds=1, _embeddedDims=Dims, _epochs=Epochs, _batchSize = Batch, _sentenceSize=SentenceSize, threshold = 0.0, _trainFilePath=augmentedTrainFilePath, _testFilePath='data/testSet.csv'))
#             testRun = Experiments(experiments)
#             testRun.run()
#             testRun.printClassResults(saveFolder=saveFolder)
#         for Dims in range(100, 1001, 100):
#             experiments = []
#             print('Experiment: ' + str(i + 1) + ' Model: ' + Model + ' Dimensions: ' + str(Dims) + ' Epochs: ' + str(Epochs) + ' Batch: ' + str(Batch) + ' Sentence Size: ' + str(SentenceSize) + ', Dataset: ' + augmentedTrainFilePath)
#             experiments.append(Experiment(_modelName=Model, _kFolds=1, _embeddedDims=Dims, _epochs=Epochs, _batchSize = Batch, _sentenceSize=SentenceSize, threshold = 0.0, _trainFilePath=augmentedTrainFilePath, _testFilePath='data/testSet.csv'))
#             testRun = Experiments(experiments)
#             testRun.run()
#             testRun.printClassResults(saveFolder=saveFolder)
#         for variation in [[10,50,100,300,600,800],[10,20,30,40,50],[60,70,80,90,100],[100,200,300,400,500],[500,600,700,800]]:
#             testRun.printCompareParameterClasses(Model, parameter='embeddedDims', parameterValues=variation, saveFolder=saveFolder)
# testRun.printParameterReports('RNN', 'embeddedDims', saveFolder=saveFolder)
# testRun.printParameterReports('LSTM', 'embeddedDims', saveFolder=saveFolder)
# testRun.printParameterReports('GRU', 'embeddedDims', saveFolder=saveFolder)

#endregion

#region Sentence Size

# saveFolder = augmentedSaveFolder + 'SentenceSize/'
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
#             print('Experiment: ' + str(i + 1) + ' Model: ' + Model + ' Dimensions: ' + str(Dims) + ' Epochs: ' + str(Epochs) + ' Batch: ' + str(Batch) + ' Sentence Size: ' + str(SentenceSize) + ', Dataset: ' + augmentedTrainFilePath)
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

KNNAugmentedLiveClasses = ['Report','Information','Admin','Availability','Template','EqualGlass','Error','Status','Colour','Weight','Access','Pricing','Delivery','Documents','Account','Leaver']
RNNAugmentedLiveClasses = ['EqualGlass','Availability','Report','Logo','Leaver','Documents','Delivery','Status','Feedback','Account']
LSTMAugmentedLiveClasses = ['Colour','Template','Availability','Gables','Error','Logo','Feedback','Pricing','Weight','Status','Access','EqualGlass','Delivery','Leaver','Documents','Report','Account']
GRUAugmentedLiveClasses = ['Error','Colour','Status','Gables','Feedback','Pricing','EqualGlass','Availability','Logo','Delivery','Access','Weight','Report','Documents','Leaver','Account']

#endregion

#region Reprint graphs with filtering

## Regenerate results to include filtered versions

# # KNN general results
# testRun = Experiments(experiments)
# for i in ('BoW','TFIDF'):
#     testRun.printParameterReports('KNN', 'nNeighbours',saveFolder='results/Augmented/KNN/',Bounds=[i], classList=KNNAugmentedLiveClasses)

# # Neural Network Dimensions results
# testRun = Experiments(experiments)
# testRun.printParameterReports('RNN', 'embeddedDims', saveFolder='results/Augmented/Dimensions/', classList=RNNAugmentedLiveClasses)
# testRun.printParameterReports('LSTM', 'embeddedDims', saveFolder='results/Augmented/Dimensions/', classList=LSTMAugmentedLiveClasses)
# testRun.printParameterReports('GRU', 'embeddedDims', saveFolder='results/Augmented/Dimensions/', classList=GRUAugmentedLiveClasses)

# # Neural Network Epochs - Batch Size results
# testRun = Experiments(experiments)
# for j in ('epochs','batchSize'):
#     testRun.printParameterReports('RNN', j,saveFolder='results/Augmented/EpochsBatchSize/',classList=RNNAugmentedLiveClasses)
#     testRun.printParameterReports('LSTM', j,saveFolder='results/Augmented/EpochsBatchSize/',classList=LSTMAugmentedLiveClasses)
#     testRun.printParameterReports('GRU', j,saveFolder='results/Augmented/EpochsBatchSize/',classList=GRUAugmentedLiveClasses)

# # Neural Network Sentence Size results
# testRun = Experiments(experiments)
# testRun.printParameterReports('RNN', 'sentenceSize', saveFolder='results/Augmented/SentenceSize/', classList=RNNAugmentedLiveClasses)
# testRun.printParameterReports('LSTM', 'sentenceSize', saveFolder='results/Augmented/SentenceSize/', classList=LSTMAugmentedLiveClasses)
# testRun.printParameterReports('GRU', 'sentenceSize', saveFolder='results/Augmented/SentenceSize/', classList=GRUAugmentedLiveClasses)

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
#     print('Model: KNN, Dataset: Augmented, nNeighbours: ' + str(neighbours) + ', Vectorization mode: ' + Mode + ', Threshold: ' + str(threshold) + ', Dataset: ' + augmentedTrainFilePath)
#     experiments = []
#     experiments.append(Experiment(_modelName="KNN", _kFolds=1, _nNeighbours=neighbours, _embeddingMode=Mode, threshold = threshold, _trainFilePath=augmentedTrainFilePath, _testFilePath='data/testSet.csv'))
#     testRun = Experiments(experiments)
#     testRun.run()
#     testRun.printClassResults(saveFolder=saveFolder)
# testRun.printParameterReports('KNN', 'threshold', classList=KNNAugmentedLiveClasses, saveFolder=saveFolder)
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
#             print('Experiment: ' + str(i + 1) + ' Model: ' + params.modelName + ' Dimensions: ' + str(params.embeddedDims) + ' Epochs: ' + str(params.epochs) + ' Batch: ' + str(params.batchSize) + ' Sentence Size: ' + str(params.sentenceSize) + ', Dataset: ' + augmentedTrainFilePath)
#             experiments.append(Experiment(_modelName=params.modelName, _kFolds=1, _embeddedDims=params.embeddedDims, _epochs=params.epochs, _batchSize = params.batchSize, _sentenceSize=params.sentenceSize, threshold = threshold, _trainFilePath='data/trainingSet_augmented.csv', _testFilePath='data/testSet.csv'))
#             testRun = Experiments(experiments)
#             testRun.run()
#             testRun.printClassResults(saveFolder=saveFolder)
# testRun.printParameterReports('RNN', 'threshold', classList=RNNAugmentedLiveClasses, saveFolder=saveFolder)
# testRun.printCompareParameterClasses(params.modelName, parameter='threshold', saveFolder=saveFolder)

# # LSTM threshold choice (also confirm sentence size and dimensions)
# for i in range(0, 1):

#     print('Experiment: ' + str(i + 1))
#     params = copy.deepcopy(LSTMParams)
#     for j in range(0, 10):
#         threshold = j / 10
#         experiments = []
#         print('Experiment: ' + str(i + 1) + ' Model: ' + params.modelName + ' Dimensions: ' + str(params.embeddedDims) + ' Epochs: ' + str(params.epochs) + ' Batch: ' + str(params.batchSize) + ' Sentence Size: ' + str(params.sentenceSize) + ', Dataset: ' + augmentedTrainFilePath)
#         experiments.append(Experiment(_modelName=params.modelName, _kFolds=1, _embeddedDims=params.embeddedDims, _epochs=params.epochs, _batchSize = params.batchSize, _sentenceSize=params.sentenceSize, threshold = threshold, _trainFilePath='data/trainingSet_augmented.csv', _testFilePath='data/testSet.csv'))
#         testRun = Experiments(experiments)
#         testRun.run()
#         testRun.printClassResults(saveFolder=saveFolder)
# testRun.printParameterReports('LSTM', 'threshold', classList=LSTMAugmentedLiveClasses, saveFolder=saveFolder)
# testRun.printCompareParameterClasses(params.modelName, parameter='threshold', saveFolder=saveFolder)

# # GRU threshold choice (also confirm sentence size and dimensions)
# for i in range(0, 1):

#     print('Experiment: ' + str(i + 1))
#     params = copy.deepcopy(GRUParams)
#     for j in range(0, 10):
#         threshold = j / 10
#         experiments = []
#         print('Experiment: ' + str(i + 1) + ' Model: ' + params.modelName + ' Dimensions: ' + str(params.embeddedDims) + ' Epochs: ' + str(params.epochs) + ' Batch: ' + str(params.batchSize) + ' Sentence Size: ' + str(params.sentenceSize) + ', Dataset: ' + augmentedTrainFilePath)
#         experiments.append(Experiment(_modelName=params.modelName, _kFolds=1, _embeddedDims=params.embeddedDims, _epochs=params.epochs, _batchSize = params.batchSize, _sentenceSize=params.sentenceSize, threshold = threshold, _trainFilePath='data/trainingSet_augmented.csv', _testFilePath='data/testSet.csv'))
#         testRun = Experiments(experiments)
#         testRun.run()
#         testRun.printClassResults(saveFolder=saveFolder)
#     testRun = Experiments(experiments)
# testRun.printParameterReports('GRU', 'threshold', classList=GRUAugmentedLiveClasses, saveFolder=saveFolder)
# testRun.printCompareParameterClasses(params.modelName, parameter='threshold', saveFolder=saveFolder)

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
#         print('Model: KNN, Dataset: Balanced, nNeighbours: ' + str(n) + ', Vectorization mode: ' + Mode + ', Dataset: ' + balancedTrainFilePath)
#         experiments = []
#         experiments.append(Experiment(_modelName="KNN", _kFolds=1, _nNeighbours=n, _embeddingMode=Mode, threshold = 0.0, _trainFilePath=balancedTrainFilePath, _testFilePath='data/testSet.csv'))
#         testRun = Experiments(experiments)
#         testRun.run()
#         testRun.printClassResults(saveFolder=saveFolder)
#     for n in range(20, 101, 10):
#         print('Model: KNN, Dataset: Balanced, nNeighbours: ' + str(n) + ', Vectorization mode: ' + Mode + ', Dataset: ' + balancedTrainFilePath)
#         experiments = []
#         experiments.append(Experiment(_modelName="KNN", _kFolds=1, _nNeighbours=n, _embeddingMode=Mode, threshold = 0.0, _trainFilePath=balancedTrainFilePath, _testFilePath='data/testSet.csv'))
#         testRun = Experiments(experiments)
#         testRun.run()
#         testRun.printClassResults(saveFolder=saveFolder)
#     for n in range(200, 1001, 100):
#         print('Model: KNN, Dataset: Balanced, nNeighbours: ' + str(n) + ', Vectorization mode: ' + Mode + ', Dataset: ' + balancedTrainFilePath)
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
#                 print('Experiment: ' + str(i + 1) + ' Model: ' + Model + ' Dimensions: ' + str(Dims) + ' Epochs: ' + str(Epochs) + ' Batch: ' + str(Batch) + ' Sentence Size: ' + str(SentenceSize) + ', Dataset: ' + balancedTrainFilePath)
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
#             print('Experiment: ' + str(i + 1) + ' Model: ' + Model + ' Dimensions: ' + str(Dims) + ' Epochs: ' + str(Epochs) + ' Batch: ' + str(Batch) + ' Sentence Size: ' + str(SentenceSize) + ', Dataset: ' + balancedTrainFilePath)
#             experiments.append(Experiment(_modelName=Model, _kFolds=1, _embeddedDims=Dims, _epochs=Epochs, _batchSize = Batch, _sentenceSize=SentenceSize, threshold = 0.0, _trainFilePath=balancedTrainFilePath, _testFilePath='data/testSet.csv'))
#             testRun = Experiments(experiments)
#             testRun.run()
#             testRun.printClassResults(saveFolder=saveFolder)
#         for Dims in range(100, 801, 100):
#             experiments = []
#             print('Experiment: ' + str(i + 1) + ' Model: ' + Model + ' Dimensions: ' + str(Dims) + ' Epochs: ' + str(Epochs) + ' Batch: ' + str(Batch) + ' Sentence Size: ' + str(SentenceSize) + ', Dataset: ' + balancedTrainFilePath)
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
#             print('Experiment: ' + str(i + 1) + ' Model: ' + Model + ' Dimensions: ' + str(Dims) + ' Epochs: ' + str(Epochs) + ' Batch: ' + str(Batch) + ' Sentence Size: ' + str(SentenceSize) + ', Dataset: ' + balancedTrainFilePath)
#             experiments.append(Experiment(_modelName=Model, _kFolds=1, _embeddedDims=Dims, _epochs=Epochs, _batchSize = Batch, _sentenceSize=SentenceSize, threshold = 0.0, _trainFilePath=balancedTrainFilePath, _testFilePath='data/testSet.csv'))
#             testRun = Experiments(experiments)
#             testRun.run()
#             testRun.printClassResults(saveFolder=saveFolder)
#         testRun.printParameterReports(Model, 'sentenceSize', saveFolder=saveFolder)
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

KNNBalancedLiveClasses = ['Project','Template','EqualGlass','Information','Colour','Report','Status','Availability','Error','Weight','Pricing','Access','Delivery','Documents','Leaver','Account']
RNNBalancedLiveClasses = ['Account']
LSTMBalancedLiveClasses = ['Status','Report','Logo','Colour','Weight','Pricing','Feedback','Gables','Documents','Delivery','Account']
GRUBalancedLiveClasses = ['EqualGlass','Colour','Status','Leaver','Availability','Pricing','Gables','Weight','Report','Feedback','Documents','Logo','Delivery','Account']

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
#     print('Model: KNN, Dataset: Balanced, nNeighbours: ' + str(neighbours) + ', Vectorization mode: ' + Mode + ', Threshold: ' + str(threshold) + ', Dataset: ' + balancedTrainFilePath + ', Dataset: ' + balancedTrainFilePath)
#     experiments = []
#     experiments.append(Experiment(_modelName="KNN", _kFolds=1, _nNeighbours=neighbours, _embeddingMode=Mode, threshold = threshold, _trainFilePath=balancedTrainFilePath, _testFilePath='data/testSet.csv'))
#     testRun = Experiments(experiments)
#     testRun.run()
#     testRun.printClassResults(saveFolder=saveFolder)
# testRun.printParameterReports('KNN', 'threshold', classList=KNNBalancedLiveClasses,saveFolder=saveFolder)
# testRun.printCompareParameterClasses('KNN', parameter='threshold', saveFolder=saveFolder)

# RNN threshold choice (also confirm sentence size and dimensions)
# for i in range(0, 1):
#     print('Experiment: ' + str(i + 1))
#     experiments = []
#     params = copy.deepcopy(RNNBalancedParams)
#     for j in range(0, 10):
#         threshold = j / 10
#         experiments = []    
#         print('Experiment: ' + str(i + 1) + ' Model: ' + params.modelName + ' Dimensions: ' + str(params.embeddedDims) + ' Epochs: ' + str(params.epochs) + ' Batch: ' + str(params.batchSize) + ' Sentence Size: ' + str(params.sentenceSize) + ', Dataset: ' + balancedTrainFilePath)
#         experiments.append(Experiment(_modelName=params.modelName, _kFolds=1, _embeddedDims=params.embeddedDims, _epochs=params.epochs, _batchSize = params.batchSize, _sentenceSize=params.sentenceSize, threshold = threshold, _trainFilePath=balancedTrainFilePath, _testFilePath='data/testSet.csv'))
#         testRun = Experiments(experiments)
#         testRun.run()
#         testRun.printClassResults(saveFolder=saveFolder)
# testRun.printParameterReports('RNN', 'threshold',classList=RNNBalancedLiveClasses, saveFolder=saveFolder)
# testRun.printCompareParameterClasses(params.modelName, parameter='threshold', saveFolder=saveFolder)

# # LSTM threshold choice (also confirm sentence size and dimensions)
# for i in range(0, 1):

#     print('Experiment: ' + str(i + 1))
#     params = copy.deepcopy(LSTMBalancedParams)
#     for j in range(0, 10):
#         threshold = j / 10
#         experiments = []
#         print('Experiment: ' + str(i + 1) + ' Model: ' + params.modelName + ' Dimensions: ' + str(params.embeddedDims) + ' Epochs: ' + str(params.epochs) + ' Batch: ' + str(params.batchSize) + ' Sentence Size: ' + str(params.sentenceSize) + ', Dataset: ' + balancedTrainFilePath)
#         experiments.append(Experiment(_modelName=params.modelName, _kFolds=1, _embeddedDims=params.embeddedDims, _epochs=params.epochs, _batchSize = params.batchSize, _sentenceSize=params.sentenceSize, threshold = threshold, _trainFilePath=balancedTrainFilePath, _testFilePath='data/testSet.csv'))
#         testRun = Experiments(experiments)
#         testRun.run()
#         testRun.printClassResults(saveFolder=saveFolder)
# testRun.printParameterReports('LSTM', 'threshold',classList=LSTMBalancedLiveClasses, saveFolder=saveFolder)
# testRun.printCompareParameterClasses(params.modelName, parameter='threshold', saveFolder=saveFolder)

# # GRU threshold choice (also confirm sentence size and dimensions)
# for i in range(0, 1):

#     print('Experiment: ' + str(i + 1))
#     params = copy.deepcopy(GRUBalancedParams)
#     for j in range(0, 10):
#         threshold = j / 10
#         experiments = []
#         print('Experiment: ' + str(i + 1) + ' Model: ' + params.modelName + ' Dimensions: ' + str(params.embeddedDims) + ' Epochs: ' + str(params.epochs) + ' Batch: ' + str(params.batchSize) + ' Sentence Size: ' + str(params.sentenceSize) + ', Dataset: ' + balancedTrainFilePath)
#         experiments.append(Experiment(_modelName=params.modelName, _kFolds=1, _embeddedDims=params.embeddedDims, _epochs=params.epochs, _batchSize = params.batchSize, _sentenceSize=params.sentenceSize, threshold = threshold, _trainFilePath=balancedTrainFilePath, _testFilePath='data/testSet.csv'))
#         testRun = Experiments(experiments)
#         testRun.run()
#         testRun.printClassResults(saveFolder=saveFolder)
# testRun.printParameterReports('GRU', 'threshold',classList=GRUBalancedLiveClasses, saveFolder=saveFolder)
# testRun.printCompareParameterClasses(params.modelName, parameter='threshold', saveFolder=saveFolder)

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
#         print('Model: KNN, Dataset: Original, nNeighbours: ' + str(n) + ', Vectorization mode: ' + Mode + ', Dataset: ' + originalTrainFilePath)
#         experiments = []
#         experiments.append(Experiment(_modelName="KNN", _kFolds=1, _nNeighbours=n, _embeddingMode=Mode, threshold = 0.0, _trainFilePath=originalTrainFilePath, _testFilePath='data/testSet.csv'))
#         testRun = Experiments(experiments)
#         testRun.run()
#         testRun.printClassResults(saveFolder=saveFolder)
#     for n in range(20, 101, 10):
#         print('Model: KNN, Dataset: Original, nNeighbours: ' + str(n) + ', Vectorization mode: ' + Mode + ', Dataset: ' + originalTrainFilePath)
#         experiments = []
#         experiments.append(Experiment(_modelName="KNN", _kFolds=1, _nNeighbours=n, _embeddingMode=Mode, threshold = 0.0, _trainFilePath=originalTrainFilePath, _testFilePath='data/testSet.csv'))
#         testRun = Experiments(experiments)
#         testRun.run()
#         testRun.printClassResults(saveFolder=saveFolder)
#     for n in range(200, 701, 100):
#         print('Model: KNN, Dataset: Original, nNeighbours: ' + str(n) + ', Vectorization mode: ' + Mode + ', Dataset: ' + originalTrainFilePath)
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
#                 print('Experiment: ' + str(i + 1) + ' Model: ' + Model + ' Dimensions: ' + str(Dims) + ' Epochs: ' + str(Epochs) + ' Batch: ' + str(Batch) + ' Sentence Size: ' + str(SentenceSize) + ', Dataset: ' + originalTrainFilePath)
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
#             print('Experiment: ' + str(i + 1) + ' Model: ' + Model + ' Dimensions: ' + str(Dims) + ' Epochs: ' + str(Epochs) + ' Batch: ' + str(Batch) + ' Sentence Size: ' + str(SentenceSize) + ', Dataset: ' + originalTrainFilePath)
#             experiments.append(Experiment(_modelName=Model, _kFolds=1, _embeddedDims=Dims, _epochs=Epochs, _batchSize = Batch, _sentenceSize=SentenceSize, threshold = 0.0, _trainFilePath=originalTrainFilePath, _testFilePath='data/testSet.csv'))
#             testRun = Experiments(experiments)
#             testRun.run()
#             testRun.printClassResults(saveFolder=saveFolder)
#         for Dims in range(100, 801, 100):
#             experiments = []
#             print('Experiment: ' + str(i + 1) + ' Model: ' + Model + ' Dimensions: ' + str(Dims) + ' Epochs: ' + str(Epochs) + ' Batch: ' + str(Batch) + ' Sentence Size: ' + str(SentenceSize) + ', Dataset: ' + originalTrainFilePath)
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
#         print('Experiment: ' + str(i + 1) + ' Model: ' + Model + ' Dimensions: ' + str(Dims) + ' Epochs: ' + str(Epochs) + ' Batch: ' + str(Batch) + ' Sentence Size: ' + str(SentenceSize) + ', Dataset: ' + originalTrainFilePath)
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

KNNOriginalLiveClasses = ['Access','Report','Availability','Leaver','Error','Logo','Delivery','Status','Documents','Weight','Pricing','Account']
RNNOriginalLiveClasses = ['Account']
LSTMOriginalLiveClasses = ['Gables','Documents','Delviery','Account']
GRUOriginalLiveClasses = ['Colour','Status','EqualGlass','Leaver','Feedback','Gables','Pricing','Weight','Logo','Report','Delivery','Documents','Account']

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
#     print('Model: KNN, nNeighbours: ' + str(neighbours) + ', Vectorization mode: ' + Mode + ', Threshold: ' + str(threshold) + ', Dataset: ' + originalTrainFilePath)
#     experiments = []
#     experiments.append(Experiment(_modelName="KNN", _kFolds=1, _nNeighbours=neighbours, _embeddingMode=Mode, threshold = threshold, _trainFilePath=originalTrainFilePath, _testFilePath='data/testSet.csv'))
#     testRun = Experiments(experiments)
#     testRun.run()
#     testRun.printClassResults(saveFolder=saveFolder)
# testRun.printParameterReports('KNN', 'threshold',classList=KNNOriginalLiveClasses,saveFolder=saveFolder)
# testRun.printCompareParameterClasses('KNN', parameter='threshold', saveFolder=saveFolder)

# RNN threshold choice
# experiments = []
# params = copy.deepcopy(RNNOriginalParams)
# for j in range(0, 10):
#     threshold = j / 10
#     experiments = []    
#     print('Model: ' + params.modelName + ' Dimensions: ' + str(params.embeddedDims) + ' Epochs: ' + str(params.epochs) + ' Batch: ' + str(params.batchSize) + ' Sentence Size: ' + str(params.sentenceSize) + ', threshold: ' + str(threshold) + ', Dataset: ' + originalTrainFilePath)
#     experiments.append(Experiment(_modelName=params.modelName, _kFolds=1, _embeddedDims=params.embeddedDims, _epochs=params.epochs, _batchSize = params.batchSize, _sentenceSize=params.sentenceSize, threshold = threshold, _trainFilePath=originalTrainFilePath, _testFilePath='data/testSet.csv'))
#     testRun = Experiments(experiments)
#     testRun.run()
#     testRun.printClassResults(saveFolder=saveFolder)
# testRun = Experiments(experiments)
# testRun.printParameterReports('RNN', 'threshold',classList=RNNOriginalLiveClasses, saveFolder=saveFolder)
# testRun.printCompareParameterClasses(params.modelName, parameter='threshold', saveFolder=saveFolder)

# # LSTM threshold choice (also confirm sentence size and dimensions)
# params = copy.deepcopy(LSTMOriginalParams)
# for j in range(0, 10):
#     threshold = j / 10
#     experiments = []
#     print('Model: ' + params.modelName + ' Dimensions: ' + str(params.embeddedDims) + ' Epochs: ' + str(params.epochs) + ' Batch: ' + str(params.batchSize) + ' Sentence Size: ' + str(params.sentenceSize) + ', threshold: ' + str(threshold) + ', Dataset: ' + originalTrainFilePath)
#     experiments.append(Experiment(_modelName=params.modelName, _kFolds=1, _embeddedDims=params.embeddedDims, _epochs=params.epochs, _batchSize = params.batchSize, _sentenceSize=params.sentenceSize, threshold = threshold, _trainFilePath=originalTrainFilePath, _testFilePath='data/testSet.csv'))
#     testRun = Experiments(experiments)
#     testRun.run()
#     testRun.printClassResults(saveFolder=saveFolder)
# testRun = Experiments(experiments)
# testRun.printParameterReports('LSTM', 'threshold',classList=LSTMOriginalLiveClasses, saveFolder=saveFolder)
# testRun.printCompareParameterClasses(params.modelName, parameter='threshold', saveFolder=saveFolder)

# # GRU threshold choice (also confirm sentence size and dimensions)
# params = copy.deepcopy(GRUOriginalParams)
# for j in range(0, 10):
#     threshold = j / 10
#     experiments = []
#     print('Model: ' + params.modelName + ' Dimensions: ' + str(params.embeddedDims) + ' Epochs: ' + str(params.epochs) + ' Batch: ' + str(params.batchSize) + ' Sentence Size: ' + str(params.sentenceSize) + ', threshold: ' + str(threshold) + ', Dataset: ' + originalTrainFilePath)
#     experiments.append(Experiment(_modelName=params.modelName, _kFolds=1, _embeddedDims=params.embeddedDims, _epochs=params.epochs, _batchSize = params.batchSize, _sentenceSize=params.sentenceSize, threshold = threshold, _trainFilePath=originalTrainFilePath, _testFilePath='data/testSet.csv'))
#     testRun = Experiments(experiments)
#     testRun.run()
#     testRun.printClassResults(saveFolder=saveFolder)
# testRun = Experiments(experiments)
# testRun.printParameterReports('GRU', 'threshold',classList=GRUOriginalLiveClasses, saveFolder=saveFolder)
# testRun.printCompareParameterClasses(params.modelName, parameter='threshold', saveFolder=saveFolder)

#endregion

#endregion

###############################
### Optimised final results ###
###############################

# Double check optimised values, fill in params lists below, check q referenced throughout

# Run through once to gauge running time, then run through multiple times
RNNAugmentedParams = Experiment(_modelName='RNN', _kFolds=1, _embeddedDims=100, _epochs=3, _batchSize = 64, _sentenceSize=6, threshold = 0.9, _trainFilePath=augmentedTrainFilePath, _testFilePath='data/testSet.csv')
LSTMAugmentedParams = Experiment(_modelName='LSTM', _kFolds=1, _embeddedDims=90, _epochs=3, _batchSize = 64, _sentenceSize=18, threshold = 0.9, _trainFilePath=augmentedTrainFilePath, _testFilePath='data/testSet.csv')
GRUAugmentedParams = Experiment(_modelName='GRU', _kFolds=1, _embeddedDims=100, _epochs=3, _batchSize = 64, _sentenceSize=21, threshold = 0.9, _trainFilePath=augmentedTrainFilePath, _testFilePath='data/testSet.csv')
RNNBalancedParams = Experiment(_modelName='RNN', _kFolds=1, _embeddedDims=100, _epochs=5, _batchSize = 32, _sentenceSize=5, threshold = 0.3, _trainFilePath=balancedTrainFilePath, _testFilePath='data/testSet.csv')
LSTMBalancedParams = Experiment(_modelName='LSTM', _kFolds=1, _embeddedDims=60, _epochs=10, _batchSize = 1, _sentenceSize=19, threshold = 0.2, _trainFilePath=balancedTrainFilePath, _testFilePath='data/testSet.csv')
GRUBalancedParams = Experiment(_modelName='GRU', _kFolds=1, _embeddedDims=70, _epochs=10, _batchSize = 1, _sentenceSize=14, threshold = 0.9, _trainFilePath=balancedTrainFilePath, _testFilePath='data/testSet.csv')
RNNOriginalParams = Experiment(_modelName='RNN', _kFolds=1, _embeddedDims=60, _epochs=70, _batchSize = 1, _sentenceSize=5, threshold = 0.3, _trainFilePath=originalTrainFilePath, _testFilePath='data/testSet.csv')
LSTMOriginalParams = Experiment(_modelName='LSTM', _kFolds=1, _embeddedDims=100, _epochs=70, _batchSize = 1, _sentenceSize=11, threshold = 0.9, _trainFilePath=originalTrainFilePath, _testFilePath='data/testSet.csv')
GRUOriginalParams = Experiment(_modelName='GRU', _kFolds=1, _embeddedDims=80, _epochs=50, _batchSize = 1, _sentenceSize=19, threshold = 0.9, _trainFilePath=originalTrainFilePath, _testFilePath='data/testSet.csv')

saveFolder= [originalSaveFolder + 'Optimal/',balancedSaveFolder + 'Optimal/',augmentedSaveFolder + 'Optimal/']
trainFilePath=[originalTrainFilePath,balancedTrainFilePath,augmentedTrainFilePath]
RNNParams = [RNNOriginalParams, RNNBalancedParams, RNNAugmentedParams]
LSTMParams = [LSTMOriginalParams, LSTMBalancedParams, LSTMAugmentedParams]
GRUParams = [GRUOriginalParams, GRUBalancedParams, GRUAugmentedParams]
ParamsList = [RNNParams,LSTMParams,GRUParams]

for i in range(0, 17):
    for q in range(0,len(saveFolder)):
        for Params in ParamsList:
            modelParams = copy.deepcopy(Params[q])
            experiments = []
            print('Experiment: ' + str(i + 1) + ' Model: ' + modelParams.modelName + ' Dimensions: ' + str(modelParams.embeddedDims) + ' Epochs: ' + str(modelParams.epochs) + ' Batch: ' + str(modelParams.batchSize) + ' Sentence Size: ' + str(modelParams.sentenceSize) + ' Threshold: ' + str(modelParams.threshold) + ' Dataset: ' + modelParams.trainFilePath)
            experiments.append(Experiment(_modelName=modelParams.modelName, _kFolds=1, _embeddedDims=modelParams.embeddedDims, _epochs=modelParams.epochs, _batchSize = modelParams.batchSize, _sentenceSize=modelParams.sentenceSize, threshold = modelParams.threshold, _trainFilePath=modelParams.trainFilePath, _testFilePath='data/testSet.csv'))
            testRun = Experiments(experiments)
            testRun.run()
            testRun.printClassResults(saveFolder=saveFolder[q])
            testRun = Experiments(experiments)

for i in range(0, 1):
    for modelParams in [LSTMBalancedParams, GRUBalancedParams]:
        experiments = []
        print('Experiment: ' + str(i + 1) + ' Model: ' + modelParams.modelName + ' Dimensions: ' + str(modelParams.embeddedDims) + ' Epochs: ' + str(modelParams.epochs) + ' Batch: ' + str(modelParams.batchSize) + ' Sentence Size: ' + str(modelParams.sentenceSize) + ' Threshold: ' + str(modelParams.threshold) + ' Dataset: ' + modelParams.trainFilePath)
        experiments.append(Experiment(_modelName=modelParams.modelName, _kFolds=1, _embeddedDims=modelParams.embeddedDims, _epochs=modelParams.epochs, _batchSize = modelParams.batchSize, _sentenceSize=modelParams.sentenceSize, threshold = modelParams.threshold, _trainFilePath=modelParams.trainFilePath, _testFilePath='data/testSet.csv'))
        testRun = Experiments(experiments)
        testRun.run()
        testRun.printClassResults(saveFolder=balancedSaveFolder)
        testRun = Experiments(experiments)

for i in range(0, 1):
    for modelParams in [RNNAugmentedParams, LSTMAugmentedParams, GRUAugmentedParams]:
        experiments = []
        print('Experiment: ' + str(i + 1) + ' Model: ' + modelParams.modelName + ' Dimensions: ' + str(modelParams.embeddedDims) + ' Epochs: ' + str(modelParams.epochs) + ' Batch: ' + str(modelParams.batchSize) + ' Sentence Size: ' + str(modelParams.sentenceSize) + ' Threshold: ' + str(modelParams.threshold) + ' Dataset: ' + modelParams.trainFilePath)
        experiments.append(Experiment(_modelName=modelParams.modelName, _kFolds=1, _embeddedDims=modelParams.embeddedDims, _epochs=modelParams.epochs, _batchSize = modelParams.batchSize, _sentenceSize=modelParams.sentenceSize, threshold = modelParams.threshold, _trainFilePath=modelParams.trainFilePath, _testFilePath='data/testSet.csv'))
        testRun = Experiments(experiments)
        testRun.run()
        testRun.printClassResults(saveFolder=augmentedSaveFolder)
        testRun = Experiments(experiments)

KNNAugmentedParams = Experiment(_modelName='KNN', _kFolds=1, _nNeighbours=10, _embeddingMode='TFIDF', threshold = 0.9, _trainFilePath=augmentedTrainFilePath, _testFilePath='data/testSet.csv')
KNNBalancedParams = Experiment(_modelName='KNN', _kFolds=1, _nNeighbours=9, _embeddingMode='TFIDF', threshold = 0.9, _trainFilePath=balancedTrainFilePath, _testFilePath='data/testSet.csv')
KNNOriginalParams = Experiment(_modelName='KNN', _kFolds=1, _nNeighbours=9, _embeddingMode='TFIDF', threshold = 0.8, _trainFilePath=originalTrainFilePath, _testFilePath='data/testSet.csv')
KNNParams = [[KNNAugmentedParams,augmentedSaveFolder + 'Optimal/'],[KNNBalancedParams,balancedSaveFolder + 'Optimal/'],[KNNOriginalParams,originalSaveFolder + 'Optimal/']]

# # KNN 
# for i in range(0,30):
#     for ModelParams, ModelFolder in KNNParams:
#         print('Experiment: ' + str(i + 1) + ' Model: ' + ModelParams.modelName + ' nNeighbours: ' + str(ModelParams.nNeighbours) + ' Vectorisation: ' + str(ModelParams.embeddingMode) + ' Threshold: ' + str(ModelParams.threshold) + ' Dataset: ' + ModelParams.trainFilePath)
#         experiments = []
#         experiments.append(Experiment(_modelName=ModelParams.modelName, _kFolds=1, _nNeighbours=ModelParams.nNeighbours, _embeddingMode=ModelParams.embeddingMode, threshold = ModelParams.threshold, _trainFilePath=ModelParams.trainFilePath, _testFilePath='data/testSet.csv'))
#         testRun = Experiments(experiments)
#         testRun.run()
#         testRun.printClassResults(saveFolder=ModelFolder)

ModelList = [['KNN'],['RNN'],['LSTM'],['GRU']]
# experiments = []
# testRun = Experiments(experiments)
# Graph augmented models against eachother
augmentedLoadFolder = augmentedSaveFolder + 'Optimal/'
augmentedLoadFolders = [augmentedLoadFolder,augmentedLoadFolder,augmentedLoadFolder,augmentedLoadFolder]
testRun.printCompareExperimentsClasses(ModelsList=ModelList, loadFolders=augmentedLoadFolders, saveFolder=augmentedLoadFolder)

# Graph balanced models against eachother
balancedLoadFolder = balancedSaveFolder + 'Optimal/'
balancedLoadFolders = [balancedLoadFolder,balancedLoadFolder,balancedLoadFolder,balancedLoadFolder]
testRun.printCompareExperimentsClasses(ModelsList=ModelList, loadFolders=balancedLoadFolders, saveFolder=balancedLoadFolder)

# Graph original models against eachother
originalLoadFolder = originalSaveFolder + 'Optimal/'
originalLoadFolders = [originalLoadFolder,originalLoadFolder,originalLoadFolder,originalLoadFolder]
testRun.printCompareExperimentsClasses(ModelsList=ModelList, loadFolders=originalLoadFolders, saveFolder=originalLoadFolder)

# Graph models against eachother across datasets
for Model in [['KNN'],['RNN'],['LSTM'],['GRU']]:
    overallLoadFolders = [originalLoadFolder,balancedLoadFolder,augmentedLoadFolder]
    testRun.printCompareExperimentsClasses(ModelsList=[Model,Model,Model], loadFolders=overallLoadFolders, saveFolder='results/Global')



    # Decide on a dataset, decide on a model / threshold