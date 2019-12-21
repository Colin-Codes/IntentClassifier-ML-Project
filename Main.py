from experimentHelper import Experiment
from experimentHelper import Experiments

experiments = []
#experiments.append(Experiment(_modelType="RNN", _kFolds=1, _embeddedDims=300, _epochs=1, _batchSize = 32, _sentenceSize=30, _trainFilePath='data/trainingSet_augmented.csv', _testFilePath='data/testSet.csv'))
#experiments.append(Experiment(_modelType="GRU", _kFolds=1, _embeddedDims=300, _epochs=1, _batchSize = 32, _sentenceSize=30, _trainFilePath='data/trainingSet_augmented.csv', _testFilePath='data/testSet.csv'))
#experiments.append(Experiment(_modelType="LSTM", _kFolds=5, _embeddedDims=300, _epochs=1, _batchSize = 32, _sentenceSize=20, _trainFilePath='data/trainingSet_augmented.csv', _testFilePath='data/testSet.csv'))
#experiments.append(Experiment(_modelType="KNN", _kFolds=5, _nNeighbours=5, _embeddingMode='TFIDF', _trainFilePath='data/trainingSet_augmented.csv', _testFilePath='data/testSet.csv'))

#experiments.append(Experiment(_modelType="KNN", _kFolds=.8, _nNeighbours=10, _embeddingMode='BoW', _trainFilePath='data/trainingSet_augmented.csv', _testFilePath='data/testSet.csv'))
#experiments.append(Experiment(_modelType="KNN", _kFolds=.8, _nNeighbours=50, _embeddingMode='BoW', _trainFilePath='data/trainingSet_augmented.csv', _testFilePath='data/testSet.csv'))
#experiments.append(Experiment(_modelType="KNN", _kFolds=.8, _nNeighbours=100, _embeddingMode='BoW', _trainFilePath='data/trainingSet_augmented.csv', _testFilePath='data/testSet.csv'))
#experiments.append(Experiment(_modelType="KNN", _kFolds=1, _nNeighbours=3, _embeddingMode='BoW', _trainFilePath='data/trainingSet_augmented.csv', _testFilePath='data/testSet.csv'))

experiments.append(Experiment(_modelType="GRU", _kFolds=1, _embeddedDims=300, _epochs=1, _batchSize = 32, _sentenceSize=20, _trainFilePath='data/trainingSet_augmented.csv', _testFilePath='data/testSet.csv'))

testRun = Experiments(experiments)
testRun.run()
testRun.evaluate()
testRun.printResults('data/GRU_300_1_32_20.csv')

