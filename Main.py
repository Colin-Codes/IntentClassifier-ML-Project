from experimentHelper import Experiment
from experimentHelper import Experiments

experiments = []
#experiments.append(Experiment(_modelType="RNN", _kFolds=1, _embeddedDims=300, _epochs=1, _batchSize = 32, _sentenceSize=30, _trainFilePath='data/trainingSet_augmented.csv', _testFilePath='data/testSet.csv'))
#experiments.append(Experiment(_modelType="GRU", _kFolds=1, _embeddedDims=300, _epochs=1, _batchSize = 32, _sentenceSize=30, _trainFilePath='data/trainingSet_augmented.csv', _testFilePath='data/testSet.csv'))
#experiments.append(Experiment(_modelType="LSTM", _kFolds=5, _embeddedDims=300, _epochs=1, _batchSize = 32, _sentenceSize=20, _trainFilePath='data/trainingSet_augmented.csv', _testFilePath='data/testSet.csv'))
#experiments.append(Experiment(_modelType="KNN", _kFolds=5, _nNeighbours=5, _embeddingMode='TFIDF', _trainFilePath='data/trainingSet_augmented.csv', _testFilePath='data/testSet.csv'))

#experiments.append(Experiment(_modelName="RNN", _kFolds=1, _embeddedDims=10, _epochs=1, _batchSize = 100, _sentenceSize=30, _trainFilePath='data/trainingSet_augmented.csv', _testFilePath='data/testSet.csv'))

testRun = Experiments(experiments)
#testRun.run()
#testRun.evaluate()
#testRun.printResults()
testRun.showResults('CR_RNN_1_10_1_100_30_20191227_231954.pickle')

