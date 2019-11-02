from experimentHelper import Experiment
from experimentHelper import Experiments

experiments = []
#experiments.append(Experiment(_modelType="LSTM", _kFolds=5, _embeddedDims=300, _epochs=1, _batchSize = 32, _sentenceSize=20, _trainFilePath='data/trainingset_augmented.csv', _testFilePath='data/testset.csv'))
#experiments.append(Experiment(_modelType="KNN", _kFolds=5, _nNeighbours=5, _trainFilePath='data/trainingset_augmented.csv', _testFilePath='data/testset.csv'))
experiments.append(Experiment("RNN", _kFolds=0.3, _embeddedDims=300, _epochs=2, _batchSize = 32, _sentenceSize=20, _trainFilePath='data/trainingset_augmented.csv', _testFilePath='data/trainingset_augmented.csv'))
experiments.append(Experiment("KNN", _kFolds=0.3, _nNeighbours=5, _trainFilePath='data/trainingset_augmented.csv', _testFilePath='data/trainingset_augmented.csv'))

testRun = Experiments(experiments)
testRun.run()
testRun.evaluate()

