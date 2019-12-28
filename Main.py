from experimentHelper import Experiment
from experimentHelper import Experiments

experiments = []
# Examples
# experiments.append(Experiment(_modelName="RNN", _kFolds=1, _embeddedDims=300, _epochs=1, _batchSize = 32, _sentenceSize=30, threshold = 0.0, _trainFilePath='data/trainingSet_augmented.csv', _testFilePath='data/testSet.csv'))
# experiments.append(Experiment(_modelName="GRU", _kFolds=1, _embeddedDims=300, _epochs=1, _batchSize = 32, _sentenceSize=30, threshold = 0.0, _trainFilePath='data/trainingSet_augmented.csv', _testFilePath='data/testSet.csv'))
# experiments.append(Experiment(_modelName="LSTM", _kFolds=1, _embeddedDims=300, _epochs=1, _batchSize = 32, _sentenceSize=20, threshold = 0.0, _trainFilePath='data/trainingSet_augmented.csv', _testFilePath='data/testSet.csv'))
# experiments.append(Experiment(_modelName="KNN", _kFolds=1, _nNeighbours=5, _embeddingMode='TFIDF', threshold = 0.0, _trainFilePath='data/trainingSet_augmented.csv', _testFilePath='data/testSet.csv'))

# testRun = Experiments(experiments)
# testRun.showClassResults('KNN/1_Folds/2_Neighbours/BoW/20191228_120212.pickle')

# testRun = Experiments(experiments)
# testRun.printMultiClassReport('RNN', 'Epochs')

# KNN general experiment
# for i in range(0, 30):
#     print('Experiment: ' + str(i + 1))
#     experiments = []
#     for n in range(1, 16):
#         experiments.append(Experiment(_modelName="KNN", _kFolds=1, _nNeighbours=n, _embeddingMode='TFIDF', threshold = 0.0, _trainFilePath='data/trainingSet_augmented.csv', _testFilePath='data/testSet.csv'))
#         experiments.append(Experiment(_modelName="KNN", _kFolds=1, _nNeighbours=n, _embeddingMode='BoW', threshold = 0.0, _trainFilePath='data/trainingSet_augmented.csv', _testFilePath='data/testSet.csv'))
#     testRun = Experiments(experiments)
#     testRun.run()
#     testRun.printClassResults()
# print('Experiment complete')

# Neural Network general experiment
for i in range(0, 30):
    print('Experiment: ' + str(i + 1))
    experiments = []
    for Model in ('RNN','LSTM','GRU'):
        for Dims in range(100, 1001, 100):
            for SentenceSize in (10, 15, 20, 25):
                for Epochs in range(1, 6):
                    for Batch in (32,64,128,37250):
                        experiments = []
                        print('Model: ' + Model + ' Dimensions: ' + str(Dims) + ' Epochs: ' + str(Epochs) + ' Batch: ' + str(Batch) + ' Sentence Size: ' + str(SentenceSize))
                        experiments.append(Experiment(_modelName=Model, _kFolds=1, _embeddedDims=Dims, _epochs=Epochs, _batchSize = Batch, _sentenceSize=SentenceSize, threshold = 0.0, _trainFilePath='data/trainingSet_augmented.csv', _testFilePath='data/testSet.csv'))
                        testRun = Experiments(experiments)
                        testRun.run()
                        testRun.printClassResults()
print('Experiment complete')

#testRun = Experiments(experiments)
#testRun.run()
#testRun.evaluate()
#testRun.printClassResults()
#testRun.showResults('CR_RNN_1_300_1_32_20_20191228_005825.pickle')

