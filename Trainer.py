from ModelBuilder import modelTrainer

embeddedDims = 300
split = 0.3
epochs = 1
batchSize = 32
minWords = 20
maxWords = 20

models = modelTrainer("RNN", embeddedDims, split, epochs, batchSize, minWords, maxWords)
models = modelTrainer("LSTM", embeddedDims, split, epochs, batchSize, minWords, maxWords)
models = modelTrainer("GRU", embeddedDims, split, epochs, batchSize, minWords, maxWords)