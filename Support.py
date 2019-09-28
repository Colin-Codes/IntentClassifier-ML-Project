import numpy as np
import csv
import itertools

from keras.utils import to_categorical

class Indexed:
    def __init__(self, TrainCSVpath, TestCSVPath, inputLength):
        self.inputLength = inputLength

        # Import data
        with open(TrainCSVpath, "r") as importTrainCSV:
            self.trainTexts = [i[1:2] for i in list(csv.reader(importTrainCSV))[1:]]
        with open(TestCSVPath, "r") as importTestCSV:
            self.testTexts = [i[1:2] for i in list(csv.reader(importTestCSV))[1:]]

        # Indexing and de-Indexing tools
        self.text_wordlists = [self.splitSentence(text) for text in self.trainTexts]
        self.text_words = set(itertools.chain(*self.text_wordlists))
        self.word2int = dict((word,enum) for enum, word in enumerate(self.text_words))
        self.int2word = list(self.text_words)

        # Encode data
        self.X = self.encodeExamples(self.trainTexts)
        self.X_test = self.encodeExamples(self.testTexts)

        # List of target classes
        with open(TrainCSVpath, "r") as importTestCSV:
            self.targets = [i[:1] for i in list(csv.reader(importTestCSV))[1:]]
        self.target_words = set(itertools.chain(*self.targets))
        self.target2int = dict((word,enum) for enum, word in enumerate(self.target_words))
        self.int2target = list(self.target_words)
        self.y = to_categorical([self.target2int[target[0]] for target in self.targets])

    def encodeExamples(self, examples):
        examples = [self.encodeSentence(example) for example in examples]
        for example in examples:
            while(len(example)) < self.inputLength:
                example.append(0) 
        examples = [example[:self.inputLength] for example in examples]
        return np.array(examples)

    def encodeSentence(self, sentence):
        return [self.word2int.get(word,-1) for word in self.splitSentence(sentence)]

    def splitSentence(self, sentence):
        return [word.lower() for word in sentence[0].split(' ')]

    def decodePredictions(self, predictions):
        return [self.decodePrediction(prediction) for prediction in predictions]

    def decodePrediction(self, prediction):
        return self.int2target[np.argmax(prediction)]

    def npX(self):
        return np.array(self.X, dtype='int32')

    def npX_test(self):
        return np.array(self.X_test, dtype='int32')

    def npy(self):
        return np.array(self.y, dtype='int32')
    
    def vocabSize(self):
        return len(self.word2int)

    def targetSize(self):
        return len(self.target2int)