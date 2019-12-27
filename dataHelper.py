from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
import pandas as pd
import random
import csv
import itertools

from keras.utils import to_categorical

class Data:
    def __init__(self, _parameters):
        self.parameters = _parameters

        if self.parameters.shuffleData == True:
            self.df = pd.read_csv(self.parameters.trainFilePath).sample(frac=1)
            self.df_test = pd.read_csv(self.parameters.testFilePath).sample(frac=1)
        else:
            self.df = pd.read_csv(self.parameters.trainFilePath)
            self.df_test = pd.read_csv(self.parameters.trainFilePath)            

        # Shallow models
        if self.parameters.sentenceSize < 0:

            # Create the BoW embedding
            BoWModel = CountVectorizer(stop_words='english')
            BagOfWords = BoWModel.fit(self.df['Email'])
            self.X_BoW = BagOfWords.transform(self.df['Email'])
            self.X_test_BoW = BagOfWords.transform(self.df_test['Email'])

            # Create the TFIDF embedding
            TFIDFModel = TfidfVectorizer(stop_words='english')
            TFIDF = TFIDFModel.fit(self.df['Email'])
            self.X_TFIDF = TFIDF.transform(self.df['Email'])
            self.X_test_TFIDF = TFIDF.transform(self.df_test['Email'])
            
        # Deep models
        with open(self.parameters.trainFilePath, "r") as importTrainCSV:
            self.trainSamples = [i for i in list(csv.reader(importTrainCSV))[1:]]
        with open(self.parameters.testFilePath, "r") as importTestCSV:
            self.testSamples = [i for i in list(csv.reader(importTestCSV))[1:]]

        # Shuffle
        if self.parameters.shuffleData == True:
            random.shuffle(self.trainSamples)
            random.shuffle(self.testSamples)
        
        # Assign
        self.trainTexts = [i[1:2] for i in self.trainSamples]
        self.testTexts = [i[1:2] for i in self.testSamples]

        # Indexing and de-Indexing tools
        self.text_wordlists = [self.splitSentence(text) for text in self.trainTexts]
        self.text_words = set(itertools.chain(*self.text_wordlists))
        self.word2int = dict((word,enum) for enum, word in enumerate(self.text_words))
        self.int2word = list(self.text_words)

        # Encode data
        self.X = self.encodeExamples(self.trainTexts)
        self.X_test = self.encodeExamples(self.testTexts)

        # List of target classes
        self.targets = [i[:1] for i in self.trainSamples]
        self.target_words = set(itertools.chain(*self.targets))
        self.target2int = dict((word,enum) for enum, word in enumerate(self.target_words))
        self.int2target = list(self.target_words)
        self.y = to_categorical([self.target2int[target[0]] for target in self.targets])

        # Test target classes
        self.testTargets = [i[:1] for i in self.testSamples]
        self.y_test = to_categorical([self.target2int[target[0]] for target in self.testTargets])

    def encodeExamples(self, examples):
        examples = [self.encodeSentence(example) for example in examples]
        for example in examples:
            while(len(example)) < self.parameters.sentenceSize:
                example.append(0) 
        examples = [example[:self.parameters.sentenceSize] for example in examples]
        return np.array(examples)

    def encodeSentence(self, sentence):
        return [self.word2int.get(word,0) for word in self.splitSentence(sentence)]

    def splitSentence(self, sentence):
        return [word.lower() for word in sentence[0].split(' ')]

    def decodeExamples(self, examples):
        examples = [self.joinSentence(self.decodeSentence(example)) for example in examples]
        return examples

    def decodeSentence(self, sentence):
        return [self.int2word[word] if word != 0 else '---' for word in sentence ]

    def joinSentence(self, sentence):
        return " ".join(sentence)

    def decodePredictions(self, predictions):
        return [self.decodePrediction(prediction) for prediction in predictions]

    def labelPredictions(self):
        return self.int2target

    def decodePrediction(self, prediction):
        return self.int2target[np.argmax(prediction)]

    def npX(self):
        return np.array(self.X, dtype='int32')

    def npX_test(self):
        return np.array(self.X_test, dtype='int32')

    def npy(self):
        return np.array(self.y, dtype='int32')

    def npy_test(self):
        return np.array(self.y_test, dtype='int32')

    def indexPredictions(self, y_pred):
        predictions = [np.argmax(prediction) for prediction in y_pred]
        return predictions
        
    def pdDataFrame(self):
        return self.df
        
    def pdDataFrame_test(self):
        return self.df_test

    def Xtext(self):
        return self.df['Email']

    def Xtext_test(self):
        return self.df_test['Email']
    
    def x_BagOfWords(self):
        return self.X_BoW
    
    def x_test_BagOfWords(self):
        return self.X_test_BoW
    
    def x_TFIDF(self):
        return self.X_TFIDF
    
    def x_test_TFIDF(self):
        return self.X_test_TFIDF
    
    def y_labels(self):
        return self.df['Class']
    
    def y_test_labels(self):
        return self.df_test['Class']

    def modelType(self):
        if self.parameters.sentenceSize < 0:
            return "Shallow"
        else:
            return "Deep"
    
    def vocabSize(self):
        return len(self.word2int)

    def targetSize(self):
        return len(self.target2int)

    def classDistribution(self):
        dist = self.df.groupby('Class').size()
        dist.sort_values(ascending=False, inplace=True)
        return dist