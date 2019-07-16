import csv
import itertools
import numpy as np
import pickle

from keras.layers import Input, Embedding, merge, Flatten, SimpleRNN, Dense, LSTM, SpatialDropout1D
from keras.models import Model, Sequential

with open("prototyping/data/dummy_train.txt", "r") as importCSV:
    questions = list(csv.reader(importCSV))

intents = np.array([[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,0,1],[0,0,0,1],[0,0,0,1],[0,0,0,1],[0,0,0,1],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,0,1],[0,0,0,1],[0,0,0,1],[0,0,0,1],[0,0,0,1],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,0,1],[0,0,0,1],[0,0,0,1],[0,0,0,1],[0,0,0,1]])

#lemmatize and catalog the words
question2wordlist = lambda question: question.lower().split(' ')
questions_wordlists = [question2wordlist(question[0]) for question in questions]
questions_words = set(itertools.chain(*questions_wordlists))
word2int = dict((word,enum) for enum, word in enumerate(questions_words))
int2word = list(questions_words)

with open("prototyping/pkl/word2int.pkl","wb") as saveword2intPKL:
    pickle.dump(word2int, saveword2intPKL)

with open("prototyping/pkl/int2word.pkl","wb") as saveint2wordPKL:
    pickle.dump(int2word, saveint2wordPKL)

#parameters
maxQuestionLength = 20
numWords = len(word2int)
numEmbeddedDims = 3

#create a numpy array from the sentences
question2intlist = lambda question: [word2int[word] for word in question]
questions_intlists = [question2intlist(question) for question in questions_wordlists]
for question in questions_intlists:
    while(len(question)) < maxQuestionLength:
        question.append(0)
questions_nparray = np.array(questions_intlists, dtype='int32')

input_question = Input(shape=(maxQuestionLength,), dtype='int32')
input_embedding = Embedding(numWords, numEmbeddedDims)(input_question)
intent_prediction = SimpleRNN(1)(input_embedding)

model = Sequential()
model.add(Embedding(numWords, numEmbeddedDims, input_length=maxQuestionLength))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(20, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(4, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs = 300
batch_size = 64

history = model.fit(questions_nparray, intents, epochs=epochs, batch_size=batch_size,validation_split=0.1)

#print training predictions
y_predict = model.predict(questions_nparray)
print(y_predict)

with open("prototyping/pkl/model.pkl","wb") as savemodelPKL:
    pickle.dump(model, savemodelPKL)