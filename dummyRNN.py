import numpy as np
import csv
import itertools

from keras.layers import Input, Embedding, merge, Flatten, SimpleRNN, Dense, LSTM, SpatialDropout1D
from keras.models import Model, Sequential
with open("data/dummy_train.txt", "r") as importCSV:
    questions = list(csv.reader(importCSV))

with open("data/dummy_test.txt", "r") as importCSV:
    test_questions = list(csv.reader(importCSV))

intents = np.array([[1,0,0,0],
[1,0,0,0],
[1,0,0,0],
[1,0,0,0],
[1,0,0,0],
[0,1,0,0],
[0,1,0,0],
[0,1,0,0],
[0,1,0,0],
[0,1,0,0],
[0,0,1,0],
[0,0,1,0],
[0,0,1,0],
[0,0,1,0],
[0,0,1,0],
[0,0,0,1],
[0,0,0,1],
[0,0,0,1],
[0,0,0,1],
[0,0,0,1],
[1,0,0,0],
[1,0,0,0],
[1,0,0,0],
[1,0,0,0],
[1,0,0,0],
[0,1,0,0],
[0,1,0,0],
[0,1,0,0],
[0,1,0,0],
[0,1,0,0],
[0,0,1,0],
[0,0,1,0],
[0,0,1,0],
[0,0,1,0],
[0,0,1,0],
[0,0,0,1],
[0,0,0,1],
[0,0,0,1],
[0,0,0,1],
[0,0,0,1],
[1,0,0,0],
[1,0,0,0],
[1,0,0,0],
[1,0,0,0],
[1,0,0,0],
[0,1,0,0],
[0,1,0,0],
[0,1,0,0],
[0,1,0,0],
[0,1,0,0],
[0,0,1,0],
[0,0,1,0],
[0,0,1,0],
[0,0,1,0],
[0,0,1,0],
[0,0,0,1],
[0,0,0,1],
[0,0,0,1],
[0,0,0,1],
[0,0,0,1]])

test_intents = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])

#lemmatize and catalog the words
question2wordlist = lambda question: question.lower().split(' ')
questions_wordlists = [question2wordlist(question[0]) for question in questions]
questions_words = set(itertools.chain(*questions_wordlists))
word2int = dict((word,enum) for enum, word in enumerate(questions_words))
int2word = list(questions_words)

#lemmatize and catalog the words
test_question2wordlist = lambda question: question.lower().split(' ')
test_questions_wordlists = [test_question2wordlist(question[0]) for question in test_questions]

#parameters
maxQuestionLength = 20
numWords = len(questions_words)
numEmbeddedDims = 3

#create a numpy array from the sentences
question2intlist = lambda question: [word2int[word] for word in question]
questions_intlists = [question2intlist(question) for question in questions_wordlists]
for question in questions_intlists:
    while(len(question)) < maxQuestionLength:
        question.append(0)
questions_nparray = np.array(questions_intlists, dtype='int32')

#create a numpy array from the sentences
test_question2intlist = lambda question: [word2int[word] for word in question]
test_questions_intlists = [test_question2intlist(question) for question in test_questions_wordlists]
for question in test_questions_intlists:
    while(len(question)) < maxQuestionLength:
        question.append(0)
test_questions_nparray = np.array(test_questions_intlists, dtype='int32')

input_question = Input(shape=(maxQuestionLength,), dtype='int32')
input_embedding = Embedding(numWords, numEmbeddedDims)(input_question)
intent_prediction = SimpleRNN(1)(input_embedding)

# predict_intent = Model(input=[input_question], output=[intent_prediction])
# predict_intent.compile(optimizer='sgd', loss='binary_crossentropy')

# # fit the model to predict what color each person is
# predict_intent.fit([questions_nparray], [intents], nb_epoch=200, verbose=1)
# embeddings = predict_intent.layers[1].get_weights()[0]

model = Sequential()
model.add(Embedding(numWords, numEmbeddedDims, input_length=maxQuestionLength))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(20, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(4, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs = 200
batch_size = 64

history = model.fit(questions_nparray, intents, epochs=epochs, batch_size=batch_size,validation_split=0.1)


# print embeddings for each word
#for i in range(numWords):
	#print('{}: {}'.format(int2word[i], embeddings[i]))

#y_predict = predict_intent.predict(test_questions_nparray)
y_predict = model.predict(test_questions_nparray)
print(y_predict)