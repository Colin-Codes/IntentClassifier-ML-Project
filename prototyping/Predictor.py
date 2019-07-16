import numpy as np
import csv
import pickle

from keras.layers import Input, Embedding, merge, Flatten, SimpleRNN, Dense, LSTM, SpatialDropout1D
from keras.models import Model, Sequential

def predict(question):

    with open("prototyping/pkl/word2int.pkl","rb") as loadword2intPKL:
        word2int = pickle.load(loadword2intPKL)

    with open("prototyping/pkl/model.pkl","rb") as loadmodelPKL:
        model = pickle.load(loadmodelPKL)

    #parameters
    maxQuestionLength = 20

    #create a numpy array from the sentences
    question_wordlist = question.lower().replace('/n',' ').replace('  ', ' ').split(' ')
    question_intlist = [word2int.get(word, 0) for word in question_wordlist]
    while(len(question_intlist) < maxQuestionLength):
        question_intlist.append(0)
    questions_nparray = np.array([question_intlist,], dtype='int32')
    y_predict = model.predict(questions_nparray)
    return y_predict