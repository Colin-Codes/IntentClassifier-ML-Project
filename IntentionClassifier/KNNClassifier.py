import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

#Import the data, split into X, y
ClassEmailPairs = pd.read_csv('data/trainingset.csv', sep=',', engine='python', quotechar='"')
X = ClassEmailPairs['Email']
y = ClassEmailPairs['Class']

#Create the word embedding
Vectorizer = CountVectorizer(max_features=100, stop_words='english')
BagOfWords = Vectorizer.fit(X)
X_train = BagOfWords.transform(X)

#List of unique intentions
intents = ClassEmailPairs['Class'].unique()

print(intents)

#Score the model
scores = []
for i in range(1,25):
    model = KNeighborsClassifier(n_neighbors=i)
    CVScores = cross_val_score(model, X_train, y, cv=25)
    scores.append(CVScores.mean())

plt.scatter(range(1,25), scores)
plt.show()


# model = KNeighborsClassifier(n_neighbors=7)
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# X_test_data.head()
# print(y_pred)