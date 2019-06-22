import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer

QuestionAnswerPairs = pd.read_csv('data/dummy_train.csv', sep=',', engine='python', quotechar='"')
X_train_data = QuestionAnswerPairs['Question']
y_train = QuestionAnswerPairs['Answer']

QuestionAnswerPairs = pd.read_csv('data/dummy_test.csv', sep=',', engine='python', quotechar='"')
X_test_data = QuestionAnswerPairs['Question']
y_test = QuestionAnswerPairs['Answer']

Vectorizer = CountVectorizer(max_features=100, stop_words='english')
BagOfWords = Vectorizer.fit(X_train)
X_train = BagOfWords.transform(X_train_data)
X_test = BagOfWords.transform(X_test_data)

# scores = []
# for i in range(1,25):
#     model = KNeighborsClassifier(n_neighbors=i)
#     model.fit(X_train, y_train)
#     scores.append(model.score(X_test, y_test))

# plt.scatter(range(1,25), scores)
# plt.show()


model = KNeighborsClassifier(n_neighbors=7)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
X_test_data.head()
print(y_pred)
