import pandas as pd
import numpy as np
import os
from dataHelper import Data
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler

# Raw dataset
data = Data('data/originalData.csv','data/originalData.csv', -1)
originalSet = data.pdDataFrame()
X = originalSet['Email']
y = originalSet['Class']
print("Original class distribution: \n" + str(data.classDistribution()))

# Split data across test (80%) and train (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
testSet = pd.DataFrame(list(zip(y_test, X_test)), columns=['Class','Email'])
trainingSet = pd.DataFrame(list(zip(y_train, X_train)), columns=['Class','Email'])

# Generate test set
testSet.to_csv(r'data/testSet.csv', index = None, header=True)
testSetDistribution = testSet.groupby('Class').size()
testSetDistribution.sort_values(ascending=False, inplace=True)
print("Test class distribution: \n" + str(testSetDistribution))

# Generate training set
trainingSet.to_csv(r'data/trainingSet.csv', index = None, header=True)
trainingSetDistribution = trainingSet.groupby('Class').size()
trainingSetDistribution.sort_values(ascending=False, inplace=True)
print("Training class distribution: \n" + str(trainingSetDistribution))

# Balanced dataset - randomly over-sample smaller classes until all classes are equally represented
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(np.array(X_train).reshape(-1,1), y_train)
balancedTrainingSet = pd.DataFrame(list(zip(y_resampled, X_resampled)), columns=['Class','Email'])
balancedTrainingSet.to_csv(r'data/trainingSet_balanced.csv', index = None, header=True)
balancedTrainingSetDistribution = balancedTrainingSet.groupby('Class').size()
balancedTrainingSetDistribution.sort_values(ascending=False, inplace=True)
print("Balanced class distribution: \n" + str(balancedTrainingSetDistribution))

#Augmented dataset
currentPath = os.getcwd()
os.chdir('EDA/code')
os.system('augment.py --input=../../data/trainingSet_balanced.csv --output=../../data/trainingSet_augmented.csv --num_aug=4 --alpha=0.1')
os.chdir(currentPath)
data_augmented = Data('data/trainingset_augmented.csv','data/testset.csv', -1)
print("Augmented class distribution: \n" + str(data_augmented.classDistribution()))