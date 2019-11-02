import pandas as pd
import os
from dataHelper import Data
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler

#Raw dataset
data = Data('data/trainingset.csv','data/testset.csv', 20)
X = data.npX()
y = data.npy()
print(data.classDistribution())

#Balanced dataset
ros = RandomOverSampler(random_state=42)
X_res, y_res = ros.fit_resample(X, y)
new_df = pd.DataFrame(list(zip(data.decodePredictions(y_res), data.decodeExamples(X_res))), columns=['Class','Email'])
print(new_df.head())
new_classDistribution = new_df.groupby('Class').size()
new_classDistribution.sort_values(ascending=False, inplace=True)
print(new_classDistribution)
new_df.to_csv(r'data/trainingset_balanced.csv', index = None, header=True)

#Augmented dataset
currentPath = os.getcwd()
os.chdir('eda_nlp/code')
os.system('augment.py --input=../../data/trainingset_balanced.csv --output=../../data/trainingset_augmented.csv --num_aug=4 --alpha=0.1')
os.chdir(currentPath)
data_augmented = Data('data/trainingset_augmented.csv','data/testset.csv', 20)
print(data_augmented.classDistribution())