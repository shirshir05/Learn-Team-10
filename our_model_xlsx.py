import pandas as pd

from random_forest import random_forest
# Feature Extraction with RFE
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

training_feature = pd.read_excel('training_feature_all.xlsx', index_col=0)
training = pd.read_excel('training_all.xlsx', index_col=0)
testing_feature = pd.read_excel('testing_feature_all.xlsx', index_col=0)
test = pd.read_excel('test_all.xlsx', index_col=0)


# training_feature
dataframe = training_feature
array = dataframe.values
X = array[0:, 0: 38]
Y = training['result']
# feature extraction
model = LogisticRegression(solver='liblinear')
rfe = RFE(model)
fit = rfe.fit(X, Y)
print("training -  Num Features: %d" % fit.n_features_)
print("training - Selected Features: %s" % fit.support_)
print("training - Feature Ranking: %s" % fit.ranking_)
training_feature = training_feature.iloc[:, fit.support_]
testing_feature = testing_feature.iloc[:, fit.support_]
print("shir")


randomForest = random_forest()
randomForest.main(training_feature, training['result'], testing_feature, test['result'])
