import pandas as pd
from sklearn.model_selection import cross_val_score


import sys
import os
from pathlib import Path
path = '../../../secret/data/'

#count number of icd10 categories
icd = pd.read_csv(path+'trainingset/raw/reg.csv', index_col=0, low_memory=False)[['icd10']]
data = pd.read_csv(path+'trainingset/vec/reg.csv', index_col=0, low_memory=False).drop(columns=['icd10'])
result = pd.concat([data,icd], axis=1, sort=False)
#print(result)
result['icd10_1'] = result['icd10'].str[:1]
result['icd10_2'] = result['icd10'].str[:2]
result['icd10_3'] = result['icd10'].str[:3]
result['icd10_4'] = result['icd10'].str[:4]

df = result.groupby(['icd10_1']).size().reset_index(name='count')
df['ratio'] = df['count']*100/df['count'].sum()
df = df[df['ratio']>0.01]
result = result[result['icd10_1'].isin(df['icd10_1'])]
label = result[['icd10_1']]
result.drop(columns=['txn','dx_type','icd10','icd10_1','icd10_2','icd10_3','icd10_4'], inplace=True)
print(result)
'''
for f1 in df['icd10_1']:
	d1 = result[result['icd10_1']==f1]
	df1 = d1.groupby(['icd10_2']).size().reset_index(name='count')
	print(df1)
'''

s = int(len(result)*0.7)
X_train = result[:s]
y_train = label[:s].values.ravel()
X_test = result[s:]
y_test = label[s:].values.ravel()

from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

classifiers = [
	KNeighborsClassifier(25),
	SVC(kernel="rbf", C=0.025, probability=True),
	NuSVC(probability=True),
	DecisionTreeClassifier(),
	RandomForestClassifier(),
	AdaBoostClassifier(),
	GradientBoostingClassifier(),
	GaussianNB(),
	LinearDiscriminantAnalysis(),
	QuadraticDiscriminantAnalysis()]

# Logging for Visual Comparison
log_cols = ["Classifier", "Accuracy", "Log Loss"]
log = pd.DataFrame(columns=log_cols)

for clf in classifiers:
	clf.fit(X_train, y_train)
	name = clf.__class__.__name__

	print("=" * 30)
	print(name)

	print('****Results****')
	train_predictions = clf.predict(X_test)
	acc = accuracy_score(y_test, train_predictions)
	print("Accuracy: {:.4%}".format(acc))

	train_predictions = clf.predict_proba(X_test)
	ll = log_loss(y_test, train_predictions)
	print("Log Loss: {}".format(ll))

	log_entry = pd.DataFrame([[name, acc * 100, ll]], columns=log_cols)
	log = log.append(log_entry)

print("=" * 30)
