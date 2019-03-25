import mysql.connector as sql
import pandas as pd
from pathlib import Path
import numpy as np

from sklearn import preprocessing
from sklearn import model_selection
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import PassiveAggressiveClassifier


def get_dataset(trainingset, validation_size):
	n = len(trainingset.columns)-1
	array = trainingset.values
	X = array[:,0:n]
	Y = array[:,n]
	seed = 7
	X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
	X_train, Y_train = shuffle(X_train, Y_train, random_state=seed)
	return X_train, X_validation, Y_train, Y_validation

def eval(testset,model):
	X_train, X_validation, Y_train, Y_validation = get_dataset(testset, 0.0)
	p = model.predict(X_train)
	cf = confusion_matrix(Y_train, p)
	print(cf)
	cr = classification_report(Y_train, p)
	print(cr)

def get_target_class(feature):
	p = '../../secret/data/drug/drug_onehot.csv'
	value = []
	for df in  pd.read_csv(p, chunksize=10000):
		v = df[feature].unique().tolist()
		value = value + v
		value = list(set(value))
		value.sort()
		print(len(value))

	df = pd.DataFrame.from_dict({feature:value})
	df.to_csv('../../secret/data/drug/target_class.csv')
	print(df)

def get_small_sample():
	p = '../../secret/data/drug_onehot.csv'
	value = []
	feature = 'icd10'
	for df in  pd.read_csv(p, chunksize=50000):
		df.to_csv('../../secret/data/drug_onehot_s.csv')
		v = df[feature].unique().tolist()
		value = value + v
		value = list(set(value))
		value.sort()
		df = pd.DataFrame.from_dict({feature:value})
		df.to_csv('../../secret/data/target_class_s.csv')
		break

def icd10_head(x):
	return x[0]

def train_model(feature):
	p = '../../secret/data/'+feature+'_onehot.csv'
	df = pd.read_csv('../../secret/data/target_class.csv')
	df['icd10'] = df['icd10'].apply(icd10_head)
	class_list = df['icd10'].tolist()
	#c = MultinomialNB()
	#c = BernoulliNB()
	c = PassiveAggressiveClassifier(n_jobs=-1, warm_start=True)
	#c = SGDClassifier(loss='log')
	#c = Perceptron(n_jobs=-1,warm_start=True)
	n = 0
	testset = None
	l = None
	chunk = 100
	
	for df in  pd.read_csv(p, chunksize=chunk):
		if l is None:
			l = df.columns.tolist()
			l.remove('TXN')
			l.remove('icd10')
			l = l + ['icd10']

		df = df[l]
		df['icd10'] = df['icd10'].apply(icd10_head)
		if testset is None:
			testset = df.tail(1)
		else:
			testset = pd.concat([testset,df.tail(1)], ignore_index=True)
		
		X_train, X_validation, Y_train, Y_validation = get_dataset(df.head(chunk-1), 0.0)
		c.partial_fit(X_train, Y_train, classes=class_list)
		n = n+1
		print('Batch '+str(n))
		if n%10 == 0:
			eval(testset,c)
	eval(testset,c)

