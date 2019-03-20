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
	p = '../../secret/data/drug_onehot.csv'
	value = []
	for df in  pd.read_csv(p, chunksize=1000000):
		df = df[df[feature].notnull()]
		v = df[feature].unique().tolist()
		value = value + v
		value = list(set(value))       
		value.sort()
	df = pd.DataFrame([value],columns=[feature])
	df.to_csv('../../secret/data/target_class.csv')
	print(df)
		
def train_model(feature):
	p = '../../secret/data/'+feature+'_onehot.csv'
	c = GaussianNB()
	n = 0
	testset = None
	l = None
	chunk = 5000
	for df in  pd.read_csv(p, chunksize=chunk):
		if l is None:
			l = df.columns.tolist()
			l.remove('TXN')
			l.remove('icd10')
			l = l + ['icd10']

		df = df[l]
		
		if testset is None:
			testset = df.tail(1)
		else:
			testset = pd.concat([testset,df.tail(1)], ignore_index=True)
		
		X_train, X_validation, Y_train, Y_validation = get_dataset(df.head(chunk-1), 0.0)
		c.partial_fit(X_train, Y_train, classes=xx)
		n = n+1
		print('Batch '+str(n))
		if n%10 == 0:
			eval(testset,c)
	eval(testset,c)
