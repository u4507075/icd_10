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

def train_model(feature):
	p = '../../secret/data/'+feature+'_onehot.csv'
	c = GaussianNB()
	n = 0
	for df in  pd.read_csv(p, chunksize=1000):
		l = df.columns.tolist()
		l.remove('TXN')
		l.remove('icd10')
		l = l + ['icd10']

		df = df[l]
		X_train, X_validation, Y_train, Y_validation = get_dataset(df, 0.0)
		if n < 1000:
			c.fit(X_train, Y_train)
		else:
			p = c.predict(X_train)
			cf = confusion_matrix(Y_train, p)
			print(cf)
			cr = classification_report(Y_train, p)
			print(cr)

			break
		n = n+1
		print('Batch '+str(n))
