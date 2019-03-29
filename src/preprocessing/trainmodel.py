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
from xgboost import XGBClassifier

def get_dataset(trainingset, validation_size):
	n = len(trainingset.columns)-1
	array = trainingset.values
	X = array[:,0:n]
	Y = array[:,n]
	seed = 7
	X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
	X_train, Y_train = shuffle(X_train, Y_train, random_state=seed)
	return X_train, X_validation, Y_train, Y_validation

def eval(testset_path,model):
	X_train, X_validation, Y_train, Y_validation = get_dataset(testset, 0.0)
	p = model.predict(X_train)
	cf = confusion_matrix(Y_train, p)
	print(cf)
	cr = classification_report(Y_train, p)
	print(cr)

def get_target_class(p,name):
	p = '../../secret/data/drug/'+name
	value = []
	for df in  pd.read_csv(p, chunksize=10000):
		v = df[feature].unique().tolist()
		value = value + v
		value = list(set(value))
		value.sort()
		print(len(value))

	df = pd.DataFrame.from_dict({feature:value})
	df.to_csv('../../secret/data/test/'+name+'_class.csv')
	print(df)



def train_model(p):

	#c = MultinomialNB()
	c = BernoulliNB()
	#c = PassiveAggressiveClassifier(n_jobs=-1, warm_start=True)
	#c = SGDClassifier(loss='log')
	#c = Perceptron(n_jobs=-1,warm_start=True)

	#c = SVC()
	c = XGBClassifier(max_depth=100)
	
	targets = []
	for df in  pd.read_csv(p, chunksize=100000, index_col=0):
		v = df['icd10'].unique().tolist()
		target = target + v
		target = list(set(target))
	target.sort()

	for target in targets:
		data = None
		chunk = 10000
		for df in  pd.read_csv(p, chunksize=chunk, index_col=0):
			df.drop(['TXN'], axis=1, inplace=True)

			t = df[df['icd10']==target]
			nt = df[df['icd10']!=target]
			nt = nt.assign(icd10 = 'not_'+target)
			if len(nt) > len(t):
				nt = nt.sample(frac=1).reset_index(drop=True)
				nt = nt.head(len(t))
			t = t.append(nt, ignore_index = True)
			t = t.reset_index(drop=True)
			if data is None:
				data = t
			else:
				data = data.append(t).reset_index(drop=True)
			#print(data)
		X_train, X_validation, Y_train, Y_validation = get_dataset(data, 0.1)
		c.fit(X_train, Y_train)
		p = c.predict(X_validation)
		print(target)
		cf = confusion_matrix(Y_validation, p)
		print(cf)
		cr = classification_report(Y_validation, p)
		print(cr)

'''
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

def train_model_onetime(target):
	df = pd.read_csv(target)
	X_train, X_validation, Y_train, Y_validation = get_dataset(df, 0.1)
	
	c = MultinomialNB()
	c.fit(X_train, Y_train)

	p = model.predict(X_validation)
	cf = confusion_matrix(Y_validation, p)
	print(cf)
	cr = classification_report(Y_validation, p)
	print(cr)
'''
