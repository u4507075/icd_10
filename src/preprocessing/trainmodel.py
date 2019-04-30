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

from sklearn.metrics import precision_recall_fscore_support

import os
import re
import pickle
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

def get_dataset(trainingset, validation_size):
	for name in trainingset.columns:
		if name != 'icd10' and str(trainingset[name].dtype) == 'object':
			trainingset[name] = pd.to_numeric(trainingset[name], errors='coerce')
	trainingset.fillna(0,inplace=True)
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

def save_file(df,p):
	file = Path(p)
	if file.is_file():
		with open(p, 'a') as f:
			df.to_csv(f, header=False)
	else:
		df.to_csv(p)

def train_model(filename):

	#c = MultinomialNB()
	#c = BernoulliNB()
	#c = PassiveAggressiveClassifier(n_jobs=-1, warm_start=True)
	#c = SGDClassifier(loss='log')
	#c = Perceptron(n_jobs=-1,warm_start=True)

	#c = SVC()


	print(filename)
	p = '../../secret/data/trainingset_clean/'+filename+'.csv'
	targets = []
	for df in  pd.read_csv(p, chunksize=100000, index_col=0):
		df['icd10'] = df['icd10'].apply(str)
		v = df['icd10'].unique().tolist()
		targets = targets + v
		targets = list(set(targets))
	targets.sort()

	if not os.path.exists('../../secret/data/model_performance/'):
		os.makedirs('../../secret/data/model_performance/')

	if not os.path.exists('../../secret/data/model/'):
		os.makedirs('../../secret/data/model/')

	if not os.path.exists('../../secret/data/model/'+filename):
		os.makedirs('../../secret/data/model/'+filename)

	regex = re.compile('[A-Z]')
	target_classes = [i for i in targets if regex.match(i)]

	if not Path('../../secret/data/model_performance/training_record.csv').is_file():
		save_file(pd.DataFrame(columns=['feature','icd10']),'../../secret/data/model_performance/training_record.csv')
	hx = pd.read_csv('../../secret/data/model_performance/training_record.csv', index_col=0)

	print("Start saving models")

	for target in target_classes:
		if len(hx[(hx['feature'] == filename) & (hx['icd10'] == target)]) == 0:
			model_file = '../../secret/data/model/'+filename+'/'+target+'.sav'
			if not Path(model_file).is_file():
				print(target)
				data = None
				chunk = 100000
				c = XGBClassifier(max_depth=100)

				for df in  pd.read_csv(p, chunksize=chunk, index_col=0):
					df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
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
				if len(data) >= 100:

					X_train, X_validation, Y_train, Y_validation = get_dataset(data, 0.1)
					c.fit(X_train, Y_train)
					pre = c.predict(X_validation)
					print(target)
					cf = confusion_matrix(Y_validation, pre)
					print(cf)
					cr = classification_report(Y_validation, pre)
					print(cr)
					v = precision_recall_fscore_support(Y_validation, pre, average='weighted')
					X_train, X_validation, Y_train, Y_validation = get_dataset(data, 0.0)
					c.fit(X_train, Y_train)
					pickle.dump(c, open(model_file, 'wb'))
					dfp = pd.DataFrame([[filename,target,v[0],v[1],v[2],len(data)]],columns=['feature','icd10','precision','recall','fscore','n'])
					save_file(dfp,'../../secret/data/model_performance/model_performance.csv')

				save_file(pd.DataFrame([[filename,target]],columns=['feature','icd10']),'../../secret/data/model_performance/training_record.csv')
		else:
			print(filename + ' and ' + target + ' ' + 'already exist.')


def save_history():
	files = os.listdir('../../secret/data/trainingset/')

	stop = False

	for filename in files:
		filename = filename.replace('.csv','')
		if stop:
			break
		print(filename)
		p = '../../secret/data/trainingset/'+filename+'.csv'
		targets = []
		for df in  pd.read_csv(p, chunksize=100000, index_col=0):
			df['icd10'] = df['icd10'].apply(str)
			v = df['icd10'].unique().tolist()
			targets = targets + v
			targets = list(set(targets))
		targets.sort()

		if not os.path.exists('../../secret/data/model_performance/'):
			os.makedirs('../../secret/data/model_performance/')

		if not os.path.exists('../../secret/data/model/'):
			os.makedirs('../../secret/data/model/')

		if not os.path.exists('../../secret/data/model/'+filename):
			os.makedirs('../../secret/data/model/'+filename)

		regex = re.compile('[A-Z]')
		target_classes = [i for i in targets if regex.match(i)]
		
		for target in target_classes:
			if stop:
				break
			
			if not Path('../../secret/data/model_performance/training_record.csv').is_file():
				save_file(pd.DataFrame(columns=['feature','icd10']),'../../secret/data/model_performance/training_record.csv')

			save_file(pd.DataFrame([[filename,target]],columns=['feature','icd10']),'../../secret/data/model_performance/training_record.csv')
			if filename == 'L1901' and target == 'M8445':
				stop = True



def train_model2():
	c = XGBClassifier(max_depth=100)
	chunk = 10000
	for df in  pd.read_csv('../../secret/data/vec/adm.csv', chunksize=chunk, index_col=0):
		df.drop(['txn'], axis=1, inplace=True)
		X_train, X_validation, Y_train, Y_validation = get_dataset(df, 0.1)
		#X_train = sc.fit_transform(X_train)
		X_train = X_train.reshape(len(X_train),len(df.columns)-1,1)
		X_validation = X_validation.reshape(len(X_validation),len(df.columns)-1,1)

		regressor = Sequential()

		regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
		regressor.add(Dropout(0.2))

		regressor.add(LSTM(units = 50, return_sequences = True))
		regressor.add(Dropout(0.2))

		regressor.add(LSTM(units = 50, return_sequences = True))
		regressor.add(Dropout(0.2))

		regressor.add(LSTM(units = 50))
		regressor.add(Dropout(0.2))

		regressor.add(Dense(units = 1))

		regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

		regressor.fit(X_train, Y_train, epochs = 100, batch_size = 32)

		pre = regressor.predict(X_validation)
		print(pre)
		#pre = sc.inverse_transform(pre)
		
		plt.plot(Y_validation, color = 'black', label = 'Actual icd10')
		plt.plot(pre, color = 'green', label = 'Predicted icd10')
		plt.title('Actual vs Predicted icd10')
		plt.xlabel('feature')
		plt.ylabel('value')
		plt.legend()
		plt.show()

		break


