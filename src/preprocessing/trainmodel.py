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

from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, Normalizer

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.optimizers import SGD

from sklearn.externals import joblib
from keras.models import model_from_json

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

def scale_data(path,filename):
	# scale to 0 - 1 without changing the distribution pattern, outlier still affects
	mmsc = MinMaxScaler(feature_range = (0, 1))
	# (x - mean)/sd = makes mean close to 0
	ssc = StandardScaler()
	chunk = 100000
	for df in  pd.read_csv(path+filename+'.csv', chunksize=chunk, index_col=0):
		df.drop(['txn'], axis=1, inplace=True)
		X_train, X_validation, Y_train, Y_validation = get_dataset(df, 0.0)
		ssc.partial_fit(X_train)
		mmsc.partial_fit(X_train)
		print('fit scaler')

	joblib.dump(mmsc, path+filename+'_minmaxscaler.save') 
	joblib.dump(ssc, path+filename+'_standardscaler.save') 

def predict(testset,testvalue,ssc,regressor):
	pre = regressor.predict(testset)
	print(pre)
	pre = ssc.inverse_transform(pre)
	plt.plot(testvalue, color = 'black', label = 'Actual icd10')
	plt.plot(pre, color = 'green', label = 'Predicted icd10')
	plt.title('Actual vs Predicted icd10')
	plt.xlabel('feature')
	plt.ylabel('value')
	plt.legend()
	plt.show()

def history_loss(loss,val_loss):
	plt.plot(loss)
	plt.plot(val_loss)
	plt.title('model train vs validation loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'validation'], loc='upper right')
	plt.show()

def save_model(model,filename):
	if not os.path.exists('../../secret/data/model/'):
		os.makedirs('../../secret/data/model/')
	# serialize model to JSON
	model_json = model.to_json()
	with open('../../secret/data/model/'+filename+".json", "w") as json_file:
		 json_file.write(model_json)
	# serialize weights to HDF5
	model.save_weights('../../secret/data/model/'+filename+".h5")
	print("save model")

def load_model(filename):
	# load json and create model
	json_file = open('../../secret/data/model/'+filename+'.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights('../../secret/data/model/'+filename+'.h5')
	print("Loaded model from disk")
	return loaded_model

def train_model2(name,f):
	c = XGBClassifier(max_depth=100)
	chunk = 10000
	r = 1
	#f = 12
	#f = 7
	#f = 2
	testset = None
	testvalue = None
	
	regressor = Sequential()

	file = Path('../../secret/data/model/'+name+'.h5')
	total_history = None
	if file.is_file():
		regressor = load_model(name)
	else:
		'''
		regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (f, 1)))
		regressor.add(Dropout(0.2))

		regressor.add(LSTM(units = 50, return_sequences = True))
		regressor.add(Dropout(0.2))

		regressor.add(LSTM(units = 50, return_sequences = True))
		regressor.add(Dropout(0.2))

		regressor.add(LSTM(units = 50))
		regressor.add(Dropout(0.2))

		regressor.add(Dense(units = 1))

		regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
		'''

		regressor.add(LSTM(512, return_sequences=True,
		            input_shape=(f, 1)))  # returns a sequence of vectors of dimension 32
		regressor.add(LSTM(512, return_sequences=True))  # returns a sequence of vectors of dimension 32
		regressor.add(LSTM(512))  # return a single vector of dimension 32
		regressor.add(Dense(1, activation='softmax'))

	regressor.compile(loss='mean_squared_error',
				        optimizer='adam',
				        metrics=['accuracy'])

	#ssc = joblib.load('../../secret/data/vec/'+name+'_standardscaler.save') 
	ssc = joblib.load('../../secret/data/vec/'+name+'_minmaxscaler.save') 

	for df in  pd.read_csv('../../secret/data/vec/'+name+'.csv', chunksize=chunk, index_col=0):
		df.drop(['txn'], axis=1, inplace=True)
		X_train, X_validation, Y_train, Y_validation = get_dataset(df, 0.0)
		X_train = ssc.fit_transform(X_train)
		X_train = X_train.reshape(len(X_train),len(df.columns)-1,1)
		'''
		X_validation = ssc.fit_transform(X_validation)
		X_validation = X_validation.reshape(len(X_validation),len(df.columns)-1,1)

		if testset is None:
			testset = np.append(testset,X_validation)
			testvalue = np.append(testvalue,Y_validation)
		else:
			testset = X_validation
			testvalue = Y_validation
		'''
		print('Chunk '+str(r))
		history = regressor.fit(X_train, Y_train, epochs = 100, batch_size = 32, validation_split=0.1)
		#regressor.reset_states()
		r = r+1
		#if r == 10:
		#	break
		#predict(testset,testvalue,ssc,regressor)
		save_file(pd.DataFrame([[history.history['loss'],history.history['val_loss']]], columns=['loss','val_loss']),
					'../../secret/data/model/'+name+'_history.csv')


		save_model(regressor,name)
		if Path('../../secret/data/model/'+name+'_history.csv').is_file():
			total_history = pd.read_csv('../../secret/data/model/'+name+'_history.csv', index_col=0)
			history_loss(total_history['loss'].values.tolist(), total_history['val_loss'].values.tolist())
		#break



