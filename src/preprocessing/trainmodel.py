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
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
from sklearn.cluster import Birch
from sklearn.decomposition import MiniBatchDictionaryLearning

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
from dask_ml.wrappers import Incremental

from sklearn.externals import joblib
from keras.models import model_from_json
import joblib as jl
'''
from creme import linear_model
from creme import naive_bayes
from creme import metrics
from creme import optim
from creme import tree
'''
import math
from sklearn.metrics import accuracy_score

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
	if validation_size == None:
		X_train = np.concatenate((X_train, X_validation))
		Y_train = np.concatenate((Y_train, Y_validation))
	X_train, Y_train = shuffle(X_train, Y_train, random_state=seed)
	return X_train, X_validation, Y_train, Y_validation

def get_testset(trainingset):
	for name in trainingset.columns:
		if name != 'icd10' and str(trainingset[name].dtype) == 'object':
			trainingset[name] = pd.to_numeric(trainingset[name], errors='coerce')
	trainingset.fillna(0,inplace=True)
	n = len(trainingset.columns)-1
	array = trainingset.values
	X = array[:,0:n]
	Y = array[:,n]

	X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, shuffle=False)
	X_train = np.concatenate((X_train, X_validation))
	Y_train = np.concatenate((Y_train, Y_validation))

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

def remove_file(p):
	file = Path(p)
	if file.is_file():
		os.remove(p)

def train_model(models, X_train, Y_train, classes):
	for m in models:
		if str(m.estimator).startswith('SGDRegressor') or str(m.estimator).startswith('PassiveAggressiveRegressor'):
			m.partial_fit(X_train, Y_train)
		else:
			m.partial_fit(X_train, Y_train, classes=classes)
			print('Loss :'+str(m.loss_))
	return models

def test(m, X_validation, Y_validation):
	if str(m.estimator).startswith('SGDRegressor') or str(m.estimator).startswith('PassiveAggressiveRegressor'):
		m.partial_fit(X_train, Y_train)
	else:
		m.partial_fit(X_train, Y_train, classes=classes)

	print(m.predict(X_validation)[:len(X_validation)])
	print('Score: ',m.score(X_validation, Y_validation))
	
def dask_model(train, modelname):
	#Good for discrete feature
	#c = MultinomialNB()
	#Good for binary feature
	#c = BernoulliNB()
	#c = PassiveAggressiveClassifier(n_jobs=-1, warm_start=True)
	#c = SGDClassifier(loss='log', penalty='l2', tol=1e-3)
	#c = Perceptron(n_jobs=-1,warm_start=True)
	models = [	#Incremental(MultinomialNB(), scoring='accuracy'),
			#Incremental(PassiveAggressiveClassifier(n_jobs=-1, warm_start=True), scoring='accuracy'),
			#Incremental(SGDClassifier(loss='log', penalty='l2', tol=1e-3), scoring='accuracy'),
			#Incremental(Perceptron(n_jobs=-1,warm_start=True), scoring='accuracy'),
			#Incremental(SGDRegressor(warm_start=True), scoring='accuracy'),
			#Incremental(PassiveAggressiveRegressor(warm_start=True), scoring='accuracy'),
			Incremental(MLPClassifier())]
	#model_names = ['passive-aggrassive-classifier','sgd-classifier','perceptron','sgd-regressor','passive-aggrassive-regressor']
	model_names = ['mlpclassifier']
	#ssc = joblib.load('../../secret/data/vec/'+name+'_standardscaler.save')
	chunk = 10000
	n = 0
	#inc = Incremental(c, scoring='accuracy')
	classes = pd.read_csv('../../secret/data/raw/icd10.csv', index_col=0).index.values
	for name in train:
		ssc = joblib.load('../../secret/data/vec/'+name+'_standardscaler.save')
		for df in  pd.read_csv('../../secret/data/trainingset/vec/'+name+'.csv', chunksize=chunk, index_col=0):
			df.drop(['txn'], axis=1, inplace=True)
			X_train, X_validation, Y_train, Y_validation = get_dataset(df, None)
			X_train = ssc.transform(X_train)
			#X_validation = ssc.transform(X_validation)
			models = train_model(models, X_train, Y_train, classes)
			n = n + chunk
			print('Train models '+name+' '+str(n))
			#break

	for i in range(len(models)):
		save_model(models[i],modelname+'_'+model_names[i])
		#test(models, X_validation, Y_validation)
		#inc.partial_fit(X_train, Y_train, classes=classes)
		#print(inc.predict(X_validation)[:len(X_validation)])
		#print('Score: ',inc.score(X_validation, Y_validation))

def eval_model(name):
	chunk = 10000
	ssc = joblib.load('../../secret/data/vec/'+name+'_standardscaler.save')
	#model_names = ['passive-aggrassive-classifier','sgd-classifier','perceptron','sgd-regressor','passive-aggrassive-regressor']
	model_names = ['mlpclassifier']
	if not os.path.exists('../../secret/data/model_prediction/'):
		os.makedirs('../../secret/data/model_prediction/')
	for df in pd.read_csv('../../secret/data/testset/vec/'+name+'.csv', chunksize=chunk, index_col=0):
		dftest = df.copy()
		dftest.drop(['txn'], axis=1, inplace=True)
		X_train, X_validation, Y_train, Y_validation = get_testset(dftest)
		X_train = ssc.transform(X_train)
		#print(len(X_train))
		for modelname in model_names:
			loaded_model = pickle.load(open('../../secret/data/model/'+name+'_'+modelname+'.pkl', 'rb'))
			#result = loaded_model.score(X_train, Y_train)
			#print(result)
			df[modelname] = loaded_model.predict(X_train)[:len(X_train)]
		save_file(df,'../../secret/data/model_prediction/'+name+'.csv')
		print('Predict '+name)

'''
def creme_model(name):
	#Need python >= 3.6
	ssc = joblib.load('../../secret/data/vec/'+name+'_standardscaler.save')
	chunk = 10000
	optimizer = optim.VanillaSGD(lr=0.01)
	#model = linear_model.LinearRegression(optimizer)
	model = naive_bayes.GaussianNB()
	#model = tree.MondrianTreeClassifier(lifetime=1, max_depth=100, min_samples_split=1, random_state=16)
	#model = tree.MondrianTreeRegressor(lifetime=1, max_depth=100, min_samples_split=1, random_state=16)


	y_true = []
	y_pred = []
	metric = metrics.Accuracy()

	for df in  pd.read_csv('../../secret/data/vec/'+name+'.csv', chunksize=chunk, index_col=0):
		df.drop(['txn'], axis=1, inplace=True)
		X_train, X_validation, Y_train, Y_validation = get_dataset(df, 0.1)
		X_train = ssc.transform(X_train)
		X_validation = ssc.transform(X_validation)
		for i in range(len(X_train.tolist())):
			# Fit the linear regression
			X = dict(zip(df.columns,X_train[i])) 
			model.fit_one(X, Y_train[i])
		for i in range(len(X_validation.tolist())):
			X = dict(zip(df.columns,X_validation[i])) 
			yi_pred = model.predict_one(X)
			# Store the truth and the prediction
			y_true.append(Y_validation[i])
			y_pred.append(round(yi_pred[True].item(0)))
		acc = accuracy_score(y_true, y_pred)
		print(acc)
'''
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
		X_train, X_validation, Y_train, Y_validation = get_dataset(df, None)
		ssc.partial_fit(X_train)
		mmsc.partial_fit(X_train)
	print('fit scaler '+filename)

	joblib.dump(mmsc, path+filename+'_minmaxscaler.save') 
	joblib.dump(ssc, path+filename+'_standardscaler.save') 

def predict(testset,testvalue,ssc,regressor):
	pre = regressor.predict(testset)
	print(pre)
	#pre = ssc.inverse_transform(pre)
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
	pkl_filename = '../../secret/data/model/'+filename+".pkl"  
	with open(pkl_filename, 'wb') as file:  
		 pickle.dump(model, file)
	print("save model")

def save_lstm_model(model,filename):
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

def lstm_model(name,f):
	c = XGBClassifier(max_depth=100)
	chunk = 10000
	r = 1
	#f = 12
	#f = 7
	#f = 2
	testset = None
	testvalue = None
	
	regressor = Sequential()

	file = Path('../../secret/data/model/'+name+'_lstm.h5')
	total_history = None
	if file.is_file():
		regressor = load_model(name)
	else:

		regressor.add(LSTM(512, return_sequences=True,
		            input_shape=(f, 1)))  # returns a sequence of vectors of dimension 32
		regressor.add(LSTM(512, return_sequences=True))  # returns a sequence of vectors of dimension 32
		regressor.add(LSTM(512))  # return a single vector of dimension 32
		regressor.add(Dense(1, activation='softmax'))

	regressor.compile(loss='mean_squared_error',
				        optimizer='adam',
				        metrics=['accuracy'])

	ssc = joblib.load('../../secret/data/vec/'+name+'_standardscaler.save') 
	#ssc = joblib.load('../../secret/data/vec/'+name+'_minmaxscaler.save') 

	for df in  pd.read_csv('../../secret/data/trainingset/vec/'+name+'.csv', chunksize=chunk, index_col=0):
		df.drop(['txn'], axis=1, inplace=True)
		X_train, X_validation, Y_train, Y_validation = get_dataset(df, None)
		X_train = ssc.fit_transform(X_train)
		X_train = X_train.reshape(len(X_train),len(df.columns)-1,1)

		print('Chunk '+str(r))
		history = regressor.fit(X_train, Y_train, epochs = 10, batch_size = 32, validation_split=0.1)
		#regressor.reset_states()
		r = r+1
		#if r == 10:
		#	break
		#predict(testset,testvalue,ssc,regressor)
		save_file(pd.DataFrame([[history.history['loss'],history.history['val_loss']]], columns=['loss','val_loss']),
					'../../secret/data/model/'+name+'_history.csv')


		save_lstm_model(regressor,name+'_lstm')
		if Path('../../secret/data/model/'+name+'_history.csv').is_file():
			total_history = pd.read_csv('../../secret/data/model/'+name+'_history.csv', index_col=0)
			history_loss(total_history['loss'].values.tolist(), total_history['val_loss'].values.tolist())
		#break

def evaluate_lstm_model(name):
	file = Path('../../secret/data/model/'+name+'.h5')
	regressor = load_model(name)
	regressor.compile(loss='mean_squared_error',
				        optimizer='adam',
				        metrics=['accuracy'])
	ssc = joblib.load('../../secret/data/vec/'+name+'_standardscaler.save') 
	chunk = 10000

	for df in  pd.read_csv('../../secret/data/vec/'+name+'.csv', chunksize=chunk, index_col=0):
		df.drop(['txn'], axis=1, inplace=True)
		X_train, X_validation, Y_train, Y_validation = get_dataset(df, 0.1)
		X_validation = ssc.fit_transform(X_validation)
		X_validation = X_validation.reshape(len(X_validation),len(df.columns)-1,1)
		predict(X_validation,Y_validation,ssc,regressor)


def kmean(train,modelname):
	chunk = 30000
	'''
	reg
	100: err 1695
	200: err 1141
	300: err 878
	400: err 935
	500: err 887
	600: err 660
	700: err 602
	800: err 477
	900: err 637
	1000:err 585

	dru
	100:   err 822
       	1000:  err 163
	5000:  err 50
	10000: err 
	15000: err 
	20000: err 
	25000: err 
	30000: err 
	'''
	#n = [100,1000,5000,10000,15000,20000,25000,30000]
	n = [20000]
	for i in n:
		print('Number of Cluster :'+str(i))
		kmeans = MiniBatchKMeans(n_clusters=i, random_state=0, batch_size=6)
		for name in train:
			ssc = joblib.load('../../secret/data/vec/'+name+'_standardscaler.save')

			for df in  pd.read_csv('../../secret/data/trainingset/vec/'+name+'.csv', chunksize=chunk, index_col=0):
				df.drop(['txn'], axis=1, inplace=True)
				X_train, X_validation, Y_train, Y_validation = get_dataset(df, None)
				X_train = ssc.transform(X_train)
				kmeans = kmeans.partial_fit(X_train)
				print('train')
				#break
		save_model(kmeans,modelname+'_kmean_'+str(i))
		print(kmeans.inertia_)
def predict_kmean(name,modelname):
	chunk = 10000
	ssc = joblib.load('../../secret/data/vec/'+name+'_standardscaler.save')
	if not os.path.exists('../../secret/data/model_prediction/'):
		os.makedirs('../../secret/data/model_prediction/')
	#n = [100,1000,5000,10000,15000,20000,25000,30000]
	n = [100,1000,10000]
	for df in  pd.read_csv('../../secret/data/testset/vec/'+name+'.csv', chunksize=chunk, index_col=0):
		dftest = df.copy()
		dftest.drop(['txn'], axis=1, inplace=True)
		X_train, X_validation, Y_train, Y_validation = get_dataset(dftest, None)
		X_train = ssc.transform(X_train)
		for i in n:
			kmeans = pickle.load(open('../../secret/data/model/'+modelname+'_kmean_'+str(i)+'.pkl', 'rb'))
			df['kmean_'+str(i)] = kmeans.predict(X_train)[:len(X_train)]
		save_file(df,'../../secret/data/model_prediction/'+name+'_kmean.csv')
		print('save result')

def top(x):
	return x.value_counts().head(10)

def topsum(x):
	return x.sum().head(5)

def get_neighbour(train,modelname):
	chunk = 100000
	results = []
	model = pickle.load(open('../../secret/data/model/'+modelname+'.pkl', 'rb'))
	for name in train:
		#ssc = joblib.load('../../secret/data/vec/'+name+'_standardscaler.save')
		for df in  pd.read_csv('../../secret/data/trainingset/vec/'+name+'.csv', chunksize=chunk, index_col=0):
			df.drop(['txn'], axis=1, inplace=True)
			X_train, X_validation, Y_train, Y_validation = get_testset(df)
			#X_train = ssc.transform(X_train)
			df['cluster'] = model.predict(X_train)[:len(X_train)]
			result = df['icd10'].groupby(df['cluster']).apply(top).to_frame()
			result = result.rename(columns={'icd10':'icd10_count'})
			result.reset_index(inplace=True)
			result = result.rename(columns={'level_1':'icd10'})
			results.append(result)
			#result['kmean_'+str(n)] = result['kmean_'+str(n)].apply(top)
			print('append neighbour')
			#if len(results) == 2:
			#	break
	total = pd.concat(results)
	total = total.groupby(['cluster','icd10']).sum()
	total.reset_index(inplace=True)
	total = total.sort_values(by=['cluster','icd10_count'], ascending=[True,False])
	total = total.groupby(['cluster']).head(10)
	total.to_csv('../../secret/data/model_prediction/'+modelname+'_neighbour.csv')

def get_weight(modelname):
	df = pd.read_csv('../../secret/data/model_prediction/'+modelname+'_neighbour.csv', index_col=0)
	df['total_count'] = df.groupby('icd10')['icd10_count'].transform('sum')
	df['cluster_count'] = df.groupby('cluster')['icd10_count'].transform('sum')
	df['weight'] = df['icd10_count']/(df['total_count']*df['cluster_count'])
	df = df.sort_values(by=['cluster','weight'], ascending=[True,False])
	df.to_csv('../../secret/data/model_prediction/'+modelname+'_neighbour.csv')
	print('save weight to '+modelname)

def birch_predict(filenames):
	mypath = '../../secret/data/'
	mypath = '/media/bon/My Passport/data/'
	chunk = 1000
	for name in filenames:
		id = []
		for df in pd.read_csv(mypath+'testset/vec/'+name+'.csv', chunksize=100000, index_col=0):
			id = id + df.index.values.tolist()

		for i in range(0, len(id), chunk):
			index = id[i:i + chunk]
			result = []
			for df in pd.read_csv(mypath+'result/'+name+'.csv', chunksize=chunk, index_col=0):
				df = df[df['index'].isin(index)]
				result.append(df)
			total = pd.concat(result)
			total = total[['index','predicted_icd10','weight']]
			total = total.groupby(['index', 'predicted_icd10'])['weight'].agg('sum').reset_index()
			total = total.sort_values(by=['index','weight'], ascending=[True,False])
			save_file(total,mypath+'result/'+name+'_prediction.csv')
			print('append prediction')
	print('complete')

def distance(x):
	return x
def birch_finetune(train):
	mypath = '../../secret/data/'
	mypath = '/media/bon/My Passport/data/'
	chunk = 10000
	samples = []
	for name in train:
		for df in  pd.read_csv(mypath+'trainingset/vec/'+name+'.csv', chunksize=chunk, index_col=0):
			df.drop(['txn'], axis=1, inplace=True)
			samples.append(df.sample(frac=1.0))
			break
		break

	df1 = pd.concat(samples)
	X_train, X_validation, Y_train, Y_validation = get_testset(df)
	
	for i in range(1,1000,1):
		i = i*0.01
		b = Birch(n_clusters=None,threshold=i)
		b = b.fit(X_train)
		df = df1.copy()
		df['cluster'] = b.predict(X_train)[:len(X_train)]
		t = b.transform(X_train)[:len(X_train)]
		df['distances'] = t.tolist()
		df['distance'] = df.apply(lambda row: row['distances'][row['cluster']], axis=1)
		df['distance'] = df['distance']*df['distance']
		df['square'] = df.groupby('cluster')['distance'].transform('sum')
		#df['distance'] = df[['index','cluster']].apply(lambda x: t[x[0]][x[1]])
		#df['center'] = df['cluster'].apply(lambda x: b.subcluster_centers_[x])
		#df['variance'] = df.groupby('cluster')['drug'].transform('var')
		df = df[['cluster','square']]
		df = df.drop_duplicates()
		df = df.fillna(0)
		s = df['square'].sum()
		print(str(i)+','+str(len(b.subcluster_centers_))+','+str(s))
		#break

def birch_train(train,modelname,n,threshold):
	chunk = 10000
	for t in threshold:
		b = Birch(n_clusters=n,threshold=t)
		for name in train:
			#ssc = jl.load('../../secret/data/vec/'+name+'_standardscaler.save')

			for df in  pd.read_csv('../../secret/data/trainingset/vec/'+name+'.csv', chunksize=chunk, index_col=0):
				df.drop(['txn'], axis=1, inplace=True)
				X_train, X_validation, Y_train, Y_validation = get_dataset(df, None)
				#X_train = ssc.transform(X_train)
				b = b.partial_fit(X_train)
				print('Number of cluster: '+str(len(b.subcluster_centers_)))
				#break
		save_model(b,modelname+'_'+str(t))
		print('save birch model')
	print('complete')

def birch_test(train,modelname):
	mypath = '../../secret/data/'
	mypath = '/media/bon/My Passport/data/'
	chunk = 10000
	model = pickle.load(open(mypath+'model/'+modelname+'.pkl', 'rb'))
	neighbour = pd.read_csv(mypath+'model_prediction/'+modelname+'_neighbour.csv', index_col=0)
	neighbour = neighbour.rename(columns={'icd10':'predicted_icd10'})
	for name in train:
		#remove_file('../../secret/data/result/'+name+'.csv')
		for df in pd.read_csv(mypath+'testset/vec/'+name+'.csv', chunksize=chunk, index_col=0):
			df.drop(['txn'], axis=1, inplace=True)
			index = df.index
			X_train, X_validation, Y_train, Y_validation = get_testset(df)
			#X_train = ssc.transform(X_train)
			df.insert(0,'index',index)
			df['model_name'] = modelname
			df['cluster'] = model.predict(X_train)[:len(X_train)]
			result = pd.merge(df,neighbour, how='left', on='cluster')
			#print(result)
			save_file(result,mypath+'result/'+name+'.csv')
			print('append result')
			#print(df)
	print('complete')

def train_had():
	p = '/media/bon/My Passport/data/'
	icd10 =  pd.read_csv('../../secret/data/raw/icd10.csv', index_col=0)
	icd10_map = dict(zip(icd10['code'],icd10.index))
	had = pd.read_csv(p+'had.csv', index_col=0)
	had = had['drug'].values.tolist()
	chunk = 10000
	n = 0
	m = Incremental(MLPClassifier())
	for name in ['dru','idru']:
		for df in  pd.read_csv(p+'trainingset/raw/'+name+'.csv', chunksize=chunk, index_col=0, low_memory=False):	
			df = df[['icd10','drug']]
			df = pd.concat([df.drop('icd10', axis=1), df['icd10'].map(icd10_map)], axis=1)
			df['had'] = np.where(df['drug'].isin(had), 1, 0)
			df = df[['icd10','had']]
			df1 = df[df['had'] == 1]
			df1 = df1.drop_duplicates()
			df2 = df[~df['icd10'].isin(df1['icd10'].values.tolist())]
			df2 = df2.drop_duplicates()
			df = pd.concat([df1,df2])
			X_train, X_validation, Y_train, Y_validation = get_dataset(df, None)
			m.partial_fit(X_train, Y_train, classes=[0,1])
			n = n + chunk
			print('fit '+str(n))
			print('Loss :'+str(m.loss_))

	save_model(m,'had_mlpclassifier')

def eval_had(name):
	chunk = 10000
	loaded_model = pickle.load(open('../../secret/data/model/had_mlpclassifier.pkl', 'rb'))
	for df in pd.read_csv('../../secret/data/testset/vec/'+name+'.csv', chunksize=chunk, index_col=0):
		dftest = df.copy()
		dftest.drop(['txn'], axis=1, inplace=True)
		dftest = dftest[['icd10','drug']]
		X_train, X_validation, Y_train, Y_validation = get_testset(dftest)
		p = loaded_model.predict_proba(X_train)
		dfp = pd.DataFrame(data=p,columns=['no_had','yes_had'])
		df['no_had'] = dfp['no_had'].values.tolist()
		df['yes_had'] = dfp['yes_had'].values.tolist()
		#print(df)
		#df['had'] = loaded_model.predict(X_train)[:len(X_train)]
		save_file(df,'/media/bon/My Passport/data/model_prediction/'+name+'_had.csv')
		print('Predict '+name)






