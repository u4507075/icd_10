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

def convert(x):
    try:
        return x.encode('latin-1','replace').decode('tis-620','replace')
    except AttributeError:
        return x

def remove_space(x):
	try:
		return x.replace(' ','')
	except AttributeError:
		return x

def decode(df):
	for c in df.columns:
		if df[c].dtype == 'object':
			df[c] = df[c].apply(convert)
	return df
def getquery(n,f):
	# Open and read the file as a single buffer
	fd = open('../../secret/query.sql', 'r')
	sql = fd.read()
	fd.close()
	sql = sql.replace('%f',str(f))
	sql = sql.replace('%n',str(n))
	return sql
def getdata(config):

	db_connection = sql.connect(	host=config.DATABASE_CONFIG['host'], 
											database=config.DATABASE_CONFIG['dbname'], 
											user=config.DATABASE_CONFIG['user'], 
											password=config.DATABASE_CONFIG['password'], 
											port=config.DATABASE_CONFIG['port'])
	n = 1000000
	offset = 0	
	while True:
		df = pd.read_sql(getquery(n,offset), con=db_connection)
		print(len(df))
		if len(df) == 0:
			break
		df = decode(df)
		p = '../../secret/data/data.csv'
		file = Path(p)
		if file.is_file():
			with open(p, 'a') as f:
				df.to_csv(f, header=False)
		else:
			df.to_csv(p)
		offset = offset + n
		print('Save data chunk: '+str(offset))

def cleandata():
	p = '../../secret/data/data.csv'
	p2 = '../../secret/data/data_clean.csv'
	for df in  pd.read_csv(p, chunksize=1000000):
		df['drug'] = df['drug'].apply(remove_space)
		file = Path(p2)
		if file.is_file():
			with open(p2, 'a') as f:
				df.to_csv(f, header=False)
		else:
			df.to_csv(p2)
		print('Append clean data')

def readdata():
	p = '../../secret/data/data_clean.csv'
	value = []
	for df in  pd.read_csv(p, chunksize=1000000):
		df = df[df['drug'].notnull()]
		v = df['drug'].unique().tolist()
		value = value + v
		value = list(set(value))       
		value.sort()
		print(len(value))
	d = { i : value[i] for i in range(0, len(value) ) }
	df = pd.DataFrame.from_dict({'drug':value, 'code': list(range(len(value)))})
	df.to_csv('../../secret/data/drug_code.csv')

def onehotdrug():
	p = '../../secret/data/data_clean.csv'
	p2 = '../../secret/data/data_map.csv'
	p3 = '../../secret/data/drug_onehot.csv'
	drug_list = pd.read_csv('../../secret/data/drug_code.csv')['drug'].values.tolist()
	#drug_list = ['TXN']+drug_list+['DX1']
	for df in  pd.read_csv(p, chunksize=100000):
		df = df[['TXN','drug','DX1']]
		df2 = pd.get_dummies(df['drug'])
		df2['TXN'] = df['TXN'].copy() 
		df2['DX1'] = df['DX1'].copy()
		df3 = df2.groupby(['TXN','DX1']).agg('sum')
		result = df3.reindex(columns=drug_list)
		result.fillna(0, inplace=True)
		file = Path(p3)
		if file.is_file():
			with open(p3, 'a') as f:
				result.to_csv(f, header=False)
		else:
			result.to_csv(p3)
		print('Append clean data')

def mapdata():
	p = '../../secret/data/data_clean.csv'
	p2 = '../../secret/data/data_map.csv'
	d = pd.read_csv('../../secret/data/drug_code.csv')
	l = dict(zip(d.drug,d.code))
	for df in  pd.read_csv(p, chunksize=1000000):
		df['drug_code'] = df['drug'].map(l)
		file = Path(p2)
		if file.is_file():
			with open(p2, 'a') as f:
				df.to_csv(f, header=False)
		else:
			df.to_csv(p2)
		print('Append mapped data')

def get_dataset(trainingset, validation_size):
	n = len(trainingset.columns)-1
	array = trainingset.values
	X = array[:,0:n]
	Y = array[:,n]
	seed = 7
	X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
	X_train, Y_train = shuffle(X_train, Y_train, random_state=seed)
	return X_train, X_validation, Y_train, Y_validation

def train_model():
	p = '../../secret/data/drug_onehot.csv'
	for df in  pd.read_csv(p, chunksize=1000):
		df = df[df.columns.tolist().remove(['TXN','DX1'])+['DX1']]
		X_train, X_validation, Y_train, Y_validation = get_dataset(df, 0.2)
		c = SVC()
		c.fit(X_train, Y_train)
		p = c.predict(X_validation)
		cf = confusion_matrix(Y_validation, p)
		print(cf)
		cr = classification_report(Y_validation, p)
		print(cr)
		break





















