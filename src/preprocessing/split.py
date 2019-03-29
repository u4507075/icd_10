import mysql.connector as sql
import pandas as pd
from pathlib import Path
import numpy as np
import ntpath

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

def getquery(t):
	d = 'dru'
	if t == 'idx':
		d = 'idru'
	sql = 	'''
		SELECT DISTINCT dx.TXN FROM icd10.%t dx
		INNER JOIN lis lis
		ON dx.TXN = lis.TXN
		WHERE YEAR(lis.DATE) > 2017 and MONTH(lis.DATE) > 4;
		'''
	sql = sql.replace('%t',str(t))
	return sql

def get_txn_test_data(config):

	db_connection = sql.connect(	host=config.DATABASE_CONFIG['host'], 
											database=config.DATABASE_CONFIG['dbname'], 
											user=config.DATABASE_CONFIG['user'], 
											password=config.DATABASE_CONFIG['password'], 
											port=config.DATABASE_CONFIG['port'])
	table = ['odx','idx']
	for t in table:
		df = pd.read_sql(getquery(t), con=db_connection)
		print(len(df))
		if len(df) == 0:
			break
		df = decode(df)
		p = '../../secret/data/testset/txn_testset.csv'
		file = Path(p)
		if file.is_file():
			with open(p, 'a') as f:
				df.to_csv(f, header=False)
		else:
			df.to_csv(p)
		print('Save txn of testset')

def save_data(df,name):
	p = '../../secret/data/'+name
	file = Path(p)
	if file.is_file():
		with open(p, 'a') as f:
			df.to_csv(f, header=False)
	else:
		df.to_csv(p)

def split_set():
	paths = [	'../../secret/data/admit/admit_onehot.csv',
				'../../secret/data/drug/drug_numeric.csv',
				'../../secret/data/registration/registration_onehot.csv']
	txn_testset = pd.read_csv('../../secret/data/testset/txn_testset.csv',index_col=0)['TXN'].values.tolist()
	for p in paths:
		name = ntpath.basename(p)
		for df in  pd.read_csv(p, chunksize=100000, index_col=0):
			testset = df[df['TXN'].isin(txn_testset)]
			trainingset = df[~df['TXN'].isin(txn_testset)]
			if len(testset) > 0:
				save_data(testset, 'testset/'+name)
			if len(trainingset) > 0:
				save_data(trainingset, 'trainingset/'+name)
			print('Save '+name)















