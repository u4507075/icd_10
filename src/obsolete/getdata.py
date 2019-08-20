import mysql.connector as sql
import pandas as pd
from pathlib import Path
import numpy as np

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
	sql = 	'''
		SELECT 
    		dx.TXN AS txn,
    		dru.CODE AS drug,
			dru.NAME AS drug_name,
    		dx.icd10
 		
		FROM icd10.odx dx
		INNER JOIN icd10.dru dru
		ON dx.TXN = dru.TXN
		WHERE dru.CODE IS NOT NULL AND dx.icd10 IS NOT NULL
		LIMIT %n OFFSET %f;
		'''
	sql = sql.replace('%f',str(f))
	sql = sql.replace('%n',str(n))
	return sql

def getiquery(n,f):
	sql = 	'''
		SELECT 
    		dx.TXN AS txn,
    		dru.CODE AS drug,
			dru.NAME AS drug_name,
    		dx.icd10
 		
		FROM icd10.idx dx
		INNER JOIN icd10.idru dru
		ON dx.TXN = dru.TXN
		WHERE dru.CODE IS NOT NULL AND dx.icd10 IS NOT NULL
		LIMIT %n OFFSET %f;
		'''
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
		df['drug'] = df['drug'].apply(remove_space)
		p = '../../secret/data/drug/dru.csv'
		file = Path(p)
		if file.is_file():
			with open(p, 'a') as f:
				df.to_csv(f, header=False)
		else:
			df.to_csv(p)
		offset = offset + n
		print('Save data chunk: '+str(offset))

def getidata(config):

	db_connection = sql.connect(	host=config.DATABASE_CONFIG['host'], 
											database=config.DATABASE_CONFIG['dbname'], 
											user=config.DATABASE_CONFIG['user'], 
											password=config.DATABASE_CONFIG['password'], 
											port=config.DATABASE_CONFIG['port'])
	n = 1000000
	offset = 0
	while True:
		df = pd.read_sql(getiquery(n,offset), con=db_connection)
		print(len(df))
		if len(df) == 0:
			break
		df = decode(df)
		df['drug'] = df['drug'].apply(remove_space)
		p = '../../secret/data/drug/idru.csv'
		file = Path(p)
		if file.is_file():
			with open(p, 'a') as f:
				df.to_csv(f, header=False)
		else:
			df.to_csv(p)
		offset = offset + n
		print('Save data chunk: '+str(offset))



