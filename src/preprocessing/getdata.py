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

def getquery(t,n,f):
	d = 'dru'
	if t == 'idx':
		d = 'idru'
	sql = 	'''
		SELECT 
    		dx.TXN,
    		dru.CODE AS drug,
    		dx.icd10
 		
		FROM icd10.%t dx
		INNER JOIN icd10.%d dru
		ON dx.TXN = dru.TXN
		WHERE dru.CODE IS NOT NULL AND dx.icd10 IS NOT NULL
		LIMIT %n OFFSET %f;
		'''
	sql = sql.replace('%t',str(t))
	sql = sql.replace('%f',str(f))
	sql = sql.replace('%n',str(n))
	sql = sql.replace('%d',str(d))
	return sql

def getdata(config,t,feature):

	db_connection = sql.connect(	host=config.DATABASE_CONFIG['host'], 
											database=config.DATABASE_CONFIG['dbname'], 
											user=config.DATABASE_CONFIG['user'], 
											password=config.DATABASE_CONFIG['password'], 
											port=config.DATABASE_CONFIG['port'])
	n = 1000000
	offset = 0
	while True:
		df = pd.read_sql(getquery(t,n,offset), con=db_connection)
		print(len(df))
		if len(df) == 0:
			break
		df = decode(df)
		p = '../../secret/data/'+feature+'/'+feature+'.csv'
		file = Path(p)
		if file.is_file():
			with open(p, 'a') as f:
				df.to_csv(f, header=False)
		else:
			df.to_csv(p)
		offset = offset + n
		print('Save data chunk: '+str(offset))

def remove_space_data(feature):
	p = '../../secret/data/'+feature+'.csv'
	p2 = '../../secret/data/'+feature+'_clean.csv'
	for df in  pd.read_csv(p, chunksize=1000000):
		for i in feature:
			#df['drug'] = df['drug'].apply(remove_space)
			df[i] = df[i].apply(remove_space)
		file = Path(p2)
		if file.is_file():
			with open(p2, 'a') as f:
				df.to_csv(f, header=False)
		else:
			df.to_csv(p2)
		print('Append clean data')


