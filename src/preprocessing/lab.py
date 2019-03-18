import mysql.connector as sql
import pandas as pd
from pathlib import Path
import numpy as np

def convert(x):
    try:
        return x.encode('latin-1','replace').decode('tis-620','replace')
    except AttributeError:
        return x


def decode(df):
	for c in df.columns:
		if df[c].dtype == 'object':
			df[c] = df[c].apply(convert)
	return df

def getquery(file,n,f):
	# Open and read the file as a single buffer
	fd = open('../../secret/'+file+'.sql', 'r')
	sql = fd.read()
	fd.close()
	sql = sql.replace('%f',str(f))
	sql = sql.replace('%n',str(n))
	return sql

def get_lab_data(config):

	db_connection = sql.connect(	host=config.DATABASE_CONFIG['host'], 
											database=config.DATABASE_CONFIG['dbname'], 
											user=config.DATABASE_CONFIG['user'], 
											password=config.DATABASE_CONFIG['password'], 
											port=config.DATABASE_CONFIG['port'])
	q = 	'''
			(SELECT CODE,COUNT(TXN) AS n FROM icd10.lab GROUP BY CODE)
			UNION
			(SELECT CODE,COUNT(TXN) AS n FROM icd10.ilab GROUP BY CODE)
			;
			'''	
	df = pd.read_sql(q)
	df = df[df['n'] >= 500]
	for index,row in df.iterrows():	
		print(row)
	'''
	n = 1000000
	offset = 0
	while True:
		df = pd.read_sql(getquery(query_file,n,offset), con=db_connection)
		print(len(df))
		if len(df) == 0:
			break
		df = decode(df)
		p = '../../secret/data/'+featuree+'.csv'
		file = Path(p)
		if file.is_file():
			with open(p, 'a') as f:
				df.to_csv(f, header=False)
		else:
			df.to_csv(p)
		offset = offset + n
		print('Save data chunk: '+str(offset))
	'''


