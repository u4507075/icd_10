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

def getquery(t1,t2,code,n,f):
	sql = 	'''
				SELECT 
					 dx.TXN,
					 lab.LST AS name,
					 lab.REP as value,
					 dx.icd10
				 
				FROM icd10.%t1 dx
				INNER JOIN icd10.%t2 lab
				ON dx.TXN = lab.TXN
				WHERE lab.REP IS NOT NULL AND lab.REP != "" AND dx.icd10 IS NOT NULL
				AND lab.CODE = %code
				LIMIT %n OFFSET %f;
			'''
	sql = sql.replace('%t1',str(t1))	
	sql = sql.replace('%t2',str(t2))
	sql = sql.replace('%code',str(code))
	sql = sql.replace('%f',str(f))
	sql = sql.replace('%n',str(n))
	return sql

def save_data(t1,t2,code):
	n = 1000000
	offset = 0
	
	while True:
		df = pd.read_sql(getquery(query_file,n,offset), con=db_connection)
		print(len(df))
		if len(df) == 0:
			break
		df = decode(df)
		p = '../../secret/data/lab/'+code+'.csv'
		file = Path(p)
		if file.is_file():
			with open(p, 'a') as f:
				df.to_csv(f, header=False)
		else:
			df.to_csv(p)
		offset = offset + n
		print('Save '+code+' chunk: '+str(offset))

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
	df = pd.read_sql(q, con=db_connection)
	df = df[df['n'] >= 500]
	for index,row in df.iterrows():	
		save_data('idx','ilab',row['CODE'])
		save_data('odx','lab',row['CODE'])



