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

def getquery(file,n,f):
	# Open and read the file as a single buffer
	fd = open('../../secret/'+file+'.sql', 'r')
	sql = fd.read()
	fd.close()
	sql = sql.replace('%f',str(f))
	sql = sql.replace('%n',str(n))
	return sql

def getdata(config,query_file,feature):

	db_connection = sql.connect(	host=config.DATABASE_CONFIG['host'], 
											database=config.DATABASE_CONFIG['dbname'], 
											user=config.DATABASE_CONFIG['user'], 
											password=config.DATABASE_CONFIG['password'], 
											port=config.DATABASE_CONFIG['port'])
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


