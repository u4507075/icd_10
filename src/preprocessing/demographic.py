import mysql.connector as sql
import pandas as pd
from pathlib import Path
import numpy as np
import os
import re
import math

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

def getquery(t1,n,f):
	sql = 	'''
				SELECT dx.TXN,sex,YEAR(CURDATE()) - YEAR(BORN) AS age
						,wt,pulse,resp,temp,bp,blood,rh
						,room,room_dc,dx.icd10

				FROM icd10.%t1 dx
				INNER JOIN icd10.adm adm
				on dx.TXN = adm.TXN
				LIMIT %n OFFSET %f;
			'''
	sql = sql.replace('%t1',str(t1))	
	sql = sql.replace('%f',str(f))
	sql = sql.replace('%n',str(n))
	return sql

def save_data(db_connection,t1):
	n = 1000000
	offset = 0
	
	while True:
		df = pd.read_sql(getquery(t1,n,offset), con=db_connection)
		if len(df) == 0:
			break
		df = decode(df)
		p = '../../secret/data/demographic/demographic.csv'
		file = Path(p)
		if file.is_file():
			with open(p, 'a') as f:
				df.to_csv(f, header=False)
		else:
			df.to_csv(p)
		offset = offset + n
		print('Save chunk '+str(offset))

def save_demographic_data(config):
	db_connection = sql.connect(	host=config.DATABASE_CONFIG['host'], 
											database=config.DATABASE_CONFIG['dbname'], 
											user=config.DATABASE_CONFIG['user'], 
											password=config.DATABASE_CONFIG['password'], 
											port=config.DATABASE_CONFIG['port'])
	save_data(db_connection,'odx')
	save_data(db_connection,'idx')

def clean_sex(x):
	x = x.replace(' ','')
	if x == 'ช':
		return 'm'
	elif x == 'ญ':
		return 'f'
	else:
		return ''

def clean_age(x):
	if x > 150:
		return 0
	else:
		return x
def clean_roomname(x):
	x = x.lower()
	return re.sub(r'[\d-]+', '', x)
def clean_demographic_data():
	p = '../../secret/data/demographic/demographic.csv'
	for df in  pd.read_csv(p, chunksize=100):
		df = df[['TXN','sex','age','wt','pulse','resp','temp','bp','blood','rh','room','room_dc','icd10']]
		df = df.fillna(0)		
		df['sex'] = df['sex'].apply(clean_sex)		
		df['age'] = df['age'].apply(clean_age)
		df['room'] = df['room'].apply(clean_roomname)
		df['room_dc'] = df['room_dc'].apply(clean_roomname)
		print(df)
		break


		















