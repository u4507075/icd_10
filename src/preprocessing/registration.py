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

def getquery(t1,t2,n,f):
	sql = 	'''
				SELECT dx.TXN,sex,YEAR(CURDATE()) - YEAR(BORN) AS age
						,wt,pulse,resp,temp,bp,blood,rh
						,room,dx.icd10

				FROM icd10.%t1 dx
				INNER JOIN icd10.%t2 adm
				on dx.TXN = adm.TXN
				LIMIT %n OFFSET %f;
			'''
	sql = sql.replace('%t1',str(t1))	
	sql = sql.replace('%t2',str(t2))	
	sql = sql.replace('%f',str(f))
	sql = sql.replace('%n',str(n))
	return sql

def save_data(db_connection,t1,t2):
	n = 1000000
	offset = 0
	
	while True:
		df = pd.read_sql(getquery(t1,t2,n,offset), con=db_connection)
		if len(df) == 0:
			break
		df = decode(df)
		p = '../../secret/data/registration/registration.csv'
		file = Path(p)
		if file.is_file():
			with open(p, 'a') as f:
				df.to_csv(f, header=False)
		else:
			df.to_csv(p)
		offset = offset + n
		print('Save chunk '+str(offset))

def save_registration_data(config):
	db_connection = sql.connect(	host=config.DATABASE_CONFIG['host'], 
											database=config.DATABASE_CONFIG['dbname'], 
											user=config.DATABASE_CONFIG['user'], 
											password=config.DATABASE_CONFIG['password'], 
											port=config.DATABASE_CONFIG['port'])
	dx = ['odx','idx']
	reg = ['reg','reg_2005','reg_2006','reg_2007','reg_2008']
	for t1 in dx:
		for t2 in reg:
			save_data(db_connection,t1,t2)

def clean_sex(x):
	x = str(x).replace(' ','')
	if x == 'ช':
		return 'm'
	elif x == 'ญ':
		return 'f'
	else:
		return 0

def clean_age(x):
	if x > 150:
		return 0
	else:
		return x
def clean_blood_group(x):
	x = str(x).lower()
	if x == 'o' or x == 'a' or x == 'b' or x == 'ab':
		return x
	else:
		return 0
def clean_rh(x):
	x = str(x).lower()
	if x == '+' or x == 'p':
		return 'p'
	elif x == '-' or x == 'n':
		return 'n'
	else:
		return 0
def clean_roomname(x):
	x = str(x).lower()
	x = x.replace('*','')
	x = x.replace(' ','')
	if x == '':
		return 0
	else:
		return re.sub(r'[\d-]+', '', x)
def save_clean_data(df):
	p = '../../secret/data/registration/registration_clean.csv'
	file = Path(p)
	if file.is_file():
		with open(p, 'a') as f:
			df.to_csv(f, header=False)
	else:
		df.to_csv(p)
def clean_registration_data():
	p = '../../secret/data/registration/registration.csv'
	for df in  pd.read_csv(p, chunksize=100000):
		df = df[['TXN','sex','age','wt','pulse','resp','temp','bp','blood','rh','room','icd10']]
			
		df['sex'] = df['sex'].apply(clean_sex)		
		df['age'] = df['age'].apply(clean_age)
		df['blood'] = df['blood'].apply(clean_blood_group)
		df['rh'] = df['rh'].apply(clean_rh)
		df['room'] = df['room'].apply(clean_roomname)
		df = df.fillna(0)	
		save_clean_data(df)
		print('Append clean data')

def onehot_registration_data():
	df = pd.read_csv('../../secret/data/registration/registration_clean.csv', index_col=0)
	d = df['bp'].str.split('/',expand=True)
	d.columns = ['sbp','dbp']
	for c in d.columns:
		d[c] = pd.to_numeric(d[c], errors='coerce')
	d.fillna(0, inplace=True)
	
	icd10 = df['icd10']
	df.drop(columns=['bp','icd10'], inplace=True)
	df['sbp'] = d['sbp']
	df['dbp'] = d['dbp']
	df2 = pd.get_dummies(data=df, columns=['sex','blood','rh','room'])

	for c in df2.columns:
		if '0' in c and c in df2.columns:
			df2.drop(columns=[c], inplace=True)
	df2['icd10'] = icd10 
	df2.to_csv('../../secret/data/registration/registration_onehot.csv')

		















