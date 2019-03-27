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
				AND lab.CODE = "%code"
				LIMIT %n OFFSET %f;
			'''
	sql = sql.replace('%t1',str(t1))	
	sql = sql.replace('%t2',str(t2))
	sql = sql.replace('%code',str(code))
	sql = sql.replace('%f',str(f))
	sql = sql.replace('%n',str(n))
	return sql

def save_data(db_connection,t1,t2,code):
	n = 1000000
	offset = 0
	
	while True:
		df = pd.read_sql(getquery(t1,t2,code,n,offset), con=db_connection)
		print(len(df))
		if len(df) == 0:
			break
		df = decode(df)
		code_name = code.replace(' ','')
		p = '../../secret/data/lab/raw/'+code_name+'.csv'
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
		save_data(db_connection,'idx','ilab',row['CODE'])
		save_data(db_connection,'odx','lab',row['CODE'])

def split_lab_data():
	files = os.listdir('../../secret/data/lab/raw/')
	for lab in files:
		p = '../../secret/data/lab/raw/'+lab
		p2 = '../../secret/data/lab/split/'+lab
		for df in  pd.read_csv(p, chunksize=1000000):
			d = df['value'].str.split(';',expand=True)
			c = lab.replace('.csv','')
			d = d.add_prefix(c+'_')
			#for col in d.columns:
			#	d[col] = pd.to_numeric(d[col],errors='coerce')
			#d.fillna(0, inplace=True)
			d.insert(0,'TXN',df['TXN'])
			d['icd10'] = df['icd10'].copy()
			file = Path(p2)
			if file.is_file():
				with open(p2, 'a') as f:
					d.to_csv(f, header=False)
			else:
				d.to_csv(p2)
			print('Append data')
		print('Save '+lab)

def get_value(x):
	#try:
	if x == np.nan:
		return 0
	else:
		s = str(x).split(' ')
		if s[0] == np.nan or s[0] == 'nan' or s[0] == '':
			return 0
		else:
			s[0] = s[0].replace('<','')
			s[0] = s[0].replace('>','')
			s[0] = s[0].replace(',','')
			s[0] = s[0].replace('|','')
			s[0] = s[0].replace('%','')
			s[0] = s[0].replace('+','')
			s[0] = s[0].lower()
			#print(s[0])
			if '/' in s[0]:
				s[0] = split_num_from_text(s[0])
				
				if re.match('^[a-z]+', str(s[0])) is not None:
					
					return 0
				else:
					return s[0]
			elif s[0] == 'p' or s[0] == 'pos' or 'positive' in s[0]:
				return 1
			elif s[0] == 'n' or s[0] == 'neg' or  'negative' in s[0] or s[0] == 'not' or s[0] == 'no' or s[0] == 'notseen':
				return -1
			elif s[0].startswith('r='):
				return s[0].replace('r=','')
			elif str(s[0]) == '' or str(s[0]) == '%':
				return 0
			else:
				return s[0]
	#except AttributeError:
	#	print('MMM')
	#	return 0
def split_num_from_text(x):
	if x != 0:
		v = re.findall('\d+\.?\d+',str(x))
		if len(v) > 0:
			return v[0]
		else:
			return x
	return x

def save_file(df,p):
	file = Path(p)
	if file.is_file():
		with open(p, 'a') as f:
			df.to_csv(f, header=False)
	else:
		df.to_csv(p)
def clean_lab_data():

	#B05, B06, B07, B08 one hot 1 feature
	#B09, B13.1 one hot 2 feature
	
	files = ['L01','L1901','L090','L10044','L07','L10962','L1001',
				'L4301','L421','L1005','L1032','L1904','L091','L1081',
				'L1903','L422','L531','L61', 'L1022','L10041','L029',
				'L093','L10591','L1031','L107018','L073','L025','L107011',
				'L105621','L10502','L36','L0261','L0371','L0414','GMCL001',
				'L1911','L1906','L0421','L1907','L083','L071','L092',
				'L5712','L10561','L078','L0411','L551','L1905','L1902',
				'L037','L02','L1910','B06','L1052','L022','L1084','L1040',
				'L54','L024','L1914','L10501','L10221','L10042','B13.1',
				'L077','L1030','L105933','L106011','L2082','L027','L10961',
				'L0221','L074','L581','L58','L202','L105932','L072',
				'L1056221','L101763','L10981','B13','L84','L10573','L09011']
	for lab in files:
		file = Path('../../secret/data/lab/clean/'+lab+'.csv')
		if not file.is_file():
			p = '../../secret/data/lab/split/'+lab+'.csv'
			for df in  pd.read_csv(p, chunksize=100000):
				for col in df.columns:
					if col != 'TXN' and col != 'icd10':
						df[col] = df[col].apply(get_value)
						df[col] = pd.to_numeric(df[col],errors='ignore')
						if col.startswith('L1081'):
							df[col] = df[col].apply(split_num_from_text)
				#df = df.loc[:, (df != 0).any(axis=0)]
				print('Save clean data: '+str(lab))
				save_file(df,'../../secret/data/lab/clean/'+lab+'.csv')
		

def tonumeric_lab_data():
	files = os.listdir('../../secret/data/lab/clean/')
	for lab in files:
		p = '../../secret/data/lab/clean/'+lab
		for df in  pd.read_csv(p, chunksize=1000000):
			for c in df.columns:
				if c != 'TXN' and c != 'icd10':
					df[c] = df[c].apply(pd.to_numeric,errors='coerce').fillna(0)

			save_file(df,'../../secret/data/lab/numeric/'+lab)
			print('Saved '+lab)










