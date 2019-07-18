import mysql.connector as mysql
import pandas as pd
from pathlib import Path
import numpy as np
import os
import re

path = '../../secret/data/raw/'

icd_sql = '''
				(SELECT code,cdesc FROM icd10.icd10)
				UNION
				(SELECT code,cdesc FROM icd10.icd10)
				LIMIT %n OFFSET %f;
			 '''
adm_sql = '''
				SELECT dx.TXN AS txn,sex,YEAR(adm.ADM) - YEAR(BORN) AS age
						,wt,pulse,resp,temp,bp,blood,rh
						,room,room_dc,dx.dxtype AS principal_dx,dx.icd10

				FROM icd10.idx dx
				INNER JOIN icd10.adm adm
				on dx.TXN = adm.TXN
				LIMIT %n OFFSET %f;
			'''
reg_sql = '''
				SELECT dx.TXN AS txn,sex,YEAR(adm.DATE) - YEAR(BORN) AS age
						,wt,pulse,resp,temp,bp,blood,rh
						,room,0 AS room_dc,dx.dxtype AS principal_dx,dx.icd10

				FROM icd10.odx dx
				INNER JOIN icd10.reg adm
				on dx.TXN = adm.TXN
				LIMIT %n OFFSET %f;
			'''
reg_2005_sql = '''
				SELECT dx.TXN AS txn,sex,YEAR(CURDATE()) - YEAR(BORN) AS age
						,wt,pulse,resp,temp,bp,blood,rh
						,room,0 AS room_dc,dx.dxtype AS principal_dx,dx.icd10

				FROM icd10.odx dx
				INNER JOIN icd10.reg_2005 adm
				on dx.TXN = adm.TXN
				LIMIT %n OFFSET %f;
			'''
reg_2006_sql = '''
				SELECT dx.TXN AS txn,sex,YEAR(CURDATE()) - YEAR(BORN) AS age
						,wt,pulse,resp,temp,bp,blood,rh
						,room,0 AS room_dc,dx.dxtype AS principal_dx,dx.icd10

				FROM icd10.odx dx
				INNER JOIN icd10.reg_2006 adm
				on dx.TXN = adm.TXN
				LIMIT %n OFFSET %f;
			'''
reg_2007_sql = '''
				SELECT dx.TXN AS txn,sex,YEAR(CURDATE()) - YEAR(BORN) AS age
						,wt,pulse,resp,temp,bp,blood,rh
						,room,0 AS room_dc,dx.dxtype AS principal_dx,dx.icd10

				FROM icd10.odx dx
				INNER JOIN icd10.reg_2007 adm
				on dx.TXN = adm.TXN
				LIMIT %n OFFSET %f;
			'''
reg_2008_sql = '''
				SELECT dx.TXN AS txn,sex,YEAR(CURDATE()) - YEAR(BORN) AS age
						,wt,pulse,resp,temp,bp,blood,rh
						,room,0 AS room_dc,dx.dxtype AS principal_dx,dx.icd10

				FROM icd10.odx dx
				INNER JOIN icd10.reg_2008 adm
				on dx.TXN = adm.TXN
				LIMIT %n OFFSET %f;
			'''
dru_sql = '''
				SELECT 
			 		dx.TXN AS txn,
			 		dru.CODE AS drug,
					dru.NAME AS drug_name,
					dx.dxtype AS principal_dx,
			 		dx.icd10
		 		/*CODER*/
				FROM icd10.odx dx
				INNER JOIN icd10.dru dru
				ON dx.TXN = dru.TXN
				WHERE dru.CODE IS NOT NULL AND dx.icd10 IS NOT NULL
				LIMIT %n OFFSET %f;
			'''
idru_sql = '''
				SELECT 
			 		dx.TXN AS txn,
			 		dru.CODE AS drug,
					dru.NAME AS drug_name,
					dx.dxtype AS principal_dx,
			 		dx.icd10
		 		/*CODER*/
				FROM icd10.idx dx
				INNER JOIN icd10.idru dru
				ON dx.TXN = dru.TXN
				WHERE dru.CODE IS NOT NULL AND dx.icd10 IS NOT NULL
				LIMIT %n OFFSET %f;
			'''
lab_sql = '''
				SELECT 
					 dx.TXN AS txn,
					 lab.NAME AS lab_name,
					 lab.LST AS name,
					 lab.REP as value,
					 dx.dxtype AS principal_dx,
					 dx.icd10
				FROM icd10.odx dx
				INNER JOIN icd10.lab lab
				ON dx.TXN = lab.TXN
				WHERE lab.REP IS NOT NULL AND lab.REP != "" AND dx.icd10 IS NOT NULL
				LIMIT %n OFFSET %f;
			'''
ilab_sql = '''
				SELECT 
					 dx.TXN AS txn,
					 lab.NAME AS lab_name,
					 lab.LST AS name,
					 lab.REP as value,
					 dx.dxtype AS principal_dx,
					 dx.icd10
				 
				FROM icd10.idx dx
				INNER JOIN icd10.ilab lab
				ON dx.TXN = lab.TXN
				WHERE lab.REP IS NOT NULL AND lab.REP != "" AND dx.icd10 IS NOT NULL
				LIMIT %n OFFSET %f;
			'''
lis_sql = '''
				SELECT 
					 dx.TXN AS txn,
					 lis.NAME AS lab_name,
					 lis.LST AS name,
					 lis.REP as value,
					 dx.dxtype AS principal_dx,
					 dx.icd10
				 
				FROM icd10.odx dx
				INNER JOIN icd10.lis lis
				ON dx.TXN = lis.TXN
				WHERE lis.REP IS NOT NULL AND lis.REP != "" AND dx.icd10 IS NOT NULL AND lis.USER = "OPD Lab"
				LIMIT %n OFFSET %f;
			'''
ilis_sql = '''
				SELECT 
					 dx.TXN AS txn,
					 lis.NAME AS lab_name,
					 lis.LST AS name,
					 lis.REP as value,
					 dx.dxtype AS principal_dx,
					 dx.icd10
				 
				FROM icd10.idx dx
				INNER JOIN icd10.lis lis
				ON dx.TXN = lis.TXN
				WHERE lis.REP IS NOT NULL AND lis.REP != "" AND dx.icd10 IS NOT NULL AND lis.USER = "Department"
				LIMIT %n OFFSET %f;
			'''

rad_sql = '''
				SELECT 
					 rad.TXN AS txn,
					 LOWER(rad.NAME) AS location,
					 LOWER(rad.LST) AS position,
					 rad.REP as report,
					 dx.dxtype AS principal_dx,
					 dx.icd10
				 
				FROM icd10.odx dx
				INNER JOIN icd10.rad rad
				ON dx.TXN = rad.TXN
				WHERE rad.REP IS NOT NULL AND rad.REP != "" AND dx.icd10 IS NOT NULL
				LIMIT %n OFFSET %f;
			'''
irad_sql = '''
				SELECT 
					 rad.TXN AS txn,
					 LOWER(rad.NAME) AS location,
					 LOWER(rad.LST) AS position,
					 rad.REP as report,
					 dx.dxtype AS principal_dx,
					 dx.icd10
				 
				FROM icd10.idx dx
				INNER JOIN icd10.irad rad
				ON dx.TXN = rad.TXN
				WHERE rad.REP IS NOT NULL AND rad.REP != "" AND dx.icd10 IS NOT NULL
				LIMIT %n OFFSET %f;
			'''
test_txn_sql = '''
			SELECT DISTINCT dx.TXN AS txn FROM icd10.odx dx
			INNER JOIN icd10.reg reg
			ON dx.TXN = reg.TXN
			LIMIT %n OFFSET %f;
			'''
itest_txn_sql = '''
			SELECT DISTINCT dx.TXN AS txn FROM icd10.idx dx
			INNER JOIN icd10.adm reg
			ON dx.TXN = reg.TXN
			LIMIT %n OFFSET %f;
			'''
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

def remove_junk(x):
	x = x.lower()
	x = re.sub('{.*}', '', x)
	x = x.replace('\\', ' \\')
	x = re.sub(r"\\.*? ",'',x)
	x = x.replace('\r\n', '')
	x = re.sub(r"[^a-z ]",'',x)
	x = re.sub(r" +",' ',x)
	return x

def decode(df):
	for c in df.columns:
		if df[c].dtype == 'object':
			df[c] = df[c].apply(convert)
	return df

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
def regex_filter(val):
	regex = re.compile('[A-Z]')
	if val:
		mo = re.search(regex,val)
		if mo:
			return True
		else:
			return False
	else:
		return False

def getquery(sql,n,f):

	sql = sql.replace('%f',str(f))
	sql = sql.replace('%n',str(n))
	return sql

def checkpath():
	if not os.path.exists(path):
			os.makedirs(path)

def get_connection(config):
	return mysql.connect(	host=config.DATABASE_CONFIG['host'], 
											database=config.DATABASE_CONFIG['dbname'], 
											user=config.DATABASE_CONFIG['user'], 
											password=config.DATABASE_CONFIG['password'], 
											port=config.DATABASE_CONFIG['port'])

def getdata(config, sql, filename):
	checkpath()
	db_connection = get_connection(config)
	n = 100000
	offset = 0
	while True:
		df = pd.read_sql(getquery(sql,n,offset), con=db_connection)

		if len(df) == 0:
			break
		df = decode(df)

		if 'sex' in df:
			df['sex'] = df['sex'].apply(clean_sex)	
		if 'age' in df:	
			df['age'] = df['age'].apply(clean_age)
		if 'blood' in df:
			df['blood'] = df['blood'].apply(clean_blood_group)
		if 'rh' in df:
			df['rh'] = df['rh'].apply(clean_rh)
		if 'bp' in df:
			d = df['bp'].str.split('/',expand=True)
			d.columns = ['sbp','dbp']
			for c in d.columns:
				d[c] = pd.to_numeric(d[c], errors='coerce')
			d.fillna(0, inplace=True)
	
			icd10 = df['icd10']
			df.drop(columns=['bp','icd10'], inplace=True)
			df['sbp'] = d['sbp']
			df['dbp'] = d['dbp']
			df['icd10'] = icd10

		if 'drug' in df:
			df['drug'] = df['drug'].apply(remove_space)

		if 'name' in df and 'value' in df:
			d1 = df['name'].str.split(';',expand=True)
			d1 = d1.merge(df, right_index = True, left_index = True)
			d1 = d1.melt(id_vars = ['txn','lab_name','value','icd10'], value_name = 'name')
			d1 = d1.sort_values(['txn', 'icd10', 'variable'], ascending=True)
			d1 = d1.drop('value', axis=1)
			d1 = d1[d1['variable'] != 'name']
			d2 = df['value'].str.split(';',expand=True)
			d2 = d2.merge(df, right_index = True, left_index = True)
			d2 = d2.melt(id_vars = ['txn','lab_name','name','icd10'], value_name = 'value')
			d2 = d2.sort_values(['txn', 'icd10', 'variable'], ascending=True)
			d2 = d2.drop('name', axis=1)
			d2 = d2[d2['variable'] != 'name']
			df = pd.merge(d1,d2,on=['txn','lab_name','icd10','variable'])
			df = df[['txn','lab_name','name','value','icd10']]

		if 'report' in df:
			df['report'] = df['report'].apply(remove_junk)

		if 'icd10' in df:
			df = df[df['icd10'].apply(regex_filter)]

		df = df.drop_duplicates()

		if filename == 'test' or filename == 'itest':
			df = df.sample(frac=0.1).reset_index(drop=True)
		p = path+filename+'.csv'
		file = Path(p)
		if file.is_file():
			with open(p, 'a') as f:
				df.to_csv(f, header=False)
		else:
			df.to_csv(p)
		offset = offset + n
		print('Save data chunk: '+str(offset))

def save_data(df,path,name):
	p = path+'/'+name
	if not os.path.exists(path):
			os.makedirs(path)
	file = Path(p)
	if file.is_file():
		with open(p, 'a') as f:
			df.to_csv(f, header=False)
	else:
		df.to_csv(p)

def get_icd10_data(config):
	getdata(config, icd_sql, 'icd10')

def get_adm_data(config):
	getdata(config, adm_sql, 'adm')

def get_reg_data(config):
	getdata(config, reg_sql, 'reg')
	getdata(config, reg_2005_sql, 'reg')
	getdata(config, reg_2006_sql, 'reg')
	getdata(config, reg_2007_sql, 'reg')
	getdata(config, reg_2008_sql, 'reg')

def get_drug_data(config):
	getdata(config, dru_sql, 'dru')
	getdata(config, idru_sql, 'idru')

def get_lab_data(config):
	getdata(config, lab_sql, 'lab')
	getdata(config, ilab_sql, 'ilab')

def get_rad_data(config):
	getdata(config, rad_sql, 'rad')
	getdata(config, irad_sql, 'irad')

def get_txn_test_data(config):
	getdata(config, test_txn_sql, 'test')
	getdata(config, itest_txn_sql, 'itest')

def split(filename,txn,folder):
	for df in  pd.read_csv('../../secret/data/'+folder+'/'+filename+'.csv', chunksize=100000, index_col=0):
		testset = df[df['txn'].isin(txn)]
		trainingset = df[~df['txn'].isin(txn)]
		if len(testset) > 0:
			save_data(testset, '../../secret/data/testset/'+folder, filename+'.csv')
		if len(trainingset) > 0:
			save_data(trainingset, '../../secret/data/trainingset/'+folder, filename+'.csv')
		print('Save '+filename+' in '+folder)

def split_data(folder):
	#folder = 'raw' or 'vec'
	opds = ['reg','lab','dru','rad']
	ipds = ['adm','ilab','idru','irad']
	test = pd.read_csv('../../secret/data/raw/test.csv',index_col=0)['txn'].values.tolist()
	itest = pd.read_csv('../../secret/data/raw/itest.csv',index_col=0)['txn'].values.tolist()
	for f in opds:
		split(f,test,folder)
	for f in ipds:
		split(f,itest,folder)
def backslash(x):
	if isinstance(x,str):
		x = re.sub(r"[^a-zA-Z0-9]",'',x)
	return x

def csv_to_sqldb(config,folder,filename):
	dtype = {'int64':'INT', 'float64': 'FLOAT', 'object':'TEXT'}
	connection = mysql.connect(     host=config.DATABASE_CONFIG['host'], 
                                                                                        database=config.DATABASE_CONFIG['dbname'], 
                                                                                        user=config.DATABASE_CONFIG['user'], 
                                                                                        password=config.DATABASE_CONFIG['password'], 
                                                                                        port=config.DATABASE_CONFIG['port'],
											use_pure=True)

	sql = 'DROP TABLE IF EXISTS %name;'.replace('%name',folder+'_'+filename)
	cursor = connection.cursor()
	cursor.execute(sql)
	cols = []
	types = []
	for df in  pd.read_csv('../../secret/data/'+folder+'/'+filename+'.csv', chunksize=100000, index_col=0):
		cols = df.columns.tolist()
		types = df.dtypes
		break
	sql_col = 'CREATE TABLE '+folder+'_'+filename+' ('
	sql_insert_1 = 'INSERT INTO '+folder+'_'+filename+'('
	sql_insert_2 = ' VALUES ('
	for i in range(len(cols)):
		t = 'TEXT'
		if str(types[i]) in dtype:
			t = dtype[str(types[i])]
		sql_col = sql_col + cols[i]+' '+t
		sql_insert_1 = sql_insert_1 + cols[i]
		sql_insert_2 = sql_insert_2 + '%s'
		if i < len(cols)-1:
			sql_col = sql_col + ','
			sql_insert_1 = sql_insert_1 + ','
			sql_insert_2 = sql_insert_2 + ','
	sql_col = sql_col + ');'
	sql_insert_1 = sql_insert_1 + ')'
	sql_insert_2 = sql_insert_2 + ');'
	sql_insert = sql_insert_1 + sql_insert_2
	cursor.execute(sql_col)
	cursor.close()
	cursor = connection.cursor(prepared=True)
	#print(sql_insert)
	for df in  pd.read_csv('../../secret/data/'+folder+'/'+filename+'.csv', chunksize=100000, index_col=0):
		df = df.fillna(0)
		for c in df:
			df[c] = df[c].apply(backslash)
		val = list(df.itertuples(index=False,name=None))
		#print(val)
		cursor.executemany(sql_insert, val)
		print('Append table '+folder+'_'+filename)
		connection.commit()
	cursor.close()
	connection.close()
	print('Insert data to table '+folder+'_'+filename)
















