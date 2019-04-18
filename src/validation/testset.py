import pandas as pd
from pathlib import Path
import numpy as np
import ntpath
import os
import pickle
import re
from random import randint
import mysql.connector as sql

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

def getadm(t):

	sql = 	'''
		SELECT dx.TXN,adm.sex,YEAR(CURDATE()) - YEAR(adm.BORN) AS age
			,wt,pulse,resp,temp,bp,blood,rh
			,adm.room,adm.room_dc,dx.icd10

	FROM icd10.%t dx
	INNER JOIN icd10.adm adm
	on dx.TXN = adm.TXN
    INNER JOIN lis lis
	ON dx.TXN = lis.TXN
	WHERE YEAR(lis.DATE) > 2017 and MONTH(lis.DATE) > 4
		'''
	sql = sql.replace('%t',str(t))
	return sql
	

def get_validation_data(config):

	db_connection = sql.connect(	host=config.DATABASE_CONFIG['host'], 
											database=config.DATABASE_CONFIG['dbname'], 
											user=config.DATABASE_CONFIG['user'], 
											password=config.DATABASE_CONFIG['password'], 
											port=config.DATABASE_CONFIG['port'])

	df = pd.read_sql(getquery('idx'), con=db_connection)
	df = decode(df)
	print(df)






