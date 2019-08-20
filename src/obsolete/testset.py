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

def getadm():

	sql = 	'''
		SELECT adm.TXN,adm.sex,YEAR(CURDATE()) - YEAR(adm.BORN) AS age
			,wt,pulse,resp,temp,bp,blood,rh
			,adm.room,adm.room_dc

	FROM  icd10.adm adm
    	INNER JOIN lis lis
	ON adm.TXN = lis.TXN
	WHERE YEAR(lis.DATE) > 2017 and MONTH(lis.DATE) > 4
		'''

	return sql

def getreg():

	sql = 	'''
				(SELECT adm.TXN,adm.sex,YEAR(CURDATE()) - YEAR(adm.BORN) AS age
					,wt,pulse,resp,temp,bp,blood,rh
					,adm.room

					FROM  icd10.reg adm
					 	INNER JOIN lis lis
					ON adm.TXN = lis.TXN
					WHERE YEAR(lis.DATE) > 2017 and MONTH(lis.DATE) > 4
				)
				UNION
				(SELECT adm.TXN,adm.sex,YEAR(CURDATE()) - YEAR(adm.BORN) AS age
							,wt,pulse,resp,temp,bp,blood,rh
							,adm.room

					FROM  icd10.reg_2005 adm
					 	INNER JOIN lis lis
					ON adm.TXN = lis.TXN
					WHERE YEAR(lis.DATE) > 2017 and MONTH(lis.DATE) > 4
				)
				UNION
				(SELECT adm.TXN,adm.sex,YEAR(CURDATE()) - YEAR(adm.BORN) AS age
							,wt,pulse,resp,temp,bp,blood,rh
							,adm.room

					FROM  icd10.reg_2006 adm
					 	INNER JOIN lis lis
					ON adm.TXN = lis.TXN
					WHERE YEAR(lis.DATE) > 2017 and MONTH(lis.DATE) > 4
				)
				UNION
				(SELECT adm.TXN,adm.sex,YEAR(CURDATE()) - YEAR(adm.BORN) AS age
							,wt,pulse,resp,temp,bp,blood,rh
							,adm.room

					FROM  icd10.reg_2007 adm
					 	INNER JOIN lis lis
					ON adm.TXN = lis.TXN
					WHERE YEAR(lis.DATE) > 2017 and MONTH(lis.DATE) > 4
				)
				UNION
				(SELECT adm.TXN,adm.sex,YEAR(CURDATE()) - YEAR(adm.BORN) AS age
							,wt,pulse,resp,temp,bp,blood,rh
							,adm.room

					FROM  icd10.reg_2008 adm
					 	INNER JOIN lis lis
					ON adm.TXN = lis.TXN
					WHERE YEAR(lis.DATE) > 2017 and MONTH(lis.DATE) > 4
				)
		'''

	return sql

def getdrug():

	sql = 	'''
			(SELECT dru.TXN, dru.CODE AS drug, dru.NAME AS drug_name

				FROM  icd10.dru dru
				 	INNER JOIN lis lis
				ON dru.TXN = lis.TXN
				WHERE YEAR(lis.DATE) > 2017 and MONTH(lis.DATE) > 4
			)
			UNION
			(SELECT dru.TXN, dru.CODE AS drug, dru.NAME AS drug_name

				FROM  icd10.idru dru
				 	INNER JOIN lis lis
				ON dru.TXN = lis.TXN
				WHERE YEAR(lis.DATE) > 2017 and MONTH(lis.DATE) > 4
			)
		'''

	return sql

def getlab():

	sql = 	'''
			(SELECT lab.TXN, lab.NAME AS lab_name, lab.LST AS name, lab.REP AS value

				FROM  icd10.lab lab
				 	INNER JOIN lis lis
				ON lab.TXN = lis.TXN
				WHERE YEAR(lis.DATE) > 2017 and MONTH(lis.DATE) > 4
			)
			UNION
			(SELECT lab.TXN, lab.NAME AS lab_name, lab.LST AS name, lab.REP AS value

				FROM  icd10.ilab lab
				 	INNER JOIN lis lis
				ON lab.TXN = lis.TXN
				WHERE YEAR(lis.DATE) > 2017 and MONTH(lis.DATE) > 4
			)
		'''

	return sql

def getlab_code():

	sql =   '''
                        (SELECT DISTINCT lab.CODE AS code, lab.NAME AS lab_name

                                FROM  icd10.lab lab
                        )
                        UNION
                        (SELECT DISTINCT lab.CODE AS code, lab.NAME AS lab_name

                                FROM  icd10.ilab lab
                        )
                '''

	return sql


def geticd():

	sql = 	'''
		(SELECT code,cdesc FROM icd10.icd10)
		UNION
		(SELECT code,cdesc FROM icd10.icd10_2010)
		'''

	return sql

def save(db_connection,sql,name):
	df = pd.read_sql(sql, con=db_connection)
	df = decode(df)
	if not os.path.exists('../../secret/data/validation/'):
		os.makedirs('../../secret/data/validation/')
	df.to_csv('../../secret/data/validation/'+name+'.csv')
	print('Saved '+name)
	

def get_validation_data(config):

	db_connection = sql.connect(	host=config.DATABASE_CONFIG['host'], 
											database=config.DATABASE_CONFIG['dbname'], 
											user=config.DATABASE_CONFIG['user'], 
											password=config.DATABASE_CONFIG['password'], 
											port=config.DATABASE_CONFIG['port'])

	#save(db_connection,getadm(),'admit_onehot')
	#save(db_connection,getreg(),'registration_onehot')
	#save(db_connection,getdrug(),'drug_numeric')
	#save(db_connection,getlab(),'lab')
	save(db_connection,getlab_code(),'lab_code')
	#save(db_connection,geticd(),'icd')





