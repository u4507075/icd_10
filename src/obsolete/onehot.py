import mysql.connector as sql
import pandas as pd
from pathlib import Path
import numpy as np


def get_total_feature(filename):
	p = '../../secret/data/drug/'+filename+'.csv'
	value = []
	for df in  pd.read_csv(p, chunksize=1000000, index_col=0):
		df = df[df['drug'].notnull()]
		v = df['drug'].unique().tolist()
		value = value + v
		value = list(set(value))       
		value.sort()
		print(len(value))
	d = { i : value[i] for i in range(0, len(value) ) }
	df = pd.DataFrame.from_dict({'drug':value, 'code': list(range(len(value)))})
	df.to_csv('../../secret/data/drug/'+filename+'_code.csv')

def onehot(feature):
	p = '../../secret/data/'+feature+'/'+feature+'_clean.csv'
	p2 = '../../secret/data/'+feature+'/'+feature+'_onehot.csv'
	drug_list = pd.read_csv('../../secret/data/'+feature+'/'+feature+'_code.csv')[feature].values.tolist()
	#drug_list = ['TXN']+drug_list+['DX1']
	for df in  pd.read_csv(p, chunksize=100000, index_col=0):
		df = df[['TXN',feature,'icd10']]
		df2 = pd.get_dummies(df[feature])
		df2['TXN'] = df['TXN'].copy() 
		df2['icd10'] = df['icd10'].copy()
		df3 = df2.groupby(['TXN','icd10']).agg('sum')
		result = df3.reindex(columns=drug_list)
		result.fillna(0, inplace=True)
		file = Path(p2)
		if file.is_file():
			with open(p2, 'a') as f:
				result.to_csv(f, header=False)
		else:
			result.to_csv(p2)
		print('Append clean data')
'''
def mapdata():
	p = '../../secret/data/data_clean.csv'
	p2 = '../../secret/data/data_map.csv'
	d = pd.read_csv('../../secret/data/drug_code.csv')
	l = dict(zip(d.drug,d.code))
	for df in  pd.read_csv(p, chunksize=1000000):
		df['drug_code'] = df['drug'].map(l)
		file = Path(p2)
		if file.is_file():
			with open(p2, 'a') as f:
				df.to_csv(f, header=False)
		else:
			df.to_csv(p2)
		print('Append mapped data')
'''

