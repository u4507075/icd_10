import mysql.connector as sql
import pandas as pd
from pathlib import Path
import numpy as np


def get_encode_feature(filename):
	p = '../../secret/data/drug/'+filename+'.csv'
	value = []
	for df in  pd.read_csv(p, chunksize=1000000, index_col=0):
		v = df[df['drug'].str.contains('[a-zA-Z]',regex=True, na=False)]['drug'].values.tolist()
		value = value + v
		value = list(set(value))
		print(len(value))
	value.sort()
	d = { i : value[i] for i in range(0, len(value)) }
	df = pd.DataFrame.from_dict({'drug':value, 'code': list(range(len(value)))})
	df['code'] = df['code']+1
	df.to_csv('../../secret/data/drug/'+filename+'_encode.csv')

def encode_feature(filename):
	p = '../../secret/data/drug/'+filename+'.csv'
	p2 = '../../secret/data/drug/'+filename+'_numeric.csv'
	encoder = pd.read_csv('../../secret/data/drug/'+filename+'_encode.csv')
	feature_encoder = dict(zip(encoder['drug'],encoder['code']))
	for df in  pd.read_csv(p, chunksize=1000000, index_col=0):
		df['drug'] = df['drug'].map(feature_encoder)
		df['drug'] = df['drug'].apply(pd.to_numeric,errors='coerce').fillna(0)
		result = df[['txn','drug','icd10']]
		file = Path(p2)
		if file.is_file():
			with open(p2, 'a') as f:
				result.to_csv(f, header=False)
		else:
			result.to_csv(p2)
		print('Append clean data')

