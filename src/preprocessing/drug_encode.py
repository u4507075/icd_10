import mysql.connector as sql
import pandas as pd
from pathlib import Path
import numpy as np


def get_encode_feature(feature):
	p = '../../secret/data/'+feature+'/'+feature+'_clean.csv'
	value = []
	for df in  pd.read_csv(p, chunksize=1000000, index_col=0):
		v = df[df[feature].str.contains('[a-zA-Z]',regex=True, na=False)][feature].values.tolist()
		value = value + v
		value = list(set(value))
		print(len(value))
	value.sort()
	d = { i : value[i] for i in range(0, len(value)) }
	df = pd.DataFrame.from_dict({feature:value, 'code': list(range(len(value)))})
	df['code'] = df['code']+1
	df.to_csv('../../secret/data/'+feature+'/'+feature+'_encode.csv')

def encode_feature(feature):
	p = '../../secret/data/'+feature+'/'+feature+'_clean.csv'
	p2 = '../../secret/data/'+feature+'/'+feature+'_numeric.csv'
	encoder = pd.read_csv('../../secret/data/'+feature+'/'+feature+'_encode.csv')
	feature_encoder = dict(zip(encoder[feature],encoder['code']))
	for df in  pd.read_csv(p, chunksize=100000):
		df[feature] = df[feature].map(feature_encoder)
		df[feature] = df[feature].apply(pd.to_numeric,errors='coerce').fillna(0)
		result = df[['TXN',feature,'icd10']]
		file = Path(p2)
		if file.is_file():
			with open(p2, 'a') as f:
				result.to_csv(f, header=False)
		else:
			result.to_csv(p2)
		print('Append clean data')

