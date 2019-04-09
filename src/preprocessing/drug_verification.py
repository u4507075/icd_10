import pandas as pd
from pathlib import Path
import numpy as np
import ntpath
import os

dp = '../../secret/data/drug/drug_clean.csv'
chunk = 10000

def save_file(df,filename):
	if not os.path.exists('../../secret/data/drug_verification/'):
		os.makedirs('../../secret/data/drug_verification/')
	file = Path('../../secret/data/drug_verification/'+filename)
	if file.is_file():
		with open('../../secret/data/drug_verification/'+filename, 'a') as f:
			df.to_csv(f, header=False)
	else:
		df.to_csv('../../secret/data/drug_verification/'+filename)

def get_drug_verification_registration_data(p):
	name = ntpath.basename(p)
	for df in  pd.read_csv(p, chunksize=chunk, index_col=0):
		df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
		for dfp in  pd.read_csv(dp, chunksize=chunk, index_col=0):
			dfp = dfp[['TXN','drug']]
			result = pd.merge(df, dfp, how='inner', on=['TXN'])
			if len(result) > 0:
				result = result.drop_duplicates()
				save_file(result,name)
				print('Append '+name)

def get_drug_verification_lab_data():
	files = os.listdir('../../secret/data/lab/encode/')
	for lab in files:
		p = '../../secret/data/lab/encode/'+lab
		for df in pd.read_csv(p, chunksize=chunk, low_memory=False):
			df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
			for dfp in  pd.read_csv(dp, chunksize=chunk, index_col=0):
				dfp = dfp[['TXN','drug']]
				result = pd.merge(df, dfp, how='inner', on=['TXN'])
				if len(result) > 0:
					result = result.drop_duplicates()
					save_file(result,lab)
					print('Append '+lab)










