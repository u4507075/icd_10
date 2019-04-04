import pandas as pd
from pathlib import Path
import numpy as np
import ntpath
import os

dp = '../../secret/data/drug/drug_clean.csv'
chunk = 100000

def get_drug_verification_registration_data(p):
	#p = '../../secret/data/registration/registration_onehot.csv'
	for df in  pd.read_csv(p, chunksize=chunk, index_col=0):
		df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
		for dfp in  pd.read_csv(dp, chunksize=chunk, index_col=0):
			dfp = dfp[['TXN','drug']]
			result = pd.merge(df, dfp, how='inner', on=['TXN'])
			if len(result) > 0:
				result = result.drop_duplicates()
				print(result)

def get_drug_verification_lab_data():
	print('x')













