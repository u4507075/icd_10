import pandas as pd
from pathlib import Path
import numpy as np
import ntpath
import os

dp = '../../secret/data/drug/drug_numeric.csv'
chunk = 1000

def get_drug_verification_registration_data():
	p = '../../secret/data/registration/registration.csv'
	for df in  pd.read_csv(p, chunksize=chunk, index_col=0):
		df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
		for dfp in  pd.read_csv(dp, chunksize=chunk, index_col=0):
			dfp = dfp[dfp['TXN','drug']]
			result = pd.merge(df, dfp, how='inner', on=['TXN'])
			print(result)

def get_drug_verification_admission_data():
	print('x')
def get_drug_verification_lab_data():
	print('x')













