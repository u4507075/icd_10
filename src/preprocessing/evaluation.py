import pandas as pd
from pathlib import Path
import numpy as np
import ntpath
import os

def split_lab():

	txn_testset = pd.read_csv('../../secret/data/testset/txn_testset.csv',index_col=0)['TXN'].values.tolist()
	files = os.listdir('../../secret/data/lab/encode/')
	for lab in files:
		p = '../../secret/data/lab/encode/'+lab
		for df in  pd.read_csv(p, chunksize=100000, index_col=0):
			testset = df[df['TXN'].isin(txn_testset)]
			trainingset = df[~df['TXN'].isin(txn_testset)]
			if len(testset) > 0:
				save_data(testset, 'testset/'+lab)
			if len(trainingset) > 0:
				save_data(trainingset, 'trainingset/'+lab)
			print('Save '+lab)













