import pandas as pd
from pathlib import Path
import numpy as np
import ntpath
import os
import pickle
import re

def predict_testset():

	txn_testset = pd.read_csv('../../secret/data/testset/txn_testset.csv',index_col=0)['TXN'].values.tolist()
	files = os.listdir('../../secret/data/testset/')
	for txn in txn_testset:
		for filename in files:
			if filename != 'txn_testset.csv':
				print(filename)
				print(txn)
				for df in  pd.read_csv('../../secret/data/testset/'+filename, chunksize=1000000, index_col=0):
					df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
					df = df[df['TXN']==txn]
					if len(df) > 0:
						for index,row in df.iterrows():
							print('### Label : '+str(row['icd10']))
							if os.path.exists('../../secret/data/model/'+filename.replace('.csv','')):
								for feature in os.listdir('../../secret/data/model/'+filename.replace('.csv','')):
									model = pickle.load(open('../../secret/data/model/'+filename.replace('.csv','')+'/'+feature, 'rb'))
									pre = model.predict(row[1:len(row)-1].tolist())
									if not pre[0].startswith('not'):
										print(pre)
										print(model.predict_proba(row[1:len(row)-1].tolist()))












