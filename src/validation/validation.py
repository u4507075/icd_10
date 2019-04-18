import pandas as pd
from pathlib import Path
import numpy as np
import ntpath
import os
import pickle
import re
from random import randint

def read(path):
	with open(path, "r", encoding='utf-8') as f:
		text= f.read()
		return text

def write(path,text):
	with open(path, "w") as file:
		file.write(text)

def predict():
	body = read('html/body.html')
	items = ''
	for i in range(randint(3,10)):
		item = read('html/item.html')
		values = ''
		for j in range(randint(3,20)):
			value = read('html/value.html')
			value = value.replace('%ITEM', str(randint(3000,200000)))
			value = value.replace('%VALUE', str(randint(0,200)))
			values = values + value
		item = item.replace('%VALUE', values)
		items = items + item
	body = body.replace('%ITEM', items)
	write('result/test.html',body)


def predict_testset():

	txn_testset = pd.read_csv('../../secret/data/testset_clean/txn_testset.csv',index_col=0)['TXN'].values.tolist()
	files = os.listdir('../../secret/data/testset_clean/')
	for txn in txn_testset:
		print(txn)
		for filename in files:
			if filename != 'txn_testset.csv':
				for df in  pd.read_csv('../../secret/data/testset_clean/'+filename, chunksize=1000000, index_col=0):
					df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
					df = df[df['TXN']==txn]
					if len(df) > 0:
						for index,row in df.iterrows():
							#print('### Label : '+str(row['icd10']))
							if os.path.exists('../../secret/data/model/'+filename.replace('.csv','')):
								for feature in os.listdir('../../secret/data/model/'+filename.replace('.csv','')):
									model = pickle.load(open('../../secret/data/model/'+filename.replace('.csv','')+'/'+feature, 'rb'))
									pre = model.predict(row[1:len(row)-1].tolist())
									if not pre[0].startswith('not'):
										print(row)
										print(filename)
										print(feature)
										print(pre)
										print(model.predict_proba(row[1:len(row)-1].tolist()))

	










