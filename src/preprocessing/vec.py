import mysql.connector as sql
import pandas as pd
from pathlib import Path
import numpy as np
import os
import re
import spacy

nlp = spacy.load('en_core_web_md', disable=["tagger","parser", "ner"])
path = '../../secret/data/raw/'

def save_file(df,name):
	p = '../../secret/data/vec/'
	if not os.path.exists(p):
		os.makedirs(p)
	file = Path(p+name+'.csv')
	if file.is_file():
		with open(p+name+'.csv', 'a') as f:
			df.to_csv(f, header=False)
	else:
		df.to_csv(p+name+'.csv')

def to_vec(x):
	if x == 0 or x == '':
		return 0
	else:
		value = sum(nlp(str(x)).vector)
		if value == 0:
			return sum(sum([nlp(v).vector for v in str(x)]))
		else:
			return value

def word_to_vec(name):
	print("Read icd10 mapping")
	icd10 =  pd.read_csv('../../secret/data/raw/icd10.csv', index_col=0)
	icd10_map = dict(zip(icd10['code'],icd10.index))
	print("Read raw data")
	n = 0
	chunk = 10000
	for df in  pd.read_csv(path+name+'.csv', chunksize=chunk, index_col=0, low_memory=False):
		for c in df.columns:
			if c != 'txn' and c != 'icd10':
				df[c] = df[c].apply(to_vec)
		if 'report' in df:
			d1 = df['report'].str.split(' ',expand=True)
			d1 = d1.merge(df, right_index = True, left_index = True)
			d1 = d1.melt(id_vars = ['txn','location','position','icd10'], value_name = 'report')
			d1 = d1[d1['variable'] != 'report']
			df = df[['txn','location','position','report','icd10']]

		df['icd10'] = df['icd10'].map(icd10_map)
		n = n + chunk
		print("Converted "+name+' '+str(n))
		save_file(df,name)
		#print(name)

def radio_to_vec(name):
	print("Read icd10 mapping")
	#icd10 =  pd.read_csv('../../secret/data/raw/icd10.csv', index_col=0)
	#icd10_map = dict(zip(icd10['code'],icd10.index))
	print("Read raw data")
	n = 0
	chunk = 5
	for df in  pd.read_csv(path+name+'.csv', chunksize=chunk, index_col=0, low_memory=False):
		#df['icd10'] = df['icd10'].map(icd10_map)
		'''
		for index,row in df.iterrows():
			print(row['report'])
			for t in row['report'].split(' '):
				doc = nlp(t)
				print(t)
				print(sum(doc.vector))
		'''
		d1 = df['report'].str.split(' ',expand=True)
		d1 = d1.merge(df, right_index = True, left_index = True)
		print(d1)
		d1 = d1.melt(id_vars = ['txn','location','position','icd10'], value_name = 'report')
		d1 = d1.sort_values(['txn', 'icd10', 'variable'], ascending=True)
		#d1 = d1[d1['variable'] != 'report']
		#df = d1[['txn','location','position','report','icd10']]
		print(d1)
		break



















