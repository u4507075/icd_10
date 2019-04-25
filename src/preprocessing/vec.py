import mysql.connector as sql
import pandas as pd
from pathlib import Path
import numpy as np
import os
import re
import spacy

nlp = spacy.load('en_core_web_md')
path = '../../secret/data/raw/'

def save_file(df,name):
	p = '../../secret/data/vec/'
	if not os.path.exists(p):
		os.makedirs(p)
	file = Path(p+name+'.csv')
	if file.is_file():
		with open(p, 'a') as f:
			df.to_csv(f, header=False)
	else:
		df.to_csv(p)

def to_vec(x):
	value = sum(nlp(str(x)).vector)
	if value == 0:
		return sum(sum([nlp(v).vector for v in str(x)]))
	else:
		return value

def word_to_vec(name):
	for df in  pd.read_csv(path+name+'.csv', chunksize=10000, index_col=0):
		for c in df.columns:
			if c != 'txn' and c != 'icd10':
				df[c] = df[c].apply(to_vec)
		save_file(df,name)
		print(name)

