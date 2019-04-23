import pandas as pd
from pathlib import Path
import numpy as np
import re
import os
import spacy

def getdata(path):

	return pd.read_csv(path, index_col=0)

def get_description_data():
	p = '../../secret/snomed/45_THIS_SNOMED_CT_US_201903_Full_Excel/'
	files = os.listdir(p)
	#c_disorder = []
	#d_disorder = []
	for filename in files:
		filename = '02 Finding.xlsx'
		xls = pd.ExcelFile(p+filename)
		return pd.read_excel(p+filename, sheet_name=xls.sheet_names[0])
		#c_disorder.append(pd.read_excel(p+filename, sheet_name=xls.sheet_names[0]))
		#d_disorder.append(pd.read_excel(p+filename, sheet_name=xls.sheet_names[1]))
		#print(filename)
	#return c_disorder,d_disorder

def remove_junk(x):
	x = x.lower()
	x = re.sub('{.*}', '', x)
	x = x.replace('\\', ' \\')
	x = re.sub(r"\\.*? ",'',x)
	x = x.replace('\r\n', '')
	x = re.sub(r"[^a-z ]",'',x)
	x = re.sub(r" +",' ',x)
	return x

def preprocess_radio_data():
	nlp = spacy.load('en_core_web_md')
	radio = getdata('../../secret/data/radio/radio_icd10.csv')
	radio['rep_clean'] = radio['REP'].apply(remove_junk)
	c_disorder = get_description_data()
	for index,row in radio.iterrows():
		for i,r in c_disorder.iterrows():
				doc1 = nlp(row['rep_clean'])
				doc2 = nlp(r['TERM'])
				p = doc1.similarity(doc2)
				if p > 0.9:
					print(p)
					print(doc1)
					print(doc2)












