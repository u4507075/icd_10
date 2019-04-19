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

def get_data(name,txn):
	df = pd.read_csv('../../secret/data/validation/'+name+'.csv', index_col=0)
	df = df[df['TXN']==txn].drop_duplicates()
	return df
'''
def get_testset(txn):
	print(get_data('admit_onehot',txn))
	print(get_data('registration_onehot',txn))
	print(get_data('lab',txn))
	print(get_data('drug_numeric',txn))
'''
def get_value(item,value):
	v = read('html/value.html')
	v = v.replace('%ITEM', str(item))
	v = v.replace('%VALUE', str(value))
	return v

def get_reg(title,prop,df):
	item = read('html/item.html')
	item = item.replace('%TITLE', title)
	item = item.replace('%PROP', prop)
	values = ''
	for index, row in df.iterrows():
		for name in list(df):
			if name != 'TXN':
				values = values + get_value(name, row[name])
	item = item.replace('%VALUE', values)
	return item

def get_lab(df):
	items = ''
	for index, row in df.iterrows():
		item = read('html/item.html')
		item = item.replace('%TITLE', row['lab_name'])
		item = item.replace('%PROP', '78%')
		values = ''
		title = str(row['name']).split(';')
		v = str(row['value']).split(';')
		for i in range(len(title)):
			if title[i] == '':
				break
			if len(v)-1 >= i:
				values = values + get_value(title[i], v[i])
			else:
				values = values + get_value(title[i], '')
		item = item.replace('%VALUE', values)
		items = items + item
	return items

def get_drug(df):
	items = ''
	for index, row in df.iterrows():
		item = read('html/item.html')
		item = item.replace('%TITLE', row['drug_name'])
		item = item.replace('%PROP', '78%')
		item = item.replace('%VALUE', '')
		items = items + item
	return items

def get_branch(dx,item):
	branch = read('html/branch.html')
	branch = branch.replace('%DX',dx)
	branch = branch.replace('%ITEM',item)
	return branch

def get_predict_text(txn,filename,prop,index):
	
	if filename == 'admit_onehot.csv':
		return get_reg('Admission data', str(prop)+'%', get_data('admit_onehot',txn))
	elif filename == 'registration_onehot.csv':
		return get_reg('Registration data','81%',get_data('registration_onehot',txn))
	elif filename == 'drug_numeric.csv':
		return get_drug(get_data('drug_numeric',txn))
	else:
		df = pd.read_csv('../../secret/data/lab/raw/'+filename, index_col=0)
		df = df[df.index==index].drop_duplicates()
		print(filename)
		print(index)
		print(df)
		return get_lab(get_data('lab',txn))

def predict_testset():

	txn_testset = pd.read_csv('../../secret/data/testset_clean/txn_testset.csv',index_col=0)['TXN'].values.tolist()
	icd10 = pd.read_csv('../../secret/data/validation/icd.csv',index_col=0)
	icd10_map = dict(zip(icd10['code'],icd10['cdesc']))
	files = os.listdir('../../secret/data/testset_clean/')
	tt = [963366,970099,970257,970774,970578]
	for txn in txn_testset:
		if txn in tt:
			print(txn)
			body = read('html/body.html')
			body = body.replace('%TXN', str(txn))
			items = ''
			branch = ''
			dxlist = []
			predicted_dx = pd.DataFrame(columns=['actual_dx','predicted_dx','branch'])
			for filename in files:
				if filename != 'txn_testset.csv':
					df = pd.read_csv('../../secret/data/testset_clean/'+filename, index_col=0)
					df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
					df = df[df['TXN']==txn]
					if len(df) > 0:
						for index,row in df.iterrows():
							dxlist.append(row['icd10'])
							if os.path.exists('../../secret/data/model/'+filename.replace('.csv','')):
								for feature in os.listdir('../../secret/data/model/'+filename.replace('.csv','')):
									model = pickle.load(open('../../secret/data/model/'+filename.replace('.csv','')+'/'+feature, 'rb'))
									pre = model.predict(row[1:len(row)-1].tolist())
									if not pre[0].startswith('not'):
										#print(model.predict_proba(row[1:len(row)-1].tolist()))
										predicted_dx = predicted_dx.append(pd.DataFrame([[row['icd10'],
																										pre[0],
																										get_predict_text(	txn,
																																filename,
																																model.predict_proba(row[1:len(row)-1].tolist()),
																																index)
																										]], 
																										columns=['actual_dx','predicted_dx','branch']))
			'''
			dx = 	''		
			for d in dxlist:
				if d in predicted_dx:
					dx.append(get_branch(d,predicted_dx
			'''
			dxlist = list(set(dxlist))
			dxlist.sort()
			print(dxlist)
			print(predicted_dx)
	
'''
						dx = row['icd10']
						if row['icd10'] in icd10_map:
							dx = row['icd10'] + ': '+ icd10_map[row['icd10']]
						if not row['icd10'] in dxlist:
							branch = branch + get_branch(dx)
							dxlist.append(row['icd10'])
				body = body.replace('%DX', branch)
				write('../../secret/data/result/'+str(txn)+'.html',body)

'''




