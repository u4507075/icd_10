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

def remove_space(x):
	try:
		return x.replace(' ','')
	except AttributeError:
		return x

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

def get_lab(df,lab_name,prop):
	items = ''
	for index, row in df.iterrows():
		item = read('html/item.html')
		item = item.replace('%TITLE', str(lab_name))
		item = item.replace('%PROP', prop)
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

def get_drug(name,prop):
	item = read('html/item.html')
	item = item.replace('%TITLE', name)
	item = item.replace('%PROP', prop)
	item = item.replace('%VALUE', '')
	items = items + item
	return item

def get_branch(dx,item):
	branch = read('html/branch.html')
	branch = branch.replace('%DX',dx)
	branch = branch.replace('%ITEM',item)
	return branch

def get_predict_text(txn,filename,prop,index,row,drug_map,lab_map):
	if filename == 'admit_onehot.csv':
		return get_reg('Admission data', str(prop)+'%', get_data('admit_onehot',txn))
	elif filename == 'registration_onehot.csv':
		return get_reg('Registration data',str(prop)+'%',get_data('registration_onehot',txn))
	elif filename == 'drug_numeric.csv':
		df = get_data('drug_numeric',txn)
		df['code'] = df['drug'].map(drug_map)
		df = df[df['code']==row['drug']]
		drugs = ''
		for index,r in df.iterrows():
			drug_num = row['drug']
			drugs = drugs + get_drug(r['drug_name'],str(prop)+'%')
		return drugs
	else:
		df = pd.read_csv('../../secret/data/lab/raw/'+filename, index_col=0)
		df = df[df.index==index].drop_duplicates()
		code = filename.replace('.csv','')
		lab_name = code
		if code in lab_map:
			lab_name = lab_map[code]
		return get_lab(df,lab_name,str(prop)+'%')

def predict_testset():

	txn_testset = pd.read_csv('../../secret/data/testset_clean/txn_testset.csv',index_col=0)['TXN'].values.tolist()
	drug = pd.read_csv('../../secret/data/drug/drug_code.csv',index_col=0)
	drug_map = dict(zip(drug['drug'],drug['code']))
	lab = pd.read_csv('../../secret/data/validation/lab_code.csv',index_col=0)
	lab['code'] = lab['code'].apply(remove_space)
	lab = lab.drop_duplicates(subset='code')
	lab_map = dict(zip(lab['code'],lab['lab_name']))
	icd10 = pd.read_csv('../../secret/data/validation/icd.csv',index_col=0)
	icd10_map = dict(zip(icd10['code'],icd10['cdesc']))
	files = os.listdir('../../secret/data/testset_clean/')

	for txn in txn_testset:

		print(txn)
		body = read('html/body.html')
		body = body.replace('%TXN', str(txn))
		items = ''
		branch = ''
		dxlist = []
		predicted_dx = pd.DataFrame(columns=['predicted_dx','prop','branch'])
		for filename in files:
			if filename != 'txn_testset.csv':
				df = pd.read_csv('../../secret/data/testset_clean/'+filename, index_col=0)
				df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
				df = df[df['TXN']==txn]
				dxlist = dxlist + df['icd10'].values.tolist()
				df = df.drop(columns=['icd10'])
				df = df.drop_duplicates()
				print(df)
				if len(df) > 0:
					for index,row in df.iterrows():
						print(row)
						if os.path.exists('../../secret/data/model/'+filename.replace('.csv','')):
							for feature in os.listdir('../../secret/data/model/'+filename.replace('.csv','')):
								model = pickle.load(open('../../secret/data/model/'+filename.replace('.csv','')+'/'+feature, 'rb'))
								pre = model.predict(row[1:len(row)].tolist())
								if not pre[0].startswith('not'):
									print(pre[0])
									prop = int((model.predict_proba(row[1:len(row)].tolist()))[0][0]*100)
									#print(model.predict_proba(row[1:len(row)-1].tolist()))
									predicted_dx = predicted_dx.append(pd.DataFrame([[pre[0],
																									prop,
																									get_predict_text(	txn,
																															filename,
																															prop,
																															index,row,drug_map,lab_map)
																									]], 
																									columns=['predicted_dx','prop','branch']))
									#break

		dxt = ''
		dxn = ''
		dxo = ''
		dxlist = list(set(dxlist))
		dxlist.sort()
		predicted_dx = predicted_dx.drop_duplicates()
		predicted_dx = predicted_dx[predicted_dx['branch'] != '']
		print(dxlist)
		print(predicted_dx)
		for d in dxlist:
			df = predicted_dx[predicted_dx['predicted_dx']== d]
			dx = d
			if d in icd10_map:
				dx = dx + ': '+ icd10_map[d]
			for index,row in df.iterrows():
				dxt = dxt + get_branch(dx,row['branch'])
			if len(df)==0:
				dxn = dxn + get_branch(dx,'')
		dfxo = predicted_dx[~predicted_dx['predicted_dx'].isin(dxlist)]
		for index,row in dfxo.iterrows():
			dx = row['predicted_dx']
			if dx in icd10_map:
				dx = dx + ': '+ icd10_map[dx]
			dxo = dxo + get_branch(dx,row['branch'])

		body = body.replace('%DXT', dxt)
		body = body.replace('%DXN', dxn)
		body = body.replace('%DXO', dxo)
		if not os.path.exists('../../secret/data/result/'):
			os.makedirs('../../secret/data/result/')
		write('../../secret/data/result/'+str(txn)+'.html',body)






