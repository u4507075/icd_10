import mysql.connector as mysql
import pandas as pd
from pathlib import Path
import numpy as np
import os
from os.path import isfile, join
import re
import gensim

path = '../../../secret/data/'

def save_file(df,p):
	file = Path(p)
	if file.is_file():
		with open(p, 'a', encoding="utf-8") as f:
			df.to_csv(f, header=False)
	else:
		df.to_csv(p)

def remove_file(p):
	file = Path(p)
	if file.is_file():
		os.remove(p)

def split_data(replace=False):
	stop_row = 0
	last = stop_row
	n = range(1, stop_row-1)
	#n = None
	mpath = path+'split/ilab/'
	if replace:
		for c in ['lab_name', 'name', 'value', 'icd10']:
			filelist = [ f for f in os.listdir(mpath+c+'/') if f.endswith(".csv") ]
			for f in filelist:
				os.remove(os.path.join(mpath+c+'/', f))
			filelist = [ f for f in os.listdir(mpath+c+'/') if f.endswith(".csv") ]
			for f in filelist:
				os.remove(os.path.join(mpath+'icd10/', f))
		n = None
	for df in pd.read_csv(path+'trainingset/raw/ilab.csv', index_col = 0, chunksize = 1, skiprows=n):
		for c in ['lab_name','name','value','icd10']:
			k = c+'_'+str(df[c].iloc[0])
			if c is not 'icd10':
				k = k.lower()
				k = re.sub('[^a-z0-9\.\_]','',k)
			k = k[:210]
			save_file(df, mpath + c + '/' + str(k) + '.csv')
		f = open(mpath + "log.txt", "w")
		f.write(str(last))
		f.close()
		if last % 1000 == 0:
			print(last)
		last = last+1

def chain(max,df,n1,n2):
		toggle = True
		df[n1] = df[n1].astype(str)
		df[n2] = df[n2].astype(str)
		d = df.sample(n=1)
		# d = df.sample(n=1,weights=df[n1+'_weight'])
		last = d[n1].iloc[0]
		text = []
		n = 0
		for i in range(max):
			if toggle:
					#text.append([last, last])
					x = [n1+'_'+last]
					y = [n1+'_'+last]
					icd10 = df[df[n1] == last].sample(n=1)[n2].iloc[0]
					x.insert(0, n2+'_'+str(icd10))
					last = df[df[n2] == icd10].sample(n=1)[n1].iloc[0]
					y.insert(0, n1+'_'+str(last))
					text.append(x)
					text.append(y)
					last = icd10
					toggle = False
			else:
					#text.append([last, last])
					x = [n2+'_'+last]
					y = [n2+'_'+last]
					drug = df[df[n2] == last].sample(n=1)[n1].iloc[0]
					x.insert(0, n1+'_'+str(drug))
					last = df[df[n1] == drug].sample(n=1)[n2].iloc[0]
					y.insert(0, n2+'_'+str(last))
					text.append(x)
					text.append(y)
					last = drug
					toggle = True

		return text

def train_chain(filename,f1,f2):
	modelpath = path+'word2vec_model/'+filename+'_'+f1+'_'+f2
	if not os.path.exists(modelpath):
		os.makedirs(modelpath)
	df = pd.read_csv(path + 'trainingset/raw/' + filename + '.csv', index_col=0, low_memory=False)
	df = df[[f1,f2]]
	t = 1000000
	p = 100
	s = 0
	model = None
	for i in range(t, -1, -1 * p):
		file = Path(modelpath + '/' + f1+'_'+f2 + '_' + str(i))
		if file.is_file():
			model = gensim.models.Word2Vec.load(modelpath + '/' + f1+'_'+f2 + '_' + str(i))
			s = i
			break
	if model is None:
		model = gensim.models.Word2Vec(chain(1000, df, f1, f2), compute_loss=True, sg=1)
	print(s)

	for i in range(s + 1, t):
		text = chain(100, df, f1, f2)
		model.build_vocab(text, update=True)
		model.train(text, total_examples=model.corpus_count, compute_loss=True, epochs=10)
		print(i)
		print(model.get_latest_training_loss())
		if i % p == 0:
			model.save(modelpath + '/' + f1+'_'+f2 + '_' + str(i))
			print('saved ' + modelpath + '/' + f1+'_'+f2 + '_' + str(i))

def merge_result(file, total, result, row, c):
	mpath = path + 'word2vec_model/' + file + '_' + c + '_icd10/' + c + '_icd10_3000'
	if isfile(mpath):
		model = gensim.models.Word2Vec.load(mpath)
		if c + '_' + str(row[c]) in model.wv.vocab:
			similar_words = model.wv.most_similar(positive=[c + '_' + str(row[c])], topn=len(model.wv.vocab))
			result = pd.DataFrame(similar_words, columns=['icd10', 'similarity_' + c + '_' + str(row[c])])
			result = result[result['icd10'].str.startswith('icd10')]
			# result = result.sort_values(by='similarity', ascending=False)
			if total is None:
				total = result
			else:
				total = pd.merge(total, result, how='outer', on='icd10')
def test_chain(fname):
	reg = pd.read_csv(path + 'testset/raw/'+fname+'.csv', index_col=0)
	txns = reg['txn'].unique()
	rank = None
	for txn in txns[:100]:
		total = None
		test = reg[reg['txn']==txn].copy()
		label = test[['icd10']].copy()
		label['icd10'] = 'icd10_' + label['icd10'].astype(str)
		#for root, dirs, files in os.walk(path + 'testset/raw/'):
		files = ['dru.csv']
		for file in files:
			if file.endswith(".csv"):
				df = pd.read_csv(path + 'testset/raw/' + file, index_col=0)
				df = df[df['txn']==txn]
				df.drop(columns=['icd10'], inplace=True)
				df.drop_duplicates(inplace=True)
				for index,row in df.iterrows():
					for c in df.columns:
						merge_result(file.split('.')[0], total, result, row, c)

		print(txn)
		total = total.fillna(0.01)
		if len(total.columns) >= 2:
			total['total_similarity'] = total.iloc[:,1]
			for i in range(2,len(total.columns)):
				total['total_similarity'] = total['total_similarity']*total.iloc[:,i]
		total = total.sort_values(by='total_similarity', ascending=False)
		total = total.reset_index(drop=True)
		total = total.reset_index(drop=False)
		match = total[total['icd10'].isin(label['icd10'].values.tolist())]
		match = pd.merge(match, label, how='outer', on='icd10')
		match.drop_duplicates(inplace=True)
		print(match)
		if rank is None:
			rank = match[['index','icd10']]
		else:
			rank = pd.concat([rank,match[['index','icd10']]])
	stat = rank.groupby('icd10') \
		   .agg(count=('icd10','size'), mean=('index','mean')) \
		   .reset_index()
	print(stat)
	#stat.to_csv(path+'result.csv')


split_data(replace=True)
#test_chain()
#reg
#,txn,sex,age,wt,pulse,resp,temp,blood,rh,room,room_dc,dx_type,sbp,dbp,icd10
#train_chain('reg','sex','icd10')
#train_chain('reg','age','icd10')
#train_chain('reg','room','icd10')

#adm
#,txn,sex,age,wt,pulse,resp,temp,blood,rh,room,room_dc,dx_type,sbp,dbp,icd10
#train_chain('adm','sex','icd10')
#train_chain('adm','age','icd10')
#train_chain('adm','room','icd10')

#drug
#train_chain('dru','drug','icd10')
#train_chain('idru','drug','icd10')

#lab
#,txn,lab_name,name,value,dx_type,icd10
#train_chain('lab','lab_name','icd10')
#train_chain('lab','name','icd10')
#train_chain('lab','value','icd10')
#train_chain('ilab','lab_name','icd10')
#train_chain('ilab','name','icd10')
#train_chain('ilab','value','icd10')
'''
x = '100'
icd10 = pd.read_csv(path + 'raw/icd10.csv', index_col=0)
icd10_map = dict(zip(icd10['code'], icd10['cdesc']))
model = gensim.models.Word2Vec.load(path+'word2vec_model/age_icd10/age_icd10_200')
similar_words = model.wv.most_similar(positive=[x],topn=10)
for i in range(len(similar_words)):
	if similar_words[i][0] in icd10_map:
		print(icd10_map[similar_words[i][0]])
'''
'''
x = 'SGN3'
icd10 = pd.read_csv(path + 'raw/icd10.csv', index_col=0)
icd10_map = dict(zip(icd10['code'], icd10['cdesc']))
model = gensim.models.Word2Vec.load(path+'word2vec_model/room_icd10/room_icd10_100')
print(model.wv.vocab)
similar_words = model.wv.most_similar(positive=[x],topn=10)
for i in range(len(similar_words)):
	if similar_words[i][0] in icd10_map:
		print(icd10_map[similar_words[i][0]])
'''

#model = gensim.models.Word2Vec.load(path+'word2vec_model/reg_room_icd10/room_icd10_100')
#print(model.wv.vocab)