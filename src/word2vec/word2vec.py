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

def combine_lab():
	remove_file(path+'trainingset/raw/lab_c.csv')
	remove_file(path + 'trainingset/raw/ilab_c.csv')
	for df in pd.read_csv(path+'trainingset/raw/lab.csv', chunksize=10000000, index_col=0):
		df['lab'] = df['lab_name']+'_'+df['name']+'_'+df['value']
		df['lab'] = df['lab'].apply(lambda x: str(x).strip())
		df['lab'] = df['lab'].apply(lambda x: str(x).lower())
		df['lab'] = df['lab'].apply(lambda x: re.sub('^a-z0-9\!\"\#\$\%\&\'\(\)\*\+\,\-\.\/\:\;\<\>\=\?\@\[\]\{\}\\\\\^\_\`\~','',str(x)))
		save_file(df[['txn','lab','dx_type','icd10']], path+'trainingset/raw/lab_c.csv')

	for df in pd.read_csv(path+'trainingset/raw/ilab.csv', chunksize=10000000, index_col=0):
		df['lab'] = df['lab_name']+'_'+df['name']+'_'+df['value']
		df['lab'] = df['lab'].apply(lambda x: str(x).strip())
		df['lab'] = df['lab'].apply(lambda x: str(x).lower())
		df['lab'] = df['lab'].apply(lambda x: re.sub('^a-z0-9\!\"\#\$\%\&\'\(\)\*\+\,\-\.\/\:\;\<\>\=\?\@\[\]\{\}\\\\\^\_\`\~','',str(x)))
		save_file(df[['txn','lab','dx_type','icd10']], path+'trainingset/raw/ilab_c.csv')

def split_data(replace=False):
    #see log.txt in split/ilab
	stop_row = 132762010
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
			fr = open(mpath + "log_r.txt", "w")
			fr.write(str(last))
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

def merge_result(n, txn, icd10, file, total, row, c):
	mpath = path + 'word2vec_model/' + file + '_' + c + '_icd10/' + c + '_icd10_'+str(n)
	iresult = None
	if isfile(mpath):
		model = gensim.models.Word2Vec.load(mpath)
		if c + '_' + str(row[c]) in model.wv.vocab and str(row[c]) != 'nan':
			similar_words = model.wv.most_similar(positive=[c + '_' + str(row[c])], topn=len(model.wv.vocab))
			result = pd.DataFrame(similar_words, columns=['icd10', 'similarity_' + file + '_' + c])
			result = result[result['icd10'].str.startswith('icd10')]
			result['percentile_' + file + '_' + c] = result['similarity_' + file + '_' + c].rank(pct=True)
			df = result.copy().rename(columns = {'similarity_' + file + '_' + c: 'similarity', 'percentile_' + file + '_' + c: 'percentile'}, inplace = False)
			df['txn'] = txn
			df['file'] = file
			df['name'] = c
			df['value'] = row[c]
			df['w_similarity'] = df['similarity']*df['percentile']
			df = df[['txn','icd10','file','name','value','similarity','percentile','w_similarity']]
			df = df[df['icd10']=='icd10_'+icd10]
			iresult = df
			#result[file + '_' + c + '_v'] = row[c]
			# result = result.sort_values(by='similarity', ascending=False)
			if total is None:
				total = result
			else:
				total = pd.merge(total, result, how='outer', on='icd10')
	return total,iresult

def test_chain(fname, n):
	remove_file(path + 'word2vec_fresult_'+str(n)+'.csv')
	remove_file(path + 'word2vec_iresult_'+str(n)+'.csv')
	reg = pd.read_csv(path + 'testset/raw/'+fname+'.csv', index_col=0)

	for index,row in reg.iterrows():

		total = None
		#test = reg[reg['txn']==txn].copy()
		#label = test[['icd10']].copy()
		#label['icd10'] = 'icd10_' + label['icd10'].astype(str)
		#for root, dirs, files in os.walk(path + 'testset/raw/'):
		txn = row['txn']
		label = row['icd10']
		for x in ['sex','age','room']:
			total,iresult = merge_result(n, txn, label, fname, total, row, x)
			if iresult is not None:
				save_file(iresult, path + 'word2vec_iresult_'+str(n)+'.csv')
		files = ['dru.csv','lab.csv']
		for file in files:
			if file.endswith(".csv"):
				df = pd.read_csv(path + 'testset/raw/' + file, index_col=0)
				df = df[df['txn']==txn]
				df.drop(columns=['icd10'], inplace=True)
				df.drop_duplicates(inplace=True)
				for i,r in df.iterrows():
					for c in df.columns:
						total,iresult = merge_result(n, txn, label, file.split('.')[0], total, r, c)
						if iresult is not None:
							save_file(iresult, path + 'word2vec_iresult_'+str(n)+'.csv')

		print(txn)
		#total = total.fillna(0.01)
		features = []
		if len(total.columns) >= 2:
			total['total_similarity'] = total.iloc[:, 1]
			total['w_total_similarity'] = total.iloc[:, 1]*total.iloc[:, 2]
			features.append(total.columns[1])
			for i in range(2, len(total.columns)):
				if 'similarity' in total.columns[i] and total.columns[i] != 'total_similarity' and total.columns[i] != 'w_total_similarity':
					total['total_similarity'] = total['total_similarity'] * total.iloc[:, i]
					total['w_total_similarity'] = total['w_total_similarity'] * total.iloc[:, i] * total.iloc[:, i+1]
					features.append(total.columns[i])
		total = total.assign(mean=total[features].mean(axis=1))
		total = total.sort_values(by='total_similarity', ascending=False)
		total = total.reset_index(drop=True)
		total = total.reset_index(drop=False)
		match = total[total['icd10'] == 'icd10_'+label]
		if len(match) == 0:
			match = match.append(pd.Series(), ignore_index=True)
			match['index'] = np.nan
			match['icd10'] = 'icd10_'+label

		match['txn'] = txn
		save_file(match[['txn','index','icd10','total_similarity','mean','w_total_similarity']],path + 'word2vec_fresult_'+str(n)+'.csv')
		#break
		'''
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
	'''
	#stat.to_csv(path+'result.csv')

def vocab_num(n,file,c):
	for i in range(100,n,100):
		mpath = path + 'word2vec_model/' + file + '_' + c + '_icd10/' + c + '_icd10_' + str(i)
		if isfile(mpath):
			model = gensim.models.Word2Vec.load(mpath)
			#print(pd.DataFrame(model.wv.vocab))
			print(i)
			print(len(model.wv.vocab))
		#break

def age(x):
	if x.startswith('age'):
		age = x.split('_')[1]
		if len(age) == 1:
			return 'age_00'+age
		elif len(age) == 2:
			return 'age_0'+age
		else:
			return x
	else:
		return x

def get_pivot(f_list,f,model,icd10_map,drug_map,human_read=False):
	d1 = []
	for i in f_list:
		similar_words = model.wv.most_similar(positive=[i], topn=len(model.wv.vocab))
		result = pd.DataFrame(similar_words, columns=['f2', 'similarity'])
		result = result[result['f2'].str.startswith(f)]
		result['f1'] = i
		d1.append(result)
	r = pd.concat(d1)
	r = r.drop_duplicates()
	if human_read:
		r['f1'] = r['f1'].apply(lambda x: icd10_map[x.split('_')[1]] if x.split('_')[1] in icd10_map and x.startswith('icd10') else x)
		r['f2'] = r['f2'].apply(lambda x: icd10_map[x.split('_')[1]] if x.split('_')[1] in icd10_map and x.startswith('icd10') else x)
		if len(r[r.f1.str.startswith('drug')]) > 0:
			r['f1'] = r['f1'].apply(lambda x: drug_map[x.split('_')[1]] if x.split('_')[1] in drug_map and x.startswith('drug') else x)
		if len(r[r.f2.str.startswith('drug')]) > 0:
			r['f2'] = r['f2'].apply(lambda x: drug_map[x.split('_')[1]] if x.split('_')[1] in drug_map and x.startswith('drug') else x)
		r['f1'] = r['f1'].apply(age)
		r['f2'] = r['f2'].apply(age)

	r = r.pivot(index='f1', columns='f2', values='similarity').reset_index()
	return r

def save_full_map(n,file,c,human_read=False):
	mpath = path + 'word2vec_model/' + file + '_' + c + '_icd10/' + c + '_icd10_' + str(n)
	if isfile(mpath):
		icd10 = pd.read_csv(path + 'icd10.csv', index_col=0)
		icd10_map = dict(zip(icd10['code'], icd10['cdesc']))
		drug = pd.read_csv(path + 'drugname_dict.csv', index_col=0)
		drug_map = dict(zip(drug['code'], drug['name']))
		model = gensim.models.Word2Vec.load(mpath)
		word_list = [v for v in model.wv.vocab]
		f1 = [x for x in word_list if x.startswith('icd10')]
		f2 = [x for x in word_list if x.startswith(c)]

		r1 = get_pivot(f1, 'icd10', model, icd10_map, drug_map, human_read)
		r2 = get_pivot(f2, c, model, icd10_map, drug_map, human_read)
		r3 = get_pivot(f1, c, model, icd10_map, drug_map, human_read)

		r1.to_csv(path+file+'_'+c+'_icd10_icd10_'+str(n)+'.csv')
		r2.to_csv(path + file+'_'+c + '_'+c+'_'+c+'_' + str(n) + '.csv')
		r3.to_csv(path + file+'_'+c + '_icd10_'+c+'_' + str(n) + '.csv')

#def merge_result_mean(n, txn, icd10, file, row, c):

def test_chain_mean(n):
	remove_file(path + 'word2vec_fresult_mean_' + str(n) + '.csv')
	remove_file(path + 'word2vec_iresult_mean_' + str(n) + '.csv')
	reg = pd.read_csv(path + 'testset/raw/reg.csv', index_col=0)
	drug = pd.read_csv(path + 'testset/raw/dru.csv', index_col=0)
	m_age = pd.read_csv(path + 'reg_age_icd10_age_' + str(n) + '.csv', index_col=1)
	m_sex = pd.read_csv(path + 'reg_sex_icd10_sex_' + str(n) + '.csv', index_col=1)
	m_room = pd.read_csv(path + 'reg_room_icd10_room_' + str(n) + '.csv', index_col=1)
	m_drug = pd.read_csv(path + 'dru_drug_icd10_drug_' + str(n) + '.csv', index_col=1)
	m_model = [m_age,m_sex,m_room]
	m_name = ['age','sex','room']

	for index, row in reg.iterrows():
		txn = row['txn']
		label = row['icd10']
		total = []
		for i in range(len(m_name)):
			name = m_name[i]+'_'+str(row[m_name[i]])
			if name in m_model[i].columns:
				d = m_model[i][[name]]
				d[m_name[i]+'_distribution'] = (d[name]-d[name].mean())/d[name].std()
				total.append(d)

		for name in drug[drug['txn']==txn]['drug'].unique():
			name = 'drug_'+name
			if name in m_drug.columns:
				d = m_drug[[name]]
				d[name+'_distribution'] = (d[name] - d[name].mean()) / d[name].std()
				total.append(d)
		result = pd.concat(total, axis=1, sort=False)
		result['total_distribution'] = 0
		result = result.fillna(0)
		for i in result.columns:
			if i != 'total_distribution' and 'distribution' in i:
				result['total_distribution'] = result['total_distribution'] + result[i]
		result['percent_distribution'] = (result['total_distribution']-result['total_distribution'].min())/(result['total_distribution'].max()-result['total_distribution'].min())

		if 'icd10_'+label in result.index:
			df = pd.DataFrame(result.loc['icd10_'+label])
			df['txn'] = txn
			df ['feature'] = df.index
			df.rename(columns={'icd10_'+label: 'value'}, inplace=True)
			df['icd10'] = label
			df = df.reset_index(drop=True)
			print(df[['txn','icd10','feature','value']])
			save_file(df[['txn','icd10','feature','value']],path + 'word2vec_iresult_mean_' + str(n) + '.csv')
			save_file(pd.DataFrame([[txn,label,result.loc['icd10_'+label]['total_distribution'],result.loc['icd10_'+label]['percent_distribution']]],columns=['txn','icd10','total_distribution','percent_distribution']),path + 'word2vec_fresult_mean_' + str(n) + '.csv')
			#print(pd.DataFrame(result.loc['icd10_'+label].T, columns=['feature','value']))
		else:
			print('icd10_'+label+' not found')


#combine_lab()
#split_data(replace=False)
#test_chain('reg',2500)
#test_chain_mean(4900)
#save_full_map(8100,'reg','age', human_read=True)
#save_full_map(6500,'reg','sex', human_read=True)
#save_full_map(4900,'reg','room')
#save_full_map(8400,'dru','drug', human_read=True)
#vocab_num(50000,'dru','drug')
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
#train_chain('lab_c','lab','icd10')


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

