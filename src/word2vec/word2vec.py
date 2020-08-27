import mysql.connector as mysql
import pandas as pd
from pathlib import Path
import numpy as np
import os
import re
import gensim

path = '../../../secret/data/'

def chain(max,df,n1,n2):
        toggle = True
        d = df.sample(n=1)
        # d = df.sample(n=1,weights=df[n1+'_weight'])
        last = d[n1].iloc[0]
        text = []
        n = 0
        for i in range(max):
                if toggle:
                        text.append([last, last])
                        x = [last]
                        y = [last]
                        icd10 = df[df[n1] == last].sample(n=1)[n2].iloc[0]
                        x.insert(0, icd10)
                        last = df[df[n2] == icd10].sample(n=1)[n1].iloc[0]
                        y.insert(0, last)
                        text.append(x)
                        text.append(y)
                        last = icd10
                        toggle = False
                else:
                        text.append([last, last])
                        x = [last]
                        y = [last]
                        drug = df[df[n2] == last].sample(n=1)[n1].iloc[0]
                        x.insert(0, drug)
                        last = df[df[n1] == drug].sample(n=1)[n2].iloc[0]
                        y.insert(0, last)
                        text.append(x)
                        text.append(y)
                        last = drug
                        toggle = True

        return text

def train_chain(filename,f1,f2):
	modelpath = path+'word2vec_model/'+f1+'_'+f2
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

#reg
#,txn,sex,age,wt,pulse,resp,temp,blood,rh,room,room_dc,dx_type,sbp,dbp,icd10
#train_chain('reg','sex','icd10')
train_chain('reg','age','icd10')