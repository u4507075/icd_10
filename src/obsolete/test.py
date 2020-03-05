import pandas as pd
import numpy as np

def get_data(name):
	chunk = 1000000
	total = []
	txn = pd.read_csv('../../../secret/data/raw/'+name+'_txn_2018.csv')
	for df in  pd.read_csv('../../../secret/data/raw/'+name+'.csv', chunksize=chunk, index_col=0):
		df = df[df['txn'].isin(txn['txn'])]
		print(len(df))
		if len(df) > 0:
			total.append(df)
	df = pd.concat(total)
	df.to_csv(name+'_full.csv')
	print('save data')

def save(name,filename,df):
	chunk = 100000
	t = []
	for d in  pd.read_csv('../../../secret/data/raw/'+name+'.csv', chunksize=chunk, index_col=0):
		d = d[['txn','sex','age']]
		d = d.drop_duplicates()
		d = d[d['txn'].isin(df.txn)]
		if len(d)>0:
			data = pd.merge(left=d,right=df, left_on='txn', right_on='txn')
			t.append(data)
		print(len(d))
	total = pd.concat(t)
	print(total)
	total.to_csv(filename+'_full.csv')
	print('save data')

def merge(df,name):
	had = pd.read_csv('had.csv', index_col=0)
	had_dict = dict(zip(had['drug'], had['drug_group']))
	subhad_dict = dict(zip(had['drug'], had['drug_subgroup']))
	df['drug_group'] = df['drug'].apply(lambda x: had_dict[x] if x in had_dict else 'non_had')
	df['drug_subgroup'] = df['drug'].apply(lambda x: subhad_dict[x] if x in subhad_dict else 'non_had')
	df['actual_had'] = df['drug'].apply(lambda x: 1 if x in had_dict else 0)
	#print(df[df['actual_had']==1])
	df.to_csv(name+'_had_full.csv')
	print('save data')

from sklearn import preprocessing
'''
import spacy
import spacy.cli
spacy.cli.download("en_core_web_md")
nlp = spacy.load('en_core_web_md', disable=["tagger","parser", "ner"])
def to_vec(x):
	if x == 0 or x == '':
		return 0
	else:
		value = sum(nlp(str(x)).vector)
		if value == 0:
			return sum(sum([nlp(v).vector for v in str(x)]))
		else:
			return value
'''
def train(df,t):
  X = df.drop(columns=[t])
  y = df[t]

  from sklearn import tree
  from sklearn import svm
  from sklearn import linear_model
  from sklearn.neighbors import KNeighborsRegressor
  from sklearn.gaussian_process import GaussianProcessRegressor
  from sklearn.naive_bayes import GaussianNB
  from sklearn.ensemble import GradientBoostingRegressor
  from sklearn.ensemble import RandomForestRegressor
  from sklearn.linear_model import LinearRegression
  #0.56
  #clf = tree.DecisionTreeRegressor()
  #0.04
  #clf = svm.SVR()
  #0.37
  #clf = linear_model.SGDRegressor(max_iter=1000, tol=1e-3)
  #0.39
  #clf = KNeighborsRegressor(n_neighbors=2)
  #0.13
  #clf = GaussianProcessRegressor()
  #0.53
  #clf = GaussianNB()
  #0.58
  clf = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                    max_depth=5, random_state=0, loss='ls')
  #0.35
  #clf = RandomForestRegressor(random_state=0, n_estimators=100, max_depth=5)

  #-inf
  #clf = LinearRegression()

  from sklearn.model_selection import cross_val_score
  scores = cross_val_score(clf, X, y, cv=10)
  print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

#get_data('dru')
#get_data('idru')
#save('reg','reg',pd.read_csv('dru_full.csv',index_col=0))
#save('adm','adm',pd.read_csv('idru_full.csv',index_col=0))
#merge(pd.read_csv('reg_full.csv',index_col=0),'dru')
#merge(pd.read_csv('adm_full.csv',index_col=0),'idru')

'''
t = 'actual_had'
df = pd.read_csv('dru_num_full.csv',index_col=0)
print(df)
df = df.drop(columns=['non_actual_had'])
train(df,t)
'''
'''
def get_data_group(name):
	chunk = 1200000
	for df in  pd.read_csv('/media/bon/My Passport/data/raw/'+name+'.csv', chunksize=chunk, index_col=0):
		df[200000:]
		df.to_csv(name+'_group.csv')
		break

get_data_group('dru')
get_data_group('idru')
save('reg','reg_group',pd.read_csv('dru_group.csv',index_col=0))
save('adm','adm_group',pd.read_csv('idru_group.csv',index_col=0))
merge(pd.read_csv('reg_group_full.csv',index_col=0),'dru_group')
merge(pd.read_csv('adm_group_full.csv',index_col=0),'idru_group')
'''

def get_data_group(name,chunk,start,filename):
	for df in  pd.read_csv(name+'.csv', chunksize=chunk, index_col=0):
		df = df[start:]
		df.to_csv(filename+'.csv')
		print('save data')
		break

#get_data_group('dru_had_full',200000,0,'dru_had_binary')
#get_data_group('idru_had_full',400000,0,'idru_had_binary')
#get_data_group('dru_had_full',1000000,200000,'dru_had_group')
get_data_group('idru_had_full',1200000,400000,'idru_had_group')
