import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
import matplotlib.lines as mlines
path = '../../secret/data/raw/'

def visualise_cluster(names):
	data = []
	for name in names:
		df = pd.read_csv(name + '_performance_by_icd10.csv', index_col=0)
		icd10 = pd.read_csv(path + 'icd10.csv', index_col=0)
		icd10_map = dict(zip(icd10['code'], icd10['cdesc']))
		df['icd10_desc'] = df['icd10'].map(icd10_map)
		df['name'] = name
		data.append(df)
	df = pd.concat(data)
	q = 0.999

	upper_n = df['n'].quantile(q)
	print(upper_n)
	df = df[df['n'] < upper_n]
	df = df[df['n'] > 1]
	df = df.dropna()
	df = df.sort_values(by='n', ascending=False)

	p = sns.scatterplot(x=df["n"], y=df["accuracy"], hue=df['name'], size=df['n'], sizes=(1, 400), alpha=.5)

	plt.title('Distribution of model performance on the dataset')
	#plt.gca().get_legend().remove()
	#plt.xlim(0, 1.0)
	plt.ylim(0, 1.0)
	plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0, hspace=0)
	plt.show()
	'''
	data = []
	for name in names:
		df = pd.read_csv(name+'_performance_by_icd10.csv',index_col=0)
		icd10 = pd.read_csv(path + 'icd10.csv', index_col=0)
		icd10_map = dict(zip(icd10['code'],icd10['cdesc']))
		df['icd10_desc'] = df['icd10'].map(icd10_map)
		df['name'] = name
		data.append(df)
	df = pd.concat(data)
	q = 0.999
	
	upper_n = df['n'].quantile(q)
	print(upper_n)
	df = df[df['n']<upper_n]
	df = df[df['n'] > 1]
	df = df.dropna()
	df = df.sort_values(by='n', ascending=False)
	#texts = []
	#x = []
	#y = []
	#p = sns.scatterplot(x=df["accuracy"], y=df["n"], hue=df['name'], size=df['n'], sizes=(1, 400), alpha=.5)
	p = sns.scatterplot(x=df["accuracy"], y=df["f_measure"], hue=df['name'], size=df['n'], sizes=(1, 400), alpha=.5)
	p.set(ylabel='n (< '+str(q)+' quantile)')
	
	i = upper_n + (upper_n*0.03)
	print(df)
	for index,row in df.iterrows():
		if row['accuracy'] > 0.4:
			#x.append(row['accuracy'])
			#y.append(row['n'])
			#texts.append(p.text(row['accuracy'], row['n'], str(row['icd10_desc'])[:50], horizontalalignment='left', size='10', color='black', weight='normal'))
			p.text(0.75, i, str(row['icd10_desc'])[:50], horizontalalignment='left', size='12', color='black', weight='normal')
			plt.plot([row['accuracy'], 0.75],[row['n'], i], color='lightgray', linestyle='dashed', linewidth=1)
			i = i - (upper_n*0.03)
			if i < 0:
				break
	
	plt.title(names[0]+' dataset')
	plt.gca().get_legend().remove()
	plt.xlim(-0.1,1.1)
	#plt.ylim(0, upper_n+(upper_n*0.05))
	plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0, hspace=0)
	#adjust_text(texts)
	#plt.tight_layout()
	plt.show()
	'''
def color_func(word, font_size, position,orientation,random_state=None, **kwargs):
	n = abs(100 - font_size)
	return("hsl(230,60%%, %d%%)" % n)

def visualise_associate_icd10(name):
	df = pd.read_csv(name + '.csv', index_col=0)
	#dru
	#4156	J00	Acute nasopharyngitis [common cold]
	#4637 K30 Functional dyspepsia

	#idru
	#4652	K358	Acute appendicitis, other and unspecified

	#reg
	#2047	E113	Type 2 diabetes mellitus, with ophthalmic complications

	target_icd10 = 2047
	txn = df[df['actual_icd10']==target_icd10]['txn']
	txn = txn.drop_duplicates()
	df = df[df['txn'].isin(txn)]
	result = df[['predicted_icd10', 'sum_weight']].groupby('predicted_icd10').agg({'sum_weight': 'count'}).reset_index()
	result = result[result['predicted_icd10']!=0]
	result = result.sort_values(by=['sum_weight'], ascending=[False])
	result = result.head(50)
	icd10 = pd.read_csv(path + 'icd10.csv', index_col=0)
	icd10_map = dict(zip(icd10.index, icd10['cdesc']))
	result['predicted_icd10'] = result['predicted_icd10'].map(icd10_map)
	result['predicted_icd10'] = result.apply(lambda x: x['predicted_icd10'][:30], axis=1)
	print(result)
	wordcloud = WordCloud(background_color="white", width=2500, height=1100, margin=5)
	wordcloud.generate_from_frequencies(frequencies=dict(zip(result['predicted_icd10'], result['sum_weight'])))
	# change the color setting
	wordcloud.recolor(color_func=color_func)
	plt.figure()
	plt.imshow(wordcloud, interpolation="bilinear")
	plt.axis("off")
	plt.show()

def prepare_data(ax,names, prefix=''):

	data = []
	for name in names:
		df = pd.read_csv(name + '_performance_by_icd10.csv', index_col=0)
		icd10 = pd.read_csv(path + 'icd10.csv', index_col=0)
		icd10_map = dict(zip(icd10['code'], icd10['cdesc']))
		df['icd10_desc'] = df['icd10'].map(icd10_map)
		df['name'] = name
		data.append(df)
	df = pd.concat(data)
	#df['icd10_desc'] = df.apply(lambda x: str(x['icd10_desc']).strip()[:30], axis=1)
	df['accuracy'] = df.apply(lambda x: int(x['accuracy']*100), axis=1)
	# Draw a heatmap with the numeric values in each cell
	data = df[['icd10','name','accuracy']].reset_index().pivot_table(index='icd10', columns='name', values='accuracy', fill_value=0)
	for name in names:
		data[name] = data.apply(lambda x: int(x[name]), axis=1)
	#data = data.reset_index()
	data = data.sort_values(by=prefix+'combine', ascending=False)
	#data = data.head(20)
	print(data)
	data2 = df[['icd10','name','n']].reset_index().pivot_table(index='icd10', columns='name', values='n', fill_value=0)
	#data2 = data2.reset_index()
	#data2 = data2.sort_values(by='combine', ascending=False)
	data2 = data2.rename(columns={prefix+'combine': 'n'})
	print(data2)
	data2 = data2[['n']]
	d = pd.merge(data,data2, left_index=True, right_index=True)
	d = d.sort_values(by='n', ascending=False)
	print(d)
	d.drop(['n'], axis=1, inplace=True)
	d = d.head(50)
	icd10 = pd.read_csv(path + 'icd10.csv', index_col=0)
	icd10['cdesc'] = icd10.apply(lambda x: str(x['cdesc']).strip()[:50], axis=1)
	icd10_map = dict(zip(icd10['code'], icd10['cdesc']))
	icd10_map['N180'] = 'Chronic renal failure'
	icd10_name = d.index.map(icd10_map)
	sns.set(font_scale=0.7)
	d = d[names]
	hm = sns.heatmap(d.T, annot=True, fmt="d", linewidths=.5, ax=ax, square=True, cbar=False, xticklabels=icd10_name,
					 yticklabels=True, cmap="PuBu")
	hm.set_xticklabels(hm.get_xticklabels(), rotation=30, fontsize=10, ha='right')
	plt.subplots_adjust(left=0.01, bottom=0.00, right=0.99, top=1.00, wspace=0, hspace=0)
	ax.set(ylabel='dataset')

def visualise_predicted_icd10():
	'''
	df = pd.read_csv(name + '_performance_by_icd10.csv', index_col=0)
	icd10 = pd.read_csv(path + 'icd10.csv', index_col=0)
	icd10_map = dict(zip(icd10['code'], icd10['cdesc']))
	df['icd10_desc'] = df['icd10'].map(icd10_map)
	df['name'] = name
	q = 0.99

	upper_n = df['n'].quantile(q)
	print(upper_n)
	df = df[df['n'] < upper_n]
	df = df[df['n'] > 1]
	df = df.dropna()
	df = df.sort_values(by='n', ascending=False)
	df['icd10_desc'] = df.apply(lambda x: x['icd10_desc'][:30], axis=1)

	wordcloud = WordCloud(background_color="white", width=2500, height=1100, margin=5)
	wordcloud.generate_from_frequencies(frequencies=dict(zip(df['icd10_desc'], df['accuracy'])))
	# change the color setting
	wordcloud.recolor(color_func=color_func)
	plt.figure()
	plt.imshow(wordcloud, interpolation="bilinear")
	plt.axis("off")
	plt.show()
	'''


	fig, (ax1, ax2) = plt.subplots(nrows=2)
	prepare_data(ax1, ['reg','dru','rad','lab','combine'], prefix='')
	prepare_data(ax2, ['adm','idru','irad','ilab','icombine'], prefix='i')
	plt.tight_layout()
	plt.show()

	'''
	names = ['10','300','15000']
	data = []
	for name in names:
		df = pd.read_csv('dru_performance_by_icd10_'+name+'.csv', index_col=0)
		icd10 = pd.read_csv(path + 'icd10.csv', index_col=0)
		icd10_map = dict(zip(icd10['code'], icd10['cdesc']))
		df['icd10_desc'] = df['icd10'].map(icd10_map)
		df['name'] = name
		data.append(df)
	df = pd.concat(data)
	print(df)
	'''