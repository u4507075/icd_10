import pandas as pd


def csv_to_sqldb(config,folder,filename):
	for df in  pd.read_csv('../../secret/data/'+folder+'/'+filename+'.csv', chunksize=100000, index_col=0):
		df.to_sql(	folder+'_'+name,
						get_connection(config),
						if_exists='append',
						index=False,
						method='multi'				
					)

		print('Append table '+folder+'_'+name)



















