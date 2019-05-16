import pandas as pd
from sqlalchemy import create_engine

def csv_to_sqldb(config,folder,filename):
	engine = sqlite3.connect('mysql://'+config.DATABASE_CONFIG['user']+':'+config.DATABASE_CONFIG['password']+'@localhost:'+str(config.DATABASE_CONFIG['port'])+'/icd10')
	connection = engine.connect()
	for df in  pd.read_csv('../../secret/data/'+folder+'/'+filename+'.csv', chunksize=100000, index_col=0):
		df.to_sql(	folder+'_'+filename,
				connection,
				if_exists='append',
				index=False,
				method='multi'
					)

		print('Append table '+folder+'_'+filename)

	connection.close()




















