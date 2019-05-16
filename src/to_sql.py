#Installation guide
#https://www.pythoncentral.io/how-to-install-sqlalchemy/

import sys
import os
from pathlib import Path
sys.path.append(str(Path(os.path.abspath('..')).parent)+'/secret')
import config
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

#Save csv file to sql
folders = ['raw','vec']
filenames = ['reg','lab','dru','rad','adm','ilab','idru','irad']
for folder in folders:
	for filename in filenames:
		csv_to_sqldb(config,folder,filename)


