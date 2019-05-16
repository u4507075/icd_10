#Installation guide
#https://www.pythoncentral.io/how-to-install-sqlalchemy/

from preprocessing.to_sql import csv_to_sqldb
import sys
import os
from pathlib import Path
sys.path.append(str(Path(os.path.abspath('..')).parent)+'/secret')

import config
#Save csv file to sql
folders = ['raw','vec']
filenames = ['reg','lab','dru','rad','adm','ilab','idru','irad']
for folder in folders:
	for filename in filenames:
		csv_to_sqldb(config,folder,filename)


