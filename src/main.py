from preprocessing.trainmodel import scale_data
#from preprocessing.trainmodel import dask_model
#from preprocessing.trainmodel import eval_model
#from preprocessing.trainmodel import lstm_model
#from preprocessing.trainmodel import evaluate_lstm_model
from preprocessing.trainmodel import kmean
#from preprocessing.trainmodel import predict_kmean
from preprocessing.trainmodel import get_neighbour
from preprocessing.trainmodel import get_weight
from preprocessing.trainmodel import birch_train
from preprocessing.trainmodel import predict_cluster
from preprocessing.trainmodel import predict_icd10
#from preprocessing.trainmodel import train_had
#from preprocessing.trainmodel import eval_had
from preprocessing.trainmodel import birch_finetune
from preprocessing.trainmodel import kmean_finetune
#from preprocessing.trainmodel import train_lgb
#from preprocessing.trainmodel import train_xgb
from preprocessing.trainmodel import validate
from preprocessing.trainmodel import combine_prediction

from preprocessing.preprocess import get_icd10_data
from preprocessing.preprocess import get_adm_data
from preprocessing.preprocess import get_reg_data
from preprocessing.preprocess import get_drug_data
from preprocessing.preprocess import get_lab_data
from preprocessing.preprocess import get_rad_data
from preprocessing.preprocess import get_txn_test_data
from preprocessing.preprocess import split_data
from preprocessing.preprocess import csv_to_sqldb

#from preprocessing.vec import word_to_vec
#from preprocessing.vec import radio_to_vec

import sys
import os
from pathlib import Path
sys.path.append(str(Path(os.path.abspath('..')).parent)+'/secret')

import config


#get_icd10_data(config)
#get_adm_data(config)
#get_reg_data(config)
#get_drug_data(config)
#get_lab_data(config)
#get_rad_data(config)
#get_txn_test_data(config)

#word_to_vec('adm')
#word_to_vec('reg')
#word_to_vec('dru')
#word_to_vec('idru')
#word_to_vec('lab')
#word_to_vec('ilab')
#radio_to_vec('rad')
#radio_to_vec('irad')

#split_data('raw')
#split_data('vec')

'''
scale_data('../../secret/data/vec/','reg')
scale_data('../../secret/data/vec/','adm')
scale_data('../../secret/data/vec/','rad')
scale_data('../../secret/data/vec/','irad')
scale_data('../../secret/data/vec/','dru')
scale_data('../../secret/data/vec/','idru')
scale_data('../../secret/data/vec/','lab')
scale_data('../../secret/data/vec/','ilab')
'''


#kmean(['dru','idru'],'drug')
#get_neighbour(['dru','idru'],'drug_kmean_15000')
#get_weight('drug_kmean_15000')
#predict_cluster(['dru','idru'],'drug_kmean_15000')
#predict_icd10(['dru','idru'],'drug_kmean_15000')
#validate(['dru','idru'],'drug_kmean_15000')

#kmean(['reg','adm'],'reg')
#get_neighbour(['reg','adm'],'reg_kmean_15000')
#get_weight('reg_kmean_15000')
#predict_cluster(['reg','adm'],'reg_kmean_15000')
#validate(['reg','adm'],'reg_kmean_15000')

#kmean(['lab','ilab'],'lab')
#get_neighbour(['lab','ilab'],'lab_kmean_15000')
#get_weight('lab_kmean_15000')
#predict_cluster(['lab','ilab'],'lab_kmean_15000')
#validate(['lab','ilab'],'lab_kmean_15000')

#kmean(['rad','irad'],'rad')
#get_neighbour(['rad','irad'],'rad_kmean_15000')
#get_weight('rad_kmean_15000')
#predict_cluster(['rad','irad'],'rad_kmean_15000')
#validate(['rad','irad'],'rad_kmean_15000')


#combine_prediction(['dru_drug_kmean_15000','idru_drug_kmean_15000','reg_reg_kmean_15000','adm_reg_kmean_15000','rad_rad_kmean_15000','irad_rad_kmean_15000'],'kmean_15000')
#validate(['reg'],'combined_kmean_15000',combine=True)

#birch_finetune(['dru','idru'],0.1)
#birch_finetune(['reg','adm'],3)
#birch_finetune(['lab','ilab'],2)
#birch_finetune(['rad','irad'],10)

#birch_train(['reg','adm'],'reg_birch',3.5)
birch_train(['dru','idru'],'drug_birch',0.35)
##birch_train(['rad','irad'],'rad_birch',10.0)
#birch_train(['lab','ilab'],'lab_birch',1.25)

#get_neighbour(['reg','adm'],'reg_birch')
#get_neighbour(['dru','idru'],'drug_birch')
##get_neighbour(['rad','irad'],'rad_birch')
##get_neighbour(['lab','ilab'],'lab_birch')

#get_weight('reg_birch')
#get_weight('drug_birch')
##get_weight('rad_birch')
##get_weight('lab_birch')

#predict_cluster(['reg','adm'],'reg_birch')
#predict_cluster(['dru','idru'],'drug_birch')

#validate(['reg','adm'],'reg_birch')
#validate(['dru','idru'],'drug_birch')


#birch_train(['dru','idru'],'drug_birch')
#for i in [0.1,0.25,0.5,0.75,1.00]:
#	get_neighbour(['dru','idru'],'drug_birch_'+str(i))
#	get_weight('drug_birch_'+str(i))
#	predict_cluster(['dru','idru'],'drug_birch_'+str(i))
#	predict_icd10(['dru','idru'])

#birch_train(['reg','adm'],'reg_birch')
#for i in [0.1,0.25,0.5,0.75,1.00]:
#	get_neighbour(['reg','adm'],'reg_birch_'+str(i))
#	get_weight('reg_birch_'+str(i))
#	predict_cluster(['reg','adm'],'reg_birch_'+str(i))
#	predict_icd10(['reg','adm'])

#birch_train(['lab','ilab'],'lab_birch')
#for i in [0.75,1.0,1.25,1.50,1.75]:
#	get_neighbour(['lab','ilab'],'lab_birch_'+str(i))
#	get_weight('lab_birch_'+str(i))
#	predict_cluster(['lab','ilab'],'lab_birch_'+str(i))

#dask_model(['dru','idru'],'drug')
#dask_model(['reg','adm'],'reg')
#dask_model(['lab','ilab'],'lab')
#dask_model(['rad','irad'],'rad')

#eval_model('reg')
#eval_model('adm')
#eval_model('dru')

#lstm_model('reg',11)
#lstm_model('rad',7)
#lstm_model('adm',12)
#lstm_model('idru',2)
#evaluate_lstm_model('rad')
#evaluate_lstm_model('adm')
#evaluate_lstm_model('idru')

#train_had()
#eval_had('dru')

#train_xgb(['rad','ilab'])

#train_xgb(['reg','adm'])

'''

import pandas as pd
for df in pd.read_csv('../../secret/data/testset/raw/lab.csv', chunksize=100, index_col=0):
	print(df)
	break
'''









