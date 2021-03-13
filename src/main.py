#from preprocessing.trainmodel import scale_data
#from preprocessing.trainmodel import dask_model
#from preprocessing.trainmodel import eval_model
#from preprocessing.trainmodel import lstm_model
#from preprocessing.trainmodel import evaluate_lstm_model
'''
from preprocessing.trainmodel import kmean
from preprocessing.trainmodel import eval_kmean
from preprocessing.trainmodel import get_neighbour
from preprocessing.trainmodel import get_weight
from preprocessing.trainmodel import get_total_weight
from preprocessing.trainmodel import birch_train
from preprocessing.trainmodel import predict_cluster
from preprocessing.trainmodel import predict_icd10
from preprocessing.trainmodel import train_had
from preprocessing.trainmodel import eval_had
from preprocessing.trainmodel import birch_finetune
from preprocessing.trainmodel import kmean_finetune
#from preprocessing.trainmodel import train_lgb
#from preprocessing.trainmodel import train_xgb
from preprocessing.trainmodel import cluster_validate
from preprocessing.trainmodel import bag_validate
from preprocessing.trainmodel import bag_combine_validate
from preprocessing.trainmodel import bag_evaluation
from preprocessing.trainmodel import bag_performance
from preprocessing.trainmodel import bag_performance_icd10
from preprocessing.trainmodel import bag_combine_performance_icd10
from preprocessing.trainmodel import combine_prediction
from preprocessing.trainmodel import get_total_icd10_weight
from preprocessing.trainmodel import apply_pca
from preprocessing.trainmodel import test_icd10_prediction
from preprocessing.trainmodel import create_gradientboosting_group

from visualisation.visualisation import visualise_cluster
from visualisation.visualisation import visualise_associate_icd10
from visualisation.visualisation import visualise_predicted_icd10
'''
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
scale_data('reg')
scale_data('adm')
scale_data('rad')
scale_data('irad')
scale_data('dru')
scale_data('idru')
scale_data('lab')
scale_data('ilab')
'''

#apply_pca('reg')
#apply_pca('adm')
#apply_pca('rad')
#apply_pca('irad')
#apply_pca('dru')
#apply_pca('idru')
#apply_pca('lab')
#apply_pca('ilab')


n = 15000
#kmean(n,['dru','idru'],'drug')
#kmean(n,['reg','adm'],'reg')
#kmean(n,['rad','irad'],'rad')
#kmean(n,['lab','ilab'],'lab')

#get_total_weight(n,['dru','idru'],'drug',inplace=False)
#get_total_weight(n,['reg','adm'],'reg',inplace=False)
#get_total_weight(n,['rad','irad'],'rad',inplace=False)
#get_total_weight(n,['lab','ilab'],'lab',inplace=False)

#total_validate(n,['dru','idru'],'drug')
#total_validate(n,['reg','adm'],'reg')
#total_validate(n,['rad','irad'],'rad')
#total_validate(n,['lab','ilab'],'lab')

def validate_kmean(n,files,name):
	kmean(n,files,name)
	get_total_weight(n,files,name,inplace=True)
	#total_validate(n,files,name)
'''
for i in range(2,10000):
	print(i)
	kmean(i,['dru','idru'],'drug')
	kmean(i,['reg','adm'],'reg')
	kmean(i,['rad','irad'],'rad')
	kmean(i,['lab','ilab'],'lab')
'''

'''
print('dru')
eval_kmean('drug','dru',2,300,10)
print('reg')
eval_kmean('reg','reg',2,300,10)
print('rad')
eval_kmean('rad','rad',2,300,10)
print('lab')
eval_kmean('lab','lab',2,300,10)
'''
#eval_kmean('drug','dru',15000,15001,1)

#5,10,100,1000,5000,10000,15000
#validate_kmean(1000,['dru','idru'],'drug')
#validate_kmean(5000,['reg','adm'],'reg')
#validate_kmean(100,['rad','irad'],'rad')
#validate_kmean(10,['lab','ilab'],'lab')

#cluster_validate(15000,['dru'],'drug')
#cluster_validate(5,['reg','adm'],'reg')

#get_total_icd10_weight(5,'drug')

#data = ['reg','adm']
#data = ['rad','irad']
#data = ['dru','idru']
data = ['ilab']
#name = 'reg'
#name = 'rad'
#name = 'drug'
name = 'lab'
#name = 'drug'

num = 15000
#kmean(num,data,name)
#get_neighbour(data,name+'_kmean_'+str(num))
#get_weight(name+'_kmean_'+str(num))
#predict_cluster(data,name+'_kmean_'+str(num))
#optional step:  predict icd10
#predict_icd10(['dru','idru'],'drug_kmean_15000')
#optional step: onehot icd10 prediction
#test_icd10_prediction('dru','drug_kmean_10')
#bag_validate(data,name+'_kmean_'+str(num))

#bag_combine_validate(15000)
#bag_combine_validate(15000,prefix='i')

#bag_evaluation('reg_reg_kmean_15000_validation_total')
#bag_evaluation('adm_reg_kmean_15000_validation_total')
#bag_evaluation('rad_rad_kmean_15000_validation_total')
#bag_evaluation('irad_rad_kmean_15000_validation_total')
#bag_evaluation('dru_drug_kmean_15000_validation_total')
#bag_evaluation('idru_drug_kmean_15000_validation_total')
#bag_evaluation('lab_lab_kmean_15000_validation_total')
#bag_evaluation('ilab_lab_kmean_15000_validation_total')
#bag_evaluation('combine_15000_validation')
#bag_evaluation('icombine_15000_validation')

#bag_performance('reg_reg_kmean_15000_validation_total')
#bag_performance('adm_reg_kmean_15000_validation_total')
#bag_performance('rad_rad_kmean_15000_validation_total')
#bag_performance('irad_rad_kmean_15000_validation_total')
#bag_performance('dru_drug_kmean_15000_validation_total')
#bag_performance('idru_drug_kmean_15000_validation_total')
#bag_performance('lab_lab_kmean_15000_validation_total')
#bag_performance('ilab_lab_kmean_15000_validation_total')
#bag_performance('combine_15000_validation')
#bag_performance('icombine_15000_validation')

#bag_performance_icd10('reg','reg')
#bag_performance_icd10('adm','reg')
#bag_performance_icd10('rad','rad')
#bag_performance_icd10('irad','rad')
#bag_performance_icd10('dru','drug')
#bag_performance_icd10('idru','drug')
#bag_performance_icd10('lab','lab')
#bag_performance_icd10('ilab','lab')
#bag_combine_performance_icd10()
#bag_combine_performance_icd10(prefix='i')

#visualise_cluster(['reg','adm','rad','irad','dru','idru'])
name = ['reg','adm','rad','irad','dru','idru','combine']
#for n in name:
#	visualise_cluster([n])
#visualise_cluster(['adm','idru','irad','icombine'])
#visualise_cluster(['reg','dru','rad','lab','combine'])
#visualise_associate_icd10('reg_reg_kmean_15000_validation_total')
#visualise_predicted_icd10('combine')
#visualise_predicted_icd10()

'''
d = ['reg','adm','dru','idru','lab','ilab','rad','irad']
import pandas as pd
for i in d:
	n = 0
	for df in pd.read_csv('../../secret/data/testset/vec/'+i+'.csv', chunksize=100000):
		n = n+len(df)
	print(i)
	print(n)
'''



#create_gradientboosting_group('dru')

#combine_prediction(['dru_drug_kmean_15000','idru_drug_kmean_15000','reg_reg_kmean_15000','adm_reg_kmean_15000','rad_rad_kmean_15000','irad_rad_kmean_15000'],'kmean_15000')
#validate(['reg'],'combined_kmean_15000',combine=True)

#birch_finetune(['dru','idru'],0.1)
#birch_finetune(['reg','adm'],3)
#birch_finetune(['lab','ilab'],2)
#birch_finetune(['rad','irad'],10)

#birch_train(['reg','adm'],'reg_birch',3.5)
#birch_train(['dru','idru'],'drug_birch',0.35)
#birch_train(['rad','irad'],'rad_birch',15.0)
#birch_train(['lab','ilab'],'lab_birch',1.25)

#get_neighbour(['reg','adm'],'reg_birch')
#get_neighbour(['dru','idru'],'drug_birch')
#get_neighbour(['rad','irad'],'rad_birch')
#get_neighbour(['lab','ilab'],'lab_birch')

#get_weight('reg_birch')
#get_weight('drug_birch')
#get_weight('rad_birch')
##get_weight('lab_birch')

#predict_cluster(['reg','adm'],'reg_birch')
#predict_cluster(['dru','idru'],'drug_birch')
#predict_cluster(['rad','irad'],'rad_birch')

#validate(['reg','adm'],'reg_birch')
#validate(['dru','idru'],'drug_birch')
#validate(['rad','irad'],'rad_birch')

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
#eval_had('idru')

#train_xgb(['rad','ilab'])

#train_xgb(['reg','adm'])


'''
import pandas as pd
import numpy as np
mypath = '../../secret/data/'
had = pd.read_csv(mypath+'had.csv', index_col=0)
had = had['drug'].values.tolist()

for df in pd.read_csv('../../secret/data/trainingset/raw/idru.csv', chunksize=100000, index_col=0):
	df = df[df['icd10']=='S012']
	if len(df) > 0:
		df['had'] = np.where(df['drug'].isin(had), 1, 0)
		df = df[df['had'] == 1]
		if len(df) > 0:
			print(df)

'''








