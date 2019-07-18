from preprocessing.getdata import getdata
from preprocessing.getdata import getidata
from preprocessing.onehot import get_total_feature
from preprocessing.onehot import onehot
from preprocessing.drug_encode import get_encode_feature
from preprocessing.drug_encode import encode_feature
from preprocessing.trainmodel import scale_data
from preprocessing.trainmodel import dask_model
from preprocessing.trainmodel import eval_model
from preprocessing.trainmodel import lstm_model
from preprocessing.trainmodel import evaluate_lstm_model
from preprocessing.trainmodel import kmean
from preprocessing.trainmodel import predict_kmean
from preprocessing.trainmodel import get_neighbour
from preprocessing.trainmodel import get_weight
from preprocessing.trainmodel import birch_train
from preprocessing.trainmodel import birch_test
from preprocessing.trainmodel import birch_predict
from preprocessing.trainmodel import train_had
from preprocessing.trainmodel import eval_had
from preprocessing.trainmodel import birch_finetune

from preprocessing.lab import get_lab_data
from preprocessing.lab import split_lab_data
from preprocessing.lab import clean_lab_data
from preprocessing.lab import tonumeric_lab_data
from preprocessing.lab import get_encode_lab
from preprocessing.lab import encode_lab_data

from preprocessing.admit import save_admit_data
from preprocessing.admit import clean_admit_data
from preprocessing.admit import onehot_admit_data

from preprocessing.registration import save_registration_data
from preprocessing.registration import clean_registration_data
from preprocessing.registration import onehot_registration_data

from preprocessing.split import get_txn_test_data
from preprocessing.split import split_set
from preprocessing.split import split_lab
from preprocessing.split import clean_data



from preprocessing.drug_verification import get_drug_verification_registration_data
from preprocessing.drug_verification import get_drug_verification_lab_data

from validation.validation import predict_testset

from validation.testset import get_validation_data




from preprocessing.text import get_icd10_data
from preprocessing.text import get_adm_data
from preprocessing.text import get_reg_data
from preprocessing.text import get_drug_data
from preprocessing.text import get_lab_data
from preprocessing.text import get_rad_data
from preprocessing.text import get_txn_test_data
from preprocessing.text import split_data
from preprocessing.text import csv_to_sqldb

#from preprocessing.vec import word_to_vec
#from preprocessing.vec import radio_to_vec

import sys
import os
from pathlib import Path
sys.path.append(str(Path(os.path.abspath('..')).parent)+'/secret')

import config


#get_icd10_data(config)
get_adm_data(config)
get_reg_data(config)
get_drug_data(config)
get_lab_data(config)
get_rad_data(config)
#get_txn_test_data(config)

#word_to_vec('adm')
#word_to_vec('reg')
#word_to_vec('dru')
#word_to_vec('idru')
#word_to_vec('lab')
#word_to_vec('ilab')
#radio_to_vec('rad')
##radio_to_vec('irad')

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

###birch_finetune(['dru'])
#kmean('reg')
#kmean(['dru','idru'],'drug')
#predict_kmean('reg')
#kmean(['dru','idru'],'drug')
#predict_kmean('dru','drug')
#get_neighbour(['dru','idru'],'drug_birch')

#birch_train(['dru','idru'],'drug_birch',None,[0.1,0.25,0.5,0.75,1.00])
#for i in [0.1,0.25,0.5,0.75,1.00]:
#	get_neighbour(['dru','idru'],'drug_birch_'+str(i))
#	get_weight('drug_birch_'+str(i))
#	birch_test(['dru','idru'],'drug_birch_'+str(i))
#	birch_predict(['dru','idru'])

#birch_train(['reg','adm'],'reg_birch',None,[5,6,7,8,9])
#for i in [5,6,7,8,9]:
#	get_neighbour(['reg','adm'],'reg_birch_'+str(i))
#	get_weight('reg_birch_'+str(i))
#	birch_test(['reg','adm'],'reg_birch_'+str(i))
#	birch_predict(['reg','adm'])

#birch_train(['lab','ilab'],'lab_birch',None,[0.75,1.0,1.25,1.50,1.75])
#for i in [0.75,1.0,1.25,1.50,1.75]:
#	get_neighbour(['lab','ilab'],'lab_birch_'+str(i))
#	get_weight('lab_birch_'+str(i))
#	birch_test(['lab','ilab'],'lab_birch_'+str(i))

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




















