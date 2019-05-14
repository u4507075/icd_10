from preprocessing.getdata import getdata
from preprocessing.getdata import getidata
from preprocessing.onehot import get_total_feature
from preprocessing.onehot import onehot
from preprocessing.drug_encode import get_encode_feature
from preprocessing.drug_encode import encode_feature
from preprocessing.trainmodel import scale_data
from preprocessing.trainmodel import train_model2
#from preprocessing.trainmodel import save_history

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

from preprocessing.vec import word_to_vec
from preprocessing.vec import radio_to_vec

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
##word_to_vec('ilab')
#radio_to_vec('rad')
##radio_to_vec('irad')

split_data('raw')
#split_data('vec')

#scale_data('../../secret/data/vec/','rad')
#scale_data('../../secret/data/vec/','adm')
#scale_data('../../secret/data/vec/','idru')
#train_model2('rad',7)
#train_model2('adm',12)
#train_model2('idru',2)










### Drug ###
##getdata(config)
##getidata(config)
##get_total_feature('dru')
##get_total_feature('idru')
##get_encode_feature('dru')
##get_encode_feature('idru')
##encode_feature('dru')
##encode_feature('idru')

### Lab ###
##get_lab_data(config)
##split_lab_data('lab')
##split_lab_data('ilab')
##clean_lab_data('lab')
##clean_lab_data('ilab')
##get_encode_lab('lab')
##get_encode_lab('ilab')
##encode_lab_data('lab')
#!encode_lab_data('ilab')

##save_admit_data(config)
##clean_admit_data()
##onehot_admit_data()

##save_registration_data(config)
##clean_registration_data()
##onehot_registration_data()

#get_txn_test_data(config)
#split_set()
#split_lab()
#clean_data('trainingset','trainingset_clean')
#clean_data('testset','testset_clean')

'''
files = os.listdir('../../secret/data/trainingset/')
for f in files:
	train_model(f.replace('.csv',''))
'''

#get_validation_data(config)
#predict_testset()

#preprocess_radio_data()



##Drug verification dataset##
#get_drug_verification_registration_data('../../secret/data/registration/registration_onehot.csv')
#get_drug_verification_registration_data('../../secret/data/admit/admit_onehot.csv')
#get_drug_verification_lab_data()


'''
import pandas as pd

files = os.listdir('../../secret/data/lab/clean/')
for lab in files:
	n = lab.replace('.csv','')
	print('#### '+n)
	print('')
	print('Laboratory findings of '+n+'.')
	print('')
	print('<details><summary>lab metadata</summary>')
	print('<p>')
	print('')
	print('| Features | Types | Description |')
	print('| :--- | :--- | :--- |')
	print('| TXN | numeric | key identification for a patient visit |')
	
	p = '../../secret/data/lab/clean/'+lab
	for df in  pd.read_csv(p, chunksize=10, index_col=0):
		for i in df.columns:
			if i != 'TXN' and i != 'icd10' and i != 'Unnamed: 0.1':
				print('| '+i+' | binary |  |')
		break
	print('| icd10 | text | ICD-10 code (diagnosis) |')
	print('</p>')
	print('</details>')
	print('')
'''
