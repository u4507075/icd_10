from preprocessing.getdata import getdata
from preprocessing.getdata import remove_space_data
from preprocessing.onehot import get_total_feature
from preprocessing.onehot import onehot
from preprocessing.drug_encode import get_encode_feature
from preprocessing.drug_encode import encode_feature
from preprocessing.trainmodel import train_model

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

from preprocessing.evaluation import predict_testset

from preprocessing.drug_verification import get_drug_verification_registration_data
from preprocessing.drug_verification import get_drug_verification_lab_data


import sys
import os
from pathlib import Path
sys.path.append(str(Path(os.path.abspath('..')).parent)+'/secret')

import config

feature = 'drug'

#getdata(config,'odx',feature)
#getdata(config,'idx',feature)
#remove_space_data(feature)
#get_total_feature(feature)
##onehot(feature)
#get_encode_feature('drug')
#encode_feature('drug')

#get_lab_data(config)
#split_lab_data()
#clean_lab_data()
##tonumeric_lab_data()
#get_encode_lab()
#encode_lab_data()

#save_admit_data(config)
#clean_admit_data()
#onehot_admit_data()

#save_registration_data(config)
#clean_registration_data()
#onehot_registration_data()

#get_txn_test_data(config)
#split_set()
#split_lab()

'''
files = os.listdir('../../secret/data/trainingset/')
for f in files:
	train_model(f.replace('.csv',''))
'''

#predict_testset()




##Drug verification dataset##
#get_drug_verification_registration_data('../../secret/data/registration/registration_onehot.csv')
get_drug_verification_registration_data('../../secret/data/admit/admit_onehot.csv')
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
