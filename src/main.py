from preprocessing.getdata import getdata
from preprocessing.getdata import remove_space_data
from preprocessing.onehot import get_total_feature
from preprocessing.onehot import onehot
from preprocessing.trainmodel import train_model
from preprocessing.trainmodel import train_model_onetime
from preprocessing.trainmodel import get_target_class
from preprocessing.trainmodel import get_small_sample

from preprocessing.lab import get_lab_data
from preprocessing.lab import split_lab_data
from preprocessing.lab import clean_lab_data

from preprocessing.demographic import save_demographic_data
from preprocessing.demographic import clean_demographic_data
from preprocessing.demographic import onehot_demographic_data

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
#get_target_class('icd10')
#train_model(feature)
#train_model_onetime('../../secret/data/demographic/demographic_onehot.csv')

#get_lab_data(config)
#split_lab_data()
#clean_lab_data()

#save_demographic_data(config)
#clean_demographic_data()
#onehot_demographic_data()




