from preprocessing.getdata import getdata
from preprocessing.getdata import remove_space_data
from preprocessing.onehot import get_total_feature
from preprocessing.onehot import onehot
from preprocessing.trainmodel import train_model

from preprocessing.lab import get_lab_data
from preprocessing.lab import split_lab_data

import sys
import os
from pathlib import Path
sys.path.append(str(Path(os.path.abspath('..')).parent)+'/secret')

import config

feature = 'drug'

#getdata(config,'drug_opd',feature)
#getdata(config,'drug_ipd',feature)
#remove_space_data(feature)
#get_total_feature(feature)
#onehot(feature)
#train_model(feature)

#get_lab_data(config)
split_lab_data()
