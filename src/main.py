from db.getdata import getdata
from db.getdata import remove_space_data
from db.onehot import get_total_feature
from db.onehot import onehot
from db.trainmodel import train_model
import sys
import os
from pathlib import Path
sys.path.append(str(Path(os.path.abspath('..')).parent)+'/secret')

import config

feature = 'drug'

#getdata(config,feature)
#remove_space_data(feature)
#get_total_feature(feature)
#onehot(feature)
train_model(feature)

