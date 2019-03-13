from db.database import getdata
from db.database import cleandata
from db.database import readdata
from db.database import onehotdrug
from db.database import mapdata
from db.database import train_model
import sys
import os
from pathlib import Path
sys.path.append(str(Path(os.path.abspath('..')).parent)+'/secret')

import config

#getdata(config)
#cleandata()
#readdata()
#onehotdrug()
#mapdata()
train_model()

