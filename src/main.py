from db.database import getdata
import sys
import os
from pathlib import Path
sys.path.append(str(Path(os.path.abspath('..')).parent)+'/secret')

import config

getdata(config)
