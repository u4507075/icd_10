import pandas as pd
import numpy as np

def getdata():

	result = pd.read_csv('../../secret/data/radio/radio_icd10.csv')
	print(result)
