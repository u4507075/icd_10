import pandas as pd
import numpy as np

def getdata():

	radio = pd.read_csv('../../../secret/data/radio_result.csv')
	print(radio)

	icd10 = pd.read_csv('../../../secret/data/icd10_2010.csv')
	print(icd10)
	

getdata()
