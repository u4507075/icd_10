import pandas as pd
from pathlib import Path
import numpy as np
import ntpath
import os
import pickle
import re
from random import randint

def read(path):
	with open(path, "r", encoding='utf-8') as f:
		text= f.read()
		return text

def write(path,text):
	with open(path, "w") as file:
		file.write(text)

def predict():
	body = read('html/body.html')
	items = ''
	for i in range(randint(3,10)):
		item = read('html/item.html')
		values = ''
		for j in range(randint(3,20)):
			value = read('html/value.html')
			value = value.replace('%ITEM', str(randint(3000,200000)))
			value = value.replace('%VALUE', str(randint(0,200)))
			values = values + value
		item = item.replace('%VALUE', values)
		items = items + item
	body = body.replace('%ITEM', items)
	write('result/test.html',body)














