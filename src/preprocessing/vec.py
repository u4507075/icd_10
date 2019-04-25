import mysql.connector as sql
import pandas as pd
from pathlib import Path
import numpy as np
import os
import re
import spacy

nlp = spacy.load('en_core_web_md')
path = '../../secret/data/raw/'


def word_to_vec():
	print(sum(nlp('dog').vector))
	print(sum(nlp('dogs').vector))
	print(sum(nlp('cat').vector))
	print(sum(nlp('dog cat').vector))
	print(sum(nlp('1').vector))
