import textdistance
import re
import nltk
#nltk.download('punkt') #fist time only
#nltk.download('averaged_perceptron_tagger')
#nltk.download()
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
from fuzzywuzzy import fuzz
# def getdata():
df = pd.read_csv('../../../secret/data/radio/radio_icd10.csv')
a = df['REP'][19].lower()
b = df['cdesc'][19].lower()
a = re.sub('{.*}', '', a)
a = re.sub(r"\\.*? ",'',a)
a = re.sub(r"[^a-z ]",'',a)
tokens = nltk.word_tokenize(a)
tagged = nltk.pos_tag(tokens)
print(tagged)
for r in tagged:
	if not (r[1].startswith('J') or r[1].startswith('N') or r[1].startswith('V')): 
		a = re.sub(r[0],'',a)
print(a)
'''
a = re.sub('{.*}', '', a)
a = re.sub(r"\\.*? ",'',a)
print(a)
hamming = textdistance.hamming.normalized_similarity(a, b)
lvs = textdistance.levenshtein.normalized_similarity(a, b)
jw = textdistance.jaro_winkler.normalized_similarity(a, b)
jc = textdistance.jaccard.normalized_similarity(a.split(), b.split())
ros = textdistance.ratcliff_obershelp.normalized_similarity(a, b)
print(hamming)
print(lvs)
print(jw)
print(jc)
print(ros)

Ratio = fuzz.ratio(a.lower().split(),b.lower().split())
Partial_Ratio = fuzz.partial_ratio(a.lower().split(),b.lower().split())
Token_Sort_Ratio = fuzz.token_sort_ratio(a.split(),b.split())
Token_Set_Ratio = fuzz.token_set_ratio(a.split(),b.split())

print(Ratio)
print(Partial_Ratio)
print(Token_Sort_Ratio)
print(Token_Set_Ratio)
#print(df['Name'][1])
#print(df['REP'][1])
#print(df['cdesc'][1])
#length = textdistance.hamming(a, b)
#print(length)
'''