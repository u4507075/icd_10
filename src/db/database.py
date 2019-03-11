import mysql.connector as sql
import pandas as pd

def convert(x):
    try:
        return x.encode('latin-1','replace').decode('tis-620','replace')
    except AttributeError:
        return x

def decode(df):
	for c in df.columns:
		if df[c].dtype == 'object':
			df[c] = df[c].apply(convert)
	return df
def getquery():
	# Open and read the file as a single buffer
	fd = open('../../../secret/query.sql', 'r')
	sql = fd.read()
	fd.close()
	return sql
def getdata(config):

	db_connection = sql.connect(	host=config.DATABASE_CONFIG['host'], 
											database=config.DATABASE_CONFIG['dbname'], 
											user=config.DATABASE_CONFIG['user'], 
											password=config.DATABASE_CONFIG['password'], 
											port=config.DATABASE_CONFIG['port'])
	df = pd.read_sql(getquery(), con=db_connection)
	df = decode(df)
	print(df)


