import pandas as pd 
from collections import Counter

def get_stats(df, name):
	with open('stats.txt', 'a+') as infile:
		if 'cstance' in name:
			stats = dict(Counter(df['stance']))
		else:
			stats = dict(Counter(df['label']))
		print("######", name, "Statistics######\n", stats, file = infile)


cstance_train = pd.read_csv("cstance_train.csv")
cstance_test = pd.read_csv("cstance_test.csv")
cmu_train = pd.read_csv("CMU_MisCov_train.csv")
cmu_test = pd.read_csv("CMU_MisCov_test.csv")
coaid_train = pd.read_csv("UnbalTrain_CoAid_News.csv")
coaid_test = pd.read_csv("UnbalTest_CoAid_News.csv")
reco_train = pd.read_csv("UnbalTrain_Recovery_News.csv")
reco_test = pd.read_csv("UnbalTest_Recovery_News.csv")

# For covid stance
get_stats(cstance_train, "cstance_train")
get_stats(cstance_test, "cstance_test")
get_stats(cmu_train, "cmu_train")
get_stats(cmu_test, "cmu_test")
get_stats(coaid_train, "coaid_train")
get_stats(coaid_test, "coaid_test")
get_stats(reco_train, "reco_train")
get_stats(reco_test, "reco_test")
