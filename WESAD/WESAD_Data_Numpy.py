import pandas as pd
import numpy as np
from sklearn.utils import shuffle

df = pd.read_csv("../allchest.csv")
df_list = list(map(lambda x: df[df['ID'] == x], [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]))
df_list = list(map(lambda x: x[x['label'] <= 4], df_list))
del df

train = []
test = []

for df in df_list:
	df.reset_index(inplace=True, drop=True)
	df = shuffle(df, random_state=42)
	df.reset_index(inplace=True, drop=True)
	train.append(df.iloc[:int(0.6 * df.shape[0]), :])
	test.append(df.iloc[int(0.6 * df.shape[0]):, :])
del df_list

train = pd.concat(train, axis=0)
test = pd.concat(test, axis=0)
train.reset_index(inplace=True, drop=True)
test.reset_index(inplace=True, drop=True)
train = train[['label', 'ID', 'chestACCx', 'chestACCy', 'chestACCz', 'chestECG', 'chestEMG', 'chestEDA', 'chestTemp', 'chestResp']]
test = test[['label', 'ID', 'chestACCx', 'chestACCy', 'chestACCz', 'chestECG', 'chestEMG', 'chestEDA', 'chestTemp', 'chestResp']]

train = np.array(train, dtype=np.float32)
test = np.array(test, dtype=np.float32)

np.save("train.npy", train)
np.save("test.npy", test)
