import pandas as pd
import numpy as np
from sklearn.utils import shuffle

get_id = int(input())
df = pd.read_csv("../allchest.csv")
df = df[df['ID'] == get_id]
df.reset_index(inplace=True, drop=True)
df = shuffle(df)
df.reset_index(inplace=True, drop=True)

train = df.iloc[:int(df.shape[0] * 0.2), :]
test = df.iloc[int(df.shape[0] * 0.2):, :]
del df

train.reset_index(inplace=True, drop=True)
test.reset_index(inplace=True, drop=True)
train = train[['label', 'ID', 'chestACCx', 'chestACCy', 'chestACCz', 'chestECG', 'chestEMG', 'chestEDA', 'chestTemp', 'chestResp']]
test = test[['label', 'ID', 'chestACCx', 'chestACCy', 'chestACCz', 'chestECG', 'chestEMG', 'chestEDA', 'chestTemp', 'chestResp']]

train = np.array(train, dtype=np.float32)
test = np.array(test, dtype=np.float32)

np.save("train.npy", train)
np.save("test.npy", test)
