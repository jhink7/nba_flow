import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

csv = pd.read_csv('transformed_data/ws_quads_all.csv')
train, test = train_test_split(csv, train_size = 0.7)

train.to_csv("train_ws.csv", index=False)
test.to_csv("test_ws.csv", index=False)