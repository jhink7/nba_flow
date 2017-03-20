import pandas as pd
import numpy as np
#from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from modelling.marcel_projection_engine import MarcelProjectionEngine

#csv = pd.read_csv('transformed_data/ws_quads_all.csv')
#train, test = train_test_split(csv, train_size = 0.7)

#train.to_csv("train_ws.csv", index=False)
#test.to_csv("test_ws.csv", index=False)

train = pd.read_csv('transformed_data/train_ws.csv')
test = pd.read_csv('transformed_data/test_ws.csv')

training_set=train[(train.mp_target > 1000) & (train.mp_1 > 1000)]
marcel = MarcelProjectionEngine(train, 'mp', 'ws', True, 1/48.0)
projs = marcel.project_players(test)

# only evaluate for players who played 1000 or more minutes the target year
projs = projs[projs.mp_target > 1000]

pearson = projs['ws_target'].corr(projs['proj_marcel'])
mse = mean_squared_error(y_true = projs.ws_target, y_pred = projs.proj_marcel)
rmse = mse**(0.5)
mae = mean_absolute_error(y_true = projs.ws_target, y_pred = projs.proj_marcel)

print 'Marcel Performance'
print 'Rsq: {}'.format(pearson)
print 'rmse: {}'.format(rmse)
print 'mae: {}'.format(mae)



marcel = MarcelProjectionEngine(train, 'mp', 'ws', True, 1/48.0, False)
projs = marcel.project_players(test)

# only evaluate for players who played 1000 or more minutes the target year
projs = projs[projs.mp_target > 1000]

pearson = projs['ws_target'].corr(projs['proj_marcel'])
mse = mean_squared_error(y_true = projs.ws_target, y_pred = projs.proj_marcel)
rmse = mse**(0.5)
mae = mean_absolute_error(y_true = projs.ws_target, y_pred = projs.proj_marcel)

print ''
print 'Smarter Marcel Performance'
print 'Rsq: {}'.format(pearson)
print 'rmse: {}'.format(rmse)
print 'mae: {}'.format(mae)


#projs.to_csv('temp.csv')

